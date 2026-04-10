from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .alpaca_adapter import (
    AlpacaBotConfig,
    build_alpaca_credit_spread_blueprint,
    build_alpaca_credit_spread_close_blueprint,
)
from .research_engine import HeavyCreditSpreadResearchEngine, LatestSignalPlan


EASTERN_TZ = ZoneInfo("America/New_York")


@dataclass(slots=True)
class LiveBotConfig:
    api_key: str
    secret_key: str
    symbol: str
    model_path: str
    metadata_path: str
    paper: bool = True
    cycle_seconds: int = 60
    history_days: int = 540
    account_equity_override: float | None = None
    stock_feed: str = "iex"
    options_feed: str = "indicative"
    time_in_force: str = "day"
    control_token: str | None = None
    state_path: str = "outputs2/deployment/live_bot_state.json"
    event_log_path: str = "outputs2/deployment/live_bot_events.jsonl"
    entry_start_minute_after_open: int = 5
    entry_end_minute_after_open: int = 210


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _to_python(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {str(key): _to_python(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_python(inner) for inner in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return value


def _normalize_order_status(order: Any) -> str:
    return str(_extract(order, "status", "")).split(".")[-1].lower()


def _timestamp_to_eastern(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert(EASTERN_TZ)


class AlpacaRealtimeBot:
    def __init__(self, config: LiveBotConfig) -> None:
        self.config = config
        self.engine, self.bundle, self.metadata = HeavyCreditSpreadResearchEngine.from_saved_model_artifacts(
            model_path=config.model_path,
            metadata_path=config.metadata_path,
        )
        self._sdk: dict[str, Any] | None = None
        self._trading_client: Any | None = None
        self._stock_data_client: Any | None = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._run_now_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = {
            "status": "stopped",
            "symbol": config.symbol,
            "paper": config.paper,
            "cycle_seconds": config.cycle_seconds,
            "paused": False,
            "next_cycle_at": None,
            "last_cycle_started_at": None,
            "last_cycle_finished_at": None,
            "last_cycle_reason": None,
            "last_cycle_outcome": None,
            "last_error": None,
            "cycle_count": 0,
            "connection": {},
            "market": {},
            "account": {},
            "latest_signal_plan": None,
            "active_trade": None,
            "open_orders": [],
            "last_entry_signal_date": None,
        }
        self._restore_persisted_state()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._state["status"] = "starting"
            self._persist_state_locked()
        self._ensure_clients()
        self._validate_connections()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="alpaca-live-bot", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        with self._lock:
            self._state["status"] = "stopped"
            self._persist_state_locked()

    def pause(self) -> None:
        with self._lock:
            self._state["paused"] = True
            self._persist_state_locked()

    def resume(self) -> None:
        with self._lock:
            self._state["paused"] = False
            self._persist_state_locked()
        self.request_run_now()

    def request_run_now(self) -> None:
        self._run_now_event.set()
        with self._lock:
            self._state["next_cycle_at"] = _utc_now()
            self._persist_state_locked()

    def snapshot_state(self) -> dict[str, Any]:
        with self._lock:
            snapshot = json.loads(json.dumps(_to_python(self._state)))
        next_cycle_at = snapshot.get("next_cycle_at")
        if next_cycle_at:
            seconds_until = max(
                int((pd.Timestamp(next_cycle_at) - pd.Timestamp.now(tz="UTC")).total_seconds()),
                0,
            )
        else:
            seconds_until = None
        snapshot["seconds_until_next_cycle"] = seconds_until
        return snapshot

    def _load_sdk(self) -> dict[str, Any]:
        if self._sdk is None:
            from alpaca.data.enums import Adjustment, DataFeed
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from alpaca.trading.client import TradingClient
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrderByIdRequest, GetOrdersRequest

            self._sdk = {
                "Adjustment": Adjustment,
                "DataFeed": DataFeed,
                "GetOrderByIdRequest": GetOrderByIdRequest,
                "GetOrdersRequest": GetOrdersRequest,
                "QueryOrderStatus": QueryOrderStatus,
                "StockBarsRequest": StockBarsRequest,
                "StockHistoricalDataClient": StockHistoricalDataClient,
                "TimeFrame": TimeFrame,
                "TradingClient": TradingClient,
            }
        return self._sdk

    def _ensure_clients(self) -> None:
        sdk = self._load_sdk()
        if self._trading_client is None:
            self._trading_client = sdk["TradingClient"](
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper=self.config.paper,
            )
        if self._stock_data_client is None:
            self._stock_data_client = sdk["StockHistoricalDataClient"](
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
            )

    def _restore_persisted_state(self) -> None:
        path = Path(self.config.state_path)
        if not path.exists():
            return
        persisted = json.loads(path.read_text(encoding="utf-8"))
        for key in ("active_trade", "last_entry_signal_date"):
            if key in persisted:
                self._state[key] = persisted[key]

    def _persist_state_locked(self) -> None:
        path = Path(self.config.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_to_python(self._state), indent=2), encoding="utf-8")

    def _append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        path = Path(self.config.event_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": _utc_now(),
            "event_type": event_type,
            "payload": _to_python(payload),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def _validate_connections(self) -> None:
        account = self._trading_client.get_account()
        clock = self._trading_client.get_clock()
        price_data = self._fetch_stock_history()
        with self._lock:
            self._state["connection"] = {
                "validated_at": _utc_now(),
                "account_id": str(_extract(account, "id", "")),
                "account_status": str(_extract(account, "status", "")),
                "clock_is_open": bool(_extract(clock, "is_open", False)),
                "latest_bar_date": str(price_data.index[-1].date()),
            }
            self._persist_state_locked()

    def _loop(self) -> None:
        with self._lock:
            self._state["status"] = "running"
            self._state["next_cycle_at"] = _utc_now()
            self._persist_state_locked()

        while not self._stop_event.is_set():
            with self._lock:
                paused = bool(self._state["paused"])
                next_cycle_at = self._state["next_cycle_at"]

            if paused and not self._run_now_event.is_set():
                self._stop_event.wait(1.0)
                continue

            if self._run_now_event.is_set():
                self._run_now_event.clear()
                self._run_cycle("manual")
                continue

            if next_cycle_at is None or pd.Timestamp.now(tz="UTC") >= pd.Timestamp(next_cycle_at):
                self._run_cycle("scheduled")
                continue

            wait_seconds = min(
                max((pd.Timestamp(next_cycle_at) - pd.Timestamp.now(tz="UTC")).total_seconds(), 0.0),
                1.0,
            )
            self._stop_event.wait(wait_seconds)

    def _run_cycle(self, reason: str) -> None:
        started_at = _utc_now()
        with self._lock:
            self._state["last_cycle_started_at"] = started_at
            self._state["last_cycle_reason"] = reason
            self._state["last_error"] = None
            self._persist_state_locked()

        outcome = "ok"
        try:
            clock = self._trading_client.get_clock()
            account = self._trading_client.get_account()
            open_orders = self._fetch_open_orders()
            price_data = self._fetch_stock_history()
            signal_plan = self.engine.build_signal_plan_from_price_data(
                price_data=price_data,
                model_path=self.config.model_path,
                metadata_path=self.config.metadata_path,
                bundle=self.bundle,
                account_equity=self._resolve_account_equity(account),
            )

            with self._lock:
                self._state["market"] = self._clock_summary(clock)
                self._state["account"] = self._account_summary(account)
                self._state["open_orders"] = open_orders
                self._state["latest_signal_plan"] = signal_plan.to_dict()
                self._persist_state_locked()

            self._refresh_active_trade()
            self._maybe_submit_exit(clock=clock, price_data=price_data)
            self._maybe_submit_entry(clock=clock, signal_plan=signal_plan)
        except Exception as exc:
            outcome = "error"
            with self._lock:
                self._state["last_error"] = str(exc)
                self._persist_state_locked()
            self._append_event("cycle_error", {"reason": reason, "error": str(exc)})
        finally:
            with self._lock:
                self._state["cycle_count"] = int(self._state["cycle_count"]) + 1
                self._state["last_cycle_finished_at"] = _utc_now()
                self._state["last_cycle_outcome"] = outcome
                self._state["next_cycle_at"] = (datetime.now(timezone.utc) + timedelta(seconds=self.config.cycle_seconds)).isoformat()
                self._persist_state_locked()

    def _fetch_open_orders(self) -> list[dict[str, Any]]:
        sdk = self._load_sdk()
        request = sdk["GetOrdersRequest"](
            status=sdk["QueryOrderStatus"].OPEN,
            nested=True,
            limit=50,
        )
        orders = self._trading_client.get_orders(filter=request)
        return [
            {
                "id": str(_extract(order, "id", "")),
                "client_order_id": str(_extract(order, "client_order_id", "")),
                "status": _normalize_order_status(order),
                "submitted_at": str(_extract(order, "submitted_at", "")),
            }
            for order in orders
        ]

    def _fetch_stock_history(self) -> pd.DataFrame:
        sdk = self._load_sdk()
        request = sdk["StockBarsRequest"](
            symbol_or_symbols=[self.config.symbol],
            timeframe=sdk["TimeFrame"].Day,
            start=datetime.now(timezone.utc) - timedelta(days=self.config.history_days),
            end=datetime.now(timezone.utc),
            adjustment=sdk["Adjustment"].ALL,
            feed=getattr(sdk["DataFeed"], self.config.stock_feed.upper()),
        )
        response = self._stock_data_client.get_stock_bars(request)
        frame = response.df if hasattr(response, "df") else pd.DataFrame()
        if frame.empty:
            msg = f"No Alpaca stock bars returned for {self.config.symbol}."
            raise ValueError(msg)

        if isinstance(frame.index, pd.MultiIndex):
            try:
                frame = frame.xs(self.config.symbol, level=0)
            except KeyError:
                frame = frame.droplevel(0)
        if "timestamp" in frame.columns:
            frame = frame.set_index("timestamp")
        frame = frame.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        frame = frame.loc[:, ["open", "high", "low", "close", "volume"]].copy()
        frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(EASTERN_TZ).tz_localize(None)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        return frame

    def _resolve_account_equity(self, account: Any) -> float:
        if self.config.account_equity_override is not None:
            return float(self.config.account_equity_override)
        return float(_extract(account, "equity", 0.0) or 0.0)

    def _account_summary(self, account: Any) -> dict[str, Any]:
        return {
            "equity": float(_extract(account, "equity", 0.0) or 0.0),
            "cash": float(_extract(account, "cash", 0.0) or 0.0),
            "buying_power": float(_extract(account, "buying_power", 0.0) or 0.0),
            "status": str(_extract(account, "status", "")),
        }

    def _clock_summary(self, clock: Any) -> dict[str, Any]:
        return {
            "timestamp": str(_extract(clock, "timestamp", "")),
            "is_open": bool(_extract(clock, "is_open", False)),
            "next_open": str(_extract(clock, "next_open", "")),
            "next_close": str(_extract(clock, "next_close", "")),
        }

    def _within_entry_window(self, clock: Any) -> bool:
        if not bool(_extract(clock, "is_open", False)):
            return False
        timestamp = _timestamp_to_eastern(_extract(clock, "timestamp"))
        market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        minutes_after_open = int((timestamp - market_open).total_seconds() // 60)
        return self.config.entry_start_minute_after_open <= minutes_after_open <= self.config.entry_end_minute_after_open

    def _get_order_by_id(self, order_id: str) -> Any:
        sdk = self._load_sdk()
        return self._trading_client.get_order_by_id(
            order_id,
            filter=sdk["GetOrderByIdRequest"](nested=True),
        )

    def _current_position_symbols(self) -> set[str]:
        positions = self._trading_client.get_all_positions()
        return {str(_extract(position, "symbol", "")) for position in positions}

    def _refresh_active_trade(self) -> None:
        with self._lock:
            active_trade = self._state.get("active_trade")
        if not active_trade:
            return

        status = str(active_trade.get("status", ""))
        if status == "pending_entry":
            order = self._get_order_by_id(str(active_trade["entry_order_id"]))
            order_status = _normalize_order_status(order)
            if order_status == "filled":
                entry_fill_credit = abs(float(_extract(order, "filled_avg_price", active_trade["entry_target_credit"]) or active_trade["entry_target_credit"]))
                with self._lock:
                    active_trade["status"] = "open"
                    active_trade["entry_fill_credit"] = round(entry_fill_credit, 4)
                    active_trade["opened_at"] = _utc_now()
                    self._state["active_trade"] = active_trade
                    self._persist_state_locked()
                self._append_event("entry_filled", active_trade)
            elif order_status in {"canceled", "cancelled", "rejected", "expired"}:
                with self._lock:
                    self._state["active_trade"] = None
                    self._persist_state_locked()
                self._append_event("entry_aborted", {"order_status": order_status, "active_trade": active_trade})
        elif status == "pending_exit":
            order = self._get_order_by_id(str(active_trade["exit_order_id"]))
            order_status = _normalize_order_status(order)
            if order_status == "filled":
                with self._lock:
                    self._state["active_trade"] = None
                    self._persist_state_locked()
                self._append_event("exit_filled", active_trade)
            elif order_status in {"canceled", "cancelled", "rejected", "expired"}:
                with self._lock:
                    active_trade["status"] = "open"
                    active_trade["exit_order_id"] = None
                    self._state["active_trade"] = active_trade
                    self._persist_state_locked()
                self._append_event("exit_aborted", {"order_status": order_status, "active_trade": active_trade})
        elif status == "open":
            position_symbols = self._current_position_symbols()
            if active_trade["short_symbol"] not in position_symbols and active_trade["long_symbol"] not in position_symbols:
                with self._lock:
                    self._state["active_trade"] = None
                    self._persist_state_locked()
                self._append_event("manual_position_change_detected", active_trade)

    def _business_days_to_expiration(self, expiration_date: str) -> int:
        today = pd.Timestamp.now(tz=EASTERN_TZ).tz_localize(None).normalize()
        expiry = pd.Timestamp(expiration_date).normalize()
        if expiry <= today:
            return 0
        return max(len(pd.bdate_range(today, expiry)) - 1, 0)

    def _maybe_submit_entry(self, *, clock: Any, signal_plan: LatestSignalPlan) -> None:
        with self._lock:
            active_trade = self._state.get("active_trade")
            last_entry_signal_date = self._state.get("last_entry_signal_date")
        if active_trade is not None:
            return
        if signal_plan.status not in {"accepted", "accepted_but_zero_contracts"} or signal_plan.contracts < 1:
            return
        if last_entry_signal_date == signal_plan.as_of_date:
            return
        if not self._within_entry_window(clock):
            return

        blueprint = build_alpaca_credit_spread_blueprint(
            signal_plan,
            AlpacaBotConfig(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper=self.config.paper,
                time_in_force=self.config.time_in_force,
                options_feed=self.config.options_feed,
            ),
            submit=True,
        )
        submitted_order = blueprint.get("submitted_order") or {}
        if not submitted_order.get("id"):
            msg = "Alpaca entry order submission did not return an order id."
            raise ValueError(msg)
        active_trade = {
            "status": "pending_entry",
            "symbol": signal_plan.symbol,
            "signal_date": signal_plan.as_of_date,
            "side": signal_plan.side,
            "contracts": signal_plan.contracts,
            "short_strike": signal_plan.short_strike,
            "long_strike": signal_plan.long_strike,
            "expiration_date": str(blueprint["selection"]["expiration_date"]),
            "short_symbol": blueprint["selection"]["short_contract"]["symbol"],
            "long_symbol": blueprint["selection"]["long_contract"]["symbol"],
            "entry_target_credit": signal_plan.target_entry_credit,
            "entry_order_id": submitted_order.get("id"),
            "entry_submitted_at": _utc_now(),
            "max_loss_per_spread": signal_plan.max_loss_per_spread,
            "order_payload": blueprint["order_payload"],
        }
        with self._lock:
            self._state["active_trade"] = active_trade
            self._state["last_entry_signal_date"] = signal_plan.as_of_date
            self._persist_state_locked()
        self._append_event("entry_submitted", active_trade)

    def _maybe_submit_exit(self, *, clock: Any, price_data: pd.DataFrame) -> None:
        with self._lock:
            active_trade = self._state.get("active_trade")
        if not active_trade or active_trade.get("status") != "open":
            return
        if not bool(_extract(clock, "is_open", False)):
            return

        strategy_config = self.engine.strategy_config
        spot = float(price_data["close"].iloc[-1])
        ema10 = float(price_data["close"].ewm(span=10, adjust=False).mean().iloc[-1])
        close_blueprint = build_alpaca_credit_spread_close_blueprint(
            active_trade,
            AlpacaBotConfig(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper=self.config.paper,
                time_in_force=self.config.time_in_force,
                options_feed=self.config.options_feed,
            ),
            submit=False,
        )
        mark = float(close_blueprint["estimated_net_debit"])
        entry_credit = float(active_trade.get("entry_fill_credit") or active_trade.get("entry_target_credit") or 0.0)
        contracts = int(active_trade["contracts"])
        max_loss_dollars = float(active_trade["max_loss_per_spread"]) * contracts
        pnl_dollars = (entry_credit - mark) * 100.0 * contracts
        days_left = self._business_days_to_expiration(str(active_trade["expiration_date"]))
        trend_broken = spot < ema10 if active_trade["side"] == "bull_put" else spot > ema10
        should_exit = (
            mark <= entry_credit * strategy_config.take_profit_remaining_credit_pct
            or pnl_dollars <= -max_loss_dollars * strategy_config.stop_loss_fraction_of_max_loss
            or days_left <= strategy_config.exit_dte
            or trend_broken
        )
        if not should_exit:
            with self._lock:
                active_trade["last_mark"] = round(mark, 4)
                active_trade["days_left"] = days_left
                self._state["active_trade"] = active_trade
                self._persist_state_locked()
            return

        submitted_blueprint = build_alpaca_credit_spread_close_blueprint(
            active_trade,
            AlpacaBotConfig(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper=self.config.paper,
                time_in_force=self.config.time_in_force,
                options_feed=self.config.options_feed,
            ),
            submit=True,
        )
        submitted_order = submitted_blueprint.get("submitted_order") or {}
        if not submitted_order.get("id"):
            msg = "Alpaca exit order submission did not return an order id."
            raise ValueError(msg)
        with self._lock:
            active_trade["status"] = "pending_exit"
            active_trade["exit_order_id"] = submitted_order.get("id")
            active_trade["exit_reason"] = {
                "mark": round(mark, 4),
                "spot": round(spot, 4),
                "days_left": days_left,
                "trend_broken": trend_broken,
            }
            self._state["active_trade"] = active_trade
            self._persist_state_locked()
        self._append_event("exit_submitted", active_trade)
