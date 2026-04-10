from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any

from .research_engine import LatestSignalPlan


@dataclass(slots=True)
class AlpacaBotConfig:
    api_key: str | None = os.environ.get("ALPACA_API_KEY")
    secret_key: str | None = os.environ.get("ALPACA_SECRET_KEY")
    paper: bool = True
    time_in_force: str = "day"
    options_feed: str = "indicative"
    expiration_slop_days: int = 4
    strike_slop_pct: float = 0.015
    contract_limit: int = 500
    limit_price_offset_pct: float = 0.03
    market_data_sandbox: bool = False


def _extract(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _normalize_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _normalize_signal_plan(signal_plan: LatestSignalPlan | dict[str, Any]) -> dict[str, Any]:
    if isinstance(signal_plan, LatestSignalPlan):
        return signal_plan.to_dict()
    return dict(signal_plan)


def _mid_price(quote: Any) -> float:
    bid = float(_extract(quote, "bid_price", 0.0) or 0.0)
    ask = float(_extract(quote, "ask_price", 0.0) or 0.0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if ask > 0:
        return ask
    return bid


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {key: _model_dump(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_model_dump(inner) for inner in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _collect_contract_rows(response: Any) -> list[dict[str, Any]]:
    contracts = _extract(response, "option_contracts", [])
    rows: list[dict[str, Any]] = []
    for contract in contracts:
        rows.append(
            {
                "symbol": str(_extract(contract, "symbol")),
                "strike_price": float(_extract(contract, "strike_price")),
                "expiration_date": _normalize_date(_extract(contract, "expiration_date")),
                "tradable": bool(_extract(contract, "tradable", True)),
            }
        )
    return rows


def _pick_nearest_contract(
    contracts: list[dict[str, Any]],
    *,
    target_strike: float,
    used_symbols: set[str],
) -> dict[str, Any] | None:
    ranked = sorted(
        (
            contract
            for contract in contracts
            if contract["tradable"] and contract["symbol"] not in used_symbols
        ),
        key=lambda contract: abs(contract["strike_price"] - target_strike),
    )
    return ranked[0] if ranked else None


def _select_credit_spread_contracts(
    contracts: list[dict[str, Any]],
    *,
    target_expiration: date,
    short_strike: float,
    long_strike: float,
) -> dict[str, Any]:
    grouped: dict[date, list[dict[str, Any]]] = {}
    for contract in contracts:
        grouped.setdefault(contract["expiration_date"], []).append(contract)

    best_selection: dict[str, Any] | None = None
    best_score = float("inf")
    for expiration_date, expiration_contracts in grouped.items():
        used_symbols: set[str] = set()
        short_contract = _pick_nearest_contract(
            expiration_contracts,
            target_strike=short_strike,
            used_symbols=used_symbols,
        )
        if short_contract is None:
            continue
        used_symbols.add(short_contract["symbol"])
        long_contract = _pick_nearest_contract(
            expiration_contracts,
            target_strike=long_strike,
            used_symbols=used_symbols,
        )
        if long_contract is None:
            continue

        expiration_gap = abs((expiration_date - target_expiration).days)
        strike_gap = abs(short_contract["strike_price"] - short_strike) + abs(long_contract["strike_price"] - long_strike)
        score = expiration_gap * 1000.0 + strike_gap
        if score < best_score:
            best_score = score
            best_selection = {
                "expiration_date": expiration_date,
                "short_contract": short_contract,
                "long_contract": long_contract,
                "expiration_gap_days": expiration_gap,
                "short_strike_gap": round(abs(short_contract["strike_price"] - short_strike), 4),
                "long_strike_gap": round(abs(long_contract["strike_price"] - long_strike), 4),
            }

    if best_selection is None:
        msg = "No Alpaca option chain contracts matched the requested spread closely enough."
        raise ValueError(msg)
    return best_selection


def build_alpaca_credit_spread_blueprint(
    signal_plan: LatestSignalPlan | dict[str, Any],
    config: AlpacaBotConfig | None = None,
    *,
    submit: bool = False,
) -> dict[str, Any]:
    config = config or AlpacaBotConfig()
    plan = _normalize_signal_plan(signal_plan)

    if plan["status"] not in {"accepted", "accepted_but_zero_contracts"}:
        msg = "The signal plan is not accepted, so there is no Alpaca order to build."
        raise ValueError(msg)
    if int(plan["contracts"]) < 1:
        msg = "The signal plan accepted the setup, but the contract count is zero."
        raise ValueError(msg)
    if plan["side"] not in {"bull_put", "bear_call"}:
        msg = "The Alpaca adapter currently supports the credit-spread live path only."
        raise ValueError(msg)
    if not config.api_key or not config.secret_key:
        msg = "Alpaca API credentials are required to resolve contracts and quotes."
        raise ValueError(msg)

    try:
        from alpaca.data.enums import OptionsFeed
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionLatestQuoteRequest
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import AssetStatus, ContractType, OrderClass, OrderSide, OrderType, PositionIntent, TimeInForce
        from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest, OptionLegRequest
    except ImportError as exc:
        msg = "alpaca-py is not installed. Install the optional live dependency before using the Alpaca adapter."
        raise RuntimeError(msg) from exc

    trading_client = TradingClient(
        api_key=config.api_key,
        secret_key=config.secret_key,
        paper=config.paper,
    )
    option_client = OptionHistoricalDataClient(
        api_key=config.api_key,
        secret_key=config.secret_key,
    )

    option_type = ContractType.PUT if plan["side"] == "bull_put" else ContractType.CALL
    target_expiration = date.fromisoformat(str(plan["target_expiration_date"]))
    short_strike = float(plan["short_strike"])
    long_strike = float(plan["long_strike"])
    lower_bound = min(short_strike, long_strike) * (1.0 - config.strike_slop_pct)
    upper_bound = max(short_strike, long_strike) * (1.0 + config.strike_slop_pct)

    contracts_request = GetOptionContractsRequest(
        underlying_symbols=[str(plan["symbol"])],
        status=AssetStatus.ACTIVE,
        type=option_type,
        strike_price_gte=str(round(lower_bound, 2)),
        strike_price_lte=str(round(upper_bound, 2)),
        expiration_date_gte=target_expiration - timedelta(days=config.expiration_slop_days),
        expiration_date_lte=target_expiration + timedelta(days=config.expiration_slop_days),
        limit=config.contract_limit,
    )
    contract_rows = _collect_contract_rows(trading_client.get_option_contracts(contracts_request))
    selection = _select_credit_spread_contracts(
        contract_rows,
        target_expiration=target_expiration,
        short_strike=short_strike,
        long_strike=long_strike,
    )

    quote_request = OptionLatestQuoteRequest(
        symbol_or_symbols=[
            selection["short_contract"]["symbol"],
            selection["long_contract"]["symbol"],
        ],
        feed=getattr(OptionsFeed, config.options_feed.upper()),
    )
    quotes = option_client.get_option_latest_quote(quote_request)
    short_quote = quotes[selection["short_contract"]["symbol"]]
    long_quote = quotes[selection["long_contract"]["symbol"]]

    short_mid = _mid_price(short_quote)
    long_mid = _mid_price(long_quote)
    net_credit = short_mid - long_mid
    if net_credit <= 0:
        msg = "The resolved Alpaca quotes do not currently produce a positive net credit for this spread."
        raise ValueError(msg)

    adjusted_credit = max(net_credit * (1.0 - config.limit_price_offset_pct), 0.01)
    limit_price = -round(adjusted_credit, 2)
    time_in_force = getattr(TimeInForce, config.time_in_force.upper())

    legs = [
        OptionLegRequest(
            symbol=selection["short_contract"]["symbol"],
            ratio_qty=1,
            side=OrderSide.SELL,
            position_intent=PositionIntent.SELL_TO_OPEN,
        ),
        OptionLegRequest(
            symbol=selection["long_contract"]["symbol"],
            ratio_qty=1,
            side=OrderSide.BUY,
            position_intent=PositionIntent.BUY_TO_OPEN,
        ),
    ]
    order_request = LimitOrderRequest(
        qty=int(plan["contracts"]),
        type=OrderType.LIMIT,
        time_in_force=time_in_force,
        order_class=OrderClass.MLEG,
        limit_price=limit_price,
        legs=legs,
    )

    submitted_order: Any | None = None
    if submit:
        submitted_order = trading_client.submit_order(order_data=order_request)

    return {
        "signal_plan": plan,
        "alpaca_config": {
            **asdict(config),
            "api_key": "***",
            "secret_key": "***",
        },
        "selection": _model_dump(selection),
        "quotes": {
            "short_leg": _model_dump(short_quote),
            "long_leg": _model_dump(long_quote),
        },
        "estimated_net_credit": round(net_credit, 4),
        "adjusted_net_credit": round(adjusted_credit, 4),
        "order_payload": {
            "qty": int(plan["contracts"]),
            "type": "limit",
            "time_in_force": config.time_in_force.lower(),
            "order_class": "mleg",
            "limit_price": limit_price,
            "legs": [
                {
                    "symbol": selection["short_contract"]["symbol"],
                    "ratio_qty": 1,
                    "side": "sell",
                    "position_intent": "sell_to_open",
                },
                {
                    "symbol": selection["long_contract"]["symbol"],
                    "ratio_qty": 1,
                    "side": "buy",
                    "position_intent": "buy_to_open",
                },
            ],
        },
        "submitted_order": _model_dump(submitted_order) if submitted_order is not None else None,
    }


def build_alpaca_credit_spread_close_blueprint(
    active_trade: dict[str, Any],
    config: AlpacaBotConfig | None = None,
    *,
    submit: bool = False,
) -> dict[str, Any]:
    config = config or AlpacaBotConfig()
    if not config.api_key or not config.secret_key:
        msg = "Alpaca API credentials are required to resolve quotes and close the spread."
        raise ValueError(msg)

    try:
        from alpaca.data.enums import OptionsFeed
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionLatestQuoteRequest
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderClass, OrderSide, OrderType, PositionIntent, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
    except ImportError as exc:
        msg = "alpaca-py is not installed. Install the optional live dependency before using the Alpaca adapter."
        raise RuntimeError(msg) from exc

    short_symbol = str(active_trade["short_symbol"])
    long_symbol = str(active_trade["long_symbol"])
    contracts = int(active_trade["contracts"])
    if contracts < 1:
        msg = "A close blueprint requires at least one contract."
        raise ValueError(msg)

    trading_client = TradingClient(
        api_key=config.api_key,
        secret_key=config.secret_key,
        paper=config.paper,
    )
    option_client = OptionHistoricalDataClient(
        api_key=config.api_key,
        secret_key=config.secret_key,
    )

    quote_request = OptionLatestQuoteRequest(
        symbol_or_symbols=[short_symbol, long_symbol],
        feed=getattr(OptionsFeed, config.options_feed.upper()),
    )
    quotes = option_client.get_option_latest_quote(quote_request)
    short_quote = quotes[short_symbol]
    long_quote = quotes[long_symbol]

    short_mid = _mid_price(short_quote)
    long_mid = _mid_price(long_quote)
    net_debit = max(short_mid - long_mid, 0.01)
    adjusted_debit = max(net_debit * (1.0 + config.limit_price_offset_pct), 0.01)
    limit_price = round(adjusted_debit, 2)
    time_in_force = getattr(TimeInForce, config.time_in_force.upper())

    legs = [
        OptionLegRequest(
            symbol=short_symbol,
            ratio_qty=1,
            side=OrderSide.BUY,
            position_intent=PositionIntent.BUY_TO_CLOSE,
        ),
        OptionLegRequest(
            symbol=long_symbol,
            ratio_qty=1,
            side=OrderSide.SELL,
            position_intent=PositionIntent.SELL_TO_CLOSE,
        ),
    ]
    order_request = LimitOrderRequest(
        qty=contracts,
        type=OrderType.LIMIT,
        time_in_force=time_in_force,
        order_class=OrderClass.MLEG,
        limit_price=limit_price,
        legs=legs,
    )

    submitted_order: Any | None = None
    if submit:
        submitted_order = trading_client.submit_order(order_data=order_request)

    return {
        "active_trade": active_trade,
        "alpaca_config": {
            **asdict(config),
            "api_key": "***",
            "secret_key": "***",
        },
        "quotes": {
            "short_leg": _model_dump(short_quote),
            "long_leg": _model_dump(long_quote),
        },
        "estimated_net_debit": round(net_debit, 4),
        "adjusted_net_debit": round(adjusted_debit, 4),
        "order_payload": {
            "qty": contracts,
            "type": "limit",
            "time_in_force": config.time_in_force.lower(),
            "order_class": "mleg",
            "limit_price": limit_price,
            "legs": [
                {
                    "symbol": short_symbol,
                    "ratio_qty": 1,
                    "side": "buy",
                    "position_intent": "buy_to_close",
                },
                {
                    "symbol": long_symbol,
                    "ratio_qty": 1,
                    "side": "sell",
                    "position_intent": "sell_to_close",
                },
            ],
        },
        "submitted_order": _model_dump(submitted_order) if submitted_order is not None else None,
    }
