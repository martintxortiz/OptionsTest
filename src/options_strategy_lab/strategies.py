from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import fetch_price_history
from .pricing import (
    bear_call_credit_spread_value,
    black_scholes_price,
    bull_put_credit_spread_value,
)
from .reports import build_metrics


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0).rolling(window).mean()
    losses = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gains / losses.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def _realized_volatility(close: pd.Series, lookback: int = 20) -> pd.Series:
    returns = close.pct_change()
    return returns.rolling(lookback).std() * np.sqrt(252.0)


def _mark_iv(realized_vol: float, minimum: float, maximum: float, multiplier: float) -> float:
    base = multiplier * realized_vol if np.isfinite(realized_vol) else minimum
    return float(np.clip(base, minimum, maximum))


@dataclass(slots=True)
class CreditSpreadConfig:
    symbol: str = "SPY"
    start: str = "2020-01-01"
    end: str = "2026-04-09"
    initial_capital: float = 100_000.0
    risk_fraction: float = 0.15
    option_dte: int = 30
    exit_dte: int = 7
    bull_rsi_threshold: float = 35.0
    bear_rsi_threshold: float = 65.0
    bull_short_strike_pct: float = 0.96
    bull_long_strike_pct: float = 0.92
    bear_short_strike_pct: float = 1.04
    bear_long_strike_pct: float = 1.08
    implied_vol_multiplier: float = 1.15
    min_implied_vol: float = 0.18
    max_implied_vol: float = 0.55
    entry_slippage_pct: float = 0.03
    take_profit_remaining_credit_pct: float = 0.35
    stop_loss_fraction_of_max_loss: float = 0.60
    per_contract_fee: float = 1.0
    risk_free_rate: float = 0.04


def make_credit_spread_config(
    preset: str = "baseline",
    *,
    symbol: str = "SPY",
    start: str = "2020-01-01",
    end: str = "2026-04-09",
    initial_capital: float = 100_000.0,
    risk_fraction: float | None = None,
) -> CreditSpreadConfig:
    if preset == "baseline":
        config = CreditSpreadConfig(
            symbol=symbol,
            start=start,
            end=end,
            initial_capital=initial_capital,
            risk_fraction=0.15,
        )
    elif preset == "optimized":
        config = CreditSpreadConfig(
            symbol=symbol,
            start=start,
            end=end,
            initial_capital=initial_capital,
            risk_fraction=0.25,
            option_dte=14,
            exit_dte=3,
            bull_rsi_threshold=40.0,
            bear_rsi_threshold=75.0,
            bull_short_strike_pct=0.98,
            bull_long_strike_pct=0.93,
            bear_short_strike_pct=1.04,
            bear_long_strike_pct=1.06,
            implied_vol_multiplier=1.15,
            max_implied_vol=0.55,
            take_profit_remaining_credit_pct=0.25,
            stop_loss_fraction_of_max_loss=0.40,
        )
    elif preset == "max_return":
        config = CreditSpreadConfig(
            symbol=symbol,
            start=start,
            end=end,
            initial_capital=initial_capital,
            risk_fraction=1.00,
            option_dte=14,
            exit_dte=3,
            bull_rsi_threshold=40.0,
            bear_rsi_threshold=75.0,
            bull_short_strike_pct=0.98,
            bull_long_strike_pct=0.93,
            bear_short_strike_pct=1.04,
            bear_long_strike_pct=1.06,
            implied_vol_multiplier=1.15,
            max_implied_vol=0.55,
            take_profit_remaining_credit_pct=0.25,
            stop_loss_fraction_of_max_loss=0.40,
        )
    else:
        msg = f"Unknown credit spread preset: {preset}"
        raise ValueError(msg)

    if risk_fraction is not None:
        config.risk_fraction = risk_fraction
    return config


@dataclass(slots=True)
class AggressiveLongCallBreakoutConfig:
    trade_symbol: str = "TQQQ"
    regime_symbol: str = "QQQ"
    start: str = "2020-01-01"
    end: str = "2026-04-09"
    initial_capital: float = 100_000.0
    risk_fraction: float = 0.15
    option_dte: int = 45
    exit_dte: int = 5
    strike_pct_of_spot: float = 0.98
    breakout_lookback: int = 10
    max_hold_days: int = 16
    take_profit_pct: float = 1.60
    stop_loss_pct: float = -0.55
    implied_vol_multiplier: float = 1.20
    min_implied_vol: float = 0.35
    max_implied_vol: float = 1.10
    entry_slippage_pct: float = 0.03
    exit_slippage_pct: float = 0.02
    per_contract_fee: float = 1.0
    risk_free_rate: float = 0.04


def run_credit_spread_backtest(config: CreditSpreadConfig) -> dict[str, object]:
    price_data = fetch_price_history(config.symbol, config.start, config.end)
    close = price_data["close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()
    rsi3 = _rsi(close, 3)
    rv20 = _realized_volatility(close, 20)

    cash = config.initial_capital
    equity_points: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict[str, float | str]] = []
    position: dict[str, float | int | pd.Timestamp | str] | None = None

    for date in close.index:
        spot = float(close.loc[date])
        implied_vol = _mark_iv(
            realized_vol=float(rv20.loc[date]) if pd.notna(rv20.loc[date]) else np.nan,
            minimum=config.min_implied_vol,
            maximum=config.max_implied_vol,
            multiplier=config.implied_vol_multiplier,
        )

        if position is not None:
            position["days_left"] = int(position["days_left"]) - 1
            side = str(position["side"])
            if side == "bull_put":
                mark = bull_put_credit_spread_value(
                    spot=spot,
                    short_strike=float(position["short_strike"]),
                    long_strike=float(position["long_strike"]),
                    days_to_expiry=int(position["days_left"]),
                    implied_volatility=implied_vol,
                    risk_free_rate=config.risk_free_rate,
                )
                trend_broken = spot < float(ema10.loc[date])
            else:
                mark = bear_call_credit_spread_value(
                    spot=spot,
                    short_strike=float(position["short_strike"]),
                    long_strike=float(position["long_strike"]),
                    days_to_expiry=int(position["days_left"]),
                    implied_volatility=implied_vol,
                    risk_free_rate=config.risk_free_rate,
                )
                trend_broken = spot > float(ema10.loc[date])

            contracts = int(position["contracts"])
            entry_credit = float(position["entry_credit"])
            max_loss_dollars = float(position["max_loss_dollars"])
            pnl_dollars = (entry_credit - mark) * 100.0 * contracts

            should_exit = (
                mark <= entry_credit * config.take_profit_remaining_credit_pct
                or pnl_dollars <= -max_loss_dollars * config.stop_loss_fraction_of_max_loss
                or int(position["days_left"]) <= config.exit_dte
                or trend_broken
            )
            if should_exit:
                exit_cost = mark * 100.0 * contracts + config.per_contract_fee * contracts
                cash -= exit_cost
                net_pnl = float(position["entry_cash_received"]) - exit_cost
                trades.append(
                    {
                        "entry_date": str(position["entry_date"]),
                        "exit_date": str(date.date()),
                        "side": side,
                        "contracts": contracts,
                        "short_strike": float(position["short_strike"]),
                        "long_strike": float(position["long_strike"]),
                        "entry_credit": round(entry_credit, 4),
                        "exit_debit": round(mark, 4),
                        "max_loss_dollars": round(max_loss_dollars, 2),
                        "net_pnl": round(net_pnl, 2),
                        "return_on_risk_pct": round((net_pnl / max_loss_dollars) * 100.0, 2) if max_loss_dollars > 0 else 0.0,
                    }
                )
                position = None

        if (
            position is None
            and pd.notna(sma200.loc[date])
            and pd.notna(sma50.loc[date])
            and pd.notna(rsi3.loc[date])
        ):
            side: str | None = None
            short_strike = 0.0
            long_strike = 0.0
            entry_value = 0.0
            width = 0.0

            if spot > float(sma50.loc[date]) > float(sma200.loc[date]) and float(rsi3.loc[date]) < config.bull_rsi_threshold:
                side = "bull_put"
                short_strike = round(spot * config.bull_short_strike_pct)
                long_strike = round(spot * config.bull_long_strike_pct)
                if long_strike >= short_strike:
                    long_strike = short_strike - 4
                width = short_strike - long_strike
                entry_value = bull_put_credit_spread_value(
                    spot=spot,
                    short_strike=short_strike,
                    long_strike=long_strike,
                    days_to_expiry=config.option_dte,
                    implied_volatility=implied_vol,
                    risk_free_rate=config.risk_free_rate,
                )
            elif spot < float(sma50.loc[date]) < float(sma200.loc[date]) and float(rsi3.loc[date]) > config.bear_rsi_threshold:
                side = "bear_call"
                short_strike = round(spot * config.bear_short_strike_pct)
                long_strike = round(spot * config.bear_long_strike_pct)
                if long_strike <= short_strike:
                    long_strike = short_strike + 4
                width = long_strike - short_strike
                entry_value = bear_call_credit_spread_value(
                    spot=spot,
                    short_strike=short_strike,
                    long_strike=long_strike,
                    days_to_expiry=config.option_dte,
                    implied_volatility=implied_vol,
                    risk_free_rate=config.risk_free_rate,
                )

            if side is not None:
                entry_credit = entry_value * (1.0 - config.entry_slippage_pct)
                max_loss_per_spread = (width - entry_credit) * 100.0
                if entry_credit > 0.2 and max_loss_per_spread > 0:
                    contracts = int((cash * config.risk_fraction) // max_loss_per_spread)
                    if contracts >= 1:
                        entry_cash = entry_credit * 100.0 * contracts - config.per_contract_fee * contracts
                        cash += entry_cash
                        position = {
                            "entry_date": date.date(),
                            "side": side,
                            "contracts": contracts,
                            "short_strike": short_strike,
                            "long_strike": long_strike,
                            "entry_credit": entry_credit,
                            "days_left": config.option_dte,
                            "width": width,
                            "max_loss_dollars": max_loss_per_spread * contracts,
                            "entry_cash_received": entry_cash,
                        }

        marked_equity = cash
        if position is not None:
            side = str(position["side"])
            if side == "bull_put":
                liability = bull_put_credit_spread_value(
                    spot=spot,
                    short_strike=float(position["short_strike"]),
                    long_strike=float(position["long_strike"]),
                    days_to_expiry=int(position["days_left"]),
                    implied_volatility=implied_vol,
                    risk_free_rate=config.risk_free_rate,
                )
            else:
                liability = bear_call_credit_spread_value(
                    spot=spot,
                    short_strike=float(position["short_strike"]),
                    long_strike=float(position["long_strike"]),
                    days_to_expiry=int(position["days_left"]),
                    implied_volatility=implied_vol,
                    risk_free_rate=config.risk_free_rate,
                )
            marked_equity -= liability * 100.0 * int(position["contracts"])
        equity_points.append((date, marked_equity))

    equity_curve = pd.Series(
        data=[value for _, value in equity_points],
        index=[date for date, _ in equity_points],
        dtype=float,
    )
    trades_df = pd.DataFrame(trades)
    metrics = build_metrics(equity_curve=equity_curve, trades=trades_df, initial_capital=config.initial_capital)

    return {
        "strategy_name": "credit_spread",
        "config": asdict(config),
        "equity_curve": equity_curve,
        "trades": trades_df,
        "metrics": metrics,
    }


def run_aggressive_long_call_breakout(config: AggressiveLongCallBreakoutConfig) -> dict[str, object]:
    regime_data = fetch_price_history(config.regime_symbol, config.start, config.end)
    trade_data = fetch_price_history(config.trade_symbol, config.start, config.end)
    joined = trade_data.join(
        regime_data["close"].rename("regime_close"),
        how="inner",
    )

    close = joined["close"]
    regime_close = joined["regime_close"]
    regime_sma200 = regime_close.rolling(200).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    breakout = close > close.shift(1).rolling(config.breakout_lookback).max()
    pullback = close.shift(1) < ema20.shift(1)
    rv20 = _realized_volatility(close, 20)

    cash = config.initial_capital
    equity_points: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict[str, float | str]] = []
    position: dict[str, float | int | pd.Timestamp] | None = None

    for date in joined.index:
        spot = float(close.loc[date])
        implied_vol = _mark_iv(
            realized_vol=float(rv20.loc[date]) if pd.notna(rv20.loc[date]) else np.nan,
            minimum=config.min_implied_vol,
            maximum=config.max_implied_vol,
            multiplier=config.implied_vol_multiplier,
        )

        if position is not None:
            position["days_left"] = int(position["days_left"]) - 1
            position["days_held"] = int(position["days_held"]) + 1
            mark = black_scholes_price(
                spot=spot,
                strike=float(position["strike"]),
                years_to_expiry=max(int(position["days_left"]), 0) / 252.0,
                risk_free_rate=config.risk_free_rate,
                volatility=implied_vol,
                option_type="call",
            )
            pnl_pct = (mark - float(position["entry_price"])) / float(position["entry_price"])
            should_exit = (
                pnl_pct >= config.take_profit_pct
                or pnl_pct <= config.stop_loss_pct
                or int(position["days_left"]) <= config.exit_dte
                or int(position["days_held"]) >= config.max_hold_days
                or spot < float(ema10.loc[date])
            )
            if should_exit:
                contracts = int(position["contracts"])
                proceeds = mark * (1.0 - config.exit_slippage_pct) * 100.0 * contracts - config.per_contract_fee * contracts
                cash += proceeds
                net_pnl = proceeds - float(position["entry_cost"])
                trades.append(
                    {
                        "entry_date": str(position["entry_date"]),
                        "exit_date": str(date.date()),
                        "contracts": contracts,
                        "strike": float(position["strike"]),
                        "entry_price": round(float(position["entry_price"]), 4),
                        "exit_price": round(mark, 4),
                        "net_pnl": round(net_pnl, 2),
                    }
                )
                position = None

        if (
            position is None
            and pd.notna(regime_sma200.loc[date])
            and float(regime_close.loc[date]) > float(regime_sma200.loc[date])
            and bool(breakout.loc[date])
            and bool(pullback.loc[date])
            and spot > float(ema20.loc[date])
        ):
            strike = round(spot * config.strike_pct_of_spot)
            option_price = black_scholes_price(
                spot=spot,
                strike=strike,
                years_to_expiry=config.option_dte / 252.0,
                risk_free_rate=config.risk_free_rate,
                volatility=implied_vol,
                option_type="call",
            )
            entry_price = option_price * (1.0 + config.entry_slippage_pct)
            if entry_price > 0.2:
                contracts = int((cash * config.risk_fraction) // (entry_price * 100.0))
                if contracts >= 1:
                    entry_cost = entry_price * 100.0 * contracts + config.per_contract_fee * contracts
                    if entry_cost < cash:
                        cash -= entry_cost
                        position = {
                            "entry_date": date.date(),
                            "contracts": contracts,
                            "strike": strike,
                            "entry_price": entry_price,
                            "days_left": config.option_dte,
                            "days_held": 0,
                            "entry_cost": entry_cost,
                        }

        marked_equity = cash
        if position is not None:
            option_mark = black_scholes_price(
                spot=spot,
                strike=float(position["strike"]),
                years_to_expiry=max(int(position["days_left"]), 0) / 252.0,
                risk_free_rate=config.risk_free_rate,
                volatility=implied_vol,
                option_type="call",
            )
            marked_equity += option_mark * 100.0 * int(position["contracts"])
        equity_points.append((date, marked_equity))

    equity_curve = pd.Series(
        data=[value for _, value in equity_points],
        index=[date for date, _ in equity_points],
        dtype=float,
    )
    trades_df = pd.DataFrame(trades)
    metrics = build_metrics(equity_curve=equity_curve, trades=trades_df, initial_capital=config.initial_capital)

    return {
        "strategy_name": "aggressive_long_call_breakout",
        "config": asdict(config),
        "equity_curve": equity_curve,
        "trades": trades_df,
        "metrics": metrics,
    }
