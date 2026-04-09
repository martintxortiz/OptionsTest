from __future__ import annotations

import itertools
import json
import os
import sqlite3
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import fetch_price_history
from .pricing import black_scholes_price
from .reports import build_metrics
from .strategies import _mark_iv, _realized_volatility, _rsi


SEARCH_SPACE_VERSION = "2026-04-09-v2"
_WORKER_STATE: dict[str, Any] = {}
_FAILED_OBJECTIVE = -1_000_000_000.0

_FIXED_EXECUTION_PARAMS: dict[str, float] = {
    "entry_slippage_pct": 0.03,
    "exit_slippage_pct": 0.03,
    "per_contract_fee": 1.0,
    "risk_free_rate": 0.04,
}

_COMMON_SPACES: dict[str, dict[str, tuple[Any, ...]]] = {
    "directional": {
        "risk_fraction": (0.15, 0.35, 0.75),
        "option_dte": (7, 14, 21, 30),
        "exit_dte": (1, 3, 5),
        "iv_multiplier": (1.0, 1.15, 1.30),
        "min_iv": (0.18, 0.25),
        "max_iv": (0.45, 0.70),
    },
    "volatility": {
        "risk_fraction": (0.10, 0.20, 0.35),
        "option_dte": (7, 14, 21, 30),
        "exit_dte": (1, 3, 5),
        "iv_multiplier": (1.0, 1.15, 1.30),
        "min_iv": (0.18, 0.25),
        "max_iv": (0.55, 0.85),
    },
    "neutral": {
        "risk_fraction": (0.15, 0.30, 0.50),
        "option_dte": (7, 14, 21, 30),
        "exit_dte": (1, 3, 5),
        "iv_multiplier": (1.0, 1.15, 1.30),
        "min_iv": (0.18, 0.25),
        "max_iv": (0.45, 0.70),
    },
}

_SIGNAL_SPACES: dict[str, dict[str, tuple[Any, ...]]] = {
    "bull_mean_reversion": {
        "rsi_window": (3, 5),
        "bull_rsi_threshold": (30.0, 35.0, 40.0),
        "min_rv": (0.0, 0.10),
        "max_rv": (0.45, 0.60),
    },
    "bear_mean_reversion": {
        "rsi_window": (3, 5),
        "bear_rsi_threshold": (60.0, 65.0, 70.0),
        "min_rv": (0.0, 0.10),
        "max_rv": (0.45, 0.60),
    },
    "bull_breakout": {
        "lookback": (5, 10, 20),
        "min_rv": (0.0, 0.10),
        "max_rv": (0.45, 0.60),
    },
    "bear_breakdown": {
        "lookback": (5, 10, 20),
        "min_rv": (0.0, 0.10),
        "max_rv": (0.45, 0.60),
    },
    "neutral_range": {
        "neutral_band_pct": (0.005, 0.010),
        "neutral_exit_band_pct": (0.015, 0.025),
        "neutral_rv_max": (0.18, 0.24, 0.30),
        "range_lookback": (10, 20),
        "range_cap_pct": (0.03, 0.05, 0.08),
    },
    "volatility_expansion": {
        "lookback": (5, 10, 20),
        "range_lookback": (10, 20),
        "range_cap_pct": (0.03, 0.05, 0.08),
        "compression_rv_max": (0.18, 0.22, 0.26),
    },
}

_FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "long_call": {
        "base_space": "directional",
        "signals": ("bull_mean_reversion", "bull_breakout"),
        "params": {
            "call_long_pct": (0.97, 1.00, 1.03),
            "take_profit": (0.75, 1.50),
            "stop_loss": (0.30, 0.45),
        },
    },
    "long_put": {
        "base_space": "directional",
        "signals": ("bear_mean_reversion", "bear_breakdown"),
        "params": {
            "put_long_pct": (0.97, 1.00, 1.03),
            "take_profit": (0.75, 1.50),
            "stop_loss": (0.30, 0.45),
        },
    },
    "bull_call_debit_spread": {
        "base_space": "directional",
        "signals": ("bull_mean_reversion", "bull_breakout"),
        "params": {
            "call_long_pct": (0.97, 1.00),
            "call_short_pct": (1.04, 1.08, 1.12),
            "take_profit": (0.50, 1.00),
            "stop_loss": (0.25, 0.40),
        },
    },
    "bear_put_debit_spread": {
        "base_space": "directional",
        "signals": ("bear_mean_reversion", "bear_breakdown"),
        "params": {
            "put_long_pct": (1.00, 1.03),
            "put_short_pct": (0.96, 0.92, 0.88),
            "take_profit": (0.50, 1.00),
            "stop_loss": (0.25, 0.40),
        },
    },
    "bull_put_credit_spread": {
        "base_space": "directional",
        "signals": ("bull_mean_reversion", "bull_breakout"),
        "params": {
            "put_short_pct": (0.98, 0.96, 0.94),
            "put_long_pct": (0.94, 0.90),
            "take_profit": (0.20, 0.35),
            "stop_loss": (0.45, 0.60),
        },
    },
    "bear_call_credit_spread": {
        "base_space": "directional",
        "signals": ("bear_mean_reversion", "bear_breakdown"),
        "params": {
            "call_short_pct": (1.02, 1.04, 1.06),
            "call_long_pct": (1.06, 1.10),
            "take_profit": (0.20, 0.35),
            "stop_loss": (0.45, 0.60),
        },
    },
    "long_straddle": {
        "base_space": "volatility",
        "signals": ("volatility_expansion",),
        "params": {
            "call_long_pct": (0.99, 1.00),
            "put_long_pct": (0.99, 1.00),
            "take_profit": (0.50, 1.00),
            "stop_loss": (0.25, 0.40),
        },
    },
    "long_strangle": {
        "base_space": "volatility",
        "signals": ("volatility_expansion",),
        "params": {
            "call_long_pct": (1.04, 1.08),
            "put_long_pct": (0.96, 0.92),
            "take_profit": (0.50, 1.00),
            "stop_loss": (0.25, 0.40),
        },
    },
    "iron_condor": {
        "base_space": "neutral",
        "signals": ("neutral_range",),
        "params": {
            "put_long_pct": (0.92, 0.88),
            "put_short_pct": (0.96, 0.94),
            "call_short_pct": (1.04, 1.06),
            "call_long_pct": (1.08, 1.12),
            "take_profit": (0.20, 0.35),
            "stop_loss": (0.45, 0.60),
        },
    },
    "iron_butterfly": {
        "base_space": "neutral",
        "signals": ("neutral_range",),
        "params": {
            "put_long_pct": (0.94, 0.90),
            "put_short_pct": (0.99, 1.00),
            "call_short_pct": (1.00, 1.01),
            "call_long_pct": (1.06, 1.10),
            "take_profit": (0.20, 0.35),
            "stop_loss": (0.45, 0.60),
        },
    },
}


@dataclass(slots=True)
class SearchConfig:
    start: str = "2020-01-01"
    end: str = "2026-04-09"
    initial_capital: float = 100_000.0
    workers: int = max(1, os.cpu_count() or 1)
    batch_size: int = 32
    max_candidates: int | None = None
    checkpoint_db: str = "outputs/search/strategy_search.sqlite"
    output_dir: str = "outputs/search"
    cache_dir: str = "data_cache"
    top_k: int = 100
    families: tuple[str, ...] = (
        "bull_put_credit_spread",
        "bear_call_credit_spread",
        "bull_call_debit_spread",
        "bear_put_debit_spread",
        "long_call",
        "long_put",
        "long_straddle",
        "long_strangle",
        "iron_condor",
        "iron_butterfly",
    )
    symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM", "SMH", "TQQQ")
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class FamilyPlan:
    family: str
    signals: tuple[str, ...]
    common_params: tuple[dict[str, Any], ...]
    family_params: tuple[dict[str, Any], ...]
    signal_params: dict[str, tuple[dict[str, Any], ...]]
    signal_totals: dict[str, int]
    per_symbol_count: int
    total_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cpu_safe_env() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = "1"
    threadpool_limits(limits=1)


def _rolling_range_pct(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    high_roll = high.rolling(lookback).max()
    low_roll = low.rolling(lookback).min()
    return (high_roll - low_roll) / close.replace(0.0, np.nan)


def _search_identity(config: SearchConfig) -> dict[str, Any]:
    return {
        "start": config.start,
        "end": config.end,
        "initial_capital": config.initial_capital,
        "symbols": list(config.symbols),
        "families": list(config.families),
    }


def _validate_config(config: SearchConfig) -> None:
    if not config.symbols:
        msg = "At least one symbol is required."
        raise ValueError(msg)
    if not config.families:
        msg = "At least one strategy family is required."
        raise ValueError(msg)
    unknown = sorted(set(config.families) - set(_FAMILY_SPECS))
    if unknown:
        msg = f"Unknown strategy families: {', '.join(unknown)}"
        raise ValueError(msg)
    if config.workers < 1:
        msg = "workers must be at least 1."
        raise ValueError(msg)
    if config.batch_size < 1:
        msg = "batch_size must be at least 1."
        raise ValueError(msg)
    if config.top_k < 1:
        msg = "top_k must be at least 1."
        raise ValueError(msg)
    if config.max_candidates is not None and config.max_candidates < 0:
        msg = "max_candidates cannot be negative."
        raise ValueError(msg)


def _prepare_symbol_inputs(symbol: str, start: str, end: str, cache_dir: str) -> dict[str, Any]:
    price_data = fetch_price_history(symbol=symbol, start=start, end=end, cache_dir=cache_dir)
    close = price_data["close"]
    high = price_data["high"]
    low = price_data["low"]

    return {
        "symbol": symbol,
        "price_data": price_data,
        "close": close,
        "high": high,
        "low": low,
        "sma20": close.rolling(20).mean(),
        "sma50": close.rolling(50).mean(),
        "sma200": close.rolling(200).mean(),
        "ema10": close.ewm(span=10, adjust=False).mean(),
        "ema20": close.ewm(span=20, adjust=False).mean(),
        "rv20": _realized_volatility(close, 20),
        "rsi": {window: _rsi(close, window) for window in (3, 5)},
        "breakout": {
            lookback: high.shift(1).rolling(lookback).max() for lookback in (5, 10, 20)
        },
        "breakdown": {
            lookback: low.shift(1).rolling(lookback).min() for lookback in (5, 10, 20)
        },
        "range_pct": {
            lookback: _rolling_range_pct(high, low, close, lookback) for lookback in (10, 20)
        },
    }


def _warm_price_cache(config: SearchConfig) -> None:
    for symbol in config.symbols:
        fetch_price_history(
            symbol=symbol,
            start=config.start,
            end=config.end,
            cache_dir=config.cache_dir,
        )


def _worker_initializer(symbols: tuple[str, ...], start: str, end: str, cache_dir: str) -> None:
    _cpu_safe_env()
    global _WORKER_STATE
    _WORKER_STATE = {
        "symbols": {
            symbol: _prepare_symbol_inputs(symbol=symbol, start=start, end=end, cache_dir=cache_dir)
            for symbol in symbols
        }
    }


def _signed_leg_value(
    spot: float,
    strike: float,
    days_to_expiry: int,
    implied_volatility: float,
    risk_free_rate: float,
    option_type: str,
    side: str,
) -> float:
    price = black_scholes_price(
        spot=spot,
        strike=strike,
        years_to_expiry=max(days_to_expiry, 0) / 252.0,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility,
        option_type=option_type,
    )
    return price if side == "long" else -price


def _position_value(
    legs: list[dict[str, Any]],
    spot: float,
    days_to_expiry: int,
    implied_volatility: float,
    risk_free_rate: float,
) -> float:
    return sum(
        _signed_leg_value(
            spot=spot,
            strike=float(leg["strike"]),
            days_to_expiry=days_to_expiry,
            implied_volatility=implied_volatility,
            risk_free_rate=risk_free_rate,
            option_type=str(leg["option_type"]),
            side=str(leg["side"]),
        )
        for leg in legs
    )


def _family_is_credit(family: str) -> bool:
    return family in {
        "bull_put_credit_spread",
        "bear_call_credit_spread",
        "iron_condor",
        "iron_butterfly",
    }


def _build_position(candidate: dict[str, Any], spot: float) -> tuple[list[dict[str, Any]], float]:
    family = candidate["family"]
    legs: list[dict[str, Any]] = []

    def add_leg(option_type: str, side: str, strike_pct: float) -> None:
        legs.append(
            {
                "option_type": option_type,
                "side": side,
                "strike": round(spot * strike_pct),
            }
        )

    if family == "long_call":
        add_leg("call", "long", candidate["call_long_pct"])
    elif family == "long_put":
        add_leg("put", "long", candidate["put_long_pct"])
    elif family == "bull_call_debit_spread":
        add_leg("call", "long", candidate["call_long_pct"])
        add_leg("call", "short", candidate["call_short_pct"])
    elif family == "bear_put_debit_spread":
        add_leg("put", "long", candidate["put_long_pct"])
        add_leg("put", "short", candidate["put_short_pct"])
    elif family == "bull_put_credit_spread":
        add_leg("put", "short", candidate["put_short_pct"])
        add_leg("put", "long", candidate["put_long_pct"])
    elif family == "bear_call_credit_spread":
        add_leg("call", "short", candidate["call_short_pct"])
        add_leg("call", "long", candidate["call_long_pct"])
    elif family == "long_straddle":
        add_leg("call", "long", candidate["call_long_pct"])
        add_leg("put", "long", candidate["put_long_pct"])
    elif family == "long_strangle":
        add_leg("call", "long", candidate["call_long_pct"])
        add_leg("put", "long", candidate["put_long_pct"])
    elif family == "iron_condor":
        add_leg("put", "long", candidate["put_long_pct"])
        add_leg("put", "short", candidate["put_short_pct"])
        add_leg("call", "short", candidate["call_short_pct"])
        add_leg("call", "long", candidate["call_long_pct"])
    elif family == "iron_butterfly":
        add_leg("put", "long", candidate["put_long_pct"])
        add_leg("put", "short", candidate["put_short_pct"])
        add_leg("call", "short", candidate["call_short_pct"])
        add_leg("call", "long", candidate["call_long_pct"])
    else:
        msg = f"Unknown family: {family}"
        raise ValueError(msg)

    if not _family_is_credit(family):
        return legs, 0.0

    if family in {"bull_put_credit_spread", "bear_call_credit_spread"}:
        strikes = sorted(float(leg["strike"]) for leg in legs)
        return legs, abs(strikes[-1] - strikes[0]) * 100.0

    put_width = abs(candidate["put_short_pct"] - candidate["put_long_pct"])
    call_width = abs(candidate["call_long_pct"] - candidate["call_short_pct"])
    return legs, max(put_width, call_width) * spot * 100.0


def _signal_allowed(candidate: dict[str, Any], inputs: dict[str, Any], date: pd.Timestamp) -> bool:
    close = inputs["close"]
    sma20 = inputs["sma20"]
    sma50 = inputs["sma50"]
    sma200 = inputs["sma200"]
    ema20 = inputs["ema20"]
    rv20 = inputs["rv20"]
    spot = float(close.loc[date])

    signal = candidate["signal"]
    if signal == "bull_mean_reversion":
        rsi = inputs["rsi"][candidate["rsi_window"]]
        return (
            pd.notna(sma50.loc[date])
            and pd.notna(sma200.loc[date])
            and pd.notna(rsi.loc[date])
            and pd.notna(rv20.loc[date])
            and spot > float(sma50.loc[date]) > float(sma200.loc[date])
            and float(rsi.loc[date]) < candidate["bull_rsi_threshold"]
            and candidate["min_rv"] <= float(rv20.loc[date]) <= candidate["max_rv"]
        )

    if signal == "bear_mean_reversion":
        rsi = inputs["rsi"][candidate["rsi_window"]]
        return (
            pd.notna(sma50.loc[date])
            and pd.notna(sma200.loc[date])
            and pd.notna(rsi.loc[date])
            and pd.notna(rv20.loc[date])
            and spot < float(sma50.loc[date]) < float(sma200.loc[date])
            and float(rsi.loc[date]) > candidate["bear_rsi_threshold"]
            and candidate["min_rv"] <= float(rv20.loc[date]) <= candidate["max_rv"]
        )

    date_loc = close.index.get_loc(date)
    prev_date = close.index[date_loc - 1] if date_loc > 0 else None
    if signal == "bull_breakout":
        breakout = inputs["breakout"][candidate["lookback"]]
        pullback_ok = prev_date is not None and float(close.loc[prev_date]) < float(ema20.loc[prev_date])
        return (
            pd.notna(sma20.loc[date])
            and pd.notna(sma50.loc[date])
            and pd.notna(sma200.loc[date])
            and pd.notna(breakout.loc[date])
            and pd.notna(rv20.loc[date])
            and spot > float(sma20.loc[date]) > float(sma50.loc[date]) > float(sma200.loc[date])
            and spot > float(breakout.loc[date])
            and pullback_ok
            and candidate["min_rv"] <= float(rv20.loc[date]) <= candidate["max_rv"]
        )

    if signal == "bear_breakdown":
        breakdown = inputs["breakdown"][candidate["lookback"]]
        bounce_ok = prev_date is not None and float(close.loc[prev_date]) > float(ema20.loc[prev_date])
        return (
            pd.notna(sma20.loc[date])
            and pd.notna(sma50.loc[date])
            and pd.notna(sma200.loc[date])
            and pd.notna(breakdown.loc[date])
            and pd.notna(rv20.loc[date])
            and spot < float(sma20.loc[date]) < float(sma50.loc[date]) < float(sma200.loc[date])
            and spot < float(breakdown.loc[date])
            and bounce_ok
            and candidate["min_rv"] <= float(rv20.loc[date]) <= candidate["max_rv"]
        )

    if signal == "neutral_range":
        range_pct = inputs["range_pct"][candidate["range_lookback"]]
        return (
            pd.notna(sma20.loc[date])
            and pd.notna(range_pct.loc[date])
            and pd.notna(rv20.loc[date])
            and abs(spot / float(sma20.loc[date]) - 1.0) <= candidate["neutral_band_pct"]
            and float(rv20.loc[date]) <= candidate["neutral_rv_max"]
            and float(range_pct.loc[date]) <= candidate["range_cap_pct"]
        )

    if signal == "volatility_expansion":
        breakout = inputs["breakout"][candidate["lookback"]]
        breakdown = inputs["breakdown"][candidate["lookback"]]
        range_pct = inputs["range_pct"][candidate["range_lookback"]]
        broke_up = pd.notna(breakout.loc[date]) and spot > float(breakout.loc[date])
        broke_down = pd.notna(breakdown.loc[date]) and spot < float(breakdown.loc[date])
        return (
            pd.notna(rv20.loc[date])
            and pd.notna(range_pct.loc[date])
            and float(rv20.loc[date]) <= candidate["compression_rv_max"]
            and float(range_pct.loc[date]) <= candidate["range_cap_pct"]
            and (broke_up or broke_down)
        )

    return False


def _trend_broken(candidate: dict[str, Any], inputs: dict[str, Any], date: pd.Timestamp, spot: float) -> bool:
    ema10 = inputs["ema10"]
    sma20 = inputs["sma20"]
    signal = candidate["signal"]
    if signal in {"bull_mean_reversion", "bull_breakout"}:
        return pd.notna(ema10.loc[date]) and spot < float(ema10.loc[date])
    if signal in {"bear_mean_reversion", "bear_breakdown"}:
        return pd.notna(ema10.loc[date]) and spot > float(ema10.loc[date])
    if signal == "neutral_range":
        return pd.notna(sma20.loc[date]) and abs(spot / float(sma20.loc[date]) - 1.0) > candidate["neutral_exit_band_pct"]
    return False


def _failure_result(candidate: dict[str, Any], exc: Exception) -> dict[str, Any]:
    metrics = {
        "initial_capital": float(candidate["initial_capital"]),
        "ending_equity": float(candidate["initial_capital"]),
        "total_return_pct": 0.0,
        "cagr_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe": 0.0,
        "monthly_mean_return_pct": 0.0,
        "monthly_median_return_pct": 0.0,
        "monthly_positive_rate_pct": 0.0,
        "active_month_mean_return_pct": 0.0,
        "active_month_median_return_pct": 0.0,
        "best_month_return_pct": 0.0,
        "worst_month_return_pct": 0.0,
        "trade_count": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "average_trade_dollars": 0.0,
        "objective_score": _FAILED_OBJECTIVE,
        "error": f"{type(exc).__name__}: {exc}",
    }
    return {
        "candidate_index": candidate["candidate_index"],
        "symbol": candidate["symbol"],
        "family": candidate["family"],
        "signal": candidate["signal"],
        "objective_score": _FAILED_OBJECTIVE,
        "metrics": metrics,
        "params": candidate,
    }


def _evaluate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    try:
        inputs = _WORKER_STATE["symbols"][candidate["symbol"]]
        close = inputs["close"]
        rv20 = inputs["rv20"]

        cash = candidate["initial_capital"]
        equity_points: list[float] = []
        trades: list[dict[str, Any]] = []
        position: dict[str, Any] | None = None

        for date in close.index:
            spot = float(close.loc[date])
            rv_value = float(rv20.loc[date]) if pd.notna(rv20.loc[date]) else np.nan
            implied_vol = _mark_iv(
                realized_vol=rv_value,
                minimum=candidate["min_iv"],
                maximum=candidate["max_iv"],
                multiplier=candidate["iv_multiplier"],
            )

            if position is not None:
                position["days_left"] -= 1
                raw_mark_value = _position_value(
                    legs=position["legs"],
                    spot=spot,
                    days_to_expiry=position["days_left"],
                    implied_volatility=implied_vol,
                    risk_free_rate=candidate["risk_free_rate"],
                )
                if _family_is_credit(candidate["family"]):
                    adjusted_mark_value = raw_mark_value * (1.0 + candidate["exit_slippage_pct"])
                else:
                    adjusted_mark_value = raw_mark_value * (1.0 - candidate["exit_slippage_pct"])

                exit_cash_flow = adjusted_mark_value * 100.0 * position["contracts"] - candidate["per_contract_fee"] * position["contracts"]
                pnl = exit_cash_flow + position["entry_cash_flow"]

                if _family_is_credit(candidate["family"]):
                    should_exit = (
                        abs(raw_mark_value) <= abs(position["entry_value"]) * candidate["take_profit"]
                        or pnl <= -position["risk_dollars"] * candidate["stop_loss"]
                        or position["days_left"] <= candidate["exit_dte"]
                        or _trend_broken(candidate, inputs, date, spot)
                    )
                else:
                    pnl_pct = (raw_mark_value - position["entry_value"]) / max(abs(position["entry_value"]), 1e-9)
                    should_exit = (
                        pnl_pct >= candidate["take_profit"]
                        or pnl_pct <= -candidate["stop_loss"]
                        or position["days_left"] <= candidate["exit_dte"]
                        or _trend_broken(candidate, inputs, date, spot)
                    )

                if should_exit:
                    cash += exit_cash_flow
                    trades.append(
                        {
                            "entry_date": str(position["entry_date"]),
                            "exit_date": str(date.date()),
                            "contracts": position["contracts"],
                            "family": candidate["family"],
                            "signal": candidate["signal"],
                            "entry_value": round(position["entry_value"], 4),
                            "exit_value": round(adjusted_mark_value, 4),
                            "net_pnl": round(pnl, 2),
                        }
                    )
                    position = None

            if position is None and _signal_allowed(candidate, inputs, date):
                legs, base_max_loss = _build_position(candidate, spot)
                entry_value = _position_value(
                    legs=legs,
                    spot=spot,
                    days_to_expiry=candidate["option_dte"],
                    implied_volatility=implied_vol,
                    risk_free_rate=candidate["risk_free_rate"],
                )

                if _family_is_credit(candidate["family"]):
                    entry_value *= 1.0 - candidate["entry_slippage_pct"]
                else:
                    entry_value *= 1.0 + candidate["entry_slippage_pct"]

                risk_per_contract = abs(entry_value) * 100.0
                if _family_is_credit(candidate["family"]):
                    risk_per_contract = max(base_max_loss - abs(entry_value) * 100.0, 1.0)

                if abs(entry_value) > 0.05 and risk_per_contract > 0:
                    contracts = int((cash * candidate["risk_fraction"]) // risk_per_contract)
                    if contracts >= 1:
                        entry_cash_flow = -entry_value * 100.0 * contracts - candidate["per_contract_fee"] * contracts
                        projected_cash = cash + entry_cash_flow
                        if projected_cash > -candidate["initial_capital"] * 0.25:
                            cash = projected_cash
                            position = {
                                "entry_date": date.date(),
                                "legs": legs,
                                "entry_value": entry_value,
                                "days_left": candidate["option_dte"],
                                "contracts": contracts,
                                "entry_cash_flow": entry_cash_flow,
                                "risk_dollars": risk_per_contract * contracts,
                            }

            marked = cash
            if position is not None:
                marked += _position_value(
                    legs=position["legs"],
                    spot=spot,
                    days_to_expiry=position["days_left"],
                    implied_volatility=implied_vol,
                    risk_free_rate=candidate["risk_free_rate"],
                ) * 100.0 * position["contracts"]
            equity_points.append(marked)

        equity_curve = pd.Series(equity_points, index=close.index, dtype=float)
        trades_df = pd.DataFrame(trades)
        metrics = build_metrics(
            equity_curve=equity_curve,
            trades=trades_df,
            initial_capital=candidate["initial_capital"],
        )

        objective_score = (
            float(metrics["cagr_pct"])
            + 8.0 * float(metrics["sharpe"])
            + 0.8 * float(metrics["monthly_mean_return_pct"])
            - 0.45 * abs(float(metrics["max_drawdown_pct"]))
        )
        metrics["objective_score"] = round(objective_score, 4)

        return {
            "candidate_index": candidate["candidate_index"],
            "symbol": candidate["symbol"],
            "family": candidate["family"],
            "signal": candidate["signal"],
            "objective_score": metrics["objective_score"],
            "metrics": metrics,
            "params": candidate,
        }
    except Exception as exc:  # pragma: no cover - defensive for long-running search jobs
        return _failure_result(candidate, exc)


def _space_rows(
    space: dict[str, tuple[Any, ...]],
    validator: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[dict[str, Any], ...]:
    keys = tuple(space)
    rows: list[dict[str, Any]] = []
    for combo in itertools.product(*(space[key] for key in keys)):
        row = dict(zip(keys, combo, strict=True))
        if validator is not None and not validator(row):
            continue
        rows.append(row)
    return tuple(rows)


def _valid_common_params(params: dict[str, Any]) -> bool:
    return params["exit_dte"] < params["option_dte"] and params["max_iv"] > params["min_iv"]


def _valid_signal_params(signal: str, params: dict[str, Any]) -> bool:
    if signal in {"bull_mean_reversion", "bear_mean_reversion", "bull_breakout", "bear_breakdown"}:
        return params["max_rv"] > params["min_rv"]
    if signal == "neutral_range":
        return params["neutral_exit_band_pct"] > params["neutral_band_pct"]
    return True


def _valid_family_params(family: str, params: dict[str, Any]) -> bool:
    if family == "bull_call_debit_spread":
        return params["call_long_pct"] < params["call_short_pct"]
    if family == "bear_put_debit_spread":
        return params["put_short_pct"] < params["put_long_pct"]
    if family == "bull_put_credit_spread":
        return params["put_long_pct"] < params["put_short_pct"] < 1.0
    if family == "bear_call_credit_spread":
        return 1.0 < params["call_short_pct"] < params["call_long_pct"]
    if family == "long_strangle":
        return params["put_long_pct"] < 1.0 < params["call_long_pct"]
    if family == "iron_condor":
        return (
            params["put_long_pct"] < params["put_short_pct"] < 1.0
            and 1.0 < params["call_short_pct"] < params["call_long_pct"]
        )
    if family == "iron_butterfly":
        return (
            params["put_long_pct"] < params["put_short_pct"] <= 1.0
            and 1.0 <= params["call_short_pct"] < params["call_long_pct"]
        )
    return True


@lru_cache(maxsize=None)
def _common_param_rows(space_name: str) -> tuple[dict[str, Any], ...]:
    return _space_rows(_COMMON_SPACES[space_name], validator=_valid_common_params)


@lru_cache(maxsize=None)
def _signal_param_rows(signal: str) -> tuple[dict[str, Any], ...]:
    return _space_rows(
        _SIGNAL_SPACES[signal],
        validator=lambda params: _valid_signal_params(signal, params),
    )


@lru_cache(maxsize=None)
def _family_param_rows(family: str) -> tuple[dict[str, Any], ...]:
    return _space_rows(
        _FAMILY_SPECS[family]["params"],
        validator=lambda params: _valid_family_params(family, params),
    )


@lru_cache(maxsize=None)
def _family_plan(family: str, symbol_count: int) -> FamilyPlan:
    spec = _FAMILY_SPECS[family]
    common_params = _common_param_rows(spec["base_space"])
    family_params = _family_param_rows(family)
    signal_params = {
        signal: _signal_param_rows(signal) for signal in spec["signals"]
    }
    signal_totals = {
        signal: len(common_params) * len(signal_params[signal]) * len(family_params)
        for signal in spec["signals"]
    }
    per_symbol_count = sum(signal_totals.values())
    total_count = per_symbol_count * symbol_count
    return FamilyPlan(
        family=family,
        signals=spec["signals"],
        common_params=common_params,
        family_params=family_params,
        signal_params=signal_params,
        signal_totals=signal_totals,
        per_symbol_count=per_symbol_count,
        total_count=total_count,
    )


def iter_search_candidates(
    config: SearchConfig,
    *,
    start_index: int = 0,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    _validate_config(config)

    candidate_index = 0
    yielded = 0
    symbol_count = len(config.symbols)

    for family in config.families:
        plan = _family_plan(family, symbol_count)
        family_end = candidate_index + plan.total_count
        if family_end <= start_index:
            candidate_index = family_end
            continue

        for symbol in config.symbols:
            symbol_end = candidate_index + plan.per_symbol_count
            if symbol_end <= start_index:
                candidate_index = symbol_end
                continue

            for signal in plan.signals:
                signal_total = plan.signal_totals[signal]
                signal_end = candidate_index + signal_total
                if signal_end <= start_index:
                    candidate_index = signal_end
                    continue

                combos = itertools.product(
                    plan.common_params,
                    plan.signal_params[signal],
                    plan.family_params,
                )
                local_skip = max(0, start_index - candidate_index)
                if local_skip:
                    combos = itertools.islice(combos, local_skip, None)
                    candidate_index += local_skip

                for common_params, signal_params, family_params in combos:
                    if limit is not None and yielded >= limit:
                        return
                    candidate = dict(_FIXED_EXECUTION_PARAMS)
                    candidate.update(common_params)
                    candidate.update(signal_params)
                    candidate.update(family_params)
                    candidate["symbol"] = symbol
                    candidate["signal"] = signal
                    candidate["family"] = family
                    candidate["initial_capital"] = config.initial_capital
                    candidate["candidate_index"] = candidate_index
                    yield candidate
                    yielded += 1
                    candidate_index += 1


def count_candidate_space(config: SearchConfig) -> int:
    _validate_config(config)
    symbol_count = len(config.symbols)
    return sum(_family_plan(family, symbol_count).total_count for family in config.families)


class SearchStorage:
    def __init__(self, db_path: str, output_dir: str, top_k: int) -> None:
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.top_k = top_k
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS search_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    next_index INTEGER NOT NULL,
                    processed_count INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    version TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS search_results (
                    run_id TEXT NOT NULL,
                    candidate_index INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    family TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    objective_score REAL NOT NULL,
                    total_return_pct REAL NOT NULL,
                    cagr_pct REAL NOT NULL,
                    monthly_mean_return_pct REAL NOT NULL,
                    max_drawdown_pct REAL NOT NULL,
                    sharpe REAL NOT NULL,
                    trade_count INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, candidate_index)
                );

                CREATE INDEX IF NOT EXISTS idx_search_results_score
                ON search_results (run_id, objective_score DESC);
                """
            )

    def write_manifest(
        self,
        *,
        run_id: str,
        config: SearchConfig,
        candidate_space: int,
        target_candidates: int,
    ) -> str:
        manifest = {
            "run_id": run_id,
            "search_space_version": SEARCH_SPACE_VERSION,
            "search_identity": _search_identity(config),
            "runtime_config": asdict(config),
            "candidate_space": candidate_space,
            "target_candidates": target_candidates,
            "checkpoint_db": str(self.db_path.resolve()),
            "output_dir": str(self.output_dir.resolve()),
            "created_at_utc": _utc_now(),
        }
        path = self.output_dir / f"{run_id}_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return str(path.resolve())

    def create_or_resume_run(self, config: SearchConfig) -> tuple[str, int]:
        run_id = config.run_id or f"search-{uuid.uuid4().hex[:10]}"
        identity_json = json.dumps(_search_identity(config), sort_keys=True)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT next_index, config_json, version FROM search_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if existing:
                if existing["version"] != SEARCH_SPACE_VERSION or existing["config_json"] != identity_json:
                    msg = (
                        f"Run id {run_id} already exists with a different search identity or search-space version. "
                        "Use a new run id for a new search."
                    )
                    raise ValueError(msg)
                connection.execute(
                    "UPDATE search_runs SET updated_at = ?, status = ? WHERE run_id = ?",
                    (_utc_now(), "running", run_id),
                )
                return run_id, int(existing["next_index"])

            connection.execute(
                """
                INSERT INTO search_runs (
                    run_id, created_at, updated_at, status, next_index, processed_count, config_json, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _utc_now(),
                    _utc_now(),
                    "running",
                    0,
                    0,
                    identity_json,
                    SEARCH_SPACE_VERSION,
                ),
            )
        return run_id, 0

    def save_batch(
        self,
        *,
        run_id: str,
        results: list[dict[str, Any]],
        next_index: int,
        candidate_space: int,
        target_candidates: int,
    ) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO search_results (
                    run_id, candidate_index, symbol, family, signal, objective_score,
                    total_return_pct, cagr_pct, monthly_mean_return_pct, max_drawdown_pct,
                    sharpe, trade_count, metrics_json, params_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        result["candidate_index"],
                        result["symbol"],
                        result["family"],
                        result["signal"],
                        result["objective_score"],
                        result["metrics"]["total_return_pct"],
                        result["metrics"]["cagr_pct"],
                        result["metrics"]["monthly_mean_return_pct"],
                        result["metrics"]["max_drawdown_pct"],
                        result["metrics"]["sharpe"],
                        result["metrics"]["trade_count"],
                        json.dumps(result["metrics"], sort_keys=True),
                        json.dumps(result["params"], sort_keys=True),
                    )
                    for result in results
                ],
            )
            connection.execute(
                """
                UPDATE search_runs
                SET updated_at = ?, next_index = ?, processed_count = ?
                WHERE run_id = ?
                """,
                (_utc_now(), next_index, next_index, run_id),
            )
        self.export_ranked_results(run_id)
        self.export_progress(
            run_id=run_id,
            candidate_space=candidate_space,
            target_candidates=target_candidates,
        )

    def finish_run(
        self,
        *,
        run_id: str,
        status: str,
        candidate_space: int,
        target_candidates: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE search_runs SET updated_at = ?, status = ? WHERE run_id = ?",
                (_utc_now(), status, run_id),
            )
        self.export_ranked_results(run_id)
        self.export_progress(
            run_id=run_id,
            candidate_space=candidate_space,
            target_candidates=target_candidates,
        )

    def export_ranked_results(self, run_id: str) -> None:
        rankings = {
            "top_objective": "objective_score DESC, monthly_mean_return_pct DESC",
            "top_total_return": "total_return_pct DESC, objective_score DESC",
            "top_monthly_return": "monthly_mean_return_pct DESC, objective_score DESC",
        }

        for suffix, order_by in rankings.items():
            with self._connect() as connection:
                rows = connection.execute(
                    f"""
                    SELECT candidate_index, symbol, family, signal, objective_score,
                           total_return_pct, cagr_pct, monthly_mean_return_pct,
                           max_drawdown_pct, sharpe, trade_count, params_json
                    FROM search_results
                    WHERE run_id = ?
                    ORDER BY {order_by}
                    LIMIT ?
                    """,
                    (run_id, self.top_k),
                ).fetchall()
            if not rows:
                continue

            frame = pd.DataFrame(
                [
                    {
                        "candidate_index": row["candidate_index"],
                        "symbol": row["symbol"],
                        "family": row["family"],
                        "signal": row["signal"],
                        "objective_score": row["objective_score"],
                        "total_return_pct": row["total_return_pct"],
                        "cagr_pct": row["cagr_pct"],
                        "monthly_mean_return_pct": row["monthly_mean_return_pct"],
                        "max_drawdown_pct": row["max_drawdown_pct"],
                        "sharpe": row["sharpe"],
                        "trade_count": row["trade_count"],
                        "params_json": row["params_json"],
                    }
                    for row in rows
                ]
            )
            frame.to_csv(self.output_dir / f"{run_id}_{suffix}.csv", index=False)

    def export_progress(self, *, run_id: str, candidate_space: int, target_candidates: int) -> str:
        with self._connect() as connection:
            run_row = connection.execute(
                """
                SELECT run_id, updated_at, status, next_index, processed_count
                FROM search_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            best_row = connection.execute(
                """
                SELECT candidate_index, symbol, family, signal, objective_score,
                       total_return_pct, cagr_pct, monthly_mean_return_pct,
                       max_drawdown_pct, sharpe, trade_count
                FROM search_results
                WHERE run_id = ?
                ORDER BY objective_score DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()

        processed_count = int(run_row["processed_count"]) if run_row else 0
        capped_processed = min(processed_count, target_candidates)
        progress = {
            "run_id": run_id,
            "updated_at_utc": run_row["updated_at"] if run_row else _utc_now(),
            "status": run_row["status"] if run_row else "unknown",
            "processed_count": processed_count,
            "candidate_space": candidate_space,
            "target_candidates": target_candidates,
            "remaining_to_target": max(target_candidates - capped_processed, 0),
            "target_progress_pct": round((capped_processed / max(target_candidates, 1)) * 100.0, 2),
            "full_space_progress_pct": round((processed_count / max(candidate_space, 1)) * 100.0, 2),
            "checkpoint_db": str(self.db_path.resolve()),
        }
        if best_row is not None:
            progress["best_objective"] = {
                "candidate_index": best_row["candidate_index"],
                "symbol": best_row["symbol"],
                "family": best_row["family"],
                "signal": best_row["signal"],
                "objective_score": best_row["objective_score"],
                "total_return_pct": best_row["total_return_pct"],
                "cagr_pct": best_row["cagr_pct"],
                "monthly_mean_return_pct": best_row["monthly_mean_return_pct"],
                "max_drawdown_pct": best_row["max_drawdown_pct"],
                "sharpe": best_row["sharpe"],
                "trade_count": best_row["trade_count"],
            }

        path = self.output_dir / f"{run_id}_progress.json"
        path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        return str(path.resolve())


def run_strategy_search(config: SearchConfig) -> dict[str, Any]:
    _validate_config(config)
    candidate_space = count_candidate_space(config)
    target_candidates = candidate_space if config.max_candidates is None else min(candidate_space, config.max_candidates)

    storage = SearchStorage(
        db_path=config.checkpoint_db,
        output_dir=config.output_dir,
        top_k=config.top_k,
    )
    run_id, start_index = storage.create_or_resume_run(config)

    storage.write_manifest(
        run_id=run_id,
        config=config,
        candidate_space=candidate_space,
        target_candidates=target_candidates,
    )
    storage.export_progress(
        run_id=run_id,
        candidate_space=candidate_space,
        target_candidates=target_candidates,
    )

    if start_index >= target_candidates:
        storage.finish_run(
            run_id=run_id,
            status="completed",
            candidate_space=candidate_space,
            target_candidates=target_candidates,
        )
        return {
            "run_id": run_id,
            "processed_candidates": 0,
            "candidate_space": candidate_space,
            "target_candidates": target_candidates,
            "checkpoint_db": str(Path(config.checkpoint_db).resolve()),
            "manifest_json": str((Path(config.output_dir) / f"{run_id}_manifest.json").resolve()),
            "progress_json": str((Path(config.output_dir) / f"{run_id}_progress.json").resolve()),
            "top_objective_csv": str((Path(config.output_dir) / f"{run_id}_top_objective.csv").resolve()),
            "top_total_return_csv": str((Path(config.output_dir) / f"{run_id}_top_total_return.csv").resolve()),
            "top_monthly_return_csv": str((Path(config.output_dir) / f"{run_id}_top_monthly_return.csv").resolve()),
        }

    _warm_price_cache(config)
    processed = start_index
    started_at = time.time()
    remaining_limit = target_candidates - start_index
    iterator = iter_search_candidates(config, start_index=start_index, limit=remaining_limit)

    try:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=_worker_initializer,
            initargs=(config.symbols, config.start, config.end, config.cache_dir),
        ) as executor:
            while True:
                batch = list(itertools.islice(iterator, config.batch_size))
                if not batch:
                    break
                chunksize = max(1, len(batch) // max(config.workers * 4, 1))
                results = list(executor.map(_evaluate_candidate, batch, chunksize=chunksize))
                processed += len(batch)
                storage.save_batch(
                    run_id=run_id,
                    results=results,
                    next_index=processed,
                    candidate_space=candidate_space,
                    target_candidates=target_candidates,
                )

        storage.finish_run(
            run_id=run_id,
            status="completed",
            candidate_space=candidate_space,
            target_candidates=target_candidates,
        )
    except KeyboardInterrupt:
        storage.finish_run(
            run_id=run_id,
            status="interrupted",
            candidate_space=candidate_space,
            target_candidates=target_candidates,
        )
        raise
    except Exception:
        storage.finish_run(
            run_id=run_id,
            status="failed",
            candidate_space=candidate_space,
            target_candidates=target_candidates,
        )
        raise

    elapsed = time.time() - started_at
    return {
        "run_id": run_id,
        "processed_candidates": processed - start_index,
        "candidate_space": candidate_space,
        "target_candidates": target_candidates,
        "total_elapsed_seconds": round(elapsed, 2),
        "checkpoint_db": str(Path(config.checkpoint_db).resolve()),
        "manifest_json": str((Path(config.output_dir) / f"{run_id}_manifest.json").resolve()),
        "progress_json": str((Path(config.output_dir) / f"{run_id}_progress.json").resolve()),
        "top_objective_csv": str((Path(config.output_dir) / f"{run_id}_top_objective.csv").resolve()),
        "top_total_return_csv": str((Path(config.output_dir) / f"{run_id}_top_total_return.csv").resolve()),
        "top_monthly_return_csv": str((Path(config.output_dir) / f"{run_id}_top_monthly_return.csv").resolve()),
    }
