from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import fetch_price_history
from .pricing import bear_call_credit_spread_value, bull_put_credit_spread_value
from .reports import build_metrics
from .strategies import CreditSpreadConfig, _mark_iv, _realized_volatility, _rsi, make_credit_spread_config

ML_FEATURE_COLUMNS = [
    "side_flag",
    "rsi3",
    "rv20",
    "ret5",
    "ret10",
    "ret20",
    "spot_to_sma50",
    "sma50_to_sma200",
    "ema10_gap",
    "iv_proxy",
    "short_distance_pct",
    "width_pct",
    "credit_to_width",
    "entry_credit",
]


@dataclass(slots=True)
class SimpleMLFilterConfig:
    probability_threshold: float = 0.65
    min_training_samples: int = 40
    base_risk_fraction: float = 0.5
    positive_trade_threshold_dollars: float = 0.0
    max_iter: int = 1000


def make_simple_ml_filter_config(preset: str = "aggressive") -> SimpleMLFilterConfig:
    if preset == "moderate":
        return SimpleMLFilterConfig(
            probability_threshold=0.65,
            min_training_samples=40,
            base_risk_fraction=0.5,
        )
    if preset == "aggressive":
        return SimpleMLFilterConfig(
            probability_threshold=0.65,
            min_training_samples=40,
            base_risk_fraction=1.0,
        )
    msg = f"Unknown ML preset: {preset}"
    raise ValueError(msg)


def prepare_credit_spread_inputs_from_price_data(price_data: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
    close = price_data["close"]
    return {
        "price_data": price_data,
        "close": close,
        "sma50": close.rolling(50).mean(),
        "sma200": close.rolling(200).mean(),
        "ema10": close.ewm(span=10, adjust=False).mean(),
        "rsi3": _rsi(close, 3),
        "rv20": _realized_volatility(close, 20),
        "ret5": close.pct_change(5),
        "ret10": close.pct_change(10),
        "ret20": close.pct_change(20),
    }


def _prepare_credit_spread_inputs(config: CreditSpreadConfig) -> dict[str, pd.Series | pd.DataFrame]:
    price_data = fetch_price_history(config.symbol, config.start, config.end)
    return prepare_credit_spread_inputs_from_price_data(price_data)


def _credit_spread_mark(
    side: str,
    spot: float,
    short_strike: float,
    long_strike: float,
    days_to_expiry: int,
    implied_volatility: float,
    risk_free_rate: float,
) -> float:
    if side == "bull_put":
        return bull_put_credit_spread_value(
            spot=spot,
            short_strike=short_strike,
            long_strike=long_strike,
            days_to_expiry=days_to_expiry,
            implied_volatility=implied_volatility,
            risk_free_rate=risk_free_rate,
        )
    return bear_call_credit_spread_value(
        spot=spot,
        short_strike=short_strike,
        long_strike=long_strike,
        days_to_expiry=days_to_expiry,
        implied_volatility=implied_volatility,
        risk_free_rate=risk_free_rate,
    )


def _build_signal_candidates(
    config: CreditSpreadConfig,
    inputs: dict[str, pd.Series | pd.DataFrame],
    positive_trade_threshold_dollars: float,
) -> pd.DataFrame:
    close = inputs["close"]
    sma50 = inputs["sma50"]
    sma200 = inputs["sma200"]
    ema10 = inputs["ema10"]
    rsi3 = inputs["rsi3"]
    rv20 = inputs["rv20"]
    ret5 = inputs["ret5"]
    ret10 = inputs["ret10"]
    ret20 = inputs["ret20"]

    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    index_dates = list(close.index)
    date_positions = {date: idx for idx, date in enumerate(index_dates)}

    for date in close.index:
        if not (
            pd.notna(sma50.loc[date])
            and pd.notna(sma200.loc[date])
            and pd.notna(rsi3.loc[date])
        ):
            continue

        spot = float(close.loc[date])
        implied_vol = _mark_iv(
            realized_vol=float(rv20.loc[date]) if pd.notna(rv20.loc[date]) else np.nan,
            minimum=config.min_implied_vol,
            maximum=config.max_implied_vol,
            multiplier=config.implied_vol_multiplier,
        )

        side: str | None = None
        short_strike = 0.0
        long_strike = 0.0
        width = 0.0
        entry_value = 0.0

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

        if side is None:
            continue

        entry_credit = entry_value * (1.0 - config.entry_slippage_pct)
        max_loss_per_spread = (width - entry_credit) * 100.0
        if not (entry_credit > 0.2 and max_loss_per_spread > 0):
            continue

        exit_date = date
        exit_debit = entry_credit
        exit_reason = "final_bar"
        days_left = config.option_dte

        for future_date in close.index[date_positions[date] + 1 :]:
            days_left -= 1
            future_spot = float(close.loc[future_date])
            future_iv = _mark_iv(
                realized_vol=float(rv20.loc[future_date]) if pd.notna(rv20.loc[future_date]) else np.nan,
                minimum=config.min_implied_vol,
                maximum=config.max_implied_vol,
                multiplier=config.implied_vol_multiplier,
            )
            mark = _credit_spread_mark(
                side=side,
                spot=future_spot,
                short_strike=short_strike,
                long_strike=long_strike,
                days_to_expiry=days_left,
                implied_volatility=future_iv,
                risk_free_rate=config.risk_free_rate,
            )
            pnl_per_spread = (entry_credit - mark) * 100.0
            trend_broken = future_spot < float(ema10.loc[future_date]) if side == "bull_put" else future_spot > float(ema10.loc[future_date])
            should_exit = (
                mark <= entry_credit * config.take_profit_remaining_credit_pct
                or pnl_per_spread <= -max_loss_per_spread * config.stop_loss_fraction_of_max_loss
                or days_left <= config.exit_dte
                or trend_broken
            )
            if should_exit:
                exit_date = future_date
                exit_debit = mark
                exit_reason = "rule_exit"
                break

        net_pnl_per_spread = entry_credit * 100.0 - config.per_contract_fee - (exit_debit * 100.0 + config.per_contract_fee)
        rows.append(
            {
                "entry_date": pd.Timestamp(date),
                "exit_date": pd.Timestamp(exit_date),
                "side": side,
                "side_flag": int(side == "bull_put"),
                "spot": spot,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "width": width,
                "entry_credit": entry_credit,
                "max_loss_per_spread": max_loss_per_spread,
                "rsi3": float(rsi3.loc[date]),
                "rv20": float(rv20.loc[date]) if pd.notna(rv20.loc[date]) else 0.0,
                "ret5": float(ret5.loc[date]) if pd.notna(ret5.loc[date]) else 0.0,
                "ret10": float(ret10.loc[date]) if pd.notna(ret10.loc[date]) else 0.0,
                "ret20": float(ret20.loc[date]) if pd.notna(ret20.loc[date]) else 0.0,
                "spot_to_sma50": spot / float(sma50.loc[date]),
                "sma50_to_sma200": float(sma50.loc[date]) / float(sma200.loc[date]),
                "ema10_gap": spot / float(ema10.loc[date]) - 1.0,
                "iv_proxy": implied_vol,
                "short_distance_pct": abs(short_strike - spot) / spot,
                "width_pct": width / spot,
                "credit_to_width": entry_credit / width,
                "net_pnl_per_spread": net_pnl_per_spread,
                "return_on_risk_pct": (net_pnl_per_spread / max_loss_per_spread) * 100.0,
                "label": int(net_pnl_per_spread > positive_trade_threshold_dollars),
                "exit_reason": exit_reason,
            }
        )

    return pd.DataFrame(rows)


def _build_training_availability_map(candidates: pd.DataFrame, trading_dates: pd.Index) -> dict[pd.Timestamp, list[int]]:
    date_positions = {pd.Timestamp(date): idx for idx, date in enumerate(trading_dates)}
    availability: dict[pd.Timestamp, list[int]] = {}

    for row in candidates.itertuples(index=True):
        exit_position = date_positions.get(pd.Timestamp(row.exit_date))
        if exit_position is None or exit_position + 1 >= len(trading_dates):
            continue
        available_date = pd.Timestamp(trading_dates[exit_position + 1])
        availability.setdefault(available_date, []).append(int(row.Index))

    return availability


def _fit_trade_model(
    candidates: pd.DataFrame,
    training_indices: list[int],
    max_iter: int,
) -> Pipeline | None:
    labels = candidates.loc[training_indices, "label"]
    if len(training_indices) == 0 or labels.nunique() < 2:
        return None

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter, class_weight="balanced")),
        ]
    )
    model.fit(candidates.loc[training_indices, ML_FEATURE_COLUMNS], labels)
    return model


def _extract_model_coefficients(model: Pipeline | None) -> dict[str, float]:
    if model is None:
        return {}
    clf = model.named_steps["clf"]
    return {
        feature: round(float(weight), 4)
        for feature, weight in zip(ML_FEATURE_COLUMNS, clf.coef_[0], strict=True)
    }


def run_ml_credit_spread_backtest(
    strategy_config: CreditSpreadConfig | None = None,
    ml_config: SimpleMLFilterConfig | None = None,
) -> dict[str, object]:
    strategy_config = strategy_config or make_credit_spread_config("optimized")
    ml_config = ml_config or make_simple_ml_filter_config("aggressive")

    inputs = _prepare_credit_spread_inputs(strategy_config)
    candidates = _build_signal_candidates(
        config=strategy_config,
        inputs=inputs,
        positive_trade_threshold_dollars=ml_config.positive_trade_threshold_dollars,
    )

    close = inputs["close"]
    ema10 = inputs["ema10"]
    rv20 = inputs["rv20"]

    training_map = _build_training_availability_map(candidates, close.index)
    entry_map = {pd.Timestamp(row.entry_date): int(row.Index) for row in candidates.itertuples(index=True)}

    cash = strategy_config.initial_capital
    equity_points: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict[str, float | int | str]] = []
    signal_log: list[dict[str, float | int | str]] = []
    training_indices: list[int] = []
    model: Pipeline | None = None
    position: dict[str, float | int | str | pd.Timestamp] | None = None
    retrain_needed = False

    for date in close.index:
        current_date = pd.Timestamp(date)

        for candidate_index in training_map.get(current_date, []):
            training_indices.append(candidate_index)
            retrain_needed = True

        if retrain_needed and len(training_indices) >= ml_config.min_training_samples:
            trained_model = _fit_trade_model(
                candidates=candidates,
                training_indices=training_indices,
                max_iter=ml_config.max_iter,
            )
            if trained_model is not None:
                model = trained_model
            retrain_needed = False

        spot = float(close.loc[date])
        implied_vol = _mark_iv(
            realized_vol=float(rv20.loc[date]) if pd.notna(rv20.loc[date]) else np.nan,
            minimum=strategy_config.min_implied_vol,
            maximum=strategy_config.max_implied_vol,
            multiplier=strategy_config.implied_vol_multiplier,
        )

        if position is not None:
            position["days_left"] = int(position["days_left"]) - 1
            side = str(position["side"])
            mark = _credit_spread_mark(
                side=side,
                spot=spot,
                short_strike=float(position["short_strike"]),
                long_strike=float(position["long_strike"]),
                days_to_expiry=int(position["days_left"]),
                implied_volatility=implied_vol,
                risk_free_rate=strategy_config.risk_free_rate,
            )
            trend_broken = spot < float(ema10.loc[date]) if side == "bull_put" else spot > float(ema10.loc[date])
            pnl_dollars = (float(position["entry_credit"]) - mark) * 100.0 * int(position["contracts"])
            should_exit = (
                mark <= float(position["entry_credit"]) * strategy_config.take_profit_remaining_credit_pct
                or pnl_dollars <= -float(position["max_loss_dollars"]) * strategy_config.stop_loss_fraction_of_max_loss
                or int(position["days_left"]) <= strategy_config.exit_dte
                or trend_broken
            )
            if should_exit:
                contracts = int(position["contracts"])
                exit_cost = mark * 100.0 * contracts + strategy_config.per_contract_fee * contracts
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
                        "entry_credit": round(float(position["entry_credit"]), 4),
                        "exit_debit": round(mark, 4),
                        "max_loss_dollars": round(float(position["max_loss_dollars"]), 2),
                        "net_pnl": round(net_pnl, 2),
                        "predicted_win_probability": round(float(position["predicted_win_probability"]), 4),
                    }
                )
                position = None

        candidate_index = entry_map.get(current_date)
        if position is None and candidate_index is not None:
            candidate = candidates.loc[candidate_index]
            predicted_probability = 0.5
            model_ready = model is not None and len(training_indices) >= ml_config.min_training_samples
            if model_ready:
                predicted_probability = float(
                    model.predict_proba(candidates.loc[[candidate_index], ML_FEATURE_COLUMNS])[0, 1]
                )

            accepted = (not model_ready) or predicted_probability >= ml_config.probability_threshold
            signal_log.append(
                {
                    "entry_date": str(candidate["entry_date"].date()),
                    "side": str(candidate["side"]),
                    "predicted_win_probability": round(predicted_probability, 4),
                    "accepted": int(accepted),
                    "model_ready": int(model_ready),
                    "training_sample_count": len(training_indices),
                    "realized_label": int(candidate["label"]),
                    "realized_net_pnl_per_spread": round(float(candidate["net_pnl_per_spread"]), 2),
                }
            )

            if accepted:
                contracts = int((cash * ml_config.base_risk_fraction) // float(candidate["max_loss_per_spread"]))
                if contracts >= 1:
                    entry_cash_received = float(candidate["entry_credit"]) * 100.0 * contracts - strategy_config.per_contract_fee * contracts
                    cash += entry_cash_received
                    position = {
                        "entry_date": candidate["entry_date"].date(),
                        "side": str(candidate["side"]),
                        "contracts": contracts,
                        "short_strike": float(candidate["short_strike"]),
                        "long_strike": float(candidate["long_strike"]),
                        "entry_credit": float(candidate["entry_credit"]),
                        "days_left": strategy_config.option_dte,
                        "max_loss_dollars": float(candidate["max_loss_per_spread"]) * contracts,
                        "entry_cash_received": entry_cash_received,
                        "predicted_win_probability": predicted_probability,
                    }

        marked_equity = cash
        if position is not None:
            liability = _credit_spread_mark(
                side=str(position["side"]),
                spot=spot,
                short_strike=float(position["short_strike"]),
                long_strike=float(position["long_strike"]),
                days_to_expiry=int(position["days_left"]),
                implied_volatility=implied_vol,
                risk_free_rate=strategy_config.risk_free_rate,
            )
            marked_equity -= liability * 100.0 * int(position["contracts"])
        equity_points.append((current_date, marked_equity))

    equity_curve = pd.Series(
        data=[value for _, value in equity_points],
        index=[date for date, _ in equity_points],
        dtype=float,
    )
    trades_df = pd.DataFrame(trades)
    signals_df = pd.DataFrame(signal_log)
    metrics = build_metrics(equity_curve=equity_curve, trades=trades_df, initial_capital=strategy_config.initial_capital)

    accepted_signals = int(signals_df["accepted"].sum()) if not signals_df.empty else 0
    metrics.update(
        {
            "signal_count": int(len(signals_df)),
            "accepted_signal_count": accepted_signals,
            "rejected_signal_count": int(len(signals_df) - accepted_signals),
            "accepted_signal_rate_pct": round((accepted_signals / len(signals_df)) * 100.0, 2) if not signals_df.empty else 0.0,
            "model_ready_signal_rate_pct": round(float(signals_df["model_ready"].mean()) * 100.0, 2) if not signals_df.empty else 0.0,
            "label_positive_rate_pct": round(float(candidates["label"].mean()) * 100.0, 2) if not candidates.empty else 0.0,
            "probability_threshold": ml_config.probability_threshold,
            "ml_base_risk_fraction": ml_config.base_risk_fraction,
            "final_model_coefficients": _extract_model_coefficients(model),
        }
    )

    return {
        "strategy_name": "ml_credit_spread",
        "config": {
            "strategy": asdict(strategy_config),
            "ml": asdict(ml_config),
        },
        "equity_curve": equity_curve,
        "trades": trades_df,
        "metrics": metrics,
        "extra_tables": {
            "signals": signals_df,
            "candidates": candidates,
        },
    }
