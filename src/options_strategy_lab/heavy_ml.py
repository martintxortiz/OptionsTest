from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .ml import _build_signal_candidates, _build_training_availability_map, _credit_spread_mark, _prepare_credit_spread_inputs
from .reports import build_metrics
from .strategies import CreditSpreadConfig, _mark_iv, _realized_volatility, _rsi, make_credit_spread_config


HEAVY_ML_FEATURE_COLUMNS = [
    "side_flag",
    "rsi3",
    "rsi5",
    "rsi10",
    "rv10",
    "rv20",
    "rv40",
    "vol_ratio_10_40",
    "ret1",
    "ret3",
    "ret5",
    "ret10",
    "ret20",
    "spot_to_sma20",
    "spot_to_sma50",
    "sma20_to_sma50",
    "sma50_to_sma200",
    "ema10_gap",
    "ema20_gap",
    "atr14_pct",
    "range_position20",
    "drawdown20",
    "volume_ratio20",
    "trend_stretch_10",
    "iv_proxy",
    "short_distance_pct",
    "width_pct",
    "credit_to_width",
    "risk_reward_proxy",
    "credit_times_iv",
    "width_to_short_distance",
]


@dataclass(slots=True)
class HeavyMLFilterConfig:
    probability_threshold: float = 0.58
    min_training_samples: int = 96
    base_risk_fraction: float = 0.85
    max_risk_fraction: float = 1.50
    probability_sizing_boost: float = 1.10
    positive_trade_threshold_dollars: float = 0.0
    allow_untrained_entries: bool = False
    retrain_every_new_samples: int = 12
    training_window: int | None = 360
    cpu_workers: int = max(1, os.cpu_count() or 1)
    logistic_max_iter: int = 3000
    rf_estimators: int = 700
    rf_min_samples_leaf: int = 4
    et_estimators: int = 1100
    et_min_samples_leaf: int = 3
    hgb_iterations: int = 450
    hgb_learning_rate: float = 0.035
    hgb_max_depth: int = 5
    hgb_min_samples_leaf: int = 24
    walkforward_recipe_name: str = "aggressive"
    final_search_thresholds: tuple[float, ...] = (0.54, 0.58, 0.62, 0.66)
    final_search_splits: int = 5
    final_min_validation_trades: int = 16


@dataclass(frozen=True, slots=True)
class HeavyModelRecipe:
    name: str
    logistic_weight: float
    rf_weight: float
    et_weight: float
    hgb_weight: float
    rf_estimators: int
    rf_min_samples_leaf: int
    et_estimators: int
    et_min_samples_leaf: int
    hgb_iterations: int
    hgb_learning_rate: float
    hgb_max_depth: int
    hgb_min_samples_leaf: int


@dataclass(slots=True)
class HeavyEnsembleModelBundle:
    feature_columns: tuple[str, ...]
    threshold: float
    recipe_name: str
    model_weights: dict[str, float]
    estimators: dict[str, Any]
    training_rows: int
    label_positive_rate_pct: float
    feature_importance: dict[str, float]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        feature_frame = features.loc[:, list(self.feature_columns)]
        weighted_sum = np.zeros(len(feature_frame), dtype=float)
        total_weight = 0.0

        for model_name, estimator in self.estimators.items():
            weight = float(self.model_weights[model_name])
            probabilities = estimator.predict_proba(feature_frame)[:, 1]
            weighted_sum += weight * probabilities
            total_weight += weight

        if total_weight <= 0:
            return np.full(len(feature_frame), 0.5, dtype=float)
        return weighted_sum / total_weight


def make_heavy_ml_filter_config(
    preset: str = "aggressive",
    *,
    cpu_workers: int | None = None,
) -> HeavyMLFilterConfig:
    workers = max(1, cpu_workers if cpu_workers is not None else (os.cpu_count() or 1))
    if preset == "balanced":
        return HeavyMLFilterConfig(
            probability_threshold=0.60,
            min_training_samples=96,
            base_risk_fraction=0.65,
            max_risk_fraction=1.10,
            probability_sizing_boost=0.75,
            allow_untrained_entries=False,
            retrain_every_new_samples=16,
            training_window=320,
            cpu_workers=workers,
            walkforward_recipe_name="balanced",
        )
    if preset == "aggressive":
        return HeavyMLFilterConfig(
            probability_threshold=0.58,
            min_training_samples=96,
            base_risk_fraction=0.90,
            max_risk_fraction=1.60,
            probability_sizing_boost=1.20,
            allow_untrained_entries=False,
            retrain_every_new_samples=12,
            training_window=360,
            cpu_workers=workers,
            walkforward_recipe_name="aggressive",
        )
    if preset == "cpu_max":
        return HeavyMLFilterConfig(
            probability_threshold=0.56,
            min_training_samples=80,
            base_risk_fraction=1.00,
            max_risk_fraction=1.90,
            probability_sizing_boost=1.40,
            allow_untrained_entries=False,
            retrain_every_new_samples=8,
            training_window=420,
            cpu_workers=workers,
            rf_estimators=900,
            et_estimators=1400,
            hgb_iterations=550,
            walkforward_recipe_name="aggressive",
        )
    msg = f"Unknown heavy ML preset: {preset}"
    raise ValueError(msg)


def _set_training_threads(cpu_workers: int) -> None:
    thread_count = str(max(1, cpu_workers))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = thread_count


def _map_series_for_entries(entry_dates: pd.Series, values: pd.Series, default: float = 0.0) -> pd.Series:
    mapped = entry_dates.map(values)
    mapped = mapped.astype(float)
    mapped = mapped.replace([np.inf, -np.inf], np.nan)
    return mapped.fillna(default)


def _compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _heavy_model_recipes(config: HeavyMLFilterConfig) -> tuple[HeavyModelRecipe, ...]:
    return (
        HeavyModelRecipe(
            name="balanced",
            logistic_weight=1.2,
            rf_weight=1.5,
            et_weight=1.7,
            hgb_weight=1.6,
            rf_estimators=config.rf_estimators,
            rf_min_samples_leaf=config.rf_min_samples_leaf + 1,
            et_estimators=max(config.et_estimators - 200, 200),
            et_min_samples_leaf=config.et_min_samples_leaf + 1,
            hgb_iterations=max(config.hgb_iterations - 100, 150),
            hgb_learning_rate=config.hgb_learning_rate,
            hgb_max_depth=max(config.hgb_max_depth - 1, 3),
            hgb_min_samples_leaf=config.hgb_min_samples_leaf + 4,
        ),
        HeavyModelRecipe(
            name="aggressive",
            logistic_weight=0.9,
            rf_weight=1.5,
            et_weight=2.1,
            hgb_weight=2.2,
            rf_estimators=config.rf_estimators,
            rf_min_samples_leaf=config.rf_min_samples_leaf,
            et_estimators=config.et_estimators,
            et_min_samples_leaf=config.et_min_samples_leaf,
            hgb_iterations=config.hgb_iterations,
            hgb_learning_rate=config.hgb_learning_rate,
            hgb_max_depth=config.hgb_max_depth,
            hgb_min_samples_leaf=config.hgb_min_samples_leaf,
        ),
        HeavyModelRecipe(
            name="precision",
            logistic_weight=1.5,
            rf_weight=1.6,
            et_weight=1.4,
            hgb_weight=1.8,
            rf_estimators=max(config.rf_estimators - 150, 200),
            rf_min_samples_leaf=config.rf_min_samples_leaf + 2,
            et_estimators=max(config.et_estimators - 300, 200),
            et_min_samples_leaf=config.et_min_samples_leaf + 2,
            hgb_iterations=max(config.hgb_iterations - 50, 150),
            hgb_learning_rate=max(config.hgb_learning_rate - 0.005, 0.02),
            hgb_max_depth=max(config.hgb_max_depth - 1, 3),
            hgb_min_samples_leaf=config.hgb_min_samples_leaf + 8,
        ),
    )


def _prepare_heavy_candidates(
    strategy_config: CreditSpreadConfig,
    ml_config: HeavyMLFilterConfig,
) -> tuple[dict[str, pd.Series | pd.DataFrame], pd.DataFrame]:
    inputs = _prepare_credit_spread_inputs(strategy_config)
    candidates = _build_signal_candidates(
        config=strategy_config,
        inputs=inputs,
        positive_trade_threshold_dollars=ml_config.positive_trade_threshold_dollars,
    )
    if candidates.empty:
        return inputs, candidates

    price_data = inputs["price_data"]
    close = inputs["close"]
    high = price_data["high"]
    low = price_data["low"]
    volume = price_data["volume"]
    entry_dates = pd.Series(pd.to_datetime(candidates["entry_date"]), index=candidates.index)

    sma20 = close.rolling(20).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    rsi5 = _rsi(close, 5)
    rsi10 = _rsi(close, 10)
    rv10 = _realized_volatility(close, 10)
    rv40 = _realized_volatility(close, 40)
    ret1 = close.pct_change(1)
    ret3 = close.pct_change(3)
    atr14 = _compute_true_range(high, low, close).rolling(14).mean()
    range_high20 = high.rolling(20).max()
    range_low20 = low.rolling(20).min()
    range_width20 = (range_high20 - range_low20).replace(0.0, np.nan)
    volume_ratio20 = volume / volume.rolling(20).mean().replace(0.0, np.nan)
    trend_stretch_10 = close / close.rolling(10).mean().replace(0.0, np.nan) - 1.0
    drawdown20 = close / range_high20.replace(0.0, np.nan) - 1.0
    vol_ratio_10_40 = rv10 / rv40.replace(0.0, np.nan)

    candidates = candidates.copy()
    candidates["rsi5"] = _map_series_for_entries(entry_dates, rsi5, default=50.0)
    candidates["rsi10"] = _map_series_for_entries(entry_dates, rsi10, default=50.0)
    candidates["rv10"] = _map_series_for_entries(entry_dates, rv10)
    candidates["rv40"] = _map_series_for_entries(entry_dates, rv40)
    candidates["vol_ratio_10_40"] = _map_series_for_entries(entry_dates, vol_ratio_10_40, default=1.0)
    candidates["ret1"] = _map_series_for_entries(entry_dates, ret1)
    candidates["ret3"] = _map_series_for_entries(entry_dates, ret3)
    candidates["spot_to_sma20"] = _map_series_for_entries(entry_dates, close / sma20.replace(0.0, np.nan), default=1.0)
    candidates["sma20_to_sma50"] = _map_series_for_entries(entry_dates, sma20 / inputs["sma50"].replace(0.0, np.nan), default=1.0)
    candidates["ema20_gap"] = _map_series_for_entries(entry_dates, close / ema20.replace(0.0, np.nan) - 1.0)
    candidates["atr14_pct"] = _map_series_for_entries(entry_dates, atr14 / close.replace(0.0, np.nan))
    candidates["range_position20"] = _map_series_for_entries(
        entry_dates,
        (close - range_low20) / range_width20,
        default=0.5,
    )
    candidates["drawdown20"] = _map_series_for_entries(entry_dates, drawdown20)
    candidates["volume_ratio20"] = _map_series_for_entries(entry_dates, volume_ratio20, default=1.0)
    candidates["trend_stretch_10"] = _map_series_for_entries(entry_dates, trend_stretch_10)
    candidates["risk_reward_proxy"] = (
        candidates["entry_credit"] * 100.0 / candidates["max_loss_per_spread"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    candidates["credit_times_iv"] = candidates["entry_credit"] * candidates["iv_proxy"]
    candidates["width_to_short_distance"] = (
        candidates["width_pct"] / candidates["short_distance_pct"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    candidates.loc[:, HEAVY_ML_FEATURE_COLUMNS] = (
        candidates.loc[:, HEAVY_ML_FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return inputs, candidates


def _training_slice(training_indices: list[int], window: int | None) -> list[int]:
    if window is None or len(training_indices) <= window:
        return list(training_indices)
    return list(training_indices[-window:])


def _sample_weights(training_frame: pd.DataFrame) -> np.ndarray:
    abs_pnl = training_frame["net_pnl_per_spread"].abs()
    scale = max(float(abs_pnl.median()), 1.0)
    weights = 1.0 + (abs_pnl / scale).clip(upper=4.0)
    return weights.to_numpy(dtype=float)


def _estimator_templates(
    config: HeavyMLFilterConfig,
    recipe: HeavyModelRecipe,
) -> dict[str, tuple[Any, float]]:
    cpu_workers = max(1, config.cpu_workers)
    return {
        "logistic": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=config.logistic_max_iter,
                            class_weight="balanced",
                            C=0.75,
                        ),
                    ),
                ]
            ),
            recipe.logistic_weight,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=recipe.rf_estimators,
                min_samples_leaf=recipe.rf_min_samples_leaf,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=cpu_workers,
                random_state=42,
            ),
            recipe.rf_weight,
        ),
        "extra_trees": (
            ExtraTreesClassifier(
                n_estimators=recipe.et_estimators,
                min_samples_leaf=recipe.et_min_samples_leaf,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=cpu_workers,
                random_state=42,
            ),
            recipe.et_weight,
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                max_iter=recipe.hgb_iterations,
                learning_rate=recipe.hgb_learning_rate,
                max_depth=recipe.hgb_max_depth,
                min_samples_leaf=recipe.hgb_min_samples_leaf,
                random_state=42,
            ),
            recipe.hgb_weight,
        ),
    }


def _fit_estimator(
    model_name: str,
    estimator: Any,
    features: pd.DataFrame,
    labels: pd.Series,
    sample_weight: np.ndarray,
) -> Any:
    fitted = clone(estimator)
    if model_name == "logistic":
        fitted.fit(features, labels, clf__sample_weight=sample_weight)
    else:
        fitted.fit(features, labels, sample_weight=sample_weight)
    return fitted


def _extract_feature_importance(
    estimators: dict[str, Any],
    model_weights: dict[str, float],
    feature_columns: list[str],
) -> dict[str, float]:
    aggregated = {feature: 0.0 for feature in feature_columns}
    total_weight = 0.0

    for model_name, estimator in estimators.items():
        weight = float(model_weights.get(model_name, 0.0))
        if weight <= 0:
            continue

        importance: np.ndarray | None = None
        if model_name == "logistic":
            importance = np.abs(estimator.named_steps["clf"].coef_[0])
        elif hasattr(estimator, "feature_importances_"):
            importance = np.asarray(estimator.feature_importances_, dtype=float)

        if importance is None:
            continue

        norm = float(np.sum(np.abs(importance)))
        if norm <= 0:
            continue

        total_weight += weight
        for feature, value in zip(feature_columns, importance / norm, strict=True):
            aggregated[feature] += weight * float(value)

    if total_weight <= 0:
        return {}
    return {
        feature: round(value / total_weight, 6)
        for feature, value in sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
    }


def _fit_heavy_bundle(
    candidates: pd.DataFrame,
    training_indices: list[int],
    config: HeavyMLFilterConfig,
    recipe: HeavyModelRecipe,
    threshold: float,
) -> HeavyEnsembleModelBundle | None:
    sliced_indices = _training_slice(training_indices, config.training_window)
    training_frame = candidates.loc[sliced_indices]
    labels = training_frame["label"].astype(int)
    if training_frame.empty or labels.nunique() < 2:
        return None

    features = training_frame.loc[:, HEAVY_ML_FEATURE_COLUMNS]
    sample_weight = _sample_weights(training_frame)
    templates = _estimator_templates(config, recipe)

    _set_training_threads(config.cpu_workers)
    estimators: dict[str, Any] = {}
    model_weights: dict[str, float] = {}

    with threadpool_limits(limits=max(1, config.cpu_workers)):
        for model_name, (template, weight) in templates.items():
            estimators[model_name] = _fit_estimator(
                model_name=model_name,
                estimator=template,
                features=features,
                labels=labels,
                sample_weight=sample_weight,
            )
            model_weights[model_name] = weight

    return HeavyEnsembleModelBundle(
        feature_columns=tuple(HEAVY_ML_FEATURE_COLUMNS),
        threshold=threshold,
        recipe_name=recipe.name,
        model_weights=model_weights,
        estimators=estimators,
        training_rows=int(len(training_frame)),
        label_positive_rate_pct=round(float(labels.mean()) * 100.0, 2),
        feature_importance=_extract_feature_importance(estimators, model_weights, HEAVY_ML_FEATURE_COLUMNS),
    )


def _dynamic_risk_fraction(predicted_probability: float, config: HeavyMLFilterConfig) -> float:
    threshold = config.probability_threshold
    if predicted_probability <= threshold:
        return config.base_risk_fraction

    probability_edge = (predicted_probability - threshold) / max(1.0 - threshold, 1e-9)
    scaled = config.base_risk_fraction * (1.0 + config.probability_sizing_boost * probability_edge)
    return float(min(config.max_risk_fraction, scaled))


def _choose_recipe(config: HeavyMLFilterConfig) -> HeavyModelRecipe:
    recipe_lookup = {recipe.name: recipe for recipe in _heavy_model_recipes(config)}
    return recipe_lookup.get(config.walkforward_recipe_name, _heavy_model_recipes(config)[0])


def _validation_objective(
    validation_frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    min_validation_trades: int,
) -> dict[str, float]:
    accepted = validation_frame.loc[probabilities >= threshold]
    accepted_count = int(len(accepted))
    if accepted_count < min_validation_trades:
        return {
            "score": -1_000_000.0 + accepted_count,
            "accepted_trade_count": accepted_count,
            "accepted_total_pnl": 0.0,
            "accepted_mean_pnl": 0.0,
            "accepted_win_rate_pct": 0.0,
        }

    total_pnl = float(accepted["net_pnl_per_spread"].sum())
    mean_pnl = float(accepted["net_pnl_per_spread"].mean())
    win_rate = float((accepted["net_pnl_per_spread"] > 0).mean()) * 100.0
    score = total_pnl + 15.0 * mean_pnl + 60.0 * (win_rate / 100.0)
    return {
        "score": round(score, 4),
        "accepted_trade_count": accepted_count,
        "accepted_total_pnl": round(total_pnl, 2),
        "accepted_mean_pnl": round(mean_pnl, 2),
        "accepted_win_rate_pct": round(win_rate, 2),
    }


def run_heavy_ml_credit_spread_backtest(
    strategy_config: CreditSpreadConfig | None = None,
    ml_config: HeavyMLFilterConfig | None = None,
) -> dict[str, object]:
    strategy_config = strategy_config or make_credit_spread_config("optimized")
    ml_config = ml_config or make_heavy_ml_filter_config("aggressive")

    inputs, candidates = _prepare_heavy_candidates(strategy_config, ml_config)
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
    trained_bundle: HeavyEnsembleModelBundle | None = None
    position: dict[str, float | int | str | pd.Timestamp] | None = None
    pending_new_training_samples = 0
    model_refresh_count = 0
    recipe = _choose_recipe(ml_config)

    for date in close.index:
        current_date = pd.Timestamp(date)

        for candidate_index in training_map.get(current_date, []):
            training_indices.append(candidate_index)
            pending_new_training_samples += 1

        should_retrain = (
            pending_new_training_samples >= ml_config.retrain_every_new_samples
            and len(training_indices) >= ml_config.min_training_samples
        )
        if should_retrain:
            retrained_bundle = _fit_heavy_bundle(
                candidates=candidates,
                training_indices=training_indices,
                config=ml_config,
                recipe=recipe,
                threshold=ml_config.probability_threshold,
            )
            if retrained_bundle is not None:
                trained_bundle = retrained_bundle
                model_refresh_count += 1
            pending_new_training_samples = 0

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
                        "risk_fraction_used": round(float(position["risk_fraction_used"]), 4),
                    }
                )
                position = None

        candidate_index = entry_map.get(current_date)
        if position is None and candidate_index is not None:
            candidate = candidates.loc[candidate_index]
            model_ready = trained_bundle is not None and len(training_indices) >= ml_config.min_training_samples
            predicted_probability = 0.5
            accepted = False
            risk_fraction_used = 0.0

            if model_ready and trained_bundle is not None:
                predicted_probability = float(
                    trained_bundle.predict_proba(candidates.loc[[candidate_index], HEAVY_ML_FEATURE_COLUMNS])[0]
                )
                accepted = predicted_probability >= trained_bundle.threshold
                if accepted:
                    risk_fraction_used = _dynamic_risk_fraction(predicted_probability, ml_config)
            elif ml_config.allow_untrained_entries:
                accepted = True
                risk_fraction_used = ml_config.base_risk_fraction

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
                    "risk_fraction_used": round(risk_fraction_used, 4),
                    "recipe_name": trained_bundle.recipe_name if trained_bundle is not None else recipe.name,
                }
            )

            if accepted:
                contracts = int((cash * risk_fraction_used) // float(candidate["max_loss_per_spread"]))
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
                        "risk_fraction_used": risk_fraction_used,
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
            "ml_max_risk_fraction": ml_config.max_risk_fraction,
            "model_refresh_count": model_refresh_count,
            "final_model_recipe": trained_bundle.recipe_name if trained_bundle is not None else recipe.name,
            "final_model_training_rows": trained_bundle.training_rows if trained_bundle is not None else 0,
            "final_model_label_positive_rate_pct": trained_bundle.label_positive_rate_pct if trained_bundle is not None else 0.0,
            "final_model_feature_importance": trained_bundle.feature_importance if trained_bundle is not None else {},
        }
    )

    return {
        "strategy_name": "heavy_ml_credit_spread",
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


def train_heavy_credit_model(
    strategy_config: CreditSpreadConfig | None = None,
    ml_config: HeavyMLFilterConfig | None = None,
    *,
    model_dir: str = "outputs2/models",
    model_name: str | None = None,
) -> dict[str, object]:
    strategy_config = strategy_config or make_credit_spread_config("optimized")
    ml_config = ml_config or make_heavy_ml_filter_config("aggressive")

    inputs, candidates = _prepare_heavy_candidates(strategy_config, ml_config)
    if candidates.empty:
        msg = "No candidate trades were generated for heavy model training."
        raise ValueError(msg)

    all_indices = list(candidates.index.astype(int))
    if len(all_indices) < max(ml_config.min_training_samples, ml_config.final_search_splits + 5):
        msg = "Not enough candidate trades to train the heavy model."
        raise ValueError(msg)

    model_search_rows: list[dict[str, Any]] = []
    best_bundle: HeavyEnsembleModelBundle | None = None
    best_threshold = ml_config.probability_threshold
    best_recipe = _choose_recipe(ml_config)
    best_score = -np.inf

    splitter = TimeSeriesSplit(n_splits=ml_config.final_search_splits)
    recipes = _heavy_model_recipes(ml_config)

    for recipe in recipes:
        fold_rows: list[dict[str, Any]] = []
        for threshold in ml_config.final_search_thresholds:
            fold_scores: list[float] = []
            accepted_trade_counts: list[int] = []
            accepted_total_pnls: list[float] = []

            for fold_number, (train_split, validation_split) in enumerate(splitter.split(all_indices), start=1):
                train_indices = [all_indices[idx] for idx in train_split]
                validation_indices = [all_indices[idx] for idx in validation_split]
                bundle = _fit_heavy_bundle(
                    candidates=candidates,
                    training_indices=train_indices,
                    config=ml_config,
                    recipe=recipe,
                    threshold=threshold,
                )
                if bundle is None:
                    continue

                validation_frame = candidates.loc[validation_indices]
                probabilities = bundle.predict_proba(validation_frame.loc[:, HEAVY_ML_FEATURE_COLUMNS])
                fold_metrics = _validation_objective(
                    validation_frame=validation_frame,
                    probabilities=probabilities,
                    threshold=threshold,
                    min_validation_trades=ml_config.final_min_validation_trades,
                )
                fold_scores.append(float(fold_metrics["score"]))
                accepted_trade_counts.append(int(fold_metrics["accepted_trade_count"]))
                accepted_total_pnls.append(float(fold_metrics["accepted_total_pnl"]))
                fold_rows.append(
                    {
                        "recipe_name": recipe.name,
                        "threshold": threshold,
                        "fold_number": fold_number,
                        **fold_metrics,
                    }
                )

            if not fold_scores:
                continue

            mean_score = float(np.mean(fold_scores))
            mean_trade_count = float(np.mean(accepted_trade_counts))
            mean_total_pnl = float(np.mean(accepted_total_pnls))
            model_search_rows.append(
                {
                    "recipe_name": recipe.name,
                    "threshold": threshold,
                    "mean_score": round(mean_score, 4),
                    "mean_accepted_trade_count": round(mean_trade_count, 2),
                    "mean_accepted_total_pnl": round(mean_total_pnl, 2),
                }
            )

            if mean_score > best_score:
                best_score = mean_score
                best_threshold = threshold
                best_recipe = recipe

        model_search_rows.extend(fold_rows)

    final_fit_config = HeavyMLFilterConfig(**asdict(ml_config))
    final_fit_config.training_window = None

    best_bundle = _fit_heavy_bundle(
        candidates=candidates,
        training_indices=all_indices,
        config=final_fit_config,
        recipe=best_recipe,
        threshold=best_threshold,
    )
    if best_bundle is None:
        msg = "The heavy model could not be fitted on the full candidate set."
        raise ValueError(msg)

    root = Path(model_dir)
    root.mkdir(parents=True, exist_ok=True)
    resolved_model_name = model_name or f"heavy_credit_spread_{strategy_config.symbol.lower()}"
    model_path = root / f"{resolved_model_name}.joblib"
    metadata_path = root / f"{resolved_model_name}_metadata.json"
    search_path = root / f"{resolved_model_name}_validation_results.csv"
    importance_path = root / f"{resolved_model_name}_feature_importance.csv"

    joblib.dump(best_bundle, model_path)

    metadata = {
        "model_name": resolved_model_name,
        "model_path": str(model_path.resolve()),
        "strategy_config": asdict(strategy_config),
        "ml_config": asdict(ml_config),
        "feature_columns": list(best_bundle.feature_columns),
        "best_recipe_name": best_recipe.name,
        "best_threshold": best_threshold,
        "best_validation_score": round(best_score, 4),
        "training_candidate_count": int(len(candidates)),
        "training_positive_rate_pct": round(float(candidates["label"].mean()) * 100.0, 2),
        "label_positive_rate_pct": best_bundle.label_positive_rate_pct,
        "saved_at_output_dir": str(root.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pd.DataFrame(model_search_rows).to_csv(search_path, index=False)
    pd.DataFrame(
        [
            {"feature": feature, "importance": importance}
            for feature, importance in best_bundle.feature_importance.items()
        ]
    ).to_csv(importance_path, index=False)

    return {
        "model_path": str(model_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "validation_results_path": str(search_path.resolve()),
        "feature_importance_path": str(importance_path.resolve()),
        "metadata": metadata,
        "candidate_count": int(len(candidates)),
        "best_recipe_name": best_recipe.name,
        "best_threshold": best_threshold,
        "best_validation_score": round(best_score, 4),
    }
