from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .heavy_ml import (
    HEAVY_ML_FEATURE_COLUMNS,
    HeavyEnsembleModelBundle,
    HeavyMLFilterConfig,
    _dynamic_risk_fraction,
    _prepare_heavy_candidates,
    prepare_heavy_candidates_from_price_data,
    make_heavy_ml_filter_config,
    run_heavy_ml_credit_spread_backtest,
    train_heavy_credit_model,
)
from .monte_carlo import MonteCarloConfig, run_monte_carlo_analysis
from .reports import save_backtest_artifacts
from .strategies import CreditSpreadConfig, make_credit_spread_config


@dataclass(slots=True)
class LatestSignalPlan:
    symbol: str
    as_of_date: str
    status: str
    accepted: bool
    predicted_win_probability: float
    probability_threshold: float
    risk_fraction: float
    contracts: int
    side: str | None
    short_strike: float | None
    long_strike: float | None
    target_expiration_date: str | None
    target_entry_credit: float | None
    max_loss_per_spread: float | None
    spot: float | None
    recipe_name: str
    model_training_rows: int
    model_path: str | None = None
    metadata_path: str | None = None
    feature_vector: dict[str, float] | None = None
    candidate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(inner) for inner in value]
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


class HeavyCreditSpreadResearchEngine:
    def __init__(
        self,
        strategy_config: CreditSpreadConfig | None = None,
        ml_config: HeavyMLFilterConfig | None = None,
    ) -> None:
        self.strategy_config = strategy_config or make_credit_spread_config("optimized")
        self.ml_config = ml_config or make_heavy_ml_filter_config("aggressive")

    def clone_strategy_config(self) -> CreditSpreadConfig:
        return CreditSpreadConfig(**asdict(self.strategy_config))

    def clone_ml_config(self) -> HeavyMLFilterConfig:
        return HeavyMLFilterConfig(**asdict(self.ml_config))

    def apply_training_summary(self, training_summary: dict[str, Any]) -> None:
        self.ml_config.probability_threshold = float(training_summary["best_threshold"])
        self.ml_config.walkforward_recipe_name = str(training_summary["best_recipe_name"])

    def train(
        self,
        *,
        model_dir: str = "outputs2/models",
        model_name: str | None = None,
        apply_tuning: bool = True,
    ) -> dict[str, Any]:
        training_summary = train_heavy_credit_model(
            strategy_config=self.clone_strategy_config(),
            ml_config=self.clone_ml_config(),
            model_dir=model_dir,
            model_name=model_name,
        )
        if apply_tuning:
            self.apply_training_summary(training_summary)
        return training_summary

    def backtest(
        self,
        *,
        output_dir: str | None = None,
        strategy_name: str = "heavy_ml_credit_spread",
    ) -> dict[str, Any]:
        result = run_heavy_ml_credit_spread_backtest(
            strategy_config=self.clone_strategy_config(),
            ml_config=self.clone_ml_config(),
        )
        result["strategy_name"] = strategy_name
        if output_dir is not None:
            save_backtest_artifacts(
                output_dir=output_dir,
                strategy_name=strategy_name,
                equity_curve=result["equity_curve"],
                trades=result["trades"],
                metrics=result["metrics"],
                extra_tables=result.get("extra_tables"),
            )
        return result

    def monte_carlo(
        self,
        *,
        equity_curve: pd.Series,
        initial_capital: float | None = None,
        config: MonteCarloConfig | None = None,
    ) -> dict[str, Any]:
        return run_monte_carlo_analysis(
            equity_curve=equity_curve,
            initial_capital=initial_capital or self.strategy_config.initial_capital,
            config=config,
        )

    @staticmethod
    def load_model_bundle(model_path: str) -> HeavyEnsembleModelBundle:
        return joblib.load(model_path)

    @classmethod
    def from_saved_model_artifacts(
        cls,
        *,
        model_path: str,
        metadata_path: str,
    ) -> tuple["HeavyCreditSpreadResearchEngine", HeavyEnsembleModelBundle, dict[str, Any]]:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        strategy_config = CreditSpreadConfig(**metadata["strategy_config"])
        ml_config = HeavyMLFilterConfig(**metadata["ml_config"])
        engine = cls(strategy_config=strategy_config, ml_config=ml_config)
        if "best_threshold" in metadata:
            engine.ml_config.probability_threshold = float(metadata["best_threshold"])
        if "best_recipe_name" in metadata:
            engine.ml_config.walkforward_recipe_name = str(metadata["best_recipe_name"])
        bundle = cls.load_model_bundle(model_path)
        engine.ml_config.probability_threshold = float(bundle.threshold)
        engine.ml_config.walkforward_recipe_name = str(bundle.recipe_name)
        return engine, bundle, metadata

    def _build_signal_plan_from_inputs(
        self,
        *,
        inputs: dict[str, pd.Series | pd.DataFrame],
        candidates: pd.DataFrame,
        model_path: str,
        metadata_path: str | None,
        account_equity: float | None,
        bundle: HeavyEnsembleModelBundle,
        ml_config: HeavyMLFilterConfig,
        require_signal_today: bool,
    ) -> LatestSignalPlan:
        as_of_date = pd.Timestamp(inputs["close"].index[-1])

        if candidates.empty:
            return LatestSignalPlan(
                symbol=self.strategy_config.symbol,
                as_of_date=str(as_of_date.date()),
                status="no_candidates",
                accepted=False,
                predicted_win_probability=0.0,
                probability_threshold=float(bundle.threshold),
                risk_fraction=0.0,
                contracts=0,
                side=None,
                short_strike=None,
                long_strike=None,
                target_expiration_date=None,
                target_entry_credit=None,
                max_loss_per_spread=None,
                spot=float(inputs["close"].iloc[-1]),
                recipe_name=str(bundle.recipe_name),
                model_training_rows=int(bundle.training_rows),
                model_path=str(Path(model_path).resolve()),
                metadata_path=str(Path(metadata_path).resolve()) if metadata_path else None,
            )

        if require_signal_today:
            candidate_frame = candidates.loc[candidates["entry_date"] == as_of_date]
        else:
            candidate_frame = candidates.loc[[int(candidates.index[-1])]]

        if candidate_frame.empty:
            return LatestSignalPlan(
                symbol=self.strategy_config.symbol,
                as_of_date=str(as_of_date.date()),
                status="no_signal",
                accepted=False,
                predicted_win_probability=0.0,
                probability_threshold=float(bundle.threshold),
                risk_fraction=0.0,
                contracts=0,
                side=None,
                short_strike=None,
                long_strike=None,
                target_expiration_date=None,
                target_entry_credit=None,
                max_loss_per_spread=None,
                spot=float(inputs["close"].iloc[-1]),
                recipe_name=str(bundle.recipe_name),
                model_training_rows=int(bundle.training_rows),
                model_path=str(Path(model_path).resolve()),
                metadata_path=str(Path(metadata_path).resolve()) if metadata_path else None,
            )

        candidate = candidate_frame.iloc[-1]
        predicted_probability = float(
            bundle.predict_proba(candidate_frame.loc[:, HEAVY_ML_FEATURE_COLUMNS].tail(1))[0]
        )
        accepted = predicted_probability >= float(bundle.threshold)
        risk_fraction = _dynamic_risk_fraction(predicted_probability, ml_config) if accepted else 0.0
        contracts = 0
        if accepted and account_equity is not None:
            contracts = int((account_equity * risk_fraction) // float(candidate["max_loss_per_spread"]))

        if accepted and account_equity is not None and contracts < 1:
            status = "accepted_but_zero_contracts"
        elif accepted:
            status = "accepted"
        else:
            status = "rejected"

        target_expiration_date = pd.Timestamp(candidate["entry_date"]) + pd.offsets.BDay(self.strategy_config.option_dte)
        return LatestSignalPlan(
            symbol=self.strategy_config.symbol,
            as_of_date=str(as_of_date.date()),
            status=status,
            accepted=accepted,
            predicted_win_probability=round(predicted_probability, 6),
            probability_threshold=round(float(bundle.threshold), 6),
            risk_fraction=round(float(risk_fraction), 6),
            contracts=int(contracts),
            side=str(candidate["side"]),
            short_strike=float(candidate["short_strike"]),
            long_strike=float(candidate["long_strike"]),
            target_expiration_date=str(pd.Timestamp(target_expiration_date).date()),
            target_entry_credit=round(float(candidate["entry_credit"]), 4),
            max_loss_per_spread=round(float(candidate["max_loss_per_spread"]), 2),
            spot=round(float(candidate["spot"]), 4),
            recipe_name=str(bundle.recipe_name),
            model_training_rows=int(bundle.training_rows),
            model_path=str(Path(model_path).resolve()),
            metadata_path=str(Path(metadata_path).resolve()) if metadata_path else None,
            feature_vector={
                feature: round(float(candidate[feature]), 6)
                for feature in HEAVY_ML_FEATURE_COLUMNS
            },
            candidate=_json_safe(candidate.to_dict()),
        )

    def build_signal_plan_from_price_data(
        self,
        *,
        price_data: pd.DataFrame,
        model_path: str,
        metadata_path: str | None = None,
        account_equity: float | None = None,
        bundle: HeavyEnsembleModelBundle | None = None,
        require_signal_today: bool = True,
    ) -> LatestSignalPlan:
        ml_config = self.clone_ml_config()
        bundle = bundle or self.load_model_bundle(model_path)
        ml_config.probability_threshold = float(bundle.threshold)
        ml_config.walkforward_recipe_name = str(bundle.recipe_name)
        inputs, candidates = prepare_heavy_candidates_from_price_data(
            strategy_config=self.clone_strategy_config(),
            ml_config=ml_config,
            price_data=price_data,
        )
        return self._build_signal_plan_from_inputs(
            inputs=inputs,
            candidates=candidates,
            model_path=model_path,
            metadata_path=metadata_path,
            account_equity=account_equity,
            bundle=bundle,
            ml_config=ml_config,
            require_signal_today=require_signal_today,
        )

    def build_latest_signal_plan(
        self,
        *,
        model_path: str,
        metadata_path: str | None = None,
        account_equity: float | None = None,
        bundle: HeavyEnsembleModelBundle | None = None,
        require_signal_today: bool = True,
    ) -> LatestSignalPlan:
        ml_config = self.clone_ml_config()
        bundle = bundle or self.load_model_bundle(model_path)
        ml_config.probability_threshold = float(bundle.threshold)
        ml_config.walkforward_recipe_name = str(bundle.recipe_name)
        inputs, candidates = _prepare_heavy_candidates(self.clone_strategy_config(), ml_config)
        return self._build_signal_plan_from_inputs(
            inputs=inputs,
            candidates=candidates,
            model_path=model_path,
            metadata_path=metadata_path,
            account_equity=account_equity,
            bundle=bundle,
            ml_config=ml_config,
            require_signal_today=require_signal_today,
        )
