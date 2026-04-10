from __future__ import annotations

import argparse
import json
import os

from options_strategy_lab.heavy_ml import make_heavy_ml_filter_config
from options_strategy_lab.research_engine import HeavyCreditSpreadResearchEngine
from options_strategy_lab.reports import save_backtest_artifacts
from options_strategy_lab.strategies import make_credit_spread_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the CPU-heavy ensemble model and optionally run a walk-forward backtest.",
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-04-09")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--output-dir", default="outputs2/outputs")
    parser.add_argument("--model-dir", default="outputs2/models")
    parser.add_argument(
        "--preset",
        choices=("baseline", "optimized", "max_return"),
        default="optimized",
        help="Base credit spread preset used to generate trade candidates.",
    )
    parser.add_argument(
        "--heavy-preset",
        choices=("balanced", "aggressive", "cpu_max"),
        default="aggressive",
        help="Heavy ensemble preset.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="CPU workers/threads to hand to the heavy model estimators.",
    )
    parser.add_argument(
        "--risk-fraction",
        type=float,
        default=None,
        help="Override the heavy-model base risk fraction.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional saved model artifact name without extension.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Only train and save the heavy model artifact.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    strategy_config = make_credit_spread_config(
        preset=args.preset,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        initial_capital=args.initial_capital,
    )
    heavy_config = make_heavy_ml_filter_config(
        preset=args.heavy_preset,
        cpu_workers=args.cpu_workers,
    )
    if args.risk_fraction is not None:
        heavy_config.base_risk_fraction = args.risk_fraction

    engine = HeavyCreditSpreadResearchEngine(
        strategy_config=strategy_config,
        ml_config=heavy_config,
    )
    training_summary = engine.train(
        model_dir=args.model_dir,
        model_name=args.model_name,
    )

    result_summary: dict[str, object] = {"training": training_summary}
    if not args.skip_backtest:
        backtest_result = engine.backtest(
            strategy_name=f"heavy_ml_credit_spread_{args.heavy_preset}",
        )
        strategy_name = f"heavy_ml_credit_spread_{args.heavy_preset}"
        save_backtest_artifacts(
            output_dir=args.output_dir,
            strategy_name=strategy_name,
            equity_curve=backtest_result["equity_curve"],
            trades=backtest_result["trades"],
            metrics=backtest_result["metrics"],
            extra_tables=backtest_result.get("extra_tables"),
        )
        result_summary["backtest_metrics"] = backtest_result["metrics"]
        result_summary["backtest_strategy_name"] = strategy_name
        result_summary["backtest_tuned_threshold"] = engine.ml_config.probability_threshold
        result_summary["backtest_tuned_recipe_name"] = engine.ml_config.walkforward_recipe_name

    print(json.dumps(result_summary, indent=2))


if __name__ == "__main__":
    main()
