from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from options_strategy_lab.heavy_ml import make_heavy_ml_filter_config
from options_strategy_lab.monte_carlo import MonteCarloConfig
from options_strategy_lab.research_engine import HeavyCreditSpreadResearchEngine
from options_strategy_lab.strategies import make_credit_spread_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _research_name(symbol: str, heavy_preset: str) -> str:
    return f"heavy_credit_spread_{symbol.lower()}_{heavy_preset}"


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, backtest, stress-test, and package the heavy credit-spread strategy for VM research.",
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-04-09")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--account-equity", type=float, default=None, help="Equity used for the latest-signal sizing plan.")
    parser.add_argument("--output-dir", default="outputs2/outputs")
    parser.add_argument("--model-dir", default="outputs2/models")
    parser.add_argument("--research-dir", default="outputs2/research")
    parser.add_argument(
        "--preset",
        choices=("baseline", "optimized", "max_return"),
        default="optimized",
        help="Base credit-spread preset used to generate the working strategy candidate set.",
    )
    parser.add_argument(
        "--heavy-preset",
        choices=("balanced", "aggressive", "cpu_max"),
        default="cpu_max",
        help="Heavy ensemble preset.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="CPU workers/threads handed to the ensemble estimators.",
    )
    parser.add_argument("--risk-fraction", type=float, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--research-name", default=None)
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-monte-carlo", action="store_true")
    parser.add_argument("--skip-signal-plan", action="store_true")
    parser.add_argument("--mc-iterations", type=int, default=5000)
    parser.add_argument("--mc-block-size", type=int, default=10)
    parser.add_argument("--mc-random-seed", type=int, default=42)
    parser.add_argument("--mc-shock-std-fraction", type=float, default=0.15)
    parser.add_argument("--mc-target-monthly-return-pct", type=float, default=5.0)
    parser.add_argument("--mc-drawdown-alert-pct", type=float, default=-40.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    research_name = args.research_name or _research_name(args.symbol, args.heavy_preset)
    model_name = args.model_name or research_name
    research_root = Path(args.research_dir)
    progress_path = research_root / f"{research_name}_progress.json"
    summary_path = research_root / f"{research_name}_summary.json"
    training_summary_path = research_root / f"{research_name}_training_summary.json"
    monte_carlo_summary_path = research_root / f"{research_name}_monte_carlo_summary.json"
    monte_carlo_results_path = research_root / f"{research_name}_monte_carlo_results.csv"
    signal_plan_path = research_root / f"{research_name}_latest_signal_plan.json"

    progress: dict[str, Any] = {
        "research_name": research_name,
        "started_at_utc": _utc_now(),
        "status": "running",
        "stages": {
            "training": "pending",
            "backtest": "pending",
            "monte_carlo": "pending",
            "signal_plan": "pending",
        },
        "artifacts": {},
    }
    _write_json(progress_path, progress)

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
        model_name=model_name,
    )
    _write_json(training_summary_path, training_summary)
    progress["stages"]["training"] = "completed"
    progress["artifacts"]["training_summary_json"] = str(training_summary_path.resolve())
    progress["artifacts"]["model_path"] = str(training_summary["model_path"])
    progress["artifacts"]["metadata_path"] = str(training_summary["metadata_path"])
    _write_json(progress_path, progress)

    backtest_result: dict[str, Any] | None = None
    if args.skip_backtest:
        progress["stages"]["backtest"] = "skipped"
    else:
        backtest_result = engine.backtest(
            output_dir=args.output_dir,
            strategy_name=research_name,
        )
        progress["stages"]["backtest"] = "completed"
        progress["artifacts"]["backtest_metrics_json"] = str((Path(args.output_dir) / f"{research_name}_metrics.json").resolve())
        progress["artifacts"]["backtest_trades_csv"] = str((Path(args.output_dir) / f"{research_name}_trades.csv").resolve())
        progress["artifacts"]["backtest_equity_curve_csv"] = str((Path(args.output_dir) / f"{research_name}_equity_curve.csv").resolve())
        _write_json(progress_path, progress)

    monte_carlo_output: dict[str, Any] | None = None
    if args.skip_monte_carlo:
        progress["stages"]["monte_carlo"] = "skipped"
    else:
        if backtest_result is None:
            msg = "Monte Carlo requires the walk-forward backtest. Remove --skip-backtest or add --skip-monte-carlo."
            raise ValueError(msg)
        monte_carlo_output = engine.monte_carlo(
            equity_curve=backtest_result["equity_curve"],
            config=MonteCarloConfig(
                iterations=args.mc_iterations,
                block_size=args.mc_block_size,
                random_seed=args.mc_random_seed,
                shock_std_fraction=args.mc_shock_std_fraction,
                target_monthly_return_pct=args.mc_target_monthly_return_pct,
                drawdown_alert_pct=args.mc_drawdown_alert_pct,
            ),
        )
        _write_json(
            monte_carlo_summary_path,
            {
                "summary": monte_carlo_output["summary"],
                "config": monte_carlo_output["config"],
            },
        )
        monte_carlo_output["results"].to_csv(monte_carlo_results_path, index=False)
        progress["stages"]["monte_carlo"] = "completed"
        progress["artifacts"]["monte_carlo_summary_json"] = str(monte_carlo_summary_path.resolve())
        progress["artifacts"]["monte_carlo_results_csv"] = str(monte_carlo_results_path.resolve())
        _write_json(progress_path, progress)

    signal_plan = None
    if args.skip_signal_plan:
        progress["stages"]["signal_plan"] = "skipped"
    else:
        signal_plan = engine.build_latest_signal_plan(
            model_path=str(training_summary["model_path"]),
            metadata_path=str(training_summary["metadata_path"]),
            account_equity=args.account_equity or args.initial_capital,
        )
        _write_json(signal_plan_path, signal_plan.to_dict())
        progress["stages"]["signal_plan"] = "completed"
        progress["artifacts"]["latest_signal_plan_json"] = str(signal_plan_path.resolve())
        _write_json(progress_path, progress)

    summary: dict[str, Any] = {
        "research_name": research_name,
        "training": training_summary,
        "backtest_metrics": backtest_result["metrics"] if backtest_result is not None else None,
        "monte_carlo_summary": monte_carlo_output["summary"] if monte_carlo_output is not None else None,
        "latest_signal_plan": signal_plan.to_dict() if signal_plan is not None else None,
        "artifacts": progress["artifacts"],
        "completed_at_utc": _utc_now(),
    }
    _write_json(summary_path, summary)
    progress["status"] = "completed"
    progress["artifacts"]["summary_json"] = str(summary_path.resolve())
    progress["completed_at_utc"] = summary["completed_at_utc"]
    _write_json(progress_path, progress)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
