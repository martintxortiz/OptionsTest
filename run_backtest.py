from __future__ import annotations

import argparse
import json
import os

from options_strategy_lab.heavy_ml import make_heavy_ml_filter_config, run_heavy_ml_credit_spread_backtest
from options_strategy_lab.ml import make_simple_ml_filter_config, run_ml_credit_spread_backtest
from options_strategy_lab.reports import save_backtest_artifacts
from options_strategy_lab.strategies import (
    AggressiveLongCallBreakoutConfig,
    CreditSpreadConfig,
    make_credit_spread_config,
    run_aggressive_long_call_breakout,
    run_credit_spread_backtest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a backtest-first options strategy.")
    parser.add_argument(
        "--strategy",
        choices=("credit_spread", "ml_credit_spread", "heavy_ml_credit_spread", "aggressive_long_call"),
        default="credit_spread",
        help="Which strategy to backtest.",
    )
    parser.add_argument("--start", default="2020-01-01", help="Backtest start date.")
    parser.add_argument("--end", default="2026-04-09", help="Backtest end date.")
    parser.add_argument("--initial-capital", type=float, default=100_000.0, help="Initial account equity.")
    parser.add_argument("--output-dir", default="outputs2/outputs", help="Directory for CSV and JSON artifacts.")
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol for the credit spread strategy.")
    parser.add_argument(
        "--preset",
        choices=("baseline", "optimized", "max_return"),
        default="optimized",
        help="Credit spread preset. `max_return` is extremely aggressive and can suffer huge drawdowns.",
    )
    parser.add_argument(
        "--ml-preset",
        choices=("moderate", "aggressive"),
        default="aggressive",
        help="Simple ML filter preset for `ml_credit_spread`.",
    )
    parser.add_argument(
        "--heavy-preset",
        choices=("balanced", "aggressive", "cpu_max"),
        default="aggressive",
        help="CPU-heavy ensemble preset for `heavy_ml_credit_spread`.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="CPU workers for the heavy ensemble models.",
    )
    parser.add_argument(
        "--risk-fraction",
        type=float,
        default=None,
        help="Fraction of current cash risked on each trade. If omitted, the strategy default is used.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    default_aggressive = AggressiveLongCallBreakoutConfig()

    if args.strategy == "credit_spread":
        config = make_credit_spread_config(
            preset=args.preset,
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            initial_capital=args.initial_capital,
            risk_fraction=args.risk_fraction,
        )
        result = run_credit_spread_backtest(config)
        result["strategy_name"] = f"credit_spread_{args.preset}"
    elif args.strategy == "ml_credit_spread":
        strategy_config = make_credit_spread_config(
            preset="optimized",
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            initial_capital=args.initial_capital,
        )
        ml_config = make_simple_ml_filter_config(args.ml_preset)
        if args.risk_fraction is not None:
            ml_config.base_risk_fraction = args.risk_fraction
        result = run_ml_credit_spread_backtest(strategy_config=strategy_config, ml_config=ml_config)
        result["strategy_name"] = f"ml_credit_spread_{args.ml_preset}"
    elif args.strategy == "heavy_ml_credit_spread":
        strategy_config = make_credit_spread_config(
            preset="optimized",
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            initial_capital=args.initial_capital,
        )
        heavy_config = make_heavy_ml_filter_config(
            args.heavy_preset,
            cpu_workers=args.cpu_workers,
        )
        if args.risk_fraction is not None:
            heavy_config.base_risk_fraction = args.risk_fraction
        result = run_heavy_ml_credit_spread_backtest(
            strategy_config=strategy_config,
            ml_config=heavy_config,
        )
        result["strategy_name"] = f"heavy_ml_credit_spread_{args.heavy_preset}"
    else:
        config = AggressiveLongCallBreakoutConfig(
            start=args.start,
            end=args.end,
            initial_capital=args.initial_capital,
            risk_fraction=args.risk_fraction if args.risk_fraction is not None else default_aggressive.risk_fraction,
        )
        result = run_aggressive_long_call_breakout(config)

    save_backtest_artifacts(
        output_dir=args.output_dir,
        strategy_name=str(result["strategy_name"]),
        equity_curve=result["equity_curve"],
        trades=result["trades"],
        metrics=result["metrics"],
        extra_tables=result.get("extra_tables"),
    )

    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
