from __future__ import annotations

import argparse
import json
import os

from options_strategy_lab.search import SearchConfig, count_candidate_space, run_strategy_search


def _parse_csv_tuple(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable, checkpointed multi-process options strategy search.",
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-04-09")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--checkpoint-db", default="outputs/search/strategy_search.sqlite")
    parser.add_argument("--output-dir", default="outputs/search")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--symbols",
        default="SPY,QQQ,IWM,SMH,TQQQ",
        help="Comma-separated symbols.",
    )
    parser.add_argument(
        "--families",
        default="bull_put_credit_spread,bear_call_credit_spread,bull_call_debit_spread,bear_put_debit_spread,long_call,long_put,long_straddle,long_strangle,iron_condor,iron_butterfly",
        help="Comma-separated strategy families.",
    )
    parser.add_argument("--run-id", default=None, help="Reuse an existing run id to resume.")
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count the candidate space and exit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SearchConfig(
        start=args.start,
        end=args.end,
        initial_capital=args.initial_capital,
        workers=args.workers,
        batch_size=args.batch_size,
        max_candidates=args.max_candidates,
        checkpoint_db=args.checkpoint_db,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
        symbols=_parse_csv_tuple(args.symbols),
        families=_parse_csv_tuple(args.families),
        run_id=args.run_id,
    )

    if args.count_only:
        print(json.dumps({"candidate_space": count_candidate_space(config)}, indent=2))
        return

    print(json.dumps(run_strategy_search(config), indent=2))


if __name__ == "__main__":
    main()
