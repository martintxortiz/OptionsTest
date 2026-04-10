from __future__ import annotations

import argparse
import json
from pathlib import Path

from options_strategy_lab.alpaca_adapter import AlpacaBotConfig, build_alpaca_credit_spread_blueprint
from options_strategy_lab.research_engine import HeavyCreditSpreadResearchEngine


def _research_name(symbol: str, heavy_preset: str) -> str:
    return f"heavy_credit_spread_{symbol.lower()}_{heavy_preset}"


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn the saved heavy credit-spread model into a latest-signal plan and optional Alpaca order blueprint.",
    )
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument(
        "--heavy-preset",
        choices=("balanced", "aggressive", "cpu_max"),
        default="cpu_max",
        help="Used only to infer the default research/model name.",
    )
    parser.add_argument("--research-name", default=None)
    parser.add_argument("--model-dir", default="outputs2/models")
    parser.add_argument("--output-dir", default="outputs2/deployment")
    parser.add_argument("--account-equity", type=float, default=100_000.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--secret-key", default=None)
    parser.add_argument("--paper", action="store_true", default=True)
    parser.add_argument("--live", action="store_true", help="Use the live trading endpoint instead of paper.")
    parser.add_argument("--time-in-force", choices=("day", "gtc"), default="day")
    parser.add_argument("--options-feed", choices=("indicative", "opra"), default="indicative")
    parser.add_argument("--expiration-slop-days", type=int, default=4)
    parser.add_argument("--strike-slop-pct", type=float, default=0.015)
    parser.add_argument("--contract-limit", type=int, default=500)
    parser.add_argument("--limit-price-offset-pct", type=float, default=0.03)
    parser.add_argument("--signal-only", action="store_true", help="Only write the model-side signal plan.")
    parser.add_argument("--submit", action="store_true", help="Actually submit the Alpaca order after building the blueprint.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    research_name = args.research_name or _research_name(args.symbol, args.heavy_preset)
    model_root = Path(args.model_dir)
    output_root = Path(args.output_dir)
    model_path = model_root / f"{research_name}.joblib"
    metadata_path = model_root / f"{research_name}_metadata.json"
    signal_plan_path = output_root / f"{research_name}_latest_signal_plan.json"
    blueprint_path = output_root / f"{research_name}_alpaca_order_blueprint.json"

    engine, bundle, _ = HeavyCreditSpreadResearchEngine.from_saved_model_artifacts(
        model_path=str(model_path),
        metadata_path=str(metadata_path),
    )
    signal_plan = engine.build_latest_signal_plan(
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        bundle=bundle,
        account_equity=args.account_equity,
    )
    _write_json(signal_plan_path, signal_plan.to_dict())

    default_credentials = AlpacaBotConfig()
    summary: dict[str, object] = {
        "research_name": research_name,
        "signal_plan_json": str(signal_plan_path.resolve()),
        "signal_plan_status": signal_plan.status,
        "signal_plan": signal_plan.to_dict(),
        "order_blueprint_json": None,
    }

    resolved_api_key = args.api_key or default_credentials.api_key
    resolved_secret_key = args.secret_key or default_credentials.secret_key
    has_credentials = bool(resolved_api_key and resolved_secret_key)

    can_attempt_blueprint = (
        not args.signal_only
        and signal_plan.status in {"accepted", "accepted_but_zero_contracts"}
        and signal_plan.contracts >= 1
        and has_credentials
    )
    if can_attempt_blueprint:
        blueprint = build_alpaca_credit_spread_blueprint(
            signal_plan,
            AlpacaBotConfig(
                api_key=resolved_api_key,
                secret_key=resolved_secret_key,
                paper=not args.live,
                time_in_force=args.time_in_force,
                options_feed=args.options_feed,
                expiration_slop_days=args.expiration_slop_days,
                strike_slop_pct=args.strike_slop_pct,
                contract_limit=args.contract_limit,
                limit_price_offset_pct=args.limit_price_offset_pct,
            ),
            submit=args.submit,
        )
        _write_json(blueprint_path, blueprint)
        summary["order_blueprint_json"] = str(blueprint_path.resolve())
        summary["order_blueprint"] = blueprint
    elif not args.signal_only:
        summary["order_blueprint_status"] = "skipped"
        summary["order_blueprint_reason"] = "No accepted tradable signal or Alpaca credentials were available."

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
