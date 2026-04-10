from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .reports import build_metrics


@dataclass(slots=True)
class MonteCarloConfig:
    iterations: int = 5000
    block_size: int = 10
    random_seed: int = 42
    shock_std_fraction: float = 0.15
    target_monthly_return_pct: float = 5.0
    drawdown_alert_pct: float = -40.0


def _sample_block_bootstrap(
    daily_returns: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    target_length = len(daily_returns)
    sampled_blocks: list[np.ndarray] = []

    while sum(len(block) for block in sampled_blocks) < target_length:
        start = int(rng.integers(0, max(target_length - block_size + 1, 1)))
        sampled_blocks.append(daily_returns[start : start + block_size])

    return np.concatenate(sampled_blocks)[:target_length]


def run_monte_carlo_analysis(
    equity_curve: pd.Series,
    initial_capital: float,
    config: MonteCarloConfig | None = None,
) -> dict[str, object]:
    config = config or MonteCarloConfig()
    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.empty:
        msg = "Monte Carlo analysis requires a non-empty equity curve with at least two points."
        raise ValueError(msg)

    rng = np.random.default_rng(config.random_seed)
    return_std = float(daily_returns.std())
    noise_std = return_std * config.shock_std_fraction
    index = equity_curve.index
    simulation_rows: list[dict[str, float | int]] = []

    for iteration in range(1, config.iterations + 1):
        sampled_returns = _sample_block_bootstrap(
            daily_returns.to_numpy(dtype=float),
            block_size=max(1, config.block_size),
            rng=rng,
        )
        if noise_std > 0:
            sampled_returns = sampled_returns + rng.normal(0.0, noise_std, size=len(sampled_returns))
        sampled_returns = np.clip(sampled_returns, -0.95, 5.0)

        simulated_equity = initial_capital * np.concatenate(
            [[1.0], np.cumprod(1.0 + sampled_returns)]
        )
        simulated_curve = pd.Series(simulated_equity, index=index, dtype=float)
        metrics = build_metrics(
            equity_curve=simulated_curve,
            trades=pd.DataFrame(),
            initial_capital=initial_capital,
        )
        simulation_rows.append(
            {
                "iteration": iteration,
                "ending_equity": float(metrics["ending_equity"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "cagr_pct": float(metrics["cagr_pct"]),
                "max_drawdown_pct": float(metrics["max_drawdown_pct"]),
                "monthly_mean_return_pct": float(metrics["monthly_mean_return_pct"]),
                "monthly_positive_rate_pct": float(metrics["monthly_positive_rate_pct"]),
            }
        )

    results_df = pd.DataFrame(simulation_rows)
    summary = {
        "iterations": int(config.iterations),
        "block_size": int(config.block_size),
        "shock_std_fraction": float(config.shock_std_fraction),
        "ending_equity_p05": round(float(results_df["ending_equity"].quantile(0.05)), 2),
        "ending_equity_p50": round(float(results_df["ending_equity"].quantile(0.50)), 2),
        "ending_equity_p95": round(float(results_df["ending_equity"].quantile(0.95)), 2),
        "cagr_pct_p05": round(float(results_df["cagr_pct"].quantile(0.05)), 2),
        "cagr_pct_p50": round(float(results_df["cagr_pct"].quantile(0.50)), 2),
        "cagr_pct_p95": round(float(results_df["cagr_pct"].quantile(0.95)), 2),
        "max_drawdown_pct_p05": round(float(results_df["max_drawdown_pct"].quantile(0.05)), 2),
        "max_drawdown_pct_p50": round(float(results_df["max_drawdown_pct"].quantile(0.50)), 2),
        "max_drawdown_pct_p95": round(float(results_df["max_drawdown_pct"].quantile(0.95)), 2),
        "monthly_mean_return_pct_p05": round(float(results_df["monthly_mean_return_pct"].quantile(0.05)), 2),
        "monthly_mean_return_pct_p50": round(float(results_df["monthly_mean_return_pct"].quantile(0.50)), 2),
        "monthly_mean_return_pct_p95": round(float(results_df["monthly_mean_return_pct"].quantile(0.95)), 2),
        "probability_of_loss_pct": round(float((results_df["total_return_pct"] <= 0).mean()) * 100.0, 2),
        "probability_target_monthly_mean_pct": round(
            float((results_df["monthly_mean_return_pct"] >= config.target_monthly_return_pct).mean()) * 100.0,
            2,
        ),
        "probability_drawdown_breach_pct": round(
            float((results_df["max_drawdown_pct"] <= config.drawdown_alert_pct).mean()) * 100.0,
            2,
        ),
    }

    return {
        "summary": summary,
        "results": results_df,
        "config": {
            "iterations": config.iterations,
            "block_size": config.block_size,
            "random_seed": config.random_seed,
            "shock_std_fraction": config.shock_std_fraction,
            "target_monthly_return_pct": config.target_monthly_return_pct,
            "drawdown_alert_pct": config.drawdown_alert_pct,
        },
    }
