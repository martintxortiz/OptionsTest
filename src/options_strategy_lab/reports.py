from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float,
) -> dict[str, float | int]:
    daily_returns = equity_curve.pct_change().dropna()
    total_return = float(equity_curve.iloc[-1] / initial_capital - 1.0)
    trading_years = max(len(equity_curve) / 252.0, 1e-9)
    cagr = float((equity_curve.iloc[-1] / initial_capital) ** (1.0 / trading_years) - 1.0)
    max_drawdown = float((equity_curve / equity_curve.cummax() - 1.0).min())
    sharpe = 0.0
    if not daily_returns.empty and daily_returns.std() > 0:
        sharpe = float(np.sqrt(252.0) * daily_returns.mean() / daily_returns.std())

    monthly_returns = equity_curve.resample("ME").last().pct_change().dropna()
    active_monthly_returns = monthly_returns[monthly_returns != 0]
    gross_profit = float(trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()) if not trades.empty else 0.0
    gross_loss = float(-trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()) if not trades.empty else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    win_rate = float((trades["net_pnl"] > 0).mean()) if not trades.empty else 0.0
    avg_trade = float(trades["net_pnl"].mean()) if not trades.empty else 0.0

    return {
        "initial_capital": float(initial_capital),
        "ending_equity": float(equity_curve.iloc[-1]),
        "total_return_pct": round(total_return * 100.0, 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "max_drawdown_pct": round(max_drawdown * 100.0, 2),
        "sharpe": round(sharpe, 3),
        "monthly_mean_return_pct": round(float(monthly_returns.mean()) * 100.0, 2) if not monthly_returns.empty else 0.0,
        "monthly_median_return_pct": round(float(monthly_returns.median()) * 100.0, 2) if not monthly_returns.empty else 0.0,
        "monthly_positive_rate_pct": round(float((monthly_returns > 0).mean()) * 100.0, 2) if not monthly_returns.empty else 0.0,
        "active_month_mean_return_pct": round(float(active_monthly_returns.mean()) * 100.0, 2) if not active_monthly_returns.empty else 0.0,
        "active_month_median_return_pct": round(float(active_monthly_returns.median()) * 100.0, 2) if not active_monthly_returns.empty else 0.0,
        "best_month_return_pct": round(float(monthly_returns.max()) * 100.0, 2) if not monthly_returns.empty else 0.0,
        "worst_month_return_pct": round(float(monthly_returns.min()) * 100.0, 2) if not monthly_returns.empty else 0.0,
        "trade_count": int(len(trades)),
        "win_rate_pct": round(win_rate * 100.0, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else profit_factor,
        "average_trade_dollars": round(avg_trade, 2),
    }


def save_backtest_artifacts(
    output_dir: str,
    strategy_name: str,
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    metrics: dict[str, float | int],
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    equity_path = root / f"{strategy_name}_equity_curve.csv"
    trades_path = root / f"{strategy_name}_trades.csv"
    metrics_path = root / f"{strategy_name}_metrics.json"

    equity_curve.rename("equity").to_csv(equity_path, index_label="date")
    trades.to_csv(trades_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for suffix, table in (extra_tables or {}).items():
        extra_path = root / f"{strategy_name}_{suffix}.csv"
        table.to_csv(extra_path, index=False)
