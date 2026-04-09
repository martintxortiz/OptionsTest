from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_price_history(
    symbol: str,
    start: str,
    end: str,
    cache_dir: str | None = "data_cache",
) -> pd.DataFrame:
    cache_path = None
    if cache_dir:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{symbol}_{start}_{end}.csv"
        if cache_path.exists():
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not cached.empty:
                return cached

    data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        msg = f"No price history returned for {symbol} from {start} to {end}."
        raise ValueError(msg)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    cleaned = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )[["open", "high", "low", "close", "volume"]]
    cleaned = cleaned.dropna()

    if cache_path:
        cleaned.to_csv(cache_path)
    return cleaned
