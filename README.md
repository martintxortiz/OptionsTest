# Options Strategy Lab

Backtest-first Python options strategies aimed at later Alpaca deployment.

## What is here

- `credit_spread`: trend-filtered SPY bull put / bear call credit spreads. This is the main strategy.
- `ml_credit_spread`: the optimized SPY credit-spread signals filtered by a simple walk-forward logistic model trained on past good vs bad signal outcomes.
- `aggressive_long_call`: a more convex TQQQ long-call breakout model. It is included for iteration, but it is not the baseline because its honest daily-bar backtests were weaker.
- `run_search.py`: resumable multiprocessing search across many option strategy families and parameter combinations, checkpointed to SQLite as it runs.

## Quick start

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe run_backtest.py --strategy credit_spread --preset optimized --symbol SPY
.\.venv\Scripts\python.exe run_backtest.py --strategy ml_credit_spread --ml-preset aggressive --symbol SPY
.\.venv\Scripts\python.exe run_search.py --workers 8 --batch-size 32 --max-candidates 1000
```

Artifacts are written to `outputs/`:

- equity curve CSV
- trade log CSV
- metrics JSON

## Credit spread presets

- `baseline`: original lower-return, lower-risk configuration.
- `optimized`: stronger daily-bar result from parameter search.
- `max_return`: pushes position sizing to the edge of account-level risk. This can produce much higher average monthly returns in the model, but the drawdown is severe.

## ML presets

- `moderate`: simple trade filter with lower account risk.
- `aggressive`: same simple model, but sized to chase more return after the filter removes weaker signals.

## Important limitation

This backtester uses historical underlying bars from Yahoo Finance and prices options with Black-Scholes plus a realized-volatility-based IV proxy. That makes it useful for strategy selection and execution logic, but not a substitute for historical option-chain backtests with actual quotes.

The search runner spans a wide set of common structures that can still map to Alpaca multi-leg orders: long calls and puts, debit spreads, credit spreads, straddles, strangles, iron condors, and iron butterflies. It is intentionally biased toward deployable option structures instead of unlimited-leg abstractions.

## Resumable Search

The search runner is designed for long CPU-heavy runs on a Debian VM:

- Uses one Python worker per CPU core by default.
- Forces BLAS and related math libraries to a single thread per worker to avoid oversubscription.
- Warms the Yahoo price cache once in the parent process before the worker pool starts.
- Writes every completed batch into a SQLite checkpoint database.
- Exports rolling ranked CSVs as the search continues.
- Writes a manifest JSON and a progress JSON beside the checkpoint.
- Resumes cleanly if you rerun with the same `--run-id`, even if you change worker count, batch size, or `--max-candidates`.

Example:

```powershell
.\.venv\Scripts\python.exe run_search.py `
  --run-id vm-search-001 `
  --workers 16 `
  --batch-size 64 `
  --max-candidates 500000 `
  --symbols SPY,QQQ,IWM,SMH,TQQQ `
  --checkpoint-db outputs/search/strategy_search.sqlite `
  --output-dir outputs/search
```

Important files for a search run:

- `<run_id>_manifest.json`: exact runtime config, search identity, and paths.
- `<run_id>_progress.json`: rolling processed-count, progress percent, and current best result.
- `<run_id>_top_objective.csv`: best candidates by the default risk-adjusted score.
- `<run_id>_top_total_return.csv`: best candidates by total return.
- `<run_id>_top_monthly_return.csv`: best candidates by mean monthly return.

Count the search space before launching:

```powershell
.\.venv\Scripts\python.exe run_search.py --count-only
```

The default full-family grid is intentionally huge. On a fresh VM, check the count first and usually start with a bounded slice via `--max-candidates` or a narrower `--families` list.

Resume an interrupted run:

```powershell
.\.venv\Scripts\python.exe run_search.py `
  --run-id vm-search-001 `
  --workers 16 `
  --batch-size 64 `
  --max-candidates 500000
```

## Docker

The included `Dockerfile` targets Debian Bookworm via `python:3.12-slim-bookworm`.

Build:

```bash
docker build -t options-search .
```

Run on a Debian VM, mounting outputs and using all visible CPUs:

```bash
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data_cache:/app/data_cache \
  options-search \
  python run_search.py --workers $(nproc) --batch-size 64 --run-id gcp-search-001
```

If you want the container to stop after a fixed slice of the search and resume later, add `--max-candidates` and reuse the same `--run-id` on the next launch.

## Alpaca deployment path

The clean next step is to replace the modeled entry and exit prices with Alpaca option-chain snapshots and multi-leg orders after the backtest logic is approved.
# test
# test
