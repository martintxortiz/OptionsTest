# Options Strategy Lab

Backtest-first Python options research for a working credit-spread engine that can be trained, stress-tested, and later plugged into Alpaca multi-leg order flow.

## Main path

The strongest path in this repo is the heavy ML credit-spread workflow on SPY:

- a rules engine generates bull put and bear call spread candidates,
- a CPU-heavy ensemble scores those candidates with walk-forward training,
- the tuned threshold and recipe are reused for the long backtest,
- Monte Carlo stress testing is run on the resulting equity curve,
- the same trained model can produce a latest-signal plan for an Alpaca bot.

## Scripts

- `run_backtest.py`: quick backtests for the baseline, simple ML, heavy ML, and long-call variants.
- `train_heavy_model.py`: train the heavy ensemble and optionally run the walk-forward backtest.
- `run_research_pipeline.py`: consolidated VM pipeline that trains, backtests, runs Monte Carlo, and writes a latest-signal plan as it goes.
- `prepare_alpaca_bot.py`: loads the saved model, builds the latest signal plan, and optionally resolves an Alpaca order blueprint from the live option chain.
- `run_live_bot.py`: starts a realtime Alpaca bot plus a small FastAPI control center for status, timers, pause/resume, and run-now overrides.
- `run_search.py`: resumable multiprocessing search across many option strategy families and parameter combinations.

## Quick start

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe run_backtest.py --strategy heavy_ml_credit_spread --heavy-preset cpu_max --symbol SPY
.\.venv\Scripts\python.exe train_heavy_model.py --heavy-preset cpu_max --cpu-workers 8
.\.venv\Scripts\python.exe run_research_pipeline.py --heavy-preset cpu_max --cpu-workers 8
```

Default artifact layout:

- `outputs2/models`: saved model bundle, metadata, validation results, feature importance.
- `outputs2/outputs`: backtest equity curve, trades, metrics, signals, candidates.
- `outputs2/research`: pipeline progress, pipeline summary, Monte Carlo summaries, Monte Carlo results, latest signal plan.
- `outputs2/deployment`: optional Alpaca order blueprints and deployment-side signal plans.

## Heavy research pipeline

`run_research_pipeline.py` is the consolidated long-run entry point for a Debian VM. It writes progress after each stage so training artifacts, backtest artifacts, Monte Carlo outputs, and the latest signal plan are saved incrementally instead of only at the end.

Example:

```bash
python run_research_pipeline.py \
  --symbol SPY \
  --heavy-preset cpu_max \
  --cpu-workers $(nproc) \
  --mc-iterations 20000
```

Useful output files:

- `outputs2/research/<research_name>_progress.json`
- `outputs2/research/<research_name>_summary.json`
- `outputs2/research/<research_name>_training_summary.json`
- `outputs2/research/<research_name>_monte_carlo_summary.json`
- `outputs2/research/<research_name>_monte_carlo_results.csv`
- `outputs2/research/<research_name>_latest_signal_plan.json`

## Monte Carlo

Monte Carlo uses a block bootstrap of the heavy-model daily equity returns, then adds controlled noise shocks. It is meant to stress the strategy's path dependence, drawdown profile, and monthly-return distribution, not to promise better performance than the walk-forward backtest.

Key outputs include:

- ending equity percentiles,
- CAGR percentiles,
- max drawdown percentiles,
- monthly mean return percentiles,
- probability of loss,
- probability of meeting a target monthly mean return,
- probability of breaching a drawdown alert level.

## Alpaca-ready deployment

`prepare_alpaca_bot.py` uses the same heavy credit-spread engine and saved model bundle to create a latest-signal plan. If Alpaca credentials are available, it can also resolve actual option contracts, fetch latest quotes, and build a multi-leg limit-order blueprint.

Example:

```bash
python prepare_alpaca_bot.py \
  --symbol SPY \
  --heavy-preset cpu_max \
  --account-equity 100000
```

With Alpaca credentials:

```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
python prepare_alpaca_bot.py \
  --symbol SPY \
  --heavy-preset cpu_max \
  --account-equity 100000
```

The generated blueprint follows Alpaca's current multi-leg pattern for options: `order_class = mleg`, per-leg `position_intent`, and a negative `limit_price` for credit spreads.

## Realtime bot

`run_live_bot.py` runs a simple production-style service:

- Alpaca trading connection validation on startup,
- Alpaca stock daily bars for the live feature inputs,
- the same saved heavy-model engine for signal generation,
- Alpaca option-chain resolution for entry orders,
- Alpaca quote-based spread marks for exit management,
- a control-center API with status, next-cycle timer, pause, resume, and run-now controls.

Control-center endpoints:

- `GET /health`
- `GET /state`
- `POST /control/pause`
- `POST /control/resume`
- `POST /control/run-now`
- `GET /` for a small browser dashboard

Example:

```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
export BOT_CONTROL_TOKEN=change-me
python run_live_bot.py \
  --symbol SPY \
  --heavy-preset cpu_max \
  --cycle-seconds 60 \
  --stock-feed iex \
  --options-feed indicative \
  --control-token $BOT_CONTROL_TOKEN
```

The browser control center will be at `http://localhost:8080/`.

## Search

The broad search runner is still here for large VM sweeps across long options, debit spreads, credit spreads, straddles, strangles, iron condors, and iron butterflies. It checkpoints every batch to SQLite and resumes with the same `--run-id`.

Count the search space:

```powershell
.\.venv\Scripts\python.exe run_search.py --count-only
```

Resume a long search:

```powershell
.\.venv\Scripts\python.exe run_search.py `
  --run-id vm-search-001 `
  --workers 16 `
  --batch-size 64 `
  --max-candidates 500000 `
  --preset credit_focus
```

## Docker

The image targets Debian Bookworm via `python:3.12-slim-bookworm` and installs the optional Alpaca dependency too.

Build:

```bash
docker build -t options-search .
```

Default VM run:

```bash
docker run --rm \
  -e CPU_WORKERS=$(nproc) \
  -e MONTE_CARLO_ITERATIONS=20000 \
  -v $(pwd)/outputs2:/app/outputs2 \
  -v $(pwd)/data_cache:/app/data_cache \
  options-search \
  --symbol SPY
```

The container entrypoint runs `run_research_pipeline.py` by default. Extra arguments are forwarded to that script.

If you want the Alpaca deployment blueprint from the same image:

```bash
docker run --rm \
  -e ALPACA_API_KEY=$ALPACA_API_KEY \
  -e ALPACA_SECRET_KEY=$ALPACA_SECRET_KEY \
  -v $(pwd)/outputs2:/app/outputs2 \
  -v $(pwd)/data_cache:/app/data_cache \
  --entrypoint python \
  options-search \
  prepare_alpaca_bot.py --symbol SPY --heavy-preset cpu_max --account-equity 100000
```

If you want the live bot from the same image:

```bash
docker run --rm \
  -e ALPACA_API_KEY=$ALPACA_API_KEY \
  -e ALPACA_SECRET_KEY=$ALPACA_SECRET_KEY \
  -e BOT_CONTROL_TOKEN=$BOT_CONTROL_TOKEN \
  -p 8080:8080 \
  -v $(pwd)/outputs2:/app/outputs2 \
  -v $(pwd)/data_cache:/app/data_cache \
  --entrypoint python \
  options-search \
  run_live_bot.py --symbol SPY --heavy-preset cpu_max --cycle-seconds 60 --control-token $BOT_CONTROL_TOKEN
```

## Important limitation

The backtester still prices options from underlying bars with Black-Scholes plus a realized-volatility IV proxy. That is useful for research and execution logic, but it is not the same thing as a historical option-chain backtest with real quotes and fills.
