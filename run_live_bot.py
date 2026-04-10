from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from options_strategy_lab.live_bot import AlpacaRealtimeBot, LiveBotConfig


def _research_name(symbol: str, heavy_preset: str) -> str:
    return f"heavy_credit_spread_{symbol.lower()}_{heavy_preset}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Alpaca realtime bot and its control-center API.",
    )
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument(
        "--heavy-preset",
        choices=("balanced", "aggressive", "cpu_max"),
        default="cpu_max",
        help="Used only to infer the default model artifact names.",
    )
    parser.add_argument("--research-name", default=None)
    parser.add_argument("--model-dir", default="outputs2/models")
    parser.add_argument("--state-path", default="outputs2/deployment/live_bot_state.json")
    parser.add_argument("--event-log-path", default="outputs2/deployment/live_bot_events.jsonl")
    parser.add_argument("--api-key", default=os.environ.get("ALPACA_API_KEY"))
    parser.add_argument("--secret-key", default=os.environ.get("ALPACA_SECRET_KEY"))
    parser.add_argument("--paper", action="store_true", default=True)
    parser.add_argument("--live", action="store_true", help="Use the live trading endpoint instead of paper.")
    parser.add_argument("--cycle-seconds", type=int, default=60)
    parser.add_argument("--history-days", type=int, default=540)
    parser.add_argument("--account-equity-override", type=float, default=None)
    parser.add_argument("--stock-feed", choices=("iex", "sip"), default="iex")
    parser.add_argument("--options-feed", choices=("indicative", "opra"), default="indicative")
    parser.add_argument("--time-in-force", choices=("day", "gtc"), default="day")
    parser.add_argument("--entry-start-minute", type=int, default=5)
    parser.add_argument("--entry-end-minute", type=int, default=210)
    parser.add_argument("--control-token", default=os.environ.get("BOT_CONTROL_TOKEN"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser


def build_dashboard_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Options Bot Control Center</title>
  <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #0b1220; color: #e5edf8; margin: 0; padding: 24px; }
    .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-bottom: 16px; }
    .card { background: #111a2d; border: 1px solid #243251; border-radius: 12px; padding: 16px; }
    h1, h2 { margin: 0 0 12px; }
    button { margin-right: 8px; padding: 10px 14px; background: #1a7f64; color: white; border: 0; border-radius: 8px; cursor: pointer; }
    button.alt { background: #9a3412; }
    pre { white-space: pre-wrap; word-break: break-word; margin: 0; }
  </style>
</head>
<body>
  <h1>Options Bot Control Center</h1>
  <div class="row">
    <div class="card"><h2>Status</h2><pre id="status">Loading...</pre></div>
    <div class="card"><h2>Scheduler</h2><pre id="scheduler">Loading...</pre></div>
    <div class="card"><h2>Actions</h2>
      <button onclick="postAction('/control/run-now')">Run Now</button>
      <button onclick="postAction('/control/resume')">Resume</button>
      <button class="alt" onclick="postAction('/control/pause')">Pause</button>
    </div>
  </div>
  <div class="row">
    <div class="card"><h2>Account</h2><pre id="account">Loading...</pre></div>
    <div class="card"><h2>Signal</h2><pre id="signal">Loading...</pre></div>
    <div class="card"><h2>Active Trade</h2><pre id="trade">Loading...</pre></div>
  </div>
  <div class="card"><h2>Full State</h2><pre id="state">Loading...</pre></div>
  <script>
    async function postAction(path) {
      const token = window.localStorage.getItem('controlToken') || window.prompt('Control token (leave empty if none):') || '';
      window.localStorage.setItem('controlToken', token);
      const response = await fetch(path, {method: 'POST', headers: token ? {'X-Control-Token': token} : {}});
      if (!response.ok) {
        alert('Action failed: ' + await response.text());
        return;
      }
      await refresh();
    }
    async function refresh() {
      const response = await fetch('/state');
      const state = await response.json();
      document.getElementById('status').textContent = JSON.stringify({
        status: state.status,
        paused: state.paused,
        last_cycle_outcome: state.last_cycle_outcome,
        last_error: state.last_error
      }, null, 2);
      document.getElementById('scheduler').textContent = JSON.stringify({
        next_cycle_at: state.next_cycle_at,
        seconds_until_next_cycle: state.seconds_until_next_cycle,
        cycle_count: state.cycle_count
      }, null, 2);
      document.getElementById('account').textContent = JSON.stringify(state.account || {}, null, 2);
      document.getElementById('signal').textContent = JSON.stringify(state.latest_signal_plan || {}, null, 2);
      document.getElementById('trade').textContent = JSON.stringify(state.active_trade || {}, null, 2);
      document.getElementById('state').textContent = JSON.stringify(state, null, 2);
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>"""


def create_app(bot: AlpacaRealtimeBot) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        bot.start()
        try:
            yield
        finally:
            bot.stop()

    app = FastAPI(title="Options Bot Control Center", lifespan=lifespan)

    def require_token(x_control_token: str | None) -> None:
        if bot.config.control_token and x_control_token != bot.config.control_token:
            raise HTTPException(status_code=401, detail="Invalid control token.")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return build_dashboard_html()

    @app.get("/health")
    async def health() -> dict:
        state = bot.snapshot_state()
        return {
            "status": state["status"],
            "last_cycle_outcome": state["last_cycle_outcome"],
            "last_error": state["last_error"],
            "seconds_until_next_cycle": state["seconds_until_next_cycle"],
        }

    @app.get("/state")
    async def state() -> dict:
        return bot.snapshot_state()

    @app.post("/control/pause")
    async def pause(x_control_token: str | None = Header(default=None)) -> dict:
        require_token(x_control_token)
        bot.pause()
        return {"ok": True, "action": "pause"}

    @app.post("/control/resume")
    async def resume(x_control_token: str | None = Header(default=None)) -> dict:
        require_token(x_control_token)
        bot.resume()
        return {"ok": True, "action": "resume"}

    @app.post("/control/run-now")
    async def run_now(x_control_token: str | None = Header(default=None)) -> dict:
        require_token(x_control_token)
        bot.request_run_now()
        return {"ok": True, "action": "run-now"}

    return app


def main() -> None:
    args = build_parser().parse_args()
    if not args.api_key or not args.secret_key:
        msg = "Alpaca API credentials are required. Pass --api-key/--secret-key or set ALPACA_API_KEY and ALPACA_SECRET_KEY."
        raise ValueError(msg)

    research_name = args.research_name or _research_name(args.symbol, args.heavy_preset)
    model_root = Path(args.model_dir)
    bot = AlpacaRealtimeBot(
        LiveBotConfig(
            api_key=args.api_key,
            secret_key=args.secret_key,
            symbol=args.symbol,
            model_path=str(model_root / f"{research_name}.joblib"),
            metadata_path=str(model_root / f"{research_name}_metadata.json"),
            paper=not args.live,
            cycle_seconds=args.cycle_seconds,
            history_days=args.history_days,
            account_equity_override=args.account_equity_override,
            stock_feed=args.stock_feed,
            options_feed=args.options_feed,
            time_in_force=args.time_in_force,
            control_token=args.control_token,
            state_path=args.state_path,
            event_log_path=args.event_log_path,
            entry_start_minute_after_open=args.entry_start_minute,
            entry_end_minute_after_open=args.entry_end_minute,
        )
    )
    app = create_app(bot)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
