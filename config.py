"""Central configuration.

Secrets come from `.env` (never committed). Every strategy and risk parameter lives
here so the backtester and the live bot run the exact same rules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- universe
SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

# Strategy timeframe. Alpaca's native 1Hour bars are clock-aligned (09:00-10:00 mixes
# pre-market prints into the first bar), so the client builds session-aligned hourly
# bars (09:30-10:30, ..., 15:30-16:00) from 30-minute source bars instead.
TIMEFRAME = "1Hour"
SOURCE_BAR_MINUTES = 30

TIMEZONE = "America/New_York"

JOURNAL_PATH = os.getenv("JOURNAL_PATH", "paper_trading_journal.csv")
LOG_DIR = os.getenv("LOG_DIR", "logs")


@dataclass(frozen=True)
class StrategyConfig:
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    adx_period: int = 14
    atr_period: int = 14

    # Regime ("WAIT") filters
    adx_min: float = 20.0
    natr_window: int = 100
    natr_low_pct: float = 0.15
    natr_high_pct: float = 0.92

    # Pullback trigger
    rsi_trigger: float = 42.0
    rsi_dip_lookback: int = 3

    # Bracket geometry (multiples of ATR at the signal bar)
    stop_atr_mult: float = 2.0
    target_atr_mult: float = 3.0

    # "Trailing stop if Close < EMA_fast". Pullback entries often trigger while Close is
    # still below EMA_fast, so the literal rule would exit one bar after entry. When True,
    # the exit only arms after a close above EMA_fast since the signal bar. False = literal.
    ema_exit_requires_arming: bool = True

    # Closed bars required before any signal. EMA-200 seeded at bar 0 still carries
    # ~1.8% seed weight after 400 bars; fewer bars means a biased trend filter.
    warmup_bars: int = 400


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01          # fraction of equity lost if the stop is hit
    max_position_notional: float = 0.20   # fraction of equity per position
    max_daily_drawdown: float = 0.03      # vs. previous close equity
    max_open_positions: int = 5
    kill_switch_flatten: bool = True      # flatten everything when the kill-switch trips


@dataclass(frozen=True)
class ExecutionConfig:
    trade_window_start: str = "09:45"     # ET, inclusive
    trade_window_end: str = "15:45"       # ET, inclusive
    data_feed: str = os.getenv("ALPACA_DATA_FEED", "iex")
    live_lookback_days: int = 180         # calendar days of history fetched each bar
    bar_settle_seconds: int = 20          # wait after a bar closes before fetching it
    incomplete_bar_grace_seconds: int = 120
    max_signal_staleness_seconds: int = 600
    heartbeat_seconds: int = 30           # kill-switch / fill reconciliation cadence
    fill_poll_timeout_seconds: float = 30.0
    cancel_poll_timeout_seconds: float = 10.0
    max_retries: int = 5
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 30.0


@dataclass(frozen=True)
class BacktestConfig:
    years: float = 2.0
    warmup_days: int = 120                # extra history fetched so indicators are converged
    initial_equity: float = 100_000.0
    slippage_pct: float = 0.0005          # 0.05% per side, applied to every fill
    n_folds: int = 4                      # chronological walk-forward segments


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_credentials() -> tuple[str, str, bool]:
    """Return (api_key, secret_key, paper). Raises if keys are missing."""
    key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret or key.startswith("your_"):
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. Copy .env.example to .env "
            "and add your paper-trading keys."
        )
    return key, secret, env_bool("ALPACA_PAPER", default=True)
