"""Central configuration.

Secrets come from `.env` (never committed). Every strategy and risk parameter lives
here so the backtester and the live bot run the exact same rules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- universe
# Universe presets for research (backtest.py --universe). The live bot trades SYMBOLS only;
# change it only after an expanded universe has passed a held-out --compare.
UNIVERSES = {
    "core5": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"],
    "expanded12": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "AMD", "AVGO", "NFLX", "JPM"],
}
SYMBOLS = list(UNIVERSES["core5"])

# Strategy timeframe. Alpaca's native 1Hour bars are clock-aligned (09:00-10:00 mixes
# pre-market prints into the first bar), so the client builds session-aligned hourly
# bars (09:30-10:30, ..., 15:30-16:00) from 30-minute source bars instead.
TIMEFRAME = "1Hour"
SOURCE_BAR_MINUTES = 30

TIMEZONE = "America/New_York"

JOURNAL_PATH = os.getenv("JOURNAL_PATH", "paper_trading_journal.csv")
# symbol,date,timing rows (see src/events.py and earnings_calendar.example.csv).
EARNINGS_CALENDAR_PATH = os.getenv("EARNINGS_CALENDAR", "earnings_calendar.csv")
# Funds have no earnings; everything else needs calendar dates (the live bot fails closed without them).
ETF_SYMBOLS = frozenset({"SPY", "QQQ", "DIA", "IWM"})
BACKTEST_LEDGER_PATH = os.getenv("BACKTEST_LEDGER", "backtest_ledger.csv")

# One cost assumption shared by the backtest fills and the strategy's cost floor, so they cannot
# drift apart. 2 bp per side on market and stop fills: a realistic market-order cost for these
# mega-caps, but optimistic for stops in fast markets. Check it against paper fills
# (evaluate_trust.py) and stress-test with backtest.py --slippage-bps 5.
SLIPPAGE_PER_SIDE = float(os.getenv("SLIPPAGE_PER_SIDE", "0.0002"))
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

    # Exit model (v2).
    #   "bracket"   stop + target only (default: no tuned parameters).
    #   "ratchet"   bracket, and once a close reaches entry_ref + trail_trigger_r * R the stop moves
    #               to max(breakeven, highest close - trail_atr_mult * ATR), never lower. With
    #               R = 2 ATR and a 1.5R target this is a breakeven stop that can lock at most +0.5R.
    #   "ema_cross" v1: bracket plus an exit on a close below EMA_fast (noise-triggered on 1h bars).
    # Optional time stop for any mode: exit at market after max_hold_bars closed bars in the trade.
    exit_mode: str = "ratchet"
    trail_trigger_r: float = 1.0
    trail_atr_mult: float = 2.0
    ratchet_min_step_r: float = 0.10      # ignore stop moves smaller than 0.1R (or $0.02)
    stop_buffer_pct: float = 0.002        # a new stop within 0.2% of the last close exits at market
    max_hold_bars: Optional[int] = None

    # ema_cross only. Pullback entries often fire while Close is still below EMA_fast, so the
    # literal rule would exit one bar after entry; True arms it only after a close above EMA_fast.
    ema_exit_requires_arming: bool = True

    # Cost floor (v2, OFF by default; set max_cost_r = 0.10 to enable). WAIT when the round-trip
    # cost exceeds max_cost_r of the planned 1R stop, i.e. ATR/Close < round_trip_cost /
    # (stop_atr_mult * max_cost_r). At 2 x 5 bp that floor is 0.50%: SPY is then never traded and
    # QQQ rarely, so the "fix" is a universe change that rests on an unmeasured cost. Measure real
    # slippage first (paper journal), set SLIPPAGE_PER_SIDE, then enable it.
    round_trip_cost: float = 2 * SLIPPAGE_PER_SIDE
    max_cost_r: Optional[float] = None

    # Directional trend check (v2, OFF by default, hypothesis-grade).
    #   None          v1: ADX[t-1] >= adx_min, no direction.
    #   "signal_bar"  also +DI > -DI on t-1 (the proposal). RSI and DI share Wilder-14 smoothing,
    #                 so right after RSI recovers from < 42, -DI still leads on ~90% of entries:
    #                 this deletes almost every trade.
    #   "anchor"      measure trend strength before the dip contaminates it: at the anchor bar (last
    #                 bar with RSI >= anchor_rsi, between rsi_dip_lookback and anchor_max_age bars
    #                 back) require ADX >= adx_min and +DI > -DI, INSTEAD of ADX on t-1. No anchor
    #                 in range = WAIT (PULLBACK_TOO_LONG).
    directional_check: Optional[str] = None
    anchor_rsi: float = 50.0
    anchor_max_age: int = 30

    # Cross-sectional allocation (v3). When several symbols signal on the same bar and capital or
    # max_open_positions cannot fund all of them, they are funded in descending order of this
    # metric on t-1 (symbol name breaks ties): "natr" (ATR/Close), "adx", "rs_ema" (Close/EMA_slow
    # - 1), or None for SYMBOLS list order (v2). Ranking decides WHO is funded; it does not create
    # capital. Note: in 2022-2026 "natr" mostly means "NVDA first", the hindsight winner.
    signal_rank_by: Optional[str] = "natr"

    # Relative-strength filter (v3, OFF by default, hypothesis-grade). Uses the trend spread
    # Close/EMA_slow - 1 on t-1, so it adds no new parameter:
    #   "vs_benchmark"  spread >= the benchmark's spread (rs_benchmark, SPY by default)
    #   "vs_median"     spread >= the median spread of the traded universe on that bar
    rs_filter: Optional[str] = None
    rs_benchmark: str = "SPY"

    # Earnings (see src/events.py): on the last session before the gap, open positions are closed
    # at this time (checked every heartbeat from the calendar), or at the last decision bar if that
    # session closes early. A failed exit is retried with a 30 s -> 10 min backoff.
    earnings_exit_time: str = "14:30"

    # Closed bars required before any signal. EMA-200 seeded at bar 0 still carries
    # ~1.8% seed weight after 400 bars; fewer bars means a biased trend filter.
    warmup_bars: int = 400


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01          # fraction of equity lost if the stop is hit
    max_position_notional: float = 0.33   # fraction of equity per position
    # Gross long exposure cap as a fraction of equity, applied identically in the backtest and the
    # live bot. 1.0 = no margin. Without it the live bot would size against Alpaca's margin buying
    # power (2x-4x equity) while the backtest is limited to cash: 5 x 0.33 = 165% gross.
    max_gross_exposure: float = 1.0
    max_daily_drawdown: float = 0.03      # vs. previous close equity
    max_open_positions: int = 5
    kill_switch_flatten: bool = True      # flatten everything when the kill-switch trips
    # Live only: no new entries in a stock whose calendar has no event in the next 100 days.
    earnings_fail_closed: bool = True


@dataclass(frozen=True)
class ExecutionConfig:
    trade_window_start: str = "09:45"     # ET, inclusive
    trade_window_end: str = "15:45"       # ET, inclusive
    data_feed: str = os.getenv("ALPACA_DATA_FEED", "iex")
    live_lookback_days: int = 180         # calendar days of history fetched each bar
    bar_settle_seconds: int = 20          # wait after a bar closes before fetching it
    incomplete_bar_grace_seconds: int = 120
    max_signal_staleness_seconds: int = 600
    # New entries wait up to this long after a bar closes for every symbol's bar, so ranking and
    # the RS filter see the whole cross-section (as in the backtest). After that they are ranked
    # among the symbols that have arrived; exits and stop ratchets never wait.
    rank_wait_seconds: int = 120
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
    slippage_pct: float = SLIPPAGE_PER_SIDE  # market entries, market/stop exits and gap fills
    # A sell limit cannot fill below its price, so take-profit fills get no adverse slippage, but
    # (queue position) they only count when the high trades through the target by a cent.
    limit_slippage_pct: float = 0.0
    tp_requires_trade_through: bool = True
    n_folds: int = 4                      # chronological walk-forward segments
    # Annual yield earned on idle cash (T-bills / sweep), accrued daily. Sharpe and Sortino are
    # computed on returns in excess of it, so a strategy that sits in cash 75% of the time is
    # compared with what that cash would have earned anyway. 0 = v2 behaviour.
    cash_yield: float = 0.0


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
