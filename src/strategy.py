"""Multi-factor, regime-filtered trend-pullback strategy (long only).

Anti-lookahead contract
-----------------------
`evaluate_closed_bars` computes every condition on bar k using data up to bar k's close.
`generate_signals` then applies `shift(1)`, so the row for decision bar t only carries
information from bars <= t-1. Orders for bar t are placed at (or right after) its open.
The live bot uses the same path: `decide_latest` appends a placeholder for the next bar
and reads its row, so live and backtest decisions come from identical code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import StrategyConfig
from src.indicators import add_indicators

LONG, EXIT, WAIT = "LONG", "EXIT", "WAIT"

R_WARMUP = "WARMUP"
R_CHOP = "CHOP_ADX_LOW"
R_DEAD = "VOLATILITY_DEAD"
R_SPIKE = "VOLATILITY_SPIKE"
R_TREND = "TREND_FILTER"
R_NO_PULLBACK = "NO_RSI_PULLBACK"
R_NO_CONFIRM = "NO_PRICE_CONFIRM"
R_ENTRY = "ENTRY_PULLBACK_RECLAIM"
R_EXIT = "EXIT_CLOSE_BELOW_EMA_FAST"
R_HOLD = "IN_POSITION_HOLD"

SNAPSHOT_COLUMNS = [
    "open", "high", "low", "close",
    "ema_fast", "ema_slow", "rsi", "adx", "atr", "natr", "natr_lo", "natr_hi",
]
FLAG_COLUMNS = ["warm", "chop", "dead", "spike", "trend_ok", "dipped", "crossed", "confirm", "entry", "exit",
                "above_fast"]


def evaluate_closed_bars(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Evaluate every rule on each bar, as known at that bar's close. No shift here."""
    ind = add_indicators(df, cfg)
    rsi = ind["rsi"]
    trigger = cfg.rsi_trigger

    required = ind[["ema_fast", "ema_slow", "rsi", "adx", "atr", "natr_lo", "natr_hi"]].notna().all(axis=1)
    rsi_min = rsi.rolling(cfg.rsi_dip_lookback, min_periods=cfg.rsi_dip_lookback).min()
    prev_high = ind["high"].shift(1)
    rsi_prev = rsi.shift(1)
    bar_number = np.arange(len(ind))
    warm = (bar_number < cfg.warmup_bars - 1) | ~required | rsi_min.isna() | rsi_prev.isna() | prev_high.isna()

    trend_ok = (ind["close"] > ind["ema_slow"]) & (ind["ema_fast"] > ind["ema_slow"])
    chop = ind["adx"] < cfg.adx_min
    dead = ind["natr"] < ind["natr_lo"]
    spike = ind["natr"] > ind["natr_hi"]
    dipped = rsi_min < trigger
    crossed = (rsi > trigger) & (rsi_prev <= trigger)
    confirm = ind["close"] > prev_high

    entry = ~warm & ~chop & ~dead & ~spike & trend_ok & dipped & crossed & confirm
    exit_ = ind["ema_fast"].notna() & (ind["close"] < ind["ema_fast"])
    above_fast = ind["close"] > ind["ema_fast"]

    reason = np.select(
        [warm, chop, dead, spike, ~trend_ok, ~(dipped & crossed), ~confirm],
        [R_WARMUP, R_CHOP, R_DEAD, R_SPIKE, R_TREND, R_NO_PULLBACK, R_NO_CONFIRM],
        default=R_ENTRY,
    )

    out = ind[SNAPSHOT_COLUMNS].copy()
    out["rsi_prev"] = rsi_prev
    out["rsi_min"] = rsi_min
    out["prev_high"] = prev_high
    for name, flag in zip(FLAG_COLUMNS, [warm, chop, dead, spike, trend_ok, dipped, crossed, confirm, entry, exit_,
                                       above_fast]):
        out[name] = flag.astype(bool)
    out["reason_code"] = reason
    out["ref_bar_time"] = out.index
    return out


def generate_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Decision frame indexed by decision bar t. Every column describes bar t-1."""
    closed = evaluate_closed_bars(df, cfg)
    sig = closed.shift(1)
    first = sig.index[:1]
    for name in FLAG_COLUMNS:
        sig[name] = sig[name].astype("boolean").fillna(name == "warm").astype(bool)
    sig.loc[first, "reason_code"] = R_WARMUP
    return sig


def decide_latest(closed_bars: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    """Decision for the bar that opens right after the last closed bar (live path)."""
    ohlc = closed_bars[["open", "high", "low", "close"]]
    placeholder = ohlc.index[-1] + pd.Timedelta(hours=1)
    frame = ohlc.reindex(ohlc.index.append(pd.DatetimeIndex([placeholder])))
    return generate_signals(frame, cfg).iloc[-1]


def trailing_exit_armed(closed_bars: pd.DataFrame, since: pd.Timestamp, cfg: StrategyConfig) -> bool:
    """True once any closed bar at or after `since` (the entry signal bar) closed above
    EMA_fast. Live-path equivalent of the incremental flag the backtester keeps."""
    if not cfg.ema_exit_requires_arming:
        return True
    ind = add_indicators(closed_bars[["open", "high", "low", "close"]], cfg)
    window = ind.loc[ind.index >= since]
    return bool((window["close"] > window["ema_fast"]).any())


def resolve_state(sig: pd.Series, in_position: bool, exit_armed: bool = True) -> tuple[str, str]:
    if in_position:
        return (EXIT, R_EXIT) if bool(sig["exit"]) and exit_armed else (WAIT, R_HOLD)
    if bool(sig["entry"]):
        return LONG, R_ENTRY
    return WAIT, str(sig["reason_code"])


def describe(sig: pd.Series, cfg: StrategyConfig, in_position: bool = False, exit_armed: bool = True) -> str:
    """Human-readable, numeric explanation of the decision. [t-1] = last closed bar."""
    f, s = cfg.ema_fast, cfg.ema_slow
    if in_position:
        if bool(sig["exit"]) and exit_armed:
            return f"Close[t-1] {sig['close']:.2f} < EMA{f}[t-1] {sig['ema_fast']:.2f}"
        if bool(sig["exit"]):
            return (f"Close[t-1] {sig['close']:.2f} < EMA{f}[t-1] {sig['ema_fast']:.2f} but trailing exit "
                    f"not armed (no close above EMA{f} since entry); bracket stop/target working")
        return (f"Close[t-1] {sig['close']:.2f} >= EMA{f}[t-1] {sig['ema_fast']:.2f}; "
                "bracket stop/target working")
    if bool(sig["warm"]):
        return f"Indicators not ready: need {cfg.warmup_bars} closed bars with valid EMA/RSI/ADX/ATR percentiles"

    parts = []
    if bool(sig["chop"]):
        parts.append(f"{R_CHOP}: ADX[t-1] {sig['adx']:.2f} < {cfg.adx_min:.2f}")
    if bool(sig["dead"]):
        parts.append(f"{R_DEAD}: NATR[t-1] {sig['natr']:.4%} < P{cfg.natr_low_pct * 100:.0f} "
                     f"{sig['natr_lo']:.4%} of last {cfg.natr_window} bars")
    if bool(sig["spike"]):
        parts.append(f"{R_SPIKE}: NATR[t-1] {sig['natr']:.4%} > P{cfg.natr_high_pct * 100:.0f} "
                     f"{sig['natr_hi']:.4%} of last {cfg.natr_window} bars")
    if not bool(sig["trend_ok"]):
        why = []
        if not sig["close"] > sig["ema_slow"]:
            why.append(f"Close[t-1] {sig['close']:.2f} <= EMA{s} {sig['ema_slow']:.2f}")
        if not sig["ema_fast"] > sig["ema_slow"]:
            why.append(f"EMA{f} {sig['ema_fast']:.2f} <= EMA{s} {sig['ema_slow']:.2f}")
        parts.append(f"{R_TREND}: " + " & ".join(why))
    if not bool(sig["dipped"]):
        parts.append(f"{R_NO_PULLBACK}: min RSI of last {cfg.rsi_dip_lookback} bars "
                     f"{sig['rsi_min']:.2f} >= {cfg.rsi_trigger:.2f}")
    elif not bool(sig["crossed"]):
        parts.append(f"{R_NO_PULLBACK}: RSI[t-2] {sig['rsi_prev']:.2f} -> RSI[t-1] {sig['rsi']:.2f} "
                     f"is not a cross above {cfg.rsi_trigger:.2f}")
    if not bool(sig["confirm"]):
        parts.append(f"{R_NO_CONFIRM}: Close[t-1] {sig['close']:.2f} <= High[t-2] {sig['prev_high']:.2f}")

    if parts:
        return " | ".join(parts)
    return (f"RSI {sig['rsi_prev']:.2f} -> {sig['rsi']:.2f} crossed {cfg.rsi_trigger:.2f} after dip to "
            f"{sig['rsi_min']:.2f}; Close[t-1] {sig['close']:.2f} > High[t-2] {sig['prev_high']:.2f}; "
            f"ADX {sig['adx']:.2f}; NATR {sig['natr']:.4%} in [{sig['natr_lo']:.4%}, {sig['natr_hi']:.4%}]; "
            f"Close > EMA{s} {sig['ema_slow']:.2f}, EMA{f} {sig['ema_fast']:.2f} > EMA{s}")
