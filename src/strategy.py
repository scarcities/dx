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
from src.risk_manager import round_price

LONG, EXIT, WAIT = "LONG", "EXIT", "WAIT"

R_WARMUP = "WARMUP"
R_CHOP = "CHOP_ADX_LOW"
R_DEAD = "VOLATILITY_DEAD"
R_SPIKE = "VOLATILITY_SPIKE"
R_COST = "COST_TOO_HIGH"
R_TREND = "TREND_FILTER"
R_DI = "DI_BEARISH"
R_NO_ANCHOR = "PULLBACK_TOO_LONG"
R_NO_PULLBACK = "NO_RSI_PULLBACK"
R_NO_CONFIRM = "NO_PRICE_CONFIRM"
R_ENTRY = "ENTRY_PULLBACK_RECLAIM"
R_EXIT = "EXIT_CLOSE_BELOW_EMA_FAST"
R_TRAIL = "EXIT_RATCHET_STOP_AT_PRICE"
R_TIME = "EXIT_MAX_HOLD"
R_RS = "RS_LAGGARD"
R_RS_NO_BENCHMARK = "RS_NO_BENCHMARK"
R_CALENDAR = "EARNINGS_CALENDAR_STALE"
R_EARNINGS = "EARNINGS_BLACKOUT"
R_HOLD = "IN_POSITION_HOLD"

SNAPSHOT_COLUMNS = [
    "open", "high", "low", "close",
    "ema_fast", "ema_slow", "rsi", "adx", "plus_di", "minus_di", "atr", "natr", "natr_lo", "natr_hi",
]
FLAG_COLUMNS = ["warm", "chop", "dead", "spike", "costly", "trend_ok", "no_anchor", "di_bear", "dipped", "crossed",
                "confirm", "entry", "exit", "above_fast"]


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
    n = len(ind)
    false = pd.Series(False, index=ind.index)
    # Anchor = last bar with RSI >= anchor_rsi at least rsi_dip_lookback bars back (causal: <= k-L).
    positions = pd.Series(np.where(rsi >= cfg.anchor_rsi, np.arange(n), np.nan), index=ind.index)
    anchor = positions.ffill().shift(cfg.rsi_dip_lookback)
    anchor_age = pd.Series(np.arange(n), index=ind.index) - anchor
    has_anchor = anchor.notna() & (anchor_age <= cfg.anchor_max_age)
    take = anchor.fillna(0).astype(int).to_numpy()

    def at_anchor(col: str) -> pd.Series:
        return pd.Series(ind[col].to_numpy()[take], index=ind.index).where(has_anchor)

    anchor_adx, anchor_pdi, anchor_mdi = at_anchor("adx"), at_anchor("plus_di"), at_anchor("minus_di")
    if cfg.directional_check == "anchor":
        no_anchor = ~has_anchor
        chop = has_anchor & (anchor_adx < cfg.adx_min)
        di_bear = has_anchor & (anchor_mdi >= anchor_pdi)
    elif cfg.directional_check == "signal_bar":
        no_anchor, chop = false, ind["adx"] < cfg.adx_min
        di_bear = ind["minus_di"] >= ind["plus_di"]
    elif cfg.directional_check is None:
        no_anchor, chop, di_bear = false, ind["adx"] < cfg.adx_min, false
    else:
        raise ValueError(f"directional_check must be None, 'signal_bar' or 'anchor', got {cfg.directional_check!r}")
    dead = ind["natr"] < ind["natr_lo"]
    spike = ind["natr"] > ind["natr_hi"]
    cost_r = cfg.round_trip_cost / (cfg.stop_atr_mult * ind["natr"])
    costly = cost_r > cfg.max_cost_r if cfg.max_cost_r is not None else pd.Series(False, index=ind.index)
    dipped = rsi_min < trigger
    crossed = (rsi > trigger) & (rsi_prev <= trigger)
    confirm = ind["close"] > prev_high

    entry = ~warm & ~no_anchor & ~chop & ~dead & ~spike & ~costly & trend_ok & ~di_bear & dipped & crossed & confirm
    exit_ = ind["ema_fast"].notna() & (ind["close"] < ind["ema_fast"])
    above_fast = ind["close"] > ind["ema_fast"]

    reason = np.select(
        [warm, no_anchor, chop, dead, spike, costly, ~trend_ok, di_bear, ~(dipped & crossed), ~confirm],
        [R_WARMUP, R_NO_ANCHOR, R_CHOP, R_DEAD, R_SPIKE, R_COST, R_TREND, R_DI, R_NO_PULLBACK, R_NO_CONFIRM],
        default=R_ENTRY,
    )

    out = ind[SNAPSHOT_COLUMNS].copy()
    out["rsi_prev"] = rsi_prev
    out["rsi_min"] = rsi_min
    out["prev_high"] = prev_high
    out["cost_r"] = cost_r
    out["anchor_adx"], out["anchor_plus_di"], out["anchor_minus_di"] = anchor_adx, anchor_pdi, anchor_mdi
    out["anchor_age"] = anchor_age.where(has_anchor)
    flags = {"warm": warm, "chop": chop, "dead": dead, "spike": spike, "costly": costly, "trend_ok": trend_ok,
             "no_anchor": no_anchor, "di_bear": di_bear, "dipped": dipped, "crossed": crossed, "confirm": confirm,
             "entry": entry, "exit": exit_, "above_fast": above_fast}
    assert list(flags) == FLAG_COLUMNS
    for name, flag in flags.items():
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


def ratchet_stop(current_stop: float, entry_fill: float, entry_ref: float, initial_stop: float,
                 highest_close: float, atr: float, cfg: StrategyConfig) -> float:
    """Stop for an open long after the latest close ("ratchet" mode). Never below `current_stop`.

    Inactive until the highest close since entry reaches entry_ref + trail_trigger_r * R, where
    R = entry_ref - initial_stop (the planned risk; entry_ref is Close[t-1] at the signal, so the
    trigger is identical in backtest and live). Then the stop is the higher of breakeven (the
    actual entry fill plus the exit half of the round-trip cost) and highest_close - trail_atr_mult
    * ATR. Moves smaller than max(ratchet_min_step_r * R, $0.02) are ignored to avoid order churn."""
    risk = entry_ref - initial_stop
    if cfg.exit_mode != "ratchet" or risk <= 0 or not np.isfinite(highest_close) or not np.isfinite(atr):
        return current_stop
    if highest_close < entry_ref + cfg.trail_trigger_r * risk:
        return current_stop
    breakeven = entry_fill * (1.0 + cfg.round_trip_cost / 2)
    candidate = round_price(max(breakeven, highest_close - cfg.trail_atr_mult * atr))
    if candidate - current_stop < max(cfg.ratchet_min_step_r * risk, 0.02):
        return current_stop
    return candidate


def stop_too_close(stop: float, last_close: float, cfg: StrategyConfig) -> bool:
    """A sell stop at or just under the market is rejected or fires at once: exit at market instead."""
    return stop >= last_close * (1.0 - cfg.stop_buffer_pct)


# ----------------------------------------------------------------------------- cross-sectional
# Both engines call these with the same per-symbol values from bar t-1, so the live bot and the
# backtester fund the same candidates in the same order.

def trend_spread(close: float, ema_slow: float) -> float:
    """Close / EMA_slow - 1 on the signal bar: a parameter-free measure of trend strength."""
    if not (np.isfinite(close) and np.isfinite(ema_slow)) or ema_slow <= 0:
        return float("nan")
    return close / ema_slow - 1.0


def rank_metric(sig_values: dict, cfg: StrategyConfig) -> float:
    """Value used to order simultaneous entry candidates (higher first). `sig_values` holds the
    t-1 snapshot: natr, adx, close, ema_slow."""
    if cfg.signal_rank_by is None:
        return 0.0
    if cfg.signal_rank_by == "natr":
        value = sig_values.get("natr")
    elif cfg.signal_rank_by == "adx":
        value = sig_values.get("adx")
    elif cfg.signal_rank_by == "rs_ema":
        value = trend_spread(sig_values.get("close", np.nan), sig_values.get("ema_slow", np.nan))
    else:
        raise ValueError(f"signal_rank_by must be None, 'natr', 'adx' or 'rs_ema', got {cfg.signal_rank_by!r}")
    value = float(value) if value is not None else float("nan")
    return value if np.isfinite(value) else float("-inf")


def rank_candidates(candidates: list[str], metrics: dict[str, float], order: list[str],
                    cfg: StrategyConfig) -> list[str]:
    """Candidates sorted by metric descending, ties broken by symbol name. With signal_rank_by
    None the v2 behaviour is kept: the order of `order` (the SYMBOLS list)."""
    if cfg.signal_rank_by is None:
        position = {s: i for i, s in enumerate(order)}
        return sorted(candidates, key=lambda s: position.get(s, len(position)))
    return sorted(candidates, key=lambda s: (-metrics.get(s, float("-inf")), s))


def relative_strength_ok(symbol: str, spreads: dict[str, float], cfg: StrategyConfig,
                         universe: list[str]) -> tuple[bool, str, str]:
    """(passes, reason_code, detail) for the RS filter. `spreads` are t-1 trend spreads of every
    symbol available on this bar (the benchmark included even if it is not traded)."""
    if cfg.rs_filter is None:
        return True, "", ""
    own = spreads.get(symbol, float("nan"))
    if not np.isfinite(own):
        return False, R_RS, f"{R_RS}: no trend spread for {symbol}"
    if cfg.rs_filter == "vs_benchmark":
        bench = spreads.get(cfg.rs_benchmark, float("nan"))
        if not np.isfinite(bench):
            return False, R_RS_NO_BENCHMARK, f"{R_RS_NO_BENCHMARK}: no {cfg.rs_benchmark} bar to compare with"
        ok = own >= bench
        return ok, ("" if ok else R_RS), (f"{R_RS}: Close/EMA{cfg.ema_slow}-1 {own:+.2%} < "
                                          f"{cfg.rs_benchmark} {bench:+.2%}" if not ok else "")
    if cfg.rs_filter == "vs_median":
        values = [spreads[s] for s in universe if s in spreads and np.isfinite(spreads[s])]
        median = float(np.median(values))
        ok = own >= median
        return ok, ("" if ok else R_RS), (f"{R_RS}: Close/EMA{cfg.ema_slow}-1 {own:+.2%} < universe median "
                                          f"{median:+.2%} ({len(values)} symbols)" if not ok else "")
    raise ValueError(f"rs_filter must be None, 'vs_benchmark' or 'vs_median', got {cfg.rs_filter!r}")


def resolve_state(sig: pd.Series, in_position: bool, ema_exit_allowed: bool = True,
                  blackout: bool = False) -> tuple[str, str]:
    """`ema_exit_allowed`: exit_mode is "ema_cross" and the exit is armed. `blackout`: an
    earnings event is inside its blackout window (close positions, open nothing)."""
    if in_position:
        if blackout:
            return EXIT, R_EARNINGS
        return (EXIT, R_EXIT) if bool(sig["exit"]) and ema_exit_allowed else (WAIT, R_HOLD)
    if blackout:
        return WAIT, R_EARNINGS
    if bool(sig["entry"]):
        return LONG, R_ENTRY
    return WAIT, str(sig["reason_code"])


def describe(sig: pd.Series, cfg: StrategyConfig, in_position: bool = False, exit_armed: bool = True) -> str:
    """Human-readable, numeric explanation of the decision. [t-1] = last closed bar."""
    f, s = cfg.ema_fast, cfg.ema_slow
    if in_position and cfg.exit_mode != "ema_cross":
        return f"Holding: Close[t-1] {sig['close']:.2f}, ATR {sig['atr']:.2f}; {cfg.exit_mode} exit working"
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
    anchored = cfg.directional_check == "anchor"
    if bool(sig["no_anchor"]):
        parts.append(f"{R_NO_ANCHOR}: no bar with RSI >= {cfg.anchor_rsi:.0f} between {cfg.rsi_dip_lookback} and "
                     f"{cfg.anchor_max_age} bars back")
    if bool(sig["chop"]) and anchored:
        parts.append(f"{R_CHOP}: ADX at the pre-dip anchor ({sig['anchor_age']:.0f} bars back) "
                     f"{sig['anchor_adx']:.2f} < {cfg.adx_min:.2f}")
    elif bool(sig["chop"]):
        parts.append(f"{R_CHOP}: ADX[t-1] {sig['adx']:.2f} < {cfg.adx_min:.2f}")
    if bool(sig["dead"]):
        parts.append(f"{R_DEAD}: NATR[t-1] {sig['natr']:.4%} < P{cfg.natr_low_pct * 100:.0f} "
                     f"{sig['natr_lo']:.4%} of last {cfg.natr_window} bars")
    if bool(sig["spike"]):
        parts.append(f"{R_SPIKE}: NATR[t-1] {sig['natr']:.4%} > P{cfg.natr_high_pct * 100:.0f} "
                     f"{sig['natr_hi']:.4%} of last {cfg.natr_window} bars")
    if bool(sig["costly"]):
        parts.append(f"{R_COST}: round trip {cfg.round_trip_cost:.3%} / stop {cfg.stop_atr_mult * sig['natr']:.3%} "
                     f"= {sig['cost_r']:.2f}R > {cfg.max_cost_r:.2f}R")
    if not bool(sig["trend_ok"]):
        why = []
        if not sig["close"] > sig["ema_slow"]:
            why.append(f"Close[t-1] {sig['close']:.2f} <= EMA{s} {sig['ema_slow']:.2f}")
        if not sig["ema_fast"] > sig["ema_slow"]:
            why.append(f"EMA{f} {sig['ema_fast']:.2f} <= EMA{s} {sig['ema_slow']:.2f}")
        parts.append(f"{R_TREND}: " + " & ".join(why))
    if bool(sig["di_bear"]) and anchored:
        parts.append(f"{R_DI}: at the pre-dip anchor -DI {sig['anchor_minus_di']:.2f} >= +DI "
                     f"{sig['anchor_plus_di']:.2f}")
    elif bool(sig["di_bear"]):
        parts.append(f"{R_DI}: -DI[t-1] {sig['minus_di']:.2f} >= +DI {sig['plus_di']:.2f}")
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
            f"ADX {sig['adx']:.2f} (+DI {sig['plus_di']:.1f}, -DI {sig['minus_di']:.1f}); "
            f"NATR {sig['natr']:.4%} in [{sig['natr_lo']:.4%}, {sig['natr_hi']:.4%}], cost {sig['cost_r']:.2f}R; "
            f"Close > EMA{s} {sig['ema_slow']:.2f}, EMA{f} {sig['ema_fast']:.2f} > EMA{s}")
