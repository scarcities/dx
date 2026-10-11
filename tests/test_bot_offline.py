"""Offline test suite. No network and no API keys: run `pytest -q` (or `pytest -s` to see the tear sheet).

The synthetic generator builds 2,000 session-aligned hourly bars in four regimes:
  BULL  (800 bars) strong uptrend with periodic pullbacks
  CHOP  (400 bars) sideways zig-zag, balanced directional movement (low ADX)
  CRASH (300 bars) high-volatility selloff
  DEAD  (500 bars) near-zero volatility
The regimes are deliberately unambiguous: they test the filter logic, not market edge.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import backtest
import evaluate_trust
from backtest import Backtester, format_tear_sheet, load_bars, save_bars
from bot import TradingBot
from config import BacktestConfig, ExecutionConfig, RiskConfig, StrategyConfig
from src import journal as J
from src.events import (entry_blackout, event_window, events_per_year, forced_exit, has_upcoming_event,
                        load_earnings_calendar)
from src.indicators import add_indicators
from src.journal import COLUMNS, TradeJournal, read_journal, slippage_bps
from src.risk_manager import RiskManager, client_order_id, r_multiple
from src.sessions import ET, complete_bars, in_trading_window, latest_bar_end, to_session_bars
from src.strategy import (EXIT, LONG, R_CHOP, R_COST, R_DEAD, R_DI, R_EARNINGS, R_ENTRY, R_SPIKE, WAIT,
                          decide_latest, describe, evaluate_closed_bars, generate_signals, ratchet_stop, resolve_state)

CFG = StrategyConfig()                                                    # v2 defaults
V1 = replace(CFG, exit_mode="ema_cross", max_cost_r=None, directional_check=None)  # the original spec
TRAIL = replace(CFG, exit_mode="ratchet", max_cost_r=None, directional_check=None)  # v1 entries, ratchet
LIVE_RISK = RiskConfig(earnings_fail_closed=False)   # synthetic symbols have no earnings calendar
OHLC = ["open", "high", "low", "close"]
REGIMES = [("BULL", 800), ("CHOP", 400), ("CRASH", 300), ("DEAD", 500)]
SEEDS = [7, 1, 2, 3]


# ============================================================================ synthetic data
def session_index(n: int, start: str = "2023-01-03") -> pd.DatetimeIndex:
    days = pd.bdate_range(start, periods=n // 7 + 2, tz=ET.zone)
    stamps = [d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=h) for d in days for h in range(7)]
    return pd.DatetimeIndex(stamps[:n], name="timestamp")


def make_synthetic_ohlcv(seed: int = 7, start: str = "2023-01-03", bull_noise: float = 0.0025,
                         bull_drift: float = 0.0030) -> pd.DataFrame:
    """2,000 hourly bars across the four regimes, with a `regime` label column.
    Raise `bull_noise` to make pullbacks fail more often (stop-losses actually get hit)."""
    rng = np.random.default_rng(seed)
    log_price, anchor = np.log(100.0), None
    path, sigmas, labels = [], [], []
    for name, length in REGIMES:
        for i in range(length):
            if name == "BULL":      # 20 bars up, 8 bars pullback
                sigma = bull_noise
                step = (bull_drift if i % 28 < 20 else -0.0050) + rng.normal(0, sigma)
            elif name == "CHOP":    # alternating bars pinned around the regime's starting level
                anchor = log_price if i == 0 else anchor
                sigma = 0.0035
                step = (0.004 if i % 2 == 0 else -0.004) - 0.5 * (log_price - anchor) + rng.normal(0, 0.001)
            elif name == "CRASH":   # ~8x the volatility, negative drift
                sigma = 0.02
                step = -0.004 + rng.normal(0, sigma)
            else:                   # DEAD: tiny noise around a flat level
                anchor = log_price if i == 0 else anchor
                sigma = 0.0004
                step = -0.2 * (log_price - anchor) + rng.normal(0, sigma)
            log_price += step
            path.append(log_price)
            sigmas.append(sigma)
            labels.append(name)
    close = np.exp(np.array(path))
    sigmas = np.array(sigmas)
    prev_close = np.concatenate([[100.0], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0, sigmas * 0.2))
    wicks = np.abs(rng.normal(0, sigmas * 0.5, (2, len(close))))
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) * np.exp(wicks[0]),
        "low": np.minimum(open_, close) * np.exp(-wicks[1]),
        "close": close,
        "volume": rng.integers(100_000, 1_000_000, len(close)).astype(float),
        "regime": labels,
    }, index=session_index(len(close), start))


def with_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the bar_end / n_source / n_expected columns the live client attaches."""
    out = df[OHLC + ["volume"]].copy()
    start_min = out.index.hour * 60 + out.index.minute
    end_min = np.minimum(np.asarray(start_min) + 60, 16 * 60)
    out["bar_end"] = out.index.normalize() + pd.to_timedelta(end_min, unit="min")
    out["n_expected"] = (end_min - np.asarray(start_min)) // 30
    out["n_source"] = out["n_expected"]
    return out


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    return make_synthetic_ohlcv(7)


@pytest.fixture(scope="module")
def signals(synthetic) -> pd.DataFrame:
    return generate_signals(synthetic[OHLC], CFG)


def test_synthetic_generator_shape(synthetic):
    assert len(synthetic) == 2000
    assert synthetic["regime"].value_counts().to_dict() == dict(REGIMES)
    assert (synthetic["high"] >= synthetic[["open", "close"]].max(axis=1)).all()
    assert (synthetic["low"] <= synthetic[["open", "close"]].min(axis=1)).all()
    assert synthetic.index.is_monotonic_increasing and synthetic.index.tz is not None
    natr = add_indicators(synthetic[OHLC], CFG).groupby(synthetic["regime"])["natr"].median()
    assert natr["CRASH"] > 4 * natr["BULL"] and natr["DEAD"] < natr["BULL"] / 4


# ============================================================================ indicators
def _ref_wilder(values: list[float], period: int) -> list[float]:
    out = [math.nan] * len(values)
    first = next(i for i, v in enumerate(values) if not math.isnan(v))
    seed = first + period - 1
    out[seed] = sum(values[first:seed + 1]) / period
    for i in range(seed + 1, len(values)):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return out


def _ref_indicators(df: pd.DataFrame, n: int = 14) -> dict[str, list[float]]:
    h, l, c = df["high"].tolist(), df["low"].tolist(), df["close"].tolist()
    gains, losses, tr, pdm, mdm = [math.nan], [math.nan], [h[0] - l[0]], [math.nan], [math.nan]
    for i in range(1, len(c)):
        change = c[i] - c[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
        up, down = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm.append(up if up > down and up > 0 else 0.0)
        mdm.append(down if down > up and down > 0 else 0.0)
    ag, al = _ref_wilder(gains, n), _ref_wilder(losses, n)
    rsi = [math.nan if math.isnan(g) else (100.0 if lo == 0 else 100 - 100 / (1 + g / lo)) for g, lo in zip(ag, al)]
    atr = _ref_wilder(tr, n)
    tr_dm = [math.nan] + tr[1:]
    s_tr, s_p, s_m = _ref_wilder(tr_dm, n), _ref_wilder(pdm, n), _ref_wilder(mdm, n)
    dx = []
    for t, p, m in zip(s_tr, s_p, s_m):
        if math.isnan(t):
            dx.append(math.nan)
            continue
        pdi, mdi = 100 * p / t, 100 * m / t
        dx.append(0.0 if pdi + mdi == 0 else 100 * abs(pdi - mdi) / (pdi + mdi))
    return {"rsi": rsi, "atr": atr, "adx": _ref_wilder(dx, n)}


def test_indicators_match_textbook_wilder_loops(synthetic):
    df = synthetic[OHLC].iloc[:600]
    ind = add_indicators(df, CFG)
    ref = _ref_indicators(df)
    for name in ("rsi", "atr", "adx"):
        np.testing.assert_allclose(ind[name].to_numpy(), np.array(ref[name]), rtol=1e-9, atol=1e-9,
                                   equal_nan=True, err_msg=name)
    ema = [math.nan] * len(df)
    alpha, value = 2 / (CFG.ema_fast + 1), df["close"].iloc[0]
    for i, price in enumerate(df["close"]):
        value = price if i == 0 else alpha * price + (1 - alpha) * value
        ema[i] = value if i >= CFG.ema_fast - 1 else math.nan
    np.testing.assert_allclose(ind["ema_fast"].to_numpy(), ema, rtol=1e-12, equal_nan=True)


def test_indicator_edge_cases():
    idx = session_index(60)
    rising = pd.DataFrame({"close": np.linspace(100, 160, 60)}, index=idx)
    rising["open"] = rising["close"] - 0.5
    rising["high"], rising["low"] = rising["close"] + 0.1, rising["open"] - 0.1
    ind = add_indicators(rising[OHLC], CFG)
    assert ind["rsi"].dropna().eq(100.0).all()                  # no losses -> RSI 100
    assert ind["rsi"].iloc[:CFG.rsi_period].isna().all()        # Wilder needs `period` deltas
    assert ind["adx"].dropna().iloc[-1] > 90                     # one-way trend -> ADX near 100
    flat = rising.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    assert add_indicators(flat[OHLC], CFG)["rsi"].dropna().eq(50.0).all()


# ============================================================================ look-ahead bias
LOOKAHEAD_POINTS = [450, 527, 600, 801, 1000, 1201, 1650, 1999]


def test_altering_bar_t_never_changes_decisions_up_to_t(synthetic, signals):
    ohlc = synthetic[OHLC]
    for t in LOOKAHEAD_POINTS:
        mutated = ohlc.copy()
        mutated.iloc[t:] = mutated.iloc[t:] * 1.37
        mutated.iloc[t, mutated.columns.get_loc("high")] *= 3.0     # absurd bar t
        mutated.iloc[t, mutated.columns.get_loc("low")] *= 0.2
        result = generate_signals(mutated, CFG)
        pd.testing.assert_frame_equal(result.iloc[:t + 1], signals.iloc[:t + 1], check_exact=True)
        if t + 1 < len(ohlc):  # the test is not vacuous: bar t does feed decision t+1
            assert result["close"].iloc[t + 1] != signals["close"].iloc[t + 1]


def test_lookahead_check_catches_an_unshifted_signal(synthetic):
    """Negative control: without shift(1), altering bar t changes the row for bar t."""
    ohlc = synthetic[OHLC]
    t = 600
    mutated = ohlc.copy()
    mutated.iloc[t:] = mutated.iloc[t:] * 1.37
    unshifted_a = evaluate_closed_bars(ohlc, CFG)
    unshifted_b = evaluate_closed_bars(mutated, CFG)
    assert unshifted_a["close"].iloc[t] != unshifted_b["close"].iloc[t]


def test_prefix_invariance(synthetic, signals):
    for t in LOOKAHEAD_POINTS:
        prefix = generate_signals(synthetic[OHLC].iloc[:t + 1], CFG)
        pd.testing.assert_frame_equal(prefix, signals.iloc[:t + 1], check_exact=True)


def test_live_decision_path_matches_backtest_signals(synthetic, signals):
    ohlc = synthetic[OHLC]
    long_rows = list(np.flatnonzero(signals["entry"].to_numpy()))
    for t in LOOKAHEAD_POINTS[:-1] + long_rows:
        live = decide_latest(ohlc.iloc[:t], CFG)
        pd.testing.assert_series_equal(live, signals.iloc[t], check_names=False, check_exact=True)


# ============================================================================ strategy rules
ANCHOR = replace(CFG, directional_check="anchor")


@pytest.mark.parametrize("cfg", [CFG, V1, replace(CFG, max_cost_r=0.10), ANCHOR,
                                 replace(CFG, directional_check="signal_bar")],
                         ids=["v2", "v1", "cost-floor", "anchor", "signal-bar"])
def test_long_signals_satisfy_every_rule_and_nothing_else_does(synthetic, cfg):
    ind = add_indicators(synthetic[OHLC], cfg)
    rsi, close, high = ind["rsi"].to_numpy(), ind["close"].to_numpy(), ind["high"].to_numpy()
    adx, pdi, mdi = ind["adx"].to_numpy(), ind["plus_di"].to_numpy(), ind["minus_di"].to_numpy()
    expected = np.zeros(len(ind), dtype=bool)
    for t in range(cfg.warmup_bars, len(ind)):
        k = t - 1  # last closed bar
        row = ind.iloc[k]
        cost_ok = cfg.max_cost_r is None or cfg.round_trip_cost / (cfg.stop_atr_mult * row.natr) <= cfg.max_cost_r
        if cfg.directional_check == "anchor":     # last RSI >= 50 bar, 3..30 bars back
            js = [j for j in range(k - cfg.rsi_dip_lookback, -1, -1) if rsi[j] >= cfg.anchor_rsi]
            j = js[0] if js else None
            trend_ok = j is not None and k - j <= cfg.anchor_max_age and adx[j] >= cfg.adx_min and pdi[j] > mdi[j]
        else:
            trend_ok = adx[k] >= cfg.adx_min and (cfg.directional_check != "signal_bar" or pdi[k] > mdi[k])
        expected[t] = bool(
            row.close > row.ema_slow and row.ema_fast > row.ema_slow
            and trend_ok and row.natr_lo <= row.natr <= row.natr_hi
            and cost_ok
            and rsi[k] > cfg.rsi_trigger >= rsi[k - 1]
            and min(rsi[k - 2:k + 1]) < cfg.rsi_trigger
            and close[k] > high[k - 1]
        )
    if cfg.directional_check != "signal_bar":
        assert expected.sum() >= 5
    np.testing.assert_array_equal(generate_signals(synthetic[OHLC], cfg)["entry"].to_numpy(), expected)


def test_cost_floor_uses_round_trip_cost_over_stop_width(synthetic):
    pricey = replace(CFG, round_trip_cost=0.004, max_cost_r=0.10)   # 0.4% round trip: floor at NATR 2%
    sig = generate_signals(synthetic[OHLC], pricey)
    live = sig[~sig["warm"]]
    np.testing.assert_allclose(live["cost_r"], 0.004 / (2.0 * live["natr"]))
    assert (live["costly"] == (live["cost_r"] > 0.10)).all()
    assert not live["entry"].any() and (live["reason_code"] == R_COST).any()
    row = live[live["reason_code"] == R_COST].iloc[0]
    assert describe(row, pricey).startswith(f"{R_COST}: round trip 0.400%")
    assert not generate_signals(synthetic[OHLC], replace(pricey, max_cost_r=None))["costly"].any()


def test_signal_bar_di_check_deletes_most_pullback_entries():
    """RSI and DI share Wilder-14 smoothing: just after RSI recovers from < 42, -DI usually still
    leads, so '+DI > -DI on the signal bar' removes most entries. The anchor check does not."""
    counts = {"none": 0, "signal_bar": 0, "anchor": 0}
    for seed in SEEDS:
        df = make_synthetic_ohlcv(seed)[OHLC]
        for mode in counts:
            cfg = replace(V1, directional_check=None if mode == "none" else mode)
            counts[mode] += int(generate_signals(df, cfg)["entry"].sum())
    assert counts["signal_bar"] <= 0.25 * counts["none"]
    assert counts["anchor"] >= 0.5 * counts["none"]


def test_anchor_reason_text(synthetic):
    sig = generate_signals(synthetic[OHLC], ANCHOR)
    rows = sig[~sig["warm"] & (sig["reason_code"] == "PULLBACK_TOO_LONG")]
    if len(rows):
        assert describe(rows.iloc[0], ANCHOR).startswith("PULLBACK_TOO_LONG")
    chop = sig[~sig["warm"] & sig["chop"]].iloc[0]
    assert "pre-dip anchor" in describe(chop, ANCHOR)


def test_no_signal_before_warmup(signals):
    assert not signals["entry"].iloc[:CFG.warmup_bars].any()
    assert (signals["reason_code"].iloc[:CFG.warmup_bars - 1] == "WARMUP").all()


def _regime_signals(seed: int):
    df = make_synthetic_ohlcv(seed)
    sig = generate_signals(df[OHLC], CFG)
    return df, sig


@pytest.mark.parametrize("seed", SEEDS)
def test_chop_regime_outputs_wait(seed):
    df, sig = _regime_signals(seed)
    chop = sig[df["regime"].to_numpy() == "CHOP"]
    states = {resolve_state(row, in_position=False)[0] for _, row in chop.iterrows()}
    assert states == {WAIT}
    assert (chop["reason_code"] == R_CHOP).mean() > 0.5, chop["reason_code"].value_counts()
    assert chop["adx"].iloc[2 * CFG.adx_period:].median() < CFG.adx_min


@pytest.mark.parametrize("seed", SEEDS)
def test_volatility_spike_regime_outputs_wait(seed):
    df, sig = _regime_signals(seed)
    crash = sig[df["regime"].to_numpy() == "CRASH"]
    assert {resolve_state(row, in_position=False)[0] for _, row in crash.iterrows()} == {WAIT}
    assert crash["spike"].iloc[:5].any(), "spike filter must fire as soon as the crash starts"
    assert (crash["reason_code"] == R_SPIKE).sum() > 0
    onset = crash[crash["spike"]].iloc[0]
    assert R_SPIKE in describe(onset, CFG)


@pytest.mark.parametrize("seed", SEEDS)
def test_dead_regime_flags_volatility_dead(seed):
    df, sig = _regime_signals(seed)
    dead = sig[df["regime"].to_numpy() == "DEAD"]
    assert not dead["entry"].any()
    assert dead["dead"].iloc[:10].any()
    assert R_DEAD in describe(dead[dead["dead"]].iloc[0], CFG)


@pytest.mark.parametrize("seed", SEEDS)
def test_bull_regime_produces_long_entries(seed):
    df, sig = _regime_signals(seed)
    bull = sig[df["regime"].to_numpy() == "BULL"]
    assert bull["entry"].sum() >= 5


def test_resolve_state_and_reason_text(signals):
    entry_row = signals[signals["entry"]].iloc[0]
    assert resolve_state(entry_row, in_position=False) == (LONG, R_ENTRY)
    assert resolve_state(entry_row, in_position=False, blackout=True) == (WAIT, R_EARNINGS)
    assert "crossed 42.00" in describe(entry_row, CFG)
    exit_row = signals[signals["exit"] & ~signals["warm"]].iloc[0]
    assert resolve_state(exit_row, in_position=True, ema_exit_allowed=True)[0] == EXIT
    assert resolve_state(exit_row, in_position=True, ema_exit_allowed=False)[0] == WAIT
    assert resolve_state(exit_row, in_position=True, ema_exit_allowed=False, blackout=True) == (EXIT, R_EARNINGS)
    assert "not armed" in describe(exit_row, V1, in_position=True, exit_armed=False)
    wait_row = signals[signals["reason_code"] == R_CHOP].iloc[0]
    assert describe(wait_row, CFG).startswith(f"{R_CHOP}: ADX[t-1]")


# ============================================================================ risk manager
def test_sizing_risks_exactly_one_percent():
    rm = RiskManager(RiskConfig(), StrategyConfig())
    size = rm.size_position(equity=100_000, entry_ref=50.0, atr=2.5)
    assert size.shares == 200                       # 1,000 / (2.0 * 2.5)
    assert size.stop_price == 45.0 and size.take_profit == 57.5
    assert size.risk_dollars == pytest.approx(1_000.0)
    assert size.risk_pct == pytest.approx(0.01)
    assert size.notional_pct == pytest.approx(0.10)
    assert size.binding_constraint == "RISK"


def test_sizing_never_exceeds_one_percent():
    rm = RiskManager(RiskConfig(), StrategyConfig())
    rng = np.random.default_rng(0)
    for _ in range(2_000):
        equity, price = rng.uniform(5_000, 1_000_000), rng.uniform(5, 1_500)
        atr = price * rng.uniform(0.002, 0.08)
        size = rm.size_position(equity, price, atr)
        assert size.risk_pct <= 0.01 + 1e-12
        assert size.notional_pct <= 0.20 + 1e-12
        if size.binding_constraint == "RISK":  # rounding down costs less than one share of risk
            assert size.risk_dollars > equity * 0.01 - 2 * atr


def test_notional_cap_binds_on_tight_hourly_stops():
    rm = RiskManager(RiskConfig(), StrategyConfig())
    size = rm.size_position(equity=100_000, entry_ref=500.0, atr=1.0)  # 1% risk would need 500 shares
    assert size.shares == 40 and size.binding_constraint == "NOTIONAL_CAP"
    assert size.notional_pct == pytest.approx(0.20)
    assert size.risk_pct == pytest.approx(0.0008)


def test_buying_power_and_zero_size():
    rm = RiskManager(RiskConfig(), StrategyConfig())
    assert rm.size_position(100_000, 50.0, 2.5, buying_power=4_000).shares == 80
    assert rm.size_position(1_000, 50.0, 100.0).shares == 0
    assert rm.size_position(100_000, 50.0, float("nan")).shares == 0


def test_kill_switch_trips_at_three_percent_and_latches():
    rm = RiskManager(RiskConfig(), StrategyConfig())
    day = dt.date(2026, 10, 12)
    assert not rm.check_daily_drawdown(day, 100_000, 98_000).tripped
    assert not rm.check_daily_drawdown(day, 100_000, 97_000.01).tripped
    status = rm.check_daily_drawdown(day, 100_000, 97_000)
    assert status.tripped and status.newly_tripped and status.daily_pnl_pct == pytest.approx(-0.03)
    again = rm.check_daily_drawdown(day, 100_000, 99_500)   # recovery does not re-arm trading
    assert again.tripped and not again.newly_tripped
    assert rm.entry_block_reason("SPY", day, [], [], [], "id") == "KILL_SWITCH_ACTIVE"
    tomorrow = day + dt.timedelta(days=1)
    assert not rm.check_daily_drawdown(tomorrow, 97_000, 97_000).tripped
    assert rm.entry_block_reason("SPY", tomorrow, [], [], [], "id") is None


def test_kill_switch_on_simulated_consecutive_full_losses():
    """Four 1R losers in a day: the switch must stay off after three and trip on the fourth."""
    rm = RiskManager(RiskConfig(), StrategyConfig())
    day, start = dt.date(2026, 10, 12), 100_000.0
    equity, tripped_after = start, None
    for n in range(1, 6):
        size = rm.size_position(equity, entry_ref=50.0, atr=2.5)
        equity -= size.shares * size.stop_distance        # stopped out at exactly the stop
        if rm.check_daily_drawdown(day, start, equity).tripped and tripped_after is None:
            tripped_after = n
    assert tripped_after == 4


def test_idempotency_gates():
    rm = RiskManager(RiskConfig(max_open_positions=2), StrategyConfig())
    day = dt.date(2026, 10, 12)
    cid = client_order_id("tpb1", "SPY", pd.Timestamp("2026-10-12 10:30", tz=ET.zone))
    assert cid == "tpb1-SPY-20261012T1030"
    assert rm.entry_block_reason("SPY", day, ["SPY"], [], [], cid) == "POSITION_ALREADY_OPEN"
    assert rm.entry_block_reason("SPY", day, [], ["SPY"], [], cid) == "OPEN_ORDER_EXISTS"
    assert rm.entry_block_reason("SPY", day, [], [], [cid], cid) == "DUPLICATE_CLIENT_ORDER_ID"
    assert rm.entry_block_reason("SPY", day, ["QQQ", "AAPL"], [], [], cid) == "MAX_OPEN_POSITIONS"
    assert rm.entry_block_reason("SPY", day, ["QQQ"], ["AAPL"], ["other"], cid) is None


def test_r_multiple():
    assert r_multiple(100.0, 103.0, 98.0) == pytest.approx(1.5)
    assert r_multiple(100.0, 98.0, 98.0) == pytest.approx(-1.0)
    assert r_multiple(97.0, 96.0, 98.0, planned_risk=2.0) == pytest.approx(-0.5)  # gapped through the stop


def test_ratchet_stop_breakeven_then_trail_never_lower():
    # signal close 100, initial stop 98 (R = 2, ATR = 1), filled at 100.05, round trip 0.1%
    args = dict(entry_fill=100.05, entry_ref=100.0, initial_stop=98.0, cfg=TRAIL)
    assert ratchet_stop(98.0, highest_close=101.99, atr=1.0, **args) == 98.0      # below +1R: untouched
    assert ratchet_stop(98.0, highest_close=102.0, atr=1.0, **args) == 100.10     # +1R: fill + exit cost
    assert ratchet_stop(100.10, highest_close=103.5, atr=1.0, **args) == 101.5    # trail: 103.5 - 2 * 1.0
    assert ratchet_stop(101.5, highest_close=103.6, atr=1.0, **args) == 101.5     # +0.1 < 0.1R step: no churn
    assert ratchet_stop(101.5, highest_close=103.5, atr=2.0, **args) == 101.5     # wider ATR never lowers it
    assert ratchet_stop(98.0, highest_close=105.0, atr=1.0, **dict(args, cfg=CFG)) == 98.0   # bracket mode
    assert ratchet_stop(98.0, highest_close=float("-inf"), atr=1.0, **args) == 98.0


def test_breakeven_exit_is_about_zero_r_not_a_huge_multiple():
    """R is always measured against the initial stop, never the ratcheted one."""
    fill_in, stop_be = 100.05, 100.10
    assert r_multiple(fill_in, stop_be * (1 - 0.0005), 98.0) == pytest.approx(0.0, abs=0.01)


# ============================================================================ earnings blackout
JULY_2026 = [d.date() for d in pd.bdate_range("2026-06-22", "2026-08-14") if d.date() != dt.date(2026, 7, 3)]


def test_event_window_uses_exchange_sessions():
    d = dt.date
    amc = event_window(d(2026, 7, 30), "AMC", JULY_2026)                  # Thursday after the close
    assert (amc.block_from, amc.last_hold, amc.gap) == (d(2026, 7, 29), d(2026, 7, 30), d(2026, 7, 31))
    bmo = event_window(d(2026, 7, 7), "BMO", JULY_2026)                   # Tuesday; Fri 3 Jul is a holiday
    assert (bmo.block_from, bmo.last_hold, bmo.gap) == (d(2026, 7, 2), d(2026, 7, 6), d(2026, 7, 7))
    unknown = event_window(d(2026, 7, 6), "", JULY_2026)                  # unknown timing: safe side of both
    assert (unknown.block_from, unknown.last_hold, unknown.gap) == (d(2026, 7, 1), d(2026, 7, 2), d(2026, 7, 7))
    fallback = event_window(d(2026, 8, 3), "BMO")                         # no session list: extra margin
    assert fallback.last_hold == d(2026, 7, 30)
    weekend = event_window(d(2026, 7, 4), "AMC", JULY_2026)               # Saturday report -> Monday BMO
    assert (weekend.event, weekend.timing, weekend.last_hold) == (d(2026, 7, 6), "BMO", d(2026, 7, 2))


def test_forced_exit_on_an_early_close_uses_the_last_decision_bar():
    cal = {"AAPL": [(dt.date(2026, 11, 30), "BMO")]}                     # L = Fri 27 Nov, closes 13:00
    sessions = [d.date() for d in pd.bdate_range("2026-11-16", "2026-12-11") if d.date() != dt.date(2026, 11, 26)]
    half = {dt.date(2026, 11, 27): "12:30"}
    at = lambda s: pd.Timestamp(s, tz=ET.zone)  # noqa: E731
    assert forced_exit("AAPL", at("2026-11-27 12:30"), cal, sessions) is None          # 14:30 never comes
    assert forced_exit("AAPL", at("2026-11-27 12:30"), cal, sessions, last_decision=half) == dt.date(2026, 11, 30)
    assert forced_exit("AAPL", at("2026-11-27 11:30"), cal, sessions, last_decision=half) is None


def test_entry_blackout_and_forced_exit_times():
    cal = {"AAPL": [(dt.date(2026, 7, 30), "AMC")]}
    at = lambda s: pd.Timestamp(s, tz=ET.zone)  # noqa: E731
    assert forced_exit("AAPL", at("2026-07-29 15:30"), cal, JULY_2026) is None       # can still hold
    assert forced_exit("AAPL", at("2026-07-30 13:30"), cal, JULY_2026) is None       # keeps the pre-report drift
    assert forced_exit("AAPL", at("2026-07-30 14:30"), cal, JULY_2026) == dt.date(2026, 7, 30)
    assert forced_exit("AAPL", at("2026-07-30 15:30"), cal, JULY_2026) == dt.date(2026, 7, 30)
    assert entry_blackout("AAPL", at("2026-07-28 15:30"), cal, JULY_2026) is None
    assert entry_blackout("AAPL", at("2026-07-29 10:30"), cal, JULY_2026) == dt.date(2026, 7, 30)
    assert entry_blackout("AAPL", at("2026-07-31 10:30"), cal, JULY_2026) == dt.date(2026, 7, 30)  # gap session
    assert entry_blackout("AAPL", at("2026-08-03 10:30"), cal, JULY_2026) is None
    assert entry_blackout("SPY", at("2026-07-30 10:30"), cal, JULY_2026) is None


def test_earnings_calendar_loading_and_coverage(tmp_path):
    path = tmp_path / "earnings.csv"
    path.write_text("# comment line\nsymbol,date,timing\naapl, 2026-07-30 ,AMC\nNVDA,2026-08-26,\n")
    cal = load_earnings_calendar(str(path))
    assert cal == {"AAPL": [(dt.date(2026, 7, 30), "AMC")], "NVDA": [(dt.date(2026, 8, 26), "")]}
    assert has_upcoming_event("AAPL", pd.Timestamp("2026-05-01 10:30", tz=ET.zone), cal)
    assert not has_upcoming_event("AAPL", pd.Timestamp("2026-08-01 10:30", tz=ET.zone), cal)
    assert events_per_year("AAPL", cal, pd.Timestamp("2026-01-01", tz=ET.zone),
                           pd.Timestamp("2026-12-31", tz=ET.zone)) == pytest.approx(1.0, abs=0.01)
    assert load_earnings_calendar(str(tmp_path / "missing.csv")) == {}
    path.write_text("symbol,date,timing\nAAPL,2026-07-30,LUNCH\n")
    with pytest.raises(ValueError):
        load_earnings_calendar(str(path))


@pytest.mark.slow
def test_backtester_earnings_blackout(portfolio):
    """An AMC event two sessions after an entry: the trade must be closed at 14:30 on the event
    day if it is still open, and no entry may open between the session before and the gap."""
    forced = 0
    for symbol in ("SYN1", "SYN2", "SYN4"):
        df = portfolio[symbol][OHLC]
        sessions = sorted(set(df.index.date))
        for _, held in Backtester().run({symbol: df}).trades.head(4).iterrows():
            i = sessions.index(held["entry_time"].date())
            if i + 2 >= len(sessions):
                continue
            event = sessions[i + 2]
            deadline = pd.Timestamp(f"{event} 14:30", tz=ET.zone)
            trades = Backtester(calendar={symbol: [(event, "AMC")]}).run({symbol: df}).trades
            same = trades[trades["entry_time"] == held["entry_time"]].iloc[0]
            if held["exit_time"] >= deadline:
                assert same["exit_reason"] == "EARNINGS_EXIT" and same["exit_time"] == deadline
                forced += 1
            else:
                assert same["exit_time"] == held["exit_time"]
            window = event_window(event, "AMC", sessions)
            days = trades["entry_time"].dt.date
            assert not ((days >= window.block_from) & (days <= window.gap)).any()
    assert forced >= 1


# ============================================================================ sessions
def _source_bars_for_day(day: str = "2026-07-15") -> pd.DataFrame:
    start = pd.Timestamp(f"{day} 04:00", tz=ET.zone)
    idx = pd.date_range(start, periods=32, freq="30min")              # 04:00 .. 19:30 ET
    values = np.arange(len(idx), dtype=float) + 100
    df = pd.DataFrame({"open": values, "high": values + 0.5, "low": values - 0.5, "close": values + 0.25,
                       "volume": 1_000.0}, index=idx.tz_convert("UTC"))
    return df


def test_session_bars_are_aligned_and_drop_extended_hours():
    src = _source_bars_for_day()
    bars = to_session_bars(src)
    starts = [f"{t:%H:%M}" for t in bars.index]
    assert starts == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    first = bars.iloc[0]
    src_et = src.tz_convert(ET.zone)
    open_930, bar_1000 = src_et.loc[src_et.index.strftime("%H:%M") == "09:30"].iloc[0], \
        src_et.loc[src_et.index.strftime("%H:%M") == "10:00"].iloc[0]
    assert first["open"] == open_930["open"] and first["close"] == bar_1000["close"]
    assert first["high"] == max(open_930["high"], bar_1000["high"]) and first["volume"] == 2_000.0
    assert f"{bars['bar_end'].iloc[-1]:%H:%M}" == "16:00" and bars["n_expected"].iloc[-1] == 1
    assert bars["n_source"].eq(bars["n_expected"]).all()


def test_complete_bars_drops_forming_and_partial_bars():
    bars = to_session_bars(_source_bars_for_day())
    at = lambda hhmm: pd.Timestamp(f"2026-07-15 {hhmm}", tz=ET.zone)  # noqa: E731
    assert len(complete_bars(bars, at("10:29:59"))) == 0
    assert len(complete_bars(bars, at("10:30:05"))) == 1
    assert len(complete_bars(bars, at("16:00:01"))) == 7
    partial = bars.copy()
    partial.loc[partial.index[1], "n_source"] = 1                     # 11:00 source bar not published yet
    assert len(complete_bars(partial, at("11:30:30"), grace_seconds=120)) == 1
    assert len(complete_bars(partial, at("11:32:30"), grace_seconds=120)) == 2


def test_trading_window_is_eastern_time_and_dst_safe():
    def utc(s):
        return pd.Timestamp(s, tz="UTC")
    assert in_trading_window(utc("2026-07-15 13:45"))        # 09:45 EDT
    assert not in_trading_window(utc("2026-07-15 13:44"))    # 09:44 EDT
    assert in_trading_window(utc("2026-07-15 19:45"))        # 15:45 EDT
    assert not in_trading_window(utc("2026-07-15 19:46"))
    assert in_trading_window(utc("2026-01-15 14:45"))        # 09:45 EST
    assert not in_trading_window(utc("2026-01-15 13:45"))    # 08:45 EST
    assert not in_trading_window(utc("2026-07-18 15:00"))    # Saturday
    with pytest.raises(ValueError):
        in_trading_window(pd.Timestamp("2026-07-15 10:00"))
    assert latest_bar_end(pd.Timestamp("2026-07-15 10:30:10", tz=ET.zone), 20) is None
    assert f"{latest_bar_end(pd.Timestamp('2026-07-15 10:30:25', tz=ET.zone), 20):%H:%M}" == "10:30"
    assert f"{latest_bar_end(pd.Timestamp('2026-07-15 15:59', tz=ET.zone), 20):%H:%M}" == "15:30"


# ============================================================================ journal
def test_journal_schema_append_and_reload(tmp_path, signals):
    path = tmp_path / "journal.csv"
    journal = TradeJournal(str(path))
    journal.log(J.STARTUP, equity=100_000.0)
    row = signals[signals["entry"]].iloc[0]
    journal.log_bar_check("SYN", row, row.name, LONG, R_ENTRY, describe(row, CFG), "ENTER", False,
                          equity=100_000.0)
    with pytest.raises(ValueError):
        journal.log(J.ERROR, not_a_column=1)
    df = read_journal(str(path))
    assert list(df.columns) == COLUMNS and len(df) == 2
    check = df.iloc[1]
    assert check["state"] == LONG and check["reason_code"] == R_ENTRY
    assert check["rsi"] == pytest.approx(row["rsi"]) and check["decision_bar_et"] == row.name
    assert "crossed" in check["reason_detail"]
    TradeJournal(str(path)).log(J.SHUTDOWN)                      # reopening appends, no new header
    assert len(read_journal(str(path))) == 3


def test_journal_rotates_on_schema_change(tmp_path):
    path = tmp_path / "journal.csv"
    path.write_text("old,header\n1,2\n")
    TradeJournal(str(path)).log(J.STARTUP)
    assert list(read_journal(str(path)).columns) == COLUMNS
    assert len(list(tmp_path.glob("journal.schema-*.csv"))) == 1


def test_slippage_sign_convention():
    assert slippage_bps("buy", 100.0, 100.05) == pytest.approx(5.0)
    assert slippage_bps("sell", 100.0, 99.95) == pytest.approx(5.0)
    assert slippage_bps("sell", 100.0, 100.10) == pytest.approx(-10.0)


# ============================================================================ backtester
@pytest.fixture(scope="module")
def portfolio() -> dict[str, pd.DataFrame]:
    """Five synthetic symbols with a noisier bull regime, so the backtest sees losers too."""
    return {f"SYN{seed}": make_synthetic_ohlcv(seed, bull_noise=0.006, bull_drift=0.0025)
            for seed in (7, 1, 2, 3, 4)}


CONFIGS = {"v1": V1, "trail": TRAIL, "v2": CFG, "time": replace(CFG, max_hold_bars=6)}


@pytest.fixture(scope="module")
def backtest_results(portfolio):
    data = {s: df[OHLC + ["volume"]] for s, df in portfolio.items()}
    return {name: Backtester(strategy_cfg=cfg).run(data) for name, cfg in CONFIGS.items()}


@pytest.fixture(scope="module")
def backtest_result(backtest_results):
    return backtest_results["v1"]


def test_backtester_end_to_end_tear_sheet(backtest_results, capsys):
    for name in ("v1", "v2"):
        result = backtest_results[name]
        sheet = format_tear_sheet(result, f"SYNTHETIC MULTI-REGIME TEAR SHEET, {name} (not market evidence)")
        with capsys.disabled():
            print("\n" + sheet)
        m = result.metrics
        assert m["n_trades"] > 0
        for key in ("sharpe", "profit_factor", "max_drawdown", "expectancy_r", "total_return", "cagr"):
            assert key in m and np.isfinite(m[key]), (name, key)
        assert "sortino" in m and m["max_drawdown"] <= 0
        for label in ("Sharpe", "Sortino", "Profit factor", "Max drawdown", "Expectancy", "Walk-forward folds"):
            assert label in sheet
        assert len(result.folds) == BacktestConfig().n_folds
        assert result.equity.notna().all()
    assert np.isfinite(backtest_results["v1"].metrics["sortino"])
    assert set(backtest_results["v1"].trades["exit_reason"]) >= {"STOP_LOSS", "TAKE_PROFIT", "EMA_EXIT"}
    assert "EMA_EXIT" not in set(backtest_results["trail"].trades["exit_reason"])
    assert "TRAIL_STOP" in set(backtest_results["trail"].trades["exit_reason"].str.replace("_GAP", ""))


def test_backtester_cost_model(backtest_results):
    """Market/stop fills pay 0.05% per side; take-profit limit fills cannot fill below the limit."""
    for result in backtest_results.values():
        trades = result.trades
        limit = trades["exit_reason"].str.startswith("TAKE_PROFIT")
        np.testing.assert_allclose(trades["entry_fill"], trades["entry_raw"] * 1.0005)
        np.testing.assert_allclose(trades.loc[~limit, "exit_fill"], trades.loc[~limit, "exit_raw"] * 0.9995)
        np.testing.assert_allclose(trades.loc[limit, "exit_fill"], trades.loc[limit, "exit_raw"])
        assert (trades["slippage_cost"] > 0).all()


def test_breakeven_cost_accounts_for_stop_width_and_exit_type():
    from backtest import breakeven_cost_bps

    stops = pd.DataFrame({"gross_r": [0.2, 0.2], "stop_pct": [0.006, 0.025], "exit_reason": ["STOP_LOSS"] * 2})
    expected = 0.2 / np.mean([2 / 0.006, 2 / 0.025]) * 1e4
    assert breakeven_cost_bps(stops) == pytest.approx(expected)
    targets = stops.assign(exit_reason="TAKE_PROFIT", stop_pct=0.01)          # limit exits pay one side
    assert breakeven_cost_bps(targets) == pytest.approx(0.2 * 0.01 * 1e4)


def test_cluster_random_drop_null_keeps_exactly_k_signals():
    from backtest import cluster_random_drop_percentile

    r = np.arange(20, dtype=float)                     # 4 weeks x 5 signals, distinct values
    weeks = np.repeat([1, 2, 3, 4], 5)
    overall = r.mean()
    # Dropping a single signal: draws must not all equal the full-sample mean (which would make
    # any filter look like it beats "random").
    assert cluster_random_drop_percentile(r, weeks, overall + 1e-9, k=19, draws=500) < 0.9
    assert 0.0 < cluster_random_drop_percentile(r, weeks, overall, k=10, draws=500) < 1.0


def test_take_profit_needs_a_trade_through(portfolio):
    df = portfolio["SYN7"][OHLC]
    touch = Backtester(strategy_cfg=V1, bt_cfg=BacktestConfig(tp_requires_trade_through=False)).run({"S": df})
    through = Backtester(strategy_cfg=V1).run({"S": df})
    for result in (touch, through):
        tp = result.trades[result.trades["exit_reason"] == "TAKE_PROFIT"]
        bars = df.loc[tp["exit_time"]]
        assert (bars["high"].to_numpy() >= tp["take_profit"].to_numpy()).all()
    tp = through.trades[through.trades["exit_reason"] == "TAKE_PROFIT"]
    assert (df.loc[tp["exit_time"], "high"].to_numpy() >= tp["take_profit"].to_numpy() + 0.01 - 1e-9).all()


def test_signal_outcomes_match_portfolio_trades(portfolio):
    """The paired study's per-signal simulation reproduces every portfolio trade exactly."""
    for cfg in (V1, TRAIL, replace(CFG, max_hold_bars=5)):
        bt = Backtester(strategy_cfg=cfg)
        data = {"SYN1": portfolio["SYN1"][OHLC]}
        trades = bt.run(data).trades.set_index("entry_time")
        outcomes = bt.signal_outcomes(data).set_index("entry_time")
        assert len(outcomes) >= len(trades) > 0
        for entry_time, trade in trades.iterrows():
            twin = outcomes.loc[entry_time]
            assert twin["exit_reason"] == trade["exit_reason"]
            assert twin["r_multiple"] == pytest.approx(trade["r_multiple"])


@pytest.mark.parametrize("name", list(CONFIGS))
def test_backtester_trades_follow_the_rules(backtest_results, portfolio, name):
    cfg = CONFIGS[name]
    trades = backtest_results[name].trades
    for _, trade in trades.iterrows():
        df = portfolio[trade["symbol"]]
        sig = generate_signals(df[OHLC], cfg)
        row = sig.loc[trade["entry_time"]]
        assert row["entry"], "entered without a signal"
        assert in_trading_window(trade["entry_time"]), "entered outside 09:45-15:45 ET"
        assert df.loc[trade["entry_time"], "regime"] == "BULL"
        assert trade["entry_raw"] == df.loc[trade["entry_time"], "open"]
        assert trade["stop"] == pytest.approx(round(row["close"] - 2.0 * row["atr"], 2))
        assert trade["take_profit"] == pytest.approx(round(row["close"] + 3.0 * row["atr"], 2))
        assert trade["risk_pct"] <= 0.01 + 1e-9 and trade["notional_pct"] <= 0.20 + 1e-9
        assert trade["final_stop"] >= trade["stop"]
        if cfg.exit_mode != "ratchet":
            assert trade["final_stop"] == trade["stop"]
        if trade["exit_reason"] == "STOP_LOSS":
            assert trade["exit_raw"] == trade["stop"] == trade["final_stop"]
        if trade["exit_reason"] == "TRAIL_STOP" and trade["exit_raw"] == trade["final_stop"]:
            assert trade["final_stop"] > trade["stop"]
        if trade["exit_reason"] == "TAKE_PROFIT":
            assert trade["exit_raw"] == trade["take_profit"]


def test_backtester_kill_switch_flattens_and_halts(portfolio):
    tight = RiskConfig(max_daily_drawdown=0.001)
    result = Backtester(risk_cfg=tight).run({s: df[OHLC] for s, df in portfolio.items()})
    assert result.metrics["kill_switch_events"] > 0
    trades = result.trades
    assert (trades["exit_reason"] == "KILL_SWITCH").any(), "open positions must be flattened on a trip"
    for event in result.events:
        t, day = event["time"], event["time"].tz_convert(ET.zone).date()
        assert event["daily_pnl_pct"] <= -0.001
        still_open = trades[(trades["entry_time"] <= t) & (trades["exit_time"] > t)]
        assert still_open.empty, "nothing may stay open after the kill-switch trips"
        later = trades[(trades["entry_time"] > t) & (trades["entry_time"].dt.tz_convert(ET.zone).dt.date == day)]
        assert later.empty, "entries must stay blocked for the rest of the day"


# ============================================================================ live bot (fake broker)
class FakeBroker:
    """In-memory stand-in for AlpacaClient: fills market orders at the last close + 3 bps
    and works bracket legs against the synthetic bars as time advances."""

    paper = True

    def __init__(self, bars: dict[str, pd.DataFrame], equity: float = 100_000.0):
        self.bars, self.equity, self.last_equity = bars, equity, equity
        self.now = None
        self.market_open = True
        self.positions, self.orders, self.submitted, self.closed = {}, {}, [], []
        self.flattened = 0
        self.replaced, self.fail_replace = [], False
        self.protective, self.fail_protect, self.fail_close = [], False, 0
        self.fail_fetch = False
        self.fail_flatten, self.slow_close, self.hide_fills = False, False, False
        self._n = 0

    def _order(self, **kw):
        self._n += 1
        order = SimpleNamespace(id=f"ord-{self._n}", legs=[], stop_price=None, limit_price=None, filled_at=None,
                                filled_avg_price=None, filled_qty=0, client_order_id=None, **kw)
        self.orders[order.id] = order
        return order

    def _last_close(self, symbol):
        return float(complete_bars(self.bars[symbol], self.now)["close"].iloc[-1])

    def _fill(self, order, price):
        order.status, order.filled_avg_price, order.filled_qty, order.filled_at = "filled", price, order.qty, \
            self.now.to_pydatetime()

    def _work_brackets(self):
        """Each completed bar is checked once, against the stop working at that time."""
        for symbol, pos in list(self.positions.items()):
            done = complete_bars(self.bars[symbol], self.now)
            done = done[done["bar_end"] > pos.checked_until]
            if len(done):
                pos.checked_until = done["bar_end"].iloc[-1]
            legs = [o for o in self.orders.values() if o.symbol == symbol and o.status in ("new", "held")]
            stop = next((o for o in legs if o.order_type == "stop"), None)
            target = next((o for o in legs if o.order_type == "limit"), None)
            for _, bar in done.iterrows():                       # a limit needs a 1-cent trade-through
                hit = (stop, stop.stop_price) if stop and bar["low"] <= stop.stop_price else \
                      (target, target.limit_price) if target and bar["high"] >= target.limit_price + 0.01 - 1e-9 \
                      else None
                if hit:
                    self._fill(hit[0], hit[1])
                    for leg in legs:
                        if leg is not hit[0]:
                            leg.status = "canceled"
                    del self.positions[symbol]
                    break

    # --- AlpacaClient interface
    def get_clock(self):
        return SimpleNamespace(is_open=self.market_open, timestamp=self.now, next_open=self.now + pd.Timedelta(hours=12))

    def get_account(self):
        return SimpleNamespace(equity=str(self.equity), last_equity=str(self.last_equity),
                               buying_power=str(4 * self.equity), trading_blocked=False, account_blocked=False)

    def get_positions(self):
        self._work_brackets()
        return dict(self.positions)

    def get_open_orders(self, symbols=None):
        return [o for o in self.orders.values() if o.status in ("new", "held") and (not symbols or o.symbol in symbols)]

    def fetch_session_bars(self, symbols, start, end, now=None):
        if self.fail_fetch:
            raise ConnectionError("market data unavailable (simulated)")
        return {s: complete_bars(self.bars[s], now) for s in symbols}

    def submit_bracket_order(self, symbol, qty, stop_price, take_profit, client_order_id):
        existing = [o for o in self.orders.values() if o.client_order_id == client_order_id]
        if existing:
            return existing[0]
        order = self._order(symbol=symbol, qty=qty, side="buy", order_type="market", status="new")
        order.client_order_id = client_order_id
        self._fill(order, round(self._last_close(symbol) * 1.0003, 4))
        self._order(symbol=symbol, qty=qty, side="sell", order_type="stop", status="held").stop_price = stop_price
        self._order(symbol=symbol, qty=qty, side="sell", order_type="limit", status="new").limit_price = take_profit
        self.positions[symbol] = SimpleNamespace(symbol=symbol, qty=str(qty), avg_entry_price=str(order.filled_avg_price),
                                                 checked_until=self.now)
        self.submitted.append(dict(symbol=symbol, qty=qty, stop=stop_price, target=take_profit, cid=client_order_id))
        return order

    def wait_for_terminal(self, order_id, timeout=None):
        return self.orders[order_id]

    def get_order(self, order_id):
        return self.orders[order_id]

    def get_session_closes(self, start, end):
        closes = {}
        for df in self.bars.values():
            for ts in df.index:
                if start <= ts.date() <= end:
                    end_time = f"{ts + pd.Timedelta(minutes=30):%H:%M}"     # last bar starts 30 min before close
                    closes[ts.date()] = max(closes.get(ts.date(), ""), end_time)
        return closes

    def restore_protection(self, symbol, qty, stop_price, take_profit):
        if self.fail_protect:
            raise RuntimeError("protective stop rejected (simulated)")
        for order in self.get_open_orders([symbol]):        # orphaned legs hold the shares: cancel first
            order.status = "canceled"
        self._order(symbol=symbol, qty=qty, side="sell", order_type="stop", status="held").stop_price = stop_price
        if take_profit is not None:
            self._order(symbol=symbol, qty=qty, side="sell", order_type="limit", status="new").limit_price = take_profit
        self.protective.append((symbol, stop_price))
        return "oco" if take_profit is not None else "stop"

    def get_order_by_client_id(self, client_order_id):
        return next(o for o in self.orders.values() if o.client_order_id == client_order_id)

    def replace_stop(self, symbol, stop_price):
        if self.fail_replace:
            raise RuntimeError("replace rejected (simulated)")
        leg = next(o for o in self.get_open_orders([symbol]) if o.order_type == "stop")
        leg.stop_price = stop_price
        self.replaced.append((symbol, stop_price))
        return leg

    def bracket_levels(self, symbol):
        legs = self.get_open_orders([symbol])
        stop = next((o.stop_price for o in legs if o.order_type == "stop"), None)
        target = next((o.limit_price for o in legs if o.order_type == "limit"), None)
        return stop, target

    def close_position(self, symbol):
        if symbol not in self.positions:
            return None
        for order in self.get_open_orders([symbol]):
            order.status = "canceled"
        if self.fail_close:
            self.fail_close -= 1
            raise RuntimeError("insufficient qty available (simulated: cancel still pending)")
        self.closed.append(symbol)
        if self.slow_close:                                    # accepted, fills later
            pos = self.positions[symbol]
            return self._order(symbol=symbol, qty=float(pos.qty), side="sell", order_type="market", status="accepted")
        pos = self.positions.pop(symbol)
        order = self._order(symbol=symbol, qty=float(pos.qty), side="sell", order_type="market", status="new")
        self._fill(order, round(self._last_close(symbol) * 0.9997, 4))
        return order

    def flatten_all(self):
        self.flattened += 1
        if self.fail_flatten:
            raise RuntimeError("flatten failed (simulated)")
        for symbol in list(self.positions):
            self.close_position(symbol)

    def find_latest_fill(self, symbol, side, since):
        if self.hide_fills:
            return None
        fills = [o for o in self.orders.values() if o.symbol == symbol and o.side == side and o.status == "filled"
                 and o.filled_at >= since]
        return max(fills, key=lambda o: o.filled_at) if fills else None


def _decision_times(bars: pd.DataFrame, start: int, stop: int) -> list[pd.Timestamp]:
    """Moments the live bot evaluates: 25s after each hourly close inside the trading window."""
    times = [end + pd.Timedelta(seconds=25) for end in bars["bar_end"].iloc[start:stop]]
    return [t for t in times if in_trading_window(t)]


@pytest.fixture()
def live_setup(tmp_path, synthetic, signals):
    bars = with_session_columns(synthetic)
    broker = FakeBroker({"SYN": bars})
    journal = TradeJournal(str(tmp_path / "journal.csv"))
    first_long = next(i for i in np.flatnonzero(signals["entry"].to_numpy()) if in_trading_window(signals.index[i]))
    return bars, broker, journal, first_long


def _make_bot(broker, journal, symbols=("SYN",), cfg=CFG, risk=LIVE_RISK, **kw):
    return TradingBot(broker, journal, list(symbols), cfg, risk, ExecutionConfig(),
                      now_fn=lambda: broker.now, sleep=lambda s: None, **kw)


def _step(bot, broker, when):
    broker.now = when
    return bot.run_once(when)


def test_bot_enters_once_with_exact_bracket_and_survives_restart(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal)
    when = signals.index[t] + pd.Timedelta(seconds=25)
    broker.now = when
    bot.bootstrap()
    assert _step(bot, broker, when) == "EVALUATED"

    sig = signals.iloc[t]
    expected = RiskManager(RiskConfig(), CFG).size_position(100_000.0, sig["close"], sig["atr"],
                                                            buying_power=400_000.0)
    assert len(broker.submitted) == 1
    order = broker.submitted[0]
    assert order["qty"] == expected.shares and order["stop"] == expected.stop_price
    assert order["target"] == expected.take_profit
    assert order["cid"] == client_order_id("tpb2", "SYN", signals.index[t])

    log = read_journal(journal.path)
    check = log[log["event"] == J.BAR_CHECK].iloc[-1]
    assert check["state"] == LONG and check["action"] == "ENTER"
    fill = log[log["event"] == J.ENTRY_FILL].iloc[-1]
    assert fill["slippage_bps"] == pytest.approx(3.0, abs=0.05)

    assert _step(bot, broker, when + pd.Timedelta(seconds=30)) == "IDLE"   # same bar: nothing new
    restarted = _make_bot(broker, journal)                                   # crash + restart mid-bar
    restarted.bootstrap()
    assert "SYN" in restarted.trades
    _step(restarted, broker, when + pd.Timedelta(seconds=60))
    assert len(broker.submitted) == 1, "restart must not duplicate the entry"


def test_bot_blocks_outside_window_and_when_closed(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal)
    opening = signals.index[t].normalize() + pd.Timedelta(hours=9, minutes=35)
    assert _step(bot, broker, opening) == "OUTSIDE_WINDOW"
    broker.market_open = False
    assert _step(bot, broker, signals.index[t] + pd.Timedelta(seconds=25)) == "MARKET_CLOSED"
    assert broker.submitted == []


def test_bot_dry_run_sends_no_orders(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal, dry_run=True)
    _step(bot, broker, signals.index[t] + pd.Timedelta(seconds=25))
    assert broker.submitted == []
    assert read_journal(journal.path)["action"].eq("DRY_RUN_ENTER").any()


def test_bot_trade_lifecycle_and_bracket_reconciliation(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal)
    for when in _decision_times(bars, t - 1, t + 300):
        _step(bot, broker, when)
        if broker.submitted and "SYN" not in broker.positions:
            break
    log = read_journal(journal.path)
    exits = log[log["event"] == J.EXIT_FILL]
    assert len(exits) == 1
    exit_row = exits.iloc[0]
    assert exit_row["exit_reason"] in {"TRAIL_STOP", "STOP_LOSS", "TAKE_PROFIT"}
    for _, ratchet in log[log["event"] == J.STOP_RATCHET].iterrows():   # every ratchet reached the broker
        assert (ratchet["symbol"], ratchet["stop_price"]) in broker.replaced
    if exit_row["exit_reason"] == "STOP_LOSS":
        assert exit_row["r_multiple"] == pytest.approx(-1.0, abs=0.02)
    if exit_row["exit_reason"] == "TAKE_PROFIT":
        assert exit_row["r_multiple"] > 1.0
    if exit_row["exit_reason"] == "EMA_EXIT":
        assert broker.closed == ["SYN"]
        trigger = log[(log["event"] == J.BAR_CHECK) & (log["state"] == EXIT)].iloc[-1]
        assert trigger["close"] < trigger["ema_fast"]
    holds = log[(log["event"] == J.BAR_CHECK) & (log["in_position"].astype(str) == "True")]
    assert (holds["state"] != LONG).all()


def test_bot_stop_fill_is_reconciled_as_minus_one_r(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal)
    when = signals.index[t] + pd.Timedelta(seconds=25)
    _step(bot, broker, when)
    stop_leg = next(o for o in broker.orders.values() if o.order_type == "stop")
    broker.now = when + pd.Timedelta(minutes=5)
    broker._fill(stop_leg, stop_leg.stop_price)          # stop triggers between heartbeats
    del broker.positions["SYN"]
    _step(bot, broker, when + pd.Timedelta(minutes=5))
    exit_row = read_journal(journal.path).query("event == @J.EXIT_FILL").iloc[-1]
    assert exit_row["exit_reason"] == "STOP_LOSS"
    entry = bot.journal.read().query("event == @J.ENTRY_FILL").iloc[-1]
    assert exit_row["r_multiple"] == pytest.approx(-1.0)
    assert exit_row["pnl"] == pytest.approx(entry["qty"] * (stop_leg.stop_price - entry["fill_price"]))


def test_bot_journals_a_stop_that_fills_mid_cycle_before_deciding(live_setup, signals):
    """The stop fills between the heartbeat check and the bar evaluation: the exit must be
    journaled first and the decision must see the symbol as flat."""
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal)
    when = signals.index[t] + pd.Timedelta(seconds=25)
    _step(bot, broker, when)
    stop_leg = next(o for o in broker.orders.values() if o.order_type == "stop")
    original, calls = broker.get_positions, {"n": 0}

    def positions_then_stop():
        calls["n"] += 1
        if calls["n"] == 2:                       # second call of the cycle, after bars were fetched
            broker._fill(stop_leg, stop_leg.stop_price)
            broker.positions.pop("SYN", None)
        return original()

    broker.get_positions = positions_then_stop
    next_when = next(w for w in _decision_times(bars, t, t + 20) if w > when)
    _step(bot, broker, next_when)
    log = read_journal(journal.path)
    tail = log[log["ts_utc"] == next_when]
    assert list(tail["event"])[:2] == [J.EXIT_FILL, J.BAR_CHECK]
    assert tail.iloc[0]["exit_reason"] == "STOP_LOSS"
    assert tail.iloc[1]["in_position"] in (False, "False") and "SYN" not in bot.trades


def test_bot_kill_switch_flattens_and_blocks_entries(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal)
    when = signals.index[t] + pd.Timedelta(seconds=25)
    _step(bot, broker, when)
    assert "SYN" in broker.positions
    broker.equity = 96_900.0                              # -3.1% vs previous close
    _step(bot, broker, when + pd.Timedelta(seconds=30))
    log = read_journal(journal.path)
    assert (log["event"] == J.KILL_SWITCH).sum() == 1 and broker.flattened == 1
    assert "SYN" not in broker.positions
    assert log.query("event == @J.EXIT_FILL").iloc[-1]["exit_reason"] == "KILL_SWITCH"
    bot.last_decision_bar.clear()
    bot.last_processed_end = None
    _step(bot, broker, when + pd.Timedelta(seconds=60))   # same bar re-evaluated after the trip
    blocked = read_journal(journal.path).query("event == @J.BAR_CHECK").iloc[-1]
    assert blocked["action"] == "BLOCKED_KILL_SWITCH_ACTIVE"
    assert len(broker.submitted) == 1


@pytest.mark.slow
@pytest.mark.parametrize("name, required", [("v1", {"STOP_LOSS", "TAKE_PROFIT", "EMA_EXIT"}),
                                            ("trail", {"TAKE_PROFIT", "TRAIL_STOP"}), ("v2", set()),
                                            ("time", {"TIME_EXIT"})])
def test_live_bot_reproduces_every_backtest_trade(portfolio, tmp_path, name, required):
    """Full lifecycle parity: the live bot (fake broker) must enter on the same bar, with the
    same bracket, ratchet to the same stop and exit for the same reason as the backtester."""
    cfg = CONFIGS[name]
    compared = set()
    for symbol in ("SYN7", "SYN1", "SYN2", "SYN3", "SYN4"):
        df = portfolio[symbol]
        bars = with_session_columns(df)
        trades = Backtester(strategy_cfg=cfg).run({symbol: df[OHLC]}).trades
        for n, trade in trades[trades["exit_reason"] != "END_OF_DATA"].iterrows():
            broker = FakeBroker({symbol: bars})
            journal = TradeJournal(str(tmp_path / f"{name}-{symbol}-{n}.csv"))
            bot = _make_bot(broker, journal, symbols=[symbol], cfg=cfg)
            start = bars.index.get_loc(trade["entry_time"])
            exit_bar = bars.index.get_loc(trade["exit_time"])
            for when in _decision_times(bars, start - 1, exit_bar + 8):
                _step(bot, broker, when)
                if broker.submitted and symbol not in broker.positions:
                    break
            log = read_journal(journal.path)
            entry = log[log["event"] == J.ENTRY_FILL].iloc[0]
            exit_row = log[log["event"] == J.EXIT_FILL].iloc[0]
            assert entry["ts_utc"] == trade["entry_time"] + pd.Timedelta(seconds=25)
            assert entry["stop_price"] == pytest.approx(trade["stop"])
            assert entry["take_profit"] == pytest.approx(trade["take_profit"])
            assert exit_row["exit_reason"] == trade["exit_reason"].replace("_GAP", ""), (symbol, n)
            if trade["exit_reason"] in ("EMA_EXIT", "TRAIL_STOP", "TIME_EXIT") and trade["exit_raw"] != trade["final_stop"]:
                assert exit_row["ts_utc"] == trade["exit_time"] + pd.Timedelta(seconds=25)   # market exit at t
            ratchets = log[log["event"] == J.STOP_RATCHET]
            final_live_stop = ratchets["stop_price"].iloc[-1] if len(ratchets) else entry["stop_price"]
            # breakeven is built from each engine's own entry fill, so allow a few cents
            assert final_live_stop == pytest.approx(trade["final_stop"], rel=2e-3), (symbol, n)
            compared.add(trade["exit_reason"].replace("_GAP", ""))
    assert compared >= required


def test_bot_earnings_blocks_entry_and_forces_exit(live_setup, signals, tmp_path):
    bars, broker, journal, t = live_setup
    decision = signals.index[t]
    day = decision.date()
    blocked = _make_bot(broker, journal, calendar={"SYN": [(day, "AMC")]})       # entry day inside the window
    _step(blocked, broker, decision + pd.Timedelta(seconds=25))
    row = read_journal(journal.path).query("event == @J.BAR_CHECK").iloc[-1]
    assert row["state"] == WAIT and row["reason_code"] == R_EARNINGS and broker.submitted == []

    sessions = sorted(set(bars.index.date))
    event = sessions[sessions.index(day) + 2]                     # AMC two sessions out: entry day is clear
    broker2 = FakeBroker({"SYN": bars})
    journal2 = TradeJournal(str(tmp_path / "earnings2.csv"))
    bot = _make_bot(broker2, journal2, calendar={"SYN": [(event, "AMC")]})
    deadline = pd.Timestamp(f"{event} 14:30", tz=ET.zone)
    for when in _decision_times(bars, t - 1, t + 40):
        _step(bot, broker2, when)
        if when > deadline + pd.Timedelta(minutes=5) or (broker2.submitted and "SYN" not in broker2.positions):
            break
    exits = read_journal(journal2.path).query("event == @J.EXIT_FILL")
    assert len(exits) == 1
    if exits.iloc[0]["exit_reason"] == "EARNINGS_EXIT":
        assert exits.iloc[0]["ts_utc"] == deadline + pd.Timedelta(seconds=25)
    else:                                   # the bracket closed it before the deadline
        assert exits.iloc[0]["ts_utc"] < deadline


def test_bot_earnings_exit_does_not_need_market_data(live_setup, signals, tmp_path):
    """Market data fails from the morning of the last hold day: the exit still happens at 14:30."""
    bars, broker, journal, t = live_setup
    sessions = sorted(set(bars.index.date))
    day = signals.index[t].date()
    event = sessions[sessions.index(day) + 2]
    bot = _make_bot(broker, journal, calendar={"SYN": [(event, "AMC")]})
    deadline = pd.Timestamp(f"{event} 14:30", tz=ET.zone)
    for when in _decision_times(bars, t - 1, t + 40):
        broker.fail_fetch = when.date() >= event
        try:
            _step(bot, broker, when)
        except ConnectionError:
            pass                                   # run_forever logs this and carries on
        if broker.submitted and "SYN" not in broker.positions or when > deadline + pd.Timedelta(minutes=5):
            break
    exits = read_journal(journal.path).query("event == @J.EXIT_FILL")
    assert len(exits) == 1
    if exits.iloc[0]["exit_reason"] == "EARNINGS_EXIT":
        assert exits.iloc[0]["ts_utc"] == deadline + pd.Timedelta(seconds=25)
    else:
        assert exits.iloc[0]["ts_utc"] < deadline


def test_bot_fails_closed_without_upcoming_earnings_date(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal, risk=RiskConfig())                              # fail closed (default)
    _step(bot, broker, signals.index[t] + pd.Timedelta(seconds=25))
    row = read_journal(journal.path).query("event == @J.BAR_CHECK").iloc[-1]
    assert row["reason_code"] == "EARNINGS_CALENDAR_STALE" and broker.submitted == []
    later = (signals.index[t] + pd.Timedelta(days=40)).date()
    ok = _make_bot(broker, journal, risk=RiskConfig(), calendar={"SYN": [(later, "AMC")]})
    _step(ok, broker, signals.index[t] + pd.Timedelta(seconds=30))
    assert len(broker.submitted) == 1


def test_bot_failed_exit_reprotects_then_retries(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal, cfg=replace(CFG, max_hold_bars=1))
    when = signals.index[t] + pd.Timedelta(seconds=25)
    _step(bot, broker, when)
    broker.fail_close = 1                      # legs get cancelled, then the sell is rejected
    exit_at = next(w for w in _decision_times(bars, t, t + 20) if w > when)
    _step(bot, broker, exit_at)
    log = read_journal(journal.path)
    assert log["message"].fillna("").str.contains("EXIT_FAILED").any()
    assert broker.protective and "SYN" in broker.positions and bot.exit_pending
    assert broker.protective[-1][0] == "SYN"
    assert sorted(o.order_type for o in broker.get_open_orders(["SYN"])) == ["limit", "stop"]
    _step(bot, broker, exit_at + pd.Timedelta(seconds=30))                           # next heartbeat
    log = read_journal(journal.path)
    assert "SYN" not in broker.positions and not bot.exit_pending
    assert log.query("event == @J.EXIT_FILL").iloc[-1]["exit_reason"] == "TIME_EXIT"


def test_bot_backs_off_a_persistently_failing_exit(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal, cfg=replace(CFG, max_hold_bars=1))
    when = signals.index[t] + pd.Timedelta(seconds=25)
    _step(bot, broker, when)
    broker.fail_close = 99                                   # e.g. a trading halt
    exit_at = next(w for w in _decision_times(bars, t, t + 20) if w > when)
    _step(bot, broker, exit_at)
    for k in range(1, 9):                                    # 4 minutes of heartbeats
        _step(bot, broker, exit_at + pd.Timedelta(seconds=30 * k))
    failures = read_journal(journal.path)["message"].fillna("").str.contains("EXIT_FAILED").sum()
    assert 3 <= failures <= 4                                # 0s, +30s, +90s, +210s; not 9 attempts
    assert "SYN" in broker.positions and broker.get_open_orders(["SYN"])   # still protected meanwhile


def test_protection_is_never_placed_on_a_position_that_is_gone(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal)
    when = signals.index[t] + pd.Timedelta(seconds=25)
    _step(bot, broker, when)
    broker.close_position("SYN")                             # sold behind the bot's back
    bot._ensure_protection("SYN", when + pd.Timedelta(seconds=40), immediate=True)
    assert broker.protective == []


def test_bot_kill_switch_retries_a_failed_flatten_and_survives_restart(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal)
    when = signals.index[t] + pd.Timedelta(seconds=25)
    _step(bot, broker, when)
    broker.equity, broker.fail_flatten = 96_900.0, True
    _step(bot, broker, when + pd.Timedelta(seconds=30))       # trip; flatten raises
    assert "SYN" in broker.positions and bot.exit_pending["SYN"]["reason"] == "KILL_SWITCH"
    _step(bot, broker, when + pd.Timedelta(seconds=60))       # heartbeat retry closes it
    assert "SYN" not in broker.positions and broker.protective == []
    assert read_journal(journal.path).query("event == @J.EXIT_FILL").iloc[-1]["exit_reason"] == "KILL_SWITCH"
    restarted = _make_bot(broker, journal)                    # restart the same day
    restarted.bootstrap()
    assert restarted.risk.is_halted(when.tz_convert(ET.zone).date())


def test_bot_closes_an_adopted_position_with_no_known_stop(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal)
    when = signals.index[t] + pd.Timedelta(seconds=25)
    broker.now = when
    broker.positions["SYN"] = SimpleNamespace(symbol="SYN", qty="10", avg_entry_price="100", checked_until=when)
    for k in (0, 1, 3, 4):
        _step(bot, broker, when + pd.Timedelta(seconds=30 * k))
    log = read_journal(journal.path)
    assert log["message"].fillna("").str.contains("no known stop level").any()
    assert "SYN" not in broker.positions
    assert log.query("event == @J.EXIT_FILL").iloc[-1]["exit_reason"] == "UNPROTECTED_EXIT"


def test_stale_exit_state_never_sells_a_later_position(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal)
    when = signals.index[t] + pd.Timedelta(seconds=25)
    _step(bot, broker, when)
    bot.exit_pending["SYN"] = {"reason": "EARNINGS_EXIT", "ref": None, "attempts": 1, "order_id": None,
                               "last_try": when}
    broker.hide_fills = True                                  # position gone, fill not found
    for order in broker.get_open_orders(["SYN"]):
        order.status = "canceled"
    del broker.positions["SYN"]
    _step(bot, broker, when + pd.Timedelta(seconds=30))
    assert bot.exit_pending == {} and "SYN" not in bot.trades
    bot.exit_pending["SYN"] = {"reason": "X", "ref": None, "attempts": 1, "order_id": None, "last_try": when}
    blocked = bot.risk.entry_block_reason("SYN", when.date(), set(), set(bot.exit_pending), set(), "cid")
    assert blocked == "OPEN_ORDER_EXISTS"


def test_restart_keeps_the_ratcheted_stop_when_the_broker_lost_it(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal, cfg=TRAIL)
    when = signals.index[t] + pd.Timedelta(seconds=25)
    _step(bot, broker, when)
    raised = bot.trades["SYN"].stop_price + 1.0
    journal.log(J.STOP_RATCHET, now=when + pd.Timedelta(seconds=5), symbol="SYN", stop_price=raised)
    for order in broker.get_open_orders(["SYN"]):
        order.status = "canceled"
    restarted = _make_bot(broker, journal, cfg=TRAIL)
    restarted.bootstrap()
    assert restarted.trades["SYN"].stop_price == pytest.approx(raised) and restarted.trades["SYN"].trailed


def test_a_slow_exit_fill_is_not_resubmitted_or_mislabelled(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal, cfg=replace(CFG, max_hold_bars=1))
    when = signals.index[t] + pd.Timedelta(seconds=25)
    _step(bot, broker, when)
    broker.slow_close = True
    exit_at = next(w for w in _decision_times(bars, t, t + 20) if w > when)
    _step(bot, broker, exit_at)                               # market sell accepted, not filled yet
    order = next(o for o in broker.orders.values() if o.side == "sell" and o.order_type == "market")
    broker.now = exit_at + pd.Timedelta(seconds=10)
    broker._fill(order, 101.0)                                # fills between heartbeats
    original = broker.get_positions

    def positions_then_gone():
        result = original()
        broker.positions.pop("SYN", None)                     # gone right after this snapshot
        return result

    broker.get_positions = positions_then_gone
    _step(bot, broker, exit_at + pd.Timedelta(seconds=30))
    broker.get_positions = original
    _step(bot, broker, exit_at + pd.Timedelta(seconds=60))
    assert broker.closed.count("SYN") == 1
    assert read_journal(journal.path).query("event == @J.EXIT_FILL").iloc[-1]["exit_reason"] == "TIME_EXIT"


def test_bot_restores_a_missing_stop_after_the_grace_period(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal)
    when = signals.index[t] + pd.Timedelta(seconds=25)
    _step(bot, broker, when)
    for order in broker.get_open_orders(["SYN"]):
        if order.order_type == "stop":
            order.status = "canceled"            # e.g. a broken OCO after a manual edit
    _step(bot, broker, when + pd.Timedelta(seconds=30))
    assert broker.protective == []               # inside the grace period
    _step(bot, broker, when + pd.Timedelta(seconds=95))
    assert broker.protective == [("SYN", bot.trades["SYN"].stop_price)]
    assert read_journal(journal.path)["event"].eq(J.PROTECTIVE_STOP).any()
    legs = broker.get_open_orders(["SYN"])                   # exactly one stop and one target again
    assert sorted(o.order_type for o in legs) == ["limit", "stop"]


def test_bot_closes_at_market_when_protection_cannot_be_restored(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal)
    when = signals.index[t] + pd.Timedelta(seconds=25)
    _step(bot, broker, when)
    for order in broker.get_open_orders(["SYN"]):
        order.status = "canceled"
    broker.fail_protect = True
    for k in (1, 3, 4):
        _step(bot, broker, when + pd.Timedelta(seconds=30 * k))
    log = read_journal(journal.path)
    assert log["message"].fillna("").str.contains("UNPROTECTED").any()
    assert "SYN" not in broker.positions
    assert log.query("event == @J.EXIT_FILL").iloc[-1]["exit_reason"] == "UNPROTECTED_EXIT"


def test_bot_ratchet_failure_keeps_old_stop_and_gives_up_after_three_tries(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal, cfg=TRAIL)
    when = signals.index[t] + pd.Timedelta(seconds=25)
    _step(bot, broker, when)
    broker.fail_replace = True
    initial = bot.trades["SYN"].stop_price
    bot.desired_stop["SYN"] = {"stop": initial + 0.5, "attempts": 0}
    for k in range(4):
        _step(bot, broker, when + pd.Timedelta(seconds=30 * (k + 1)))
    assert broker.bracket_levels("SYN")[0] == initial and bot.trades["SYN"].stop_price == initial
    errors = read_journal(journal.path)["message"].fillna("")
    assert errors.str.contains("Stop ratchet").sum() == 3 and errors.str.contains("giving up").sum() == 1


# ============================================================================ alpaca wrapper (offline)
def test_bracket_order_request_is_gtc_bracket():
    from src.alpaca_client import build_bracket_order

    fields = build_bracket_order("SPY", 10, 495.5, 512.25, "tpb1-SPY-20261012T1030").to_request_fields()
    assert fields["order_class"].value == "bracket" and fields["time_in_force"].value == "gtc"
    assert fields["side"].value == "buy" and fields["type"].value == "market" and fields["qty"] == 10
    assert fields["stop_loss"] == {"stop_price": 495.5} and fields["take_profit"] == {"limit_price": 512.25}
    assert fields["client_order_id"] == "tpb1-SPY-20261012T1030"


def _api_error(status: int, message: str = "boom"):
    from alpaca.common.exceptions import APIError

    return APIError(json.dumps({"code": status * 100000, "message": message}),
                    SimpleNamespace(response=SimpleNamespace(status_code=status), request=None))


def test_retry_policy_backs_off_on_transient_errors_only():
    from src.alpaca_client import with_retries

    delays, calls = [], {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("reset")
        if calls["n"] == 2:
            raise _api_error(503)
        if calls["n"] == 3:
            raise _api_error(429)
        return "ok"

    assert with_retries(flaky, "flaky", max_retries=5, base_delay=1.0, sleep=delays.append) == "ok"
    assert len(delays) == 3 and delays[0] < delays[2] <= 4.0      # exponential with jitter, capped
    calls["n"] = 0

    def forbidden():
        calls["n"] += 1
        raise _api_error(403, "forbidden")

    with pytest.raises(Exception):
        with_retries(forbidden, "forbidden", sleep=delays.append)
    assert calls["n"] == 1, "4xx business errors must not be retried"
    with pytest.raises(ConnectionError):
        with_retries(lambda: (_ for _ in ()).throw(ConnectionError("down")), "down", max_retries=2,
                     sleep=lambda s: None)


def test_duplicate_client_order_id_returns_existing_order():
    from src.alpaca_client import AlpacaClient

    existing = SimpleNamespace(id="abc", client_order_id="tpb1-SPY-20261012T1030")

    class Trading:
        def submit_order(self, request):
            raise _api_error(422, "client_order_id must be unique")

        def get_order_by_client_id(self, cid):
            return existing

    client = AlpacaClient("k", "s", True, ExecutionConfig(), trading=Trading(), data=object(), sleep=lambda s: None)
    assert client.submit_bracket_order("SPY", 1, 99.0, 103.0, existing.client_order_id) is existing


def test_fetch_session_bars_from_alpaca_barset():
    from alpaca.data.models.bars import BarSet

    from src.alpaca_client import AlpacaClient

    src = _source_bars_for_day()
    raw = {"SPY": [{"t": ts.isoformat().replace("+00:00", "Z"), "o": r.open, "h": r.high, "l": r.low,
                    "c": r.close, "v": r.volume, "n": 1, "vw": r.close} for ts, r in src.iterrows()]}

    class Data:
        def get_stock_bars(self, request):
            assert request.feed.value == "iex" and request.adjustment.value == "all"
            return BarSet(raw)

    client = AlpacaClient("k", "s", True, ExecutionConfig(data_feed="iex"), trading=object(), data=Data())
    now = pd.Timestamp("2026-07-15 12:31", tz=ET.zone)
    bars = client.fetch_session_bars(["SPY", "QQQ"], now - pd.Timedelta(days=1), now, now=now)
    assert [f"{t:%H:%M}" for t in bars["SPY"].index] == ["09:30", "10:30", "11:30"]
    assert bars["QQQ"].empty
    pd.testing.assert_frame_equal(bars["SPY"], complete_bars(to_session_bars(src), now), check_freq=False)


def test_close_position_never_sells_into_pending_leg_cancels():
    from src.alpaca_client import AlpacaClient, ExitBlocked

    leg = SimpleNamespace(id="leg", symbol="SPY", side="sell", status="held", order_type="stop", type="stop")

    class Trading:
        sold = False

        def get_orders(self, filter=None):
            return [leg]                                    # the cancel never completes

        def cancel_order_by_id(self, order_id):
            pass

        def close_position(self, symbol):
            Trading.sold = True

    exec_cfg = ExecutionConfig(cancel_poll_timeout_seconds=0.0)
    client = AlpacaClient("k", "s", True, exec_cfg, trading=Trading(), data=object(), sleep=lambda s: None)
    with pytest.raises(ExitBlocked):
        client.close_position("SPY")
    assert not Trading.sold


def test_replace_stop_is_single_shot_and_needs_exactly_one_leg():
    from src.alpaca_client import AlpacaClient

    legs = [SimpleNamespace(id="a", symbol="SPY", side="sell", status="held", order_type="stop", type="stop")]
    calls = []

    class Trading:
        def get_orders(self, filter=None):
            return list(legs)

        def replace_order_by_id(self, order_id, request):
            calls.append((order_id, request.stop_price))
            raise _api_error(503)

    client = AlpacaClient("k", "s", True, ExecutionConfig(), trading=Trading(), data=object(), sleep=lambda s: None)
    with pytest.raises(Exception):
        client.replace_stop("SPY", 101.234)
    assert calls == [("a", 101.23)], "a non-idempotent replace must not be retried"
    legs.append(SimpleNamespace(id="b", symbol="SPY", side="sell", status="new", order_type="stop", type="stop"))
    with pytest.raises(RuntimeError, match="exactly one"):
        client.replace_stop("SPY", 101.0)


def test_restore_protection_places_oco_then_falls_back_to_a_plain_stop():
    from src.alpaca_client import AlpacaClient, ExitBlocked

    open_orders, submitted, cancelled = [], [], []

    class Trading:
        reject_oco = False

        def get_orders(self, filter=None):
            return list(open_orders)

        def cancel_order_by_id(self, order_id):
            cancelled.append(order_id)
            open_orders[:] = [o for o in open_orders if o.id != order_id]

        def submit_order(self, request):
            fields = request.to_request_fields()
            if fields.get("order_class") is not None and fields["order_class"].value == "oco" and Trading.reject_oco:
                raise _api_error(422, "oco rejected")
            submitted.append(fields)
            return SimpleNamespace(id="new")

    client = AlpacaClient("k", "s", True, ExecutionConfig(cancel_poll_timeout_seconds=0.0), trading=Trading(),
                          data=object(), sleep=lambda s: None)
    open_orders.append(SimpleNamespace(id="orphan-tp", symbol="AAPL", side="sell", status="new", order_type="limit",
                                       type="limit"))
    assert client.restore_protection("AAPL", 10, 195.123, 210.0) == "oco"
    assert cancelled == ["orphan-tp"]                       # the orphaned target held the shares
    oco = submitted[-1]
    assert oco["order_class"].value == "oco" and oco["side"].value == "sell" and oco["qty"] == 10
    assert oco["stop_loss"] == {"stop_price": 195.12} and oco["take_profit"] == {"limit_price": 210.0}
    Trading.reject_oco = True
    assert client.restore_protection("AAPL", 10, 195.12, 210.0) == "stop"
    assert submitted[-1]["type"].value == "stop" and submitted[-1]["stop_price"] == 195.12

    class Stuck(Trading):
        def cancel_order_by_id(self, order_id):
            pass                                            # cancel never completes

    open_orders.append(SimpleNamespace(id="leg", symbol="AAPL", side="sell", status="held", order_type="limit",
                                       type="limit"))
    stuck = AlpacaClient("k", "s", True, ExecutionConfig(cancel_poll_timeout_seconds=0.0), trading=Stuck(),
                         data=object(), sleep=lambda s: None)
    with pytest.raises(ExitBlocked):
        stuck.restore_protection("AAPL", 10, 195.12, 210.0)


def test_restore_protection_counts_a_lost_response_as_success():
    from src.alpaca_client import AlpacaClient

    open_orders = []

    class Trading:
        def get_orders(self, filter=None):
            return list(open_orders)

        def cancel_order_by_id(self, order_id):
            pass

        def submit_order(self, request):                    # accepted, but the response is lost
            open_orders.append(SimpleNamespace(id="oco", symbol="AAPL", side="sell", status="held",
                                               order_type="stop", type="stop", stop_price=195.12, limit_price=None))
            raise ConnectionError("reset by peer")

    client = AlpacaClient("k", "s", True, ExecutionConfig(cancel_poll_timeout_seconds=0.0), trading=Trading(),
                          data=object(), sleep=lambda s: None)
    assert client.restore_protection("AAPL", 10, 195.12, 210.0) == "oco"
    assert len(open_orders) == 1                             # no blind retry, no duplicate


def test_get_sessions_reads_the_exchange_calendar():
    from src.alpaca_client import AlpacaClient

    class Trading:
        def get_calendar(self, request):
            assert request.start == dt.date(2026, 7, 1)
            return [SimpleNamespace(date=dt.date(2026, 7, 2), close=dt.datetime(2026, 7, 2, 13, 0)),
                    SimpleNamespace(date=dt.date(2026, 7, 6), close=dt.datetime(2026, 7, 6, 16, 0))]

    client = AlpacaClient("k", "s", True, ExecutionConfig(), trading=Trading(), data=object())
    assert client.get_sessions(dt.date(2026, 7, 1), dt.date(2026, 7, 7)) == [dt.date(2026, 7, 2), dt.date(2026, 7, 6)]
    assert client.get_session_closes(dt.date(2026, 7, 1), dt.date(2026, 7, 7)) == {
        dt.date(2026, 7, 2): "13:00", dt.date(2026, 7, 6): "16:00"}


# ============================================================================ backtest CLI
@pytest.mark.parametrize("target", ["bars.csv.gz", "bars_dir"])
def test_save_and_load_bars_roundtrip(tmp_path, portfolio, target):
    data = {s: with_session_columns(portfolio[s]) for s in ("SYN7", "SYN1")}
    path = str(tmp_path / target)
    save_bars(data, path)
    loaded = load_bars(path, ["SYN7", "SYN1"])
    for symbol, df in data.items():
        pd.testing.assert_frame_equal(loaded[symbol][OHLC], df[OHLC], check_freq=False, check_names=False)
        assert (loaded[symbol]["bar_end"] == df["bar_end"]).all()
    with pytest.raises(FileNotFoundError):
        load_bars(path, ["NOPE"])


def test_backtest_cli_from_bars_window_and_trade_features(tmp_path, portfolio, capsys):
    path = str(tmp_path / "bars.csv.gz")
    save_bars({s: with_session_columns(portfolio[s]) for s in ("SYN7", "SYN1")}, path)
    trades_out = str(tmp_path / "trades.csv")
    args = ["--from-bars", path, "--symbols", "SYN7", "SYN1", "--start", "2023-03-01", "--end", "2023-09-30",
            "--trades-out", trades_out, "--ledger", str(tmp_path / "ledger.csv"), "--earnings", str(tmp_path / "x")]
    assert backtest.main(args) == 0
    out = capsys.readouterr().out
    assert "BACKTEST TEAR SHEET" in out and "EARNINGS BLACKOUT INCOMPLETE" in out
    trades = pd.read_csv(trades_out)
    assert {"sig_adx", "sig_plus_di", "sig_anchor_adx", "gross_r", "nights", "final_stop"} <= set(trades.columns)
    entries = pd.to_datetime(trades["entry_time"], utc=True)
    assert (entries >= pd.Timestamp("2023-03-01", tz=ET.zone)).all()
    assert (entries <= pd.Timestamp("2023-10-01", tz=ET.zone)).all()


@pytest.mark.slow
def test_backtest_cli_compare_and_ledger_warning(tmp_path, portfolio, capsys):
    path = str(tmp_path / "bars.csv.gz")
    save_bars({s: with_session_columns(portfolio[s]) for s in ("SYN7", "SYN1")}, path)
    args = ["--from-bars", path, "--symbols", "SYN7", "SYN1", "--years", "1", "--compare",
            "--ledger", str(tmp_path / "ledger.csv"), "--earnings", str(tmp_path / "x")]
    assert backtest.main(args) == 0
    first = capsys.readouterr().out
    for section in ("VARIANT COMPARISON", "v1 (original spec)", "v2 (defaults)", "Paired exit study",
                    "ema_cross (v1)", "Entry rules", "vs random drop", "added"):
        assert section in first
    assert "already evaluated" not in first
    assert backtest.main(args) == 0
    assert "already evaluated in 1 earlier run" in capsys.readouterr().out


def test_compare_isolates_the_blackout_and_reports_added_signals(portfolio):
    data = {s: portfolio[s][OHLC] for s in ("SYN7", "SYN1", "SYN3")}
    first_day = data["SYN1"].index[450].date()
    cal = {"SYN1": [(first_day, "AMC")]}
    tables = backtest.compare_variants(data, None, BacktestConfig(), ExecutionConfig(), cal)
    names = list(tables["portfolio"]["variant"])
    assert "v1 + earnings blackout" in names and "v2 without earnings blackout" in names
    plain = backtest.compare_variants(data, None, BacktestConfig(), ExecutionConfig(), None)
    v1_cal = tables["portfolio"].set_index("variant").loc["v1 (original spec)"]
    v1_plain = plain["portfolio"].set_index("variant").loc["v1 (original spec)"]
    assert v1_cal["trades"] == v1_plain["trades"]                 # the v1 baseline ignores the calendar
    rules = tables["filters"].set_index("filter")
    assert rules.loc["+DI>-DI signal bar", "pure_filter"]
    anchor = rules.loc["ADX/DI at anchor"]
    assert anchor["kept"] + anchor["removed"] == rules.loc["cost floor 0.10R", "kept"] + \
        rules.loc["cost floor 0.10R", "removed"]
    if anchor["added"] > 0:
        assert not anchor["pure_filter"] and anchor["random_drop_pct"] != anchor["random_drop_pct"]


def test_slippage_flag_moves_the_strategy_cost_too(tmp_path, portfolio, monkeypatch):
    seen = []

    class Spy(backtest.Backtester):
        def __init__(self, strategy_cfg=StrategyConfig(), *a, **kw):
            seen.append((strategy_cfg.round_trip_cost, a[1].slippage_pct if len(a) > 1 else None))
            super().__init__(strategy_cfg, *a, **kw)

    monkeypatch.setattr(backtest, "Backtester", Spy)
    path = str(tmp_path / "bars.csv.gz")
    save_bars({"SYN1": with_session_columns(portfolio["SYN1"])}, path)
    assert backtest.main(["--from-bars", path, "--symbols", "SYN1", "--slippage-bps", "2",
                          "--ledger", str(tmp_path / "l.csv"), "--earnings", str(tmp_path / "x")]) == 0
    assert seen[-1] == (pytest.approx(0.0004), pytest.approx(0.0002))


@pytest.mark.slow
# ============================================================================ trust evaluation
def test_evaluate_trust_on_simulated_three_weeks(live_setup, signals):
    bars, broker, journal, t = live_setup
    bot = _make_bot(broker, journal)
    broker.now = signals.index[t]
    bot.bootstrap()
    for when in _decision_times(bars, t - 1, t - 1 + 15 * 7):           # 15 trading days
        _step(bot, broker, when)
    df = read_journal(journal.path)
    checks, verdict = evaluate_trust.evaluate(df, ["SYN"])
    status = {c.name: c.status for c in checks}
    assert status["Bar-check coverage"] == "PASS"
    assert status["Decision latency after bar close"] == "PASS"
    assert status["Duplicate order submissions"] == "PASS"
    assert status["Entries while already in position"] == "PASS"
    assert status["Overlapping positions per symbol"] == "PASS"
    assert status["Every entry has a stop below the fill"] == "PASS"
    assert status["Per-trade risk <= limit"] == "PASS"
    assert status["Expectancy"] in {"INCONCLUSIVE"}                    # far fewer than 30 trades
    assert verdict.startswith("NOT READY")
    report = evaluate_trust.format_report(checks, verdict, journal.path, df)
    assert "VERDICT" in report


def test_week_robust_ci_is_wide_when_there_are_few_weeks():
    from backtest import bootstrap_ci

    rng = np.random.default_rng(3)
    r = rng.normal(0.17, 0.98, 30)                       # 30 trades crammed into 3 weeks
    weeks = np.repeat([1, 2, 3], 10)
    iid_lo, iid_hi = bootstrap_ci(r)
    lo, hi = bootstrap_ci(r, clusters=weeks)
    assert lo <= iid_lo and hi >= iid_hi                 # never narrower than the trade-level CI
    assert hi - lo > 1.5 * (iid_hi - iid_lo)             # t with 2 df: much wider
    false_pass = sum(bootstrap_ci(rng.normal(0, 1, 30), clusters=weeks, n=1000, seed=i)[0] > 0 for i in range(200))
    assert false_pass / 200 <= 0.05


def test_evaluate_trust_needs_twenty_weeks_before_judging_edge():
    rows = [{"event": J.EXIT_FILL, "symbol": "SPY", "r_multiple": 0.8, "pnl": 100.0,
             "ts_utc": pd.Timestamp("2026-06-01 15:00", tz="UTC") + pd.Timedelta(days=i % 21)} for i in range(40)]
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    edge = evaluate_trust.check_performance(df, min_trades=30, backtest_expectancy=None)[0]
    assert edge.status == "INCONCLUSIVE" and "distinct weeks" in edge.detail


def test_evaluate_trust_flags_duplicate_orders():
    rows = [{"event": J.ORDER_SUBMITTED, "symbol": "SPY", "client_order_id": "x"},
            {"event": J.ORDER_SUBMITTED, "symbol": "SPY", "client_order_id": "x"}]
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    status = {c.name: c.status for c in evaluate_trust.check_idempotency(df)}
    assert status["Duplicate order submissions"] == "FAIL"


def test_gitignore_protects_secrets_and_outputs():
    lines = open(".gitignore").read().split()
    for pattern in (".env", "*.log", "*.csv", ".venv/", "__pycache__/"):
        assert pattern in lines
