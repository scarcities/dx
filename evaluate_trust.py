"""Decide whether the paper bot has earned live capital. Run after 2-3 weeks:

    python evaluate_trust.py                         # reads paper_trading_journal.csv
    python evaluate_trust.py --replay                # also re-fetches bars and re-derives every signal
    python evaluate_trust.py --backtest-expectancy 0.15

Two separate questions, answered separately:
  1. OPERATIONAL TRUST: did the machine do exactly what it was supposed to? (coverage, latency,
     errors, duplicate orders, order lifecycle, slippage, risk limits, signal replay parity)
     Three weeks is enough to answer this.
  2. STATISTICAL EDGE: is the expectancy positive with confidence? Three weeks of a 1h
     pullback strategy on 5 symbols usually produces far fewer than 30 trades, so this
     will normally be INCONCLUSIVE. That is the honest answer, not a bug.
Live capital needs both to pass.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import config
from backtest import bootstrap_ci, week_clusters
from config import RiskConfig, StrategyConfig
from src import journal as J
from src.journal import read_journal
from src.sessions import ET, scheduled_bar_ends, in_trading_window

PASS, WARN, FAIL, INFO, INCONCLUSIVE = "PASS", "WARN", "FAIL", "INFO", "INCONCLUSIVE"


def expected_decision_bars(first: pd.Timestamp, last: pd.Timestamp, days) -> list[pd.Timestamp]:
    """Decision bars (bar starts the bot acts on: 10:30 ... 15:30 ET) on the given days,
    clipped to the observed run so a mid-day start or stop is not counted as missed."""
    out = []
    for day in days:
        out += [end for end in scheduled_bar_ends(day) if in_trading_window(end) and first <= end <= last]
    return out


@dataclass
class Check:
    category: str
    name: str
    status: str
    detail: str


def _rows(df: pd.DataFrame, event: str) -> pd.DataFrame:
    return df[df["event"] == event]


def check_coverage(df: pd.DataFrame, symbols: list[str]) -> list[Check]:
    checks = _rows(df, J.BAR_CHECK).dropna(subset=["decision_bar_et"])
    if checks.empty:
        return [Check("Operations", "Bar-check coverage", FAIL, "No BAR_CHECK rows in the journal")]
    decisions = checks.drop_duplicates(["symbol", "decision_bar_et"])
    days = sorted(decisions["decision_bar_et"].dt.date.unique())
    schedule = expected_decision_bars(decisions["decision_bar_et"].min(), decisions["decision_bar_et"].max(), days)
    expected = len(schedule) * len(symbols)
    coverage = min(len(decisions) / expected, 1.0) if expected else 0.0
    weekdays = pd.bdate_range(days[0], days[-1]).date
    missing = [d for d in weekdays if d not in set(days)]
    stale = int((checks["action"] == "SKIPPED_STALE").sum())
    status = PASS if coverage >= 0.95 and len(missing) <= 1 and stale == 0 else WARN
    if coverage < 0.80:
        status = FAIL
    detail = (f"{len(decisions)}/{expected} expected decisions on {len(days)} active days ({coverage:.1%}); "
              f"weekdays with no checks: {len(missing)} {[str(d) for d in missing][:5]} (holidays count here); "
              f"stale skips: {stale}")
    return [Check("Operations", "Bar-check coverage", status, detail)]


def check_latency(df: pd.DataFrame) -> list[Check]:
    checks = _rows(df, J.BAR_CHECK).dropna(subset=["decision_bar_et", "ts_utc"])
    if checks.empty:
        return []
    lag = (checks["ts_utc"] - checks["decision_bar_et"].dt.tz_convert("UTC")).dt.total_seconds()
    p95 = lag.quantile(0.95)
    status = PASS if p95 <= 120 else (WARN if p95 <= 300 else FAIL)
    return [Check("Operations", "Decision latency after bar close", status,
                  f"median {lag.median():.0f}s, p95 {p95:.0f}s, max {lag.max():.0f}s")]


def check_protection(df: pd.DataFrame) -> list[Check]:
    """Positions that lost their stop, and exits that did not complete on the first try."""
    messages = df["message"].fillna("")
    unprotected = int(messages.str.contains("UNPROTECTED").sum())
    restored = int((df["event"] == J.PROTECTIVE_STOP).sum())
    failed_exits = int(messages.str.contains("EXIT_FAILED").sum())
    ratchet_fail = int(messages.str.contains("Stop ratchet").sum())
    status = FAIL if unprotected else (WARN if restored or failed_exits or ratchet_fail else PASS)
    return [Check("Integrity", "Positions always protected by a stop", status,
                  f"{restored} stop(s) re-placed after going missing, {unprotected} protective stop(s) rejected, "
                  f"{failed_exits} exit attempt(s) retried, {ratchet_fail} failed stop ratchet attempt(s)")]


def check_errors(df: pd.DataFrame) -> list[Check]:
    errors = _rows(df, J.ERROR)
    n = len(errors)
    status = PASS if n == 0 else (WARN if n <= 5 else FAIL)
    top = errors["message"].fillna("").str.slice(0, 80).value_counts().head(3)
    detail = f"{n} ERROR rows" + (": " + "; ".join(f"{m} (x{c})" for m, c in top.items()) if n else "")
    return [Check("Operations", "Runtime errors", status, detail)]


def check_idempotency(df: pd.DataFrame) -> list[Check]:
    out = []
    submitted = _rows(df, J.ORDER_SUBMITTED)
    dup = submitted["client_order_id"].duplicated().sum()
    out.append(Check("Integrity", "Duplicate order submissions", PASS if dup == 0 else FAIL,
                     f"{len(submitted)} entries submitted, {dup} duplicate client_order_ids"))
    checks = _rows(df, J.BAR_CHECK)
    bad = checks[(checks["action"] == "ENTER") & (checks["in_position"].astype(str) == "True")]
    out.append(Check("Integrity", "Entries while already in position", PASS if bad.empty else FAIL,
                     f"{len(bad)} violations"))
    overlaps, open_symbols = 0, set()
    for _, row in df[df["event"].isin([J.ENTRY_FILL, J.EXIT_FILL])].iterrows():
        if row["event"] == J.ENTRY_FILL:
            overlaps += row["symbol"] in open_symbols
            open_symbols.add(row["symbol"])
        else:
            open_symbols.discard(row["symbol"])
    out.append(Check("Integrity", "Overlapping positions per symbol", PASS if overlaps == 0 else FAIL,
                     f"{overlaps} entry fills while a position was already open"))
    return out


def check_lifecycle(df: pd.DataFrame) -> list[Check]:
    submitted = set(_rows(df, J.ORDER_SUBMITTED)["client_order_id"].dropna())
    filled = set(_rows(df, J.ENTRY_FILL)["client_order_id"].dropna())
    rejected_rows = _rows(df, J.ORDER_REJECTED)
    rejected = set(rejected_rows["client_order_id"].dropna())
    orphans = submitted - filled - rejected
    status = PASS if not orphans else WARN
    out = [Check("Integrity", "Order lifecycle closed", status,
                 f"{len(submitted)} submitted, {len(filled)} filled, {len(rejected_rows)} rejected/cancelled, "
                 f"{len(orphans)} with no recorded outcome")]
    if len(rejected_rows):
        msgs = rejected_rows["message"].fillna("").str.slice(0, 80).value_counts().head(3)
        out.append(Check("Integrity", "Order rejections", WARN, "; ".join(f"{m} (x{c})" for m, c in msgs.items())))
    stops = _rows(df, J.ENTRY_FILL)
    unprotected = stops[~(stops["stop_price"] < stops["fill_price"])]
    out.append(Check("Integrity", "Every entry has a stop below the fill", PASS if unprotected.empty else FAIL,
                     f"{len(unprotected)} entries without a valid stop"))
    return out


def check_execution(df: pd.DataFrame, assumed_bps: float) -> list[Check]:
    out = []
    entries = _rows(df, J.ENTRY_FILL)["slippage_bps"].dropna()
    if len(entries):
        mean = entries.mean()
        status = PASS if mean <= assumed_bps else (WARN if mean <= 2 * assumed_bps else FAIL)
        out.append(Check("Execution", "Entry slippage vs backtest", status,
                         f"mean {mean:+.1f} bps, median {entries.median():+.1f}, p95 {entries.quantile(0.95):+.1f} "
                         f"(n={len(entries)}; backtest assumes {assumed_bps:.1f} bps)"))
    else:
        out.append(Check("Execution", "Entry slippage vs backtest", INCONCLUSIVE, "no entry fills yet"))
    exits = _rows(df, J.EXIT_FILL).dropna(subset=["slippage_bps"])
    for reason, grp in exits.groupby("exit_reason"):
        out.append(Check("Execution", f"Exit slippage ({reason})", INFO,
                         f"mean {grp['slippage_bps'].mean():+.1f} bps, max {grp['slippage_bps'].max():+.1f} "
                         f"(n={len(grp)})"))
    return out


def max_entry_gross(df: pd.DataFrame) -> float:
    """Largest sum of open positions' notional_pct right after an entry fill (each position at
    its own entry price and equity; good enough to catch the bot using margin)."""
    rows = df[df["event"].isin([J.ENTRY_FILL, J.EXIT_FILL])].sort_values("ts_utc")
    open_pct, worst = {}, 0.0
    for row in rows.itertuples(index=False):
        if row.event == J.ENTRY_FILL:
            open_pct[row.symbol] = float(row.notional_pct) if pd.notna(row.notional_pct) else 0.0
            worst = max(worst, sum(open_pct.values()))
        else:
            open_pct.pop(row.symbol, None)
    return worst


def check_risk(df: pd.DataFrame, risk_cfg: RiskConfig) -> list[Check]:
    out = []
    fills = _rows(df, J.ENTRY_FILL)
    if len(fills):
        max_risk = fills["risk_pct"].max()
        out.append(Check("Risk", "Per-trade risk <= limit", PASS if max_risk <= risk_cfg.risk_per_trade * 1.10 else FAIL,
                         f"max {max_risk:.3%}, mean {fills['risk_pct'].mean():.3%} "
                         f"(limit {risk_cfg.risk_per_trade:.2%}; the notional cap usually binds first)"))
        max_notional = fills["notional_pct"].max()
        out.append(Check("Risk", "Position notional <= cap",
                         PASS if max_notional <= risk_cfg.max_position_notional * 1.05 else FAIL,
                         f"max {max_notional:.2%} (cap {risk_cfg.max_position_notional:.0%})"))
        gross = max_entry_gross(df)
        out.append(Check("Risk", "Gross exposure at entry <= cap",
                         PASS if gross <= risk_cfg.max_gross_exposure * 1.05 else FAIL,
                         f"max {gross:.0%} of equity right after an entry, at entry prices "
                         f"(cap {risk_cfg.max_gross_exposure:.0%}; margin buying power must not be used)"))
    pnl = df["daily_pnl_pct"].dropna()
    if len(pnl):
        worst = pnl.min()
        limit = -risk_cfg.max_daily_drawdown
        status = PASS if worst > limit else (WARN if worst > limit - 0.005 else FAIL)
        out.append(Check("Risk", "Daily drawdown contained", status,
                         f"worst observed {worst:+.2%} vs kill-switch {limit:+.2%}; "
                         f"kill-switch trips: {len(_rows(df, J.KILL_SWITCH))}"))
    return out


def check_replay(df: pd.DataFrame, symbols: list[str], strategy_cfg: StrategyConfig) -> list[Check]:
    """Re-fetch bars and recompute each flat-position decision with the backtest code path."""
    from src.alpaca_client import AlpacaClient
    from src.strategy import generate_signals
    from config import ExecutionConfig

    checks = _rows(df, J.BAR_CHECK).dropna(subset=["decision_bar_et"])
    checks = checks[checks["in_position"].astype(str) != "True"]
    # Calendar-driven and cross-sectional (RS filter) decisions do not come from one symbol's
    # bars, so a per-symbol bar replay cannot reproduce them.
    checks = checks[~checks["reason_code"].isin(["EARNINGS_BLACKOUT", "EARNINGS_CALENDAR_STALE",
                                                 "RS_LAGGARD", "RS_NO_BENCHMARK"])]
    if checks.empty:
        return [Check("Parity", "Live vs replay signals", INCONCLUSIVE, "no flat bar checks to compare")]
    try:
        key, secret, paper = config.get_credentials()
    except RuntimeError as exc:
        return [Check("Parity", "Live vs replay signals", INCONCLUSIVE, str(exc))]
    exec_cfg = ExecutionConfig()
    client = AlpacaClient(key, secret, paper, exec_cfg)
    start = checks["decision_bar_et"].min() - pd.Timedelta(days=exec_cfg.live_lookback_days)
    end = min(checks["decision_bar_et"].max() + pd.Timedelta(hours=2), pd.Timestamp.now(tz="UTC"))
    bars = client.fetch_session_bars(symbols, start.to_pydatetime(), end.to_pydatetime())
    compared = matched = 0
    rsi_diff, mismatches = [], []
    for symbol, group in checks.groupby("symbol"):
        if symbol not in bars or bars[symbol].empty:
            continue
        sig = generate_signals(bars[symbol][["open", "high", "low", "close"]], strategy_cfg)
        for _, row in group.iterrows():
            t = row["decision_bar_et"]
            if t not in sig.index:
                continue
            replay = sig.loc[t]
            compared += 1
            if replay["reason_code"] == row["reason_code"]:
                matched += 1
            else:
                mismatches.append(f"{symbol} {t:%m-%d %H:%M} live={row['reason_code']} replay={replay['reason_code']}")
            if pd.notna(row["rsi"]) and pd.notna(replay["rsi"]):
                rsi_diff.append(abs(float(row["rsi"]) - float(replay["rsi"])))
    if not compared:
        return [Check("Parity", "Live vs replay signals", INCONCLUSIVE, "no overlapping bars returned")]
    rate = matched / compared
    status = PASS if rate >= 0.98 else (WARN if rate >= 0.95 else FAIL)
    detail = (f"{matched}/{compared} decisions identical ({rate:.1%}); median |RSI diff| "
              f"{np.median(rsi_diff) if rsi_diff else float('nan'):.3f}")
    if mismatches:
        detail += "; e.g. " + " | ".join(mismatches[:3])
    return [Check("Parity", "Live vs replay signals", status, detail)]


def check_performance(df: pd.DataFrame, min_trades: int, backtest_expectancy: Optional[float],
                      min_weeks: int = 20) -> list[Check]:
    exits = _rows(df, J.EXIT_FILL).dropna(subset=["r_multiple"])
    r = exits["r_multiple"].to_numpy(dtype=float)
    pnl = exits["pnl"].dropna()
    if len(r) == 0:
        return [Check("Edge", "Expectancy", INCONCLUSIVE, "no closed trades yet")]
    weeks = week_clusters(exits["ts_utc"])
    n_weeks = len(np.unique(weeks))
    lo, hi = bootstrap_ci(r, clusters=weeks)
    gross_loss = -pnl[pnl < 0].sum()
    pf = pnl[pnl > 0].sum() / gross_loss if gross_loss > 0 else float("inf")
    detail = (f"{len(r)} closed trades | win rate {(r > 0).mean():.0%} | expectancy {r.mean():+.3f}R "
              f"(95% CI {lo:+.3f} .. {hi:+.3f}, week-robust) | profit factor {pf:.2f} | P&L ${pnl.sum():,.2f}")
    if len(r) < min_trades or n_weeks < min_weeks:
        status = INCONCLUSIVE
        detail += (f" | need >= {min_trades} trades over >= {min_weeks} distinct weeks before the CI means "
                   f"anything (have {len(r)} over {n_weeks})")
    elif lo > 0:
        status = PASS
    elif hi < 0:
        status = FAIL
    else:
        status = WARN
    out = [Check("Edge", "Expectancy", status, detail)]
    if backtest_expectancy is not None:
        if len(r) < 10:
            out.append(Check("Edge", "Consistent with backtest", INFO,
                             f"{len(r)} trades: too few to compare with the backtest's {backtest_expectancy:+.3f}R"))
        else:
            inside = lo <= backtest_expectancy <= hi
            out.append(Check("Edge", "Consistent with backtest", INFO if inside else WARN,
                             f"backtest expectancy {backtest_expectancy:+.3f}R is "
                             f"{'inside' if inside else 'OUTSIDE'} the live 95% CI"))
    return out


def evaluate(df: pd.DataFrame, symbols: list[str], risk_cfg: RiskConfig = RiskConfig(),
             slippage_bps: float = config.SLIPPAGE_PER_SIDE * 1e4,
             min_trades: int = 30, backtest_expectancy: Optional[float] = None, replay: bool = False,
             strategy_cfg: StrategyConfig = StrategyConfig()) -> tuple[list[Check], str]:
    checks = (check_coverage(df, symbols) + check_latency(df) + check_errors(df) + check_idempotency(df)
              + check_protection(df) + check_lifecycle(df) + check_execution(df, slippage_bps)
              + check_risk(df, risk_cfg))
    if replay:
        checks += check_replay(df, symbols, strategy_cfg)
    edge = check_performance(df, min_trades, backtest_expectancy)
    checks += edge

    operational = [c for c in checks if c.category != "Edge"]
    op_fail = [c for c in operational if c.status == FAIL]
    op_warn = [c for c in operational if c.status == WARN]
    edge_status = edge[0].status
    if op_fail:
        verdict = f"NOT READY: {len(op_fail)} operational failure(s). Fix these before anything else."
    elif edge_status == PASS and not op_warn:
        verdict = "READY for a small live allocation: operations clean and expectancy CI above zero."
    elif edge_status == FAIL:
        verdict = "NOT READY: the strategy is losing money with statistical confidence."
    elif edge_status in (INCONCLUSIVE, WARN):
        verdict = ("NOT READY YET: operations look " + ("sound" if not op_warn else f"mostly sound ({len(op_warn)} warning(s))")
                   + ", but the edge is not proven. Keep paper trading until there are enough trades.")
    else:
        verdict = f"NOT READY: {len(op_warn)} operational warning(s) to investigate."
    return checks, verdict


def format_report(checks: list[Check], verdict: str, journal_path: str, df: pd.DataFrame) -> str:
    span = ""
    if not df.empty and df["ts_utc"].notna().any():
        span = f"{df['ts_utc'].min().tz_convert(ET):%Y-%m-%d} -> {df['ts_utc'].max().tz_convert(ET):%Y-%m-%d}"
    lines = ["=" * 78, " PAPER TRADING TRUST EVALUATION", "=" * 78,
             f" Journal: {journal_path}  ({len(df)} rows, {span})"]
    category = None
    for c in checks:
        if c.category != category:
            category = c.category
            lines += ["-" * 78, f" {category}"]
        lines.append(f"  [{c.status:^12}] {c.name}: {c.detail}")
    states = df[df["event"] == J.BAR_CHECK]["reason_code"].value_counts(normalize=True)
    if len(states):
        lines += ["-" * 78, " Decision mix: " + ", ".join(f"{k} {v:.0%}" for k, v in states.head(8).items())]
    lines += ["=" * 78, f" VERDICT: {verdict}", "=" * 78]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--journal", default=config.JOURNAL_PATH)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--slippage-bps", type=float, default=config.SLIPPAGE_PER_SIDE * 1e4,
                        help="per-side slippage assumed by the backtest (default: config.SLIPPAGE_PER_SIDE)")
    parser.add_argument("--backtest-expectancy", type=float, help="expectancy (R) from backtest.py, for comparison")
    parser.add_argument("--replay", action="store_true", help="re-fetch bars and verify every live decision")
    args = parser.parse_args(argv)

    df = read_journal(args.journal)
    if df.empty:
        print(f"Journal {args.journal} is empty or missing.")
        return 1
    checks, verdict = evaluate(df, config.SYMBOLS, RiskConfig(), args.slippage_bps, args.min_trades,
                               args.backtest_expectancy, args.replay)
    print(format_report(checks, verdict, args.journal, df))
    return 0 if verdict.startswith("READY") else 1


if __name__ == "__main__":
    raise SystemExit(main())
