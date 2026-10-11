"""Portfolio-level, event-driven walk-forward backtester.

Uses the exact strategy, sizing and kill-switch code the live bot uses:
  - decisions come from `generate_signals` (shift(1): bar t only sees bars <= t-1)
  - entries/exits fill at the open of bar t, only when bar t starts inside 09:45-15:45 ET
    (the live bot acts right after the previous bar closes, inside the same window)
  - bracket stop/target are computed from Close[t-1] and ATR[t-1], like the live order
  - per-side cost BacktestConfig.slippage_pct (config.SLIPPAGE_PER_SIDE, or --slippage-bps) on
    entries and on market/stop exits; take-profit limits fill at the limit with no slippage, and
    only when the high trades through the target by a cent
  - stops and targets that gap are filled at the open; if a bar touches both, the stop wins

"Walk-forward" here means a strictly causal bar-by-bar simulation, reported per
chronological fold. Parameters are fixed in config.py, so nothing is fitted in-sample;
the folds show whether the edge is stable through time or comes from one period.

Usage (on a machine with network access and a .env):
    python backtest.py                       # 2 years, config universe
    python backtest.py --years 1 --symbols SPY QQQ --feed sip --trades-out trades.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd

import config
from config import BacktestConfig, ExecutionConfig, RiskConfig, StrategyConfig
from src.events import EarningsCalendar, entry_blackout, events_per_year, forced_exit, load_earnings_calendar
from src.risk_manager import RiskManager, r_multiple
from src.sessions import ET, in_trading_window
from src.strategy import R_EARNINGS, generate_signals, ratchet_stop, stop_too_close

TRADING_DAYS = 252
# Signal-bar (t-1) values written next to each trade so conditional effects can be studied
# before anything becomes a rule.
SIGNAL_FEATURES = ["close", "atr", "natr", "cost_r", "rsi", "rsi_min", "adx", "plus_di", "minus_di",
                   "anchor_adx", "anchor_plus_di", "anchor_minus_di", "anchor_age", "ema_fast", "ema_slow"]


@dataclass
class Position:
    symbol: str
    qty: int
    entry_time: pd.Timestamp
    entry_raw: float
    entry_fill: float
    entry_ref: float
    initial_stop: float
    stop: float
    take_profit: float
    planned_risk: float
    risk_pct: float
    notional_pct: float
    binding_constraint: str
    exit_armed: bool
    features: dict = field(default_factory=dict)  # signal-bar values, exported with --trades-out
    bars_held: int = 0
    closed_bars: int = 0                  # bars closed since entry, entry bar included
    highest_close: float = float("-inf")  # highest close of the bars since entry (excludes the signal bar)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict
    folds: pd.DataFrame
    per_symbol: pd.DataFrame
    reasons: pd.Series
    events: list = field(default_factory=list)
    config: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- metrics
# Two-sided 97.5% Student t quantiles for small degrees of freedom (no scipy dependency).
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 22: 2.074, 24: 2.064, 26: 2.056, 28: 2.048, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t975(df: int) -> float:
    if df <= 0:
        return float("inf")
    keys = sorted(_T975)
    for k in keys:
        if df <= k:
            return _T975[k]
    return 1.960


def bootstrap_ci(values, clusters=None, n: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean R. Always the trade-level percentile bootstrap; with `clusters` (e.g.
    ISO week of entry) the interval is widened to also cover a cluster-robust t interval with
    G-1 degrees of freedom. A cluster *percentile* bootstrap is far too narrow when there are
    only a handful of weeks (it spans roughly the lowest to highest weekly mean)."""
    values = np.asarray(values, dtype=float)
    keep = ~np.isnan(values)
    values = values[keep]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    lo, hi = float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
    if clusters is not None:
        labels = np.asarray(clusters)[keep]
        mean = values.mean()
        resid = pd.Series(values - mean).groupby(labels).sum().to_numpy()
        g = len(resid)
        if g < 2:
            return float("-inf"), float("inf")
        se = math.sqrt(g / (g - 1) * float((resid ** 2).sum())) / len(values)
        half = t975(g - 1) * se
        lo, hi = min(lo, mean - half), max(hi, mean + half)
    return lo, hi


def week_clusters(times) -> np.ndarray:
    stamps = pd.DatetimeIndex(pd.to_datetime(pd.Series(times), utc=True)).tz_convert(ET)
    iso = stamps.isocalendar()
    return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()


def trade_stats(r_values: pd.Series, pnl: pd.Series) -> dict:
    r_values = pd.Series(r_values, dtype=float).dropna()
    pnl = pd.Series(pnl, dtype=float).dropna()
    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = float("inf") if gross_win > 0 else float("nan")
    wins, losses = r_values[r_values > 0], r_values[r_values <= 0]
    return {
        "n_trades": int(len(pnl)),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else float("nan"),
        "profit_factor": float(profit_factor),
        "expectancy_r": float(r_values.mean()) if len(r_values) else float("nan"),
        "avg_win_r": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss_r": float(losses.mean()) if len(losses) else float("nan"),
        "total_pnl": float(pnl.sum()),
    }


def equity_stats(equity: pd.Series) -> dict:
    """Sharpe/Sortino from daily returns (annualised with sqrt(252)), drawdown from every bar."""
    if equity.empty:
        return {key: float("nan") for key in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown",
                                              "volatility")} | {"n_days": 0}
    daily = equity.groupby(equity.index.tz_convert(ET).date).last()
    returns = daily.pct_change().dropna()
    n_days = len(daily)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = max(n_days / TRADING_DAYS, 1e-9)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0
    std = returns.std(ddof=1) if len(returns) > 1 else float("nan")
    downside = math.sqrt((returns.clip(upper=0.0) ** 2).mean()) if len(returns) else float("nan")
    sharpe = returns.mean() / std * math.sqrt(TRADING_DAYS) if std and std > 0 else float("nan")
    sortino = returns.mean() / downside * math.sqrt(TRADING_DAYS) if downside and downside > 0 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_end": drawdown.idxmin(),
        "volatility": float(std * math.sqrt(TRADING_DAYS)) if std == std else float("nan"),
        "n_days": int(n_days),
    }


# ----------------------------------------------------------------------------- engine
class Backtester:
    def __init__(self, strategy_cfg: StrategyConfig = StrategyConfig(), risk_cfg: RiskConfig = RiskConfig(),
                 bt_cfg: BacktestConfig = BacktestConfig(), exec_cfg: ExecutionConfig = ExecutionConfig(),
                 calendar: Optional[EarningsCalendar] = None):
        self.scfg, self.rcfg, self.bcfg, self.ecfg = strategy_cfg, risk_cfg, bt_cfg, exec_cfg
        self.calendar = calendar or {}
        self.sessions: list = []
        self.last_decision: dict = {}  # session date -> last in-window bar start "HH:MM" (half days)

    def _entry_fill(self, price: float) -> float:
        return price * (1.0 + self.bcfg.slippage_pct)

    def _exit_fill(self, price: float, kind: str) -> float:
        slip = self.bcfg.limit_slippage_pct if kind == "limit" else self.bcfg.slippage_pct
        return price * (1.0 - slip)

    def _prepare(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        self.sessions = sorted(set().union(*[set(df.index.tz_convert(ET).date) for df in data.values()]))
        self.last_decision = {}
        for df in data.values():
            local = df.index.tz_convert(ET)
            in_window = [t for t in local if self._window_ok(t)]
            for t in in_window:
                stamp = f"{t:%H:%M}"
                if stamp > self.last_decision.get(t.date(), ""):
                    self.last_decision[t.date()] = stamp
        frames = {}
        for symbol, df in data.items():
            df = df.sort_index()
            sig = generate_signals(df[["open", "high", "low", "close"]], self.scfg).add_prefix("sig_")
            frames[symbol] = df[["open", "high", "low", "close"]].join(sig)
        return frames

    def _window_ok(self, ts) -> bool:
        return in_trading_window(ts, self.ecfg.trade_window_start, self.ecfg.trade_window_end)

    def _entry_blocked(self, symbol: str, ts) -> bool:
        return entry_blackout(symbol, ts, self.calendar, self.sessions) is not None

    def _must_exit(self, symbol: str, ts) -> bool:
        return forced_exit(symbol, ts, self.calendar, self.sessions, self.scfg.earnings_exit_time,
                           self.last_decision) is not None

    def _new_position(self, symbol: str, ts, row, size) -> Position:
        features = {f"sig_{k}": float(getattr(row, f"sig_{k}")) for k in SIGNAL_FEATURES}
        return Position(symbol, size.shares, ts, float(row.open), self._entry_fill(row.open), size.entry_ref,
                        size.stop_price, size.stop_price, size.take_profit, size.entry_ref - size.stop_price,
                        size.risk_pct, size.notional_pct, size.binding_constraint,
                        exit_armed=bool(row.sig_above_fast) or not self.scfg.ema_exit_requires_arming,
                        features=features)

    def _advance(self, pos: Position, row) -> None:
        """Bookkeeping when bar t opens: bar t-1 has closed while the position was held."""
        pos.closed_bars += 1
        pos.highest_close = max(pos.highest_close, float(row.sig_close))
        pos.exit_armed |= bool(row.sig_above_fast) or not self.scfg.ema_exit_requires_arming

    def _decision_exit(self, pos: Position, row, blackout: bool) -> Optional[tuple[float, str, str]]:
        """Market exit at the open of bar t, or a stop ratchet, decided from bar t-1 information.
        Mirrors TradingBot._evaluate_symbol; only called inside the trading window."""
        if blackout:
            return float(row.open), "EARNINGS_EXIT", "market"
        cfg = self.scfg
        if cfg.exit_mode == "ema_cross" and bool(row.sig_exit) and pos.exit_armed:
            return float(row.open), "EMA_EXIT", "market"
        if cfg.max_hold_bars and pos.closed_bars >= cfg.max_hold_bars:
            return float(row.open), "TIME_EXIT", "market"
        if cfg.exit_mode == "ratchet":
            new_stop = ratchet_stop(pos.stop, pos.entry_fill, pos.entry_ref, pos.initial_stop, pos.highest_close,
                                    float(row.sig_atr), cfg)
            if new_stop > pos.stop and stop_too_close(new_stop, float(row.sig_close), cfg):
                return float(row.open), "TRAIL_STOP", "market"
            pos.stop = new_stop
        return None

    def _bracket_exit(self, pos: Position, row) -> Optional[tuple[float, str, str]]:
        """Bracket legs during bar t. Gaps fill at the open; if a bar touches both, the stop wins."""
        stop_label = "TRAIL_STOP" if pos.stop > pos.initial_stop + 1e-9 else "STOP_LOSS"
        target_hit = pos.take_profit + (0.01 if self.bcfg.tp_requires_trade_through else 0.0)
        if row.open <= pos.stop:
            return float(row.open), f"{stop_label}_GAP", "market"
        if row.open >= pos.take_profit:
            return float(row.open), "TAKE_PROFIT_GAP", "limit"
        if row.low <= pos.stop:
            return pos.stop, stop_label, "stop"
        if row.high >= target_hit - 1e-9:
            return pos.take_profit, "TAKE_PROFIT", "limit"
        return None

    def _trade_record(self, pos: Position, ts, raw_price: float, reason: str, kind: str) -> dict:
        fill = self._exit_fill(raw_price, kind)
        gross_risk = pos.entry_raw - pos.initial_stop
        nights = len(pd.bdate_range(pos.entry_time.tz_convert(ET).date(), pd.Timestamp(ts).tz_convert(ET).date())) - 1
        return {
            "symbol": pos.symbol, "entry_time": pos.entry_time, "exit_time": ts, "qty": pos.qty,
            "entry_ref": pos.entry_ref, "entry_raw": pos.entry_raw, "entry_fill": pos.entry_fill,
            "stop": pos.initial_stop, "final_stop": pos.stop, "take_profit": pos.take_profit,
            "exit_raw": raw_price, "exit_fill": fill, "exit_reason": reason,
            "r_multiple": r_multiple(pos.entry_fill, fill, pos.initial_stop, pos.planned_risk),
            "gross_r": (raw_price - pos.entry_raw) / gross_risk if gross_risk > 0 else float("nan"),
            "stop_pct": pos.planned_risk / pos.entry_ref, "nights": max(nights, 0),
            "pnl": pos.qty * (fill - pos.entry_fill), "bars_held": pos.bars_held, "risk_pct": pos.risk_pct,
            "notional_pct": pos.notional_pct, "binding_constraint": pos.binding_constraint,
            "slippage_cost": pos.qty * (pos.entry_fill - pos.entry_raw + raw_price - fill),
            **pos.features,
        }

    def run(self, data: dict[str, pd.DataFrame], start: Optional[pd.Timestamp] = None) -> BacktestResult:
        """`data`: symbol -> session-aligned hourly OHLCV (tz-aware index = bar start).
        Bars before `start` only warm up indicators; no trades and no equity are recorded there."""
        risk = RiskManager(self.rcfg, self.scfg)
        frames = self._prepare(data)
        rows_by_symbol = {s: dict(zip(f.index, f.itertuples(index=False))) for s, f in frames.items()}
        timeline = sorted(set().union(*[f.index for f in frames.values()]))
        if start is not None:
            start = pd.Timestamp(start)
        cash = self.bcfg.initial_equity
        positions: dict[str, Position] = {}
        last_close: dict[str, float] = {}
        trades, events, curve, reasons = [], [], [], []
        equity = cash
        day_start_equity, current_day = equity, None

        def close_trade(pos: Position, ts, raw_price: float, reason: str, kind: str = "market") -> None:
            nonlocal cash
            record = self._trade_record(pos, ts, raw_price, reason, kind)
            cash += pos.qty * record["exit_fill"]
            trades.append(record)
            del positions[pos.symbol]

        for ts in timeline:
            active = start is None or ts >= start
            day = ts.tz_convert(ET).date()
            if day != current_day:
                current_day, day_start_equity = day, equity
            window_ok = self._window_ok(ts)

            for symbol, rows in rows_by_symbol.items():
                row = rows.get(ts)
                if row is None:
                    continue
                pos = positions.get(symbol)

                # 1) Decisions made at the open of bar t from bar t-1 information.
                if pos is not None:
                    self._advance(pos, row)
                    exit_ = self._decision_exit(pos, row, self._must_exit(symbol, ts)) if window_ok else None
                    if exit_:
                        close_trade(pos, ts, *exit_)
                        pos = None
                elif active:
                    blackout = self._entry_blocked(symbol, ts)
                    reasons.append(R_EARNINGS if blackout else row.sig_reason_code)
                    if (window_ok and bool(row.sig_entry) and not blackout and not risk.is_halted(day)
                            and len(positions) < self.rcfg.max_open_positions):
                        size = risk.size_position(equity, float(row.sig_close), float(row.sig_atr), buying_power=cash)
                        if size.shares >= 1:
                            pos = positions[symbol] = self._new_position(symbol, ts, row, size)
                            cash -= size.shares * pos.entry_fill

                # 2) Bracket legs during bar t.
                if pos is not None:
                    hit = self._bracket_exit(pos, row)
                    if hit:
                        close_trade(pos, ts, *hit)
                    else:
                        pos.bars_held += 1
                last_close[symbol] = float(row.close)

            # 3) Mark to market at the close of bar t, then the daily kill-switch.
            equity = cash + sum(p.qty * last_close[s] for s, p in positions.items())
            if active:
                status = risk.check_daily_drawdown(day, day_start_equity, equity)
                if status.newly_tripped:
                    events.append({"time": ts, "event": "KILL_SWITCH", "daily_pnl_pct": status.daily_pnl_pct})
                    if self.rcfg.kill_switch_flatten:
                        for pos in list(positions.values()):
                            close_trade(pos, ts, last_close[pos.symbol], "KILL_SWITCH")
                        equity = cash
                curve.append((ts, equity))

        if positions:
            final_ts = timeline[-1]
            for pos in list(positions.values()):
                close_trade(pos, final_ts, last_close[pos.symbol], "END_OF_DATA")
            equity = cash
            if curve:
                curve[-1] = (final_ts, equity)

        trades_df = pd.DataFrame(trades)
        equity_curve = pd.Series(dict(curve), dtype=float).sort_index()
        return self._summarise(trades_df, equity_curve, pd.Series(reasons, dtype=object), events, list(frames))

    def signal_outcomes(self, data: dict[str, pd.DataFrame], start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Every tradable entry signal simulated on its own (no portfolio interaction, 1 share):
        one row per signal with its R under this backtester's exit rules. Running it with
        different exit settings on the same entries gives a paired comparison."""
        results = []
        sizer = RiskManager(self.rcfg, self.scfg)
        for symbol, frame in self._prepare(data).items():
            index = frame.index
            rows = list(frame.itertuples(index=False))
            for i in np.flatnonzero(frame["sig_entry"].to_numpy()):
                ts = index[i]
                if (start is not None and ts < start) or not self._window_ok(ts) or self._entry_blocked(symbol, ts):
                    continue
                row = rows[i]
                size = sizer.size_position(1e12, float(row.sig_close), float(row.sig_atr))
                pos = self._new_position(symbol, ts, row, size)
                pos.qty = 1
                exit_ = self._bracket_exit(pos, row)
                j = i
                while exit_ is None and j + 1 < len(frame):
                    j += 1
                    nxt = rows[j]
                    pos.bars_held += 1
                    self._advance(pos, nxt)
                    if self._window_ok(index[j]):
                        exit_ = self._decision_exit(pos, nxt, self._must_exit(symbol, index[j]))
                    exit_ = exit_ or self._bracket_exit(pos, nxt)
                if exit_ is None:
                    exit_ = (float(frame["close"].iloc[-1]), "END_OF_DATA", "market")
                record = self._trade_record(pos, index[j], *exit_)
                results.append({k: record[k] for k in ("symbol", "entry_time", "exit_time", "exit_reason",
                                                    "r_multiple", "gross_r", "stop_pct", "nights", "bars_held")})
        return pd.DataFrame(results, columns=["symbol", "entry_time", "exit_time", "exit_reason", "r_multiple",
                                           "gross_r", "stop_pct", "nights", "bars_held"])

    # ------------------------------------------------------------------------- report
    def _summarise(self, trades: pd.DataFrame, equity: pd.Series, reasons: pd.Series, events: list,
                   symbols: list[str]) -> BacktestResult:
        if trades.empty:
            trades = pd.DataFrame(columns=["symbol", "entry_time", "exit_time", "r_multiple", "gross_r", "stop_pct",
                                           "nights", "pnl", "exit_reason", "bars_held", "risk_pct", "notional_pct",
                                           "binding_constraint", "slippage_cost"])
        stats = trade_stats(trades["r_multiple"], trades["pnl"])
        metrics = {**equity_stats(equity), **stats}
        metrics["initial_equity"] = float(self.bcfg.initial_equity)
        metrics["final_equity"] = float(equity.iloc[-1]) if len(equity) else float(self.bcfg.initial_equity)
        metrics["avg_bars_held"] = float(trades["bars_held"].mean()) if len(trades) else float("nan")
        metrics["avg_risk_pct"] = float(trades["risk_pct"].mean()) if len(trades) else float("nan")
        metrics["gross_expectancy_r"] = float(trades["gross_r"].mean()) if len(trades) else float("nan")
        metrics["avg_nights"] = float(trades["nights"].mean()) if len(trades) else float("nan")
        metrics["slippage_cost"] = float(trades["slippage_cost"].sum()) if len(trades) else 0.0
        metrics["kill_switch_events"] = len(events)
        metrics["exposure"] = _exposure(trades, equity)

        folds = []
        if len(equity) >= self.bcfg.n_folds:
            for i, chunk in enumerate(np.array_split(np.arange(len(equity)), self.bcfg.n_folds), start=1):
                seg = equity.iloc[chunk]
                lo, hi = seg.index[0], seg.index[-1]
                in_fold = trades[(trades["entry_time"] >= lo) & (trades["entry_time"] <= hi)] if len(trades) else trades
                fold_eq = equity_stats(seg)
                fold_tr = trade_stats(in_fold["r_multiple"], in_fold["pnl"])
                folds.append({"fold": i, "start": lo, "end": hi, "trades": fold_tr["n_trades"],
                              "return": fold_eq["total_return"], "sharpe": fold_eq["sharpe"],
                              "max_dd": fold_eq["max_drawdown"], "profit_factor": fold_tr["profit_factor"],
                              "expectancy_r": fold_tr["expectancy_r"]})
        per_symbol = []
        for symbol in symbols:
            sub = trades[trades["symbol"] == symbol]
            per_symbol.append({"symbol": symbol, **trade_stats(sub["r_multiple"], sub["pnl"])})
        reason_share = reasons.value_counts(normalize=True) if len(reasons) else pd.Series(dtype=float)
        return BacktestResult(trades, equity, metrics, pd.DataFrame(folds), pd.DataFrame(per_symbol),
                              reason_share, events, {"strategy": asdict(self.scfg), "risk": asdict(self.rcfg),
                                                     "backtest": asdict(self.bcfg)})


def _exposure(trades: pd.DataFrame, equity: pd.Series) -> float:
    if trades.empty or equity.empty:
        return 0.0
    in_market = pd.Series(False, index=equity.index)
    for _, trade in trades.iterrows():
        in_market |= (equity.index >= trade["entry_time"]) & (equity.index <= trade["exit_time"])
    return float(in_market.mean())


def _pct(x: float, signed: bool = True) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{x:+.2%}" if signed else f"{x:.2%}"


def _num(x: float, fmt: str = ".2f") -> str:
    if x is None or x != x:
        return "n/a"
    return "inf" if math.isinf(x) else format(x, fmt)


def format_tear_sheet(result: BacktestResult, title: str = "BACKTEST TEAR SHEET") -> str:
    m = result.metrics
    eq = result.equity
    lines = ["=" * 78, f" {title}", "=" * 78]
    if len(eq):
        lines.append(f" Period              {eq.index[0]:%Y-%m-%d} -> {eq.index[-1]:%Y-%m-%d}  ({m['n_days']} trading days)")
    lines += [
        f" Symbols             {', '.join(result.per_symbol['symbol']) if len(result.per_symbol) else '-'}",
        f" Bars / costs        session-aligned 1h | slippage {result.config['backtest']['slippage_pct']:.2%} per side",
        f" Equity              ${m['initial_equity']:,.2f} -> ${m['final_equity']:,.2f}",
        f" Total return        {_pct(m['total_return'])}   CAGR {_pct(m['cagr'])}   Ann. vol {_pct(m['volatility'], False)}",
        f" Sharpe (ann.)       {_num(m['sharpe'])}",
        f" Sortino (ann.)      {_num(m['sortino'])}",
        f" Max drawdown        {_pct(m['max_drawdown'])}",
        f" Profit factor       {_num(m['profit_factor'])}",
        f" Expectancy          {_num(m['expectancy_r'], '+.3f')} R per trade",
        f" Trades              {m['n_trades']}  | win rate {_pct(m['win_rate'], False)} | avg win "
        f"{_num(m['avg_win_r'], '+.2f')}R | avg loss {_num(m['avg_loss_r'], '+.2f')}R",
        f" Avg hold / exposure {_num(m['avg_bars_held'], '.1f')} bars | {m['exposure']:.1%} of bars in market",
        f" Avg risk per trade  {_pct(m['avg_risk_pct'], False)} of equity | slippage paid ${m['slippage_cost']:,.2f}",
        f" Gross expectancy    {_num(m['gross_expectancy_r'], '+.3f')} R before costs | avg nights held "
        f"{_num(m['avg_nights'], '.1f')}",
        f" Kill-switch trips   {m['kill_switch_events']}",
    ]
    trades = result.trades
    if len(trades):
        exits = trades["exit_reason"].value_counts()
        lines.append(" Exit reasons        " + ", ".join(f"{k} {v}" for k, v in exits.items()))
        binding = trades["binding_constraint"].value_counts()
        lines.append(" Size limited by     " + ", ".join(f"{k} {v}" for k, v in binding.items()))
    if len(result.folds):
        lines += ["-" * 78, " Walk-forward folds (chronological, out-of-sample: parameters are fixed)"]
        lines.append(f" {'fold':>4} {'start':>10} {'end':>10} {'trades':>6} {'return':>8} {'sharpe':>7} "
                     f"{'maxDD':>8} {'PF':>6} {'exp R':>7}")
        for _, f in result.folds.iterrows():
            lines.append(f" {f['fold']:>4} {f['start']:%Y-%m-%d} {f['end']:%Y-%m-%d} {f['trades']:>6} "
                         f"{_pct(f['return']):>8} {_num(f['sharpe']):>7} {_pct(f['max_dd']):>8} "
                         f"{_num(f['profit_factor']):>6} {_num(f['expectancy_r'], '+.3f'):>7}"
                         f"{'  n<30: noise' if f['trades'] < 30 else ''}")
    if len(result.per_symbol):
        lines += ["-" * 78, f" {'symbol':<8} {'trades':>6} {'win%':>7} {'PF':>6} {'exp R':>7} {'P&L':>12}"]
        for _, s in result.per_symbol.iterrows():
            lines.append(f" {s['symbol']:<8} {s['n_trades']:>6} {_pct(s['win_rate'], False):>7} "
                         f"{_num(s['profit_factor']):>6} {_num(s['expectancy_r'], '+.3f'):>7} {s['total_pnl']:>12,.2f}")
    if len(result.reasons):
        lines += ["-" * 78, " Decision mix while flat (share of bar checks)"]
        lines += [f"   {code:<26} {share:6.1%}" for code, share in result.reasons.items()]
    if m["n_trades"] < 30:
        lines += ["-" * 78, f" WARNING: {m['n_trades']} trades is too few for these statistics to mean much."]
    lines.append("=" * 78)
    return "\n".join(lines)


# ----------------------------------------------------------------------------- variant comparison
V1_SPEC = dict(exit_mode="ema_cross", max_cost_r=None, directional_check=None, max_hold_bars=None)
EXCLUDE = "_exclude_symbols"

# Portfolio runs: each v2 change in isolation on top of the v1 spec, then the v2 defaults.
COMPARE_VARIANTS = {
    "v1 (original spec)": V1_SPEC,
    "v1 + bracket exit": dict(V1_SPEC, exit_mode="bracket"),
    "v1 + ratchet exit": dict(V1_SPEC, exit_mode="ratchet"),
    "v1 + bracket + 14-bar time stop": dict(V1_SPEC, exit_mode="bracket", max_hold_bars=14),
    "v1 + cost floor 0.10R": dict(V1_SPEC, max_cost_r=0.10),
    "v1 without SPY, QQQ": dict(V1_SPEC, **{EXCLUDE: ("SPY", "QQQ")}),
    "v1 + ADX/DI at pre-dip anchor": dict(V1_SPEC, directional_check="anchor"),
    "v1 + +DI>-DI on signal bar": dict(V1_SPEC, directional_check="signal_bar"),
    "v2 (defaults)": {},
}
# Paired exit study: the same v1 entry signals, each exit rule simulated independently.
EXIT_VARIANTS = {
    "ema_cross (v1)": dict(exit_mode="ema_cross"),
    "bracket": dict(exit_mode="bracket"),
    "ratchet": dict(exit_mode="ratchet"),
    "bracket + 14-bar time stop": dict(exit_mode="bracket", max_hold_bars=14),
}
# Entry filters judged by the v1 signals they would remove.
FILTER_VARIANTS = {
    "cost floor 0.10R": dict(max_cost_r=0.10),
    "ADX/DI at anchor": dict(directional_check="anchor"),
    "+DI>-DI signal bar": dict(directional_check="signal_bar"),
}


def breakeven_cost_bps(trades: pd.DataFrame) -> float:
    """Per-side cost c (bp) at which mean net R reaches zero. In R units a trade pays
    k * c / stop_pct, with k = 1 when it exits on a take-profit limit (no exit slippage) and 2
    otherwise, so c = mean(gross_r) / mean(k / stop_pct)."""
    sides = np.where(trades["exit_reason"].str.startswith("TAKE_PROFIT"), 1.0, 2.0)
    load = np.mean(sides / trades["stop_pct"].to_numpy(dtype=float))
    return float(trades["gross_r"].mean() / load * 1e4) if load > 0 else float("nan")


def cluster_random_drop_percentile(r: np.ndarray, clusters: np.ndarray, kept_mean: float, k: int,
                                   draws: int = 5000, seed: int = 0) -> float:
    """Share of random subsets that keep ~k signals by dropping whole ISO weeks (signals in the
    same week are not independent) whose mean R is below `kept_mean`."""
    rng = np.random.default_rng(seed)
    labels = np.unique(clusters)
    members = [np.flatnonzero(clusters == c) for c in labels]
    means = np.empty(draws)
    for d in range(draws):
        chosen, count = [], 0
        for i in rng.permutation(len(labels)):
            if count >= k:
                break
            chosen.append(rng.permutation(members[i]))
            count += len(members[i])
        # Exactly k: the last week drawn is kept only in part, otherwise small filters would be
        # compared with draws that keep every signal.
        means[d] = r[np.concatenate(chosen)[:k]].mean()
    return float((means < kept_mean).mean())


def _variant_row(name: str, result: BacktestResult) -> dict:
    trades, m = result.trades, result.metrics
    lo, hi = bootstrap_ci(trades["r_multiple"], clusters=week_clusters(trades["entry_time"])) \
        if len(trades) > 1 else (float("nan"), float("nan"))
    years = max(m["n_days"] / TRADING_DAYS, 1e-9)
    by_symbol = trades.groupby("symbol")["r_multiple"].sum() if len(trades) else pd.Series(dtype=float)
    best = by_symbol.idxmax() if len(by_symbol) else None
    without_best = trades.loc[trades["symbol"] != best, "r_multiple"].mean() if best is not None else float("nan")
    gross = trades["gross_r"].mean() if len(trades) else float("nan")
    breakeven_bps = breakeven_cost_bps(trades) if len(trades) and gross > 0 else float("nan")
    return {"variant": name, "trades": m["n_trades"], "per_year": m["n_trades"] / years,
            "expectancy_r": m["expectancy_r"], "ci_lo": lo, "ci_hi": hi, "gross_r": gross,
            "breakeven_bps": breakeven_bps, "profit_factor": m["profit_factor"], "max_drawdown": m["max_drawdown"],
            "best_symbol": best, "exp_without_best": without_best,
            "nights": trades["nights"].mean() if len(trades) else float("nan")}


def compare_variants(data: dict[str, pd.DataFrame], start, bt_cfg: BacktestConfig, exec_cfg: ExecutionConfig,
                     calendar: Optional[EarningsCalendar] = None,
                     base: Optional[StrategyConfig] = None) -> dict[str, pd.DataFrame]:
    base = base or StrategyConfig()
    portfolio = []
    runs = []
    for name, overrides in COMPARE_VARIANTS.items():
        overrides = dict(overrides)
        excluded = overrides.pop(EXCLUDE, ())
        if excluded and not set(excluded) & set(data):
            continue
        # The v1 rows (baseline and "v1 + one change") run without the earnings blackout, so each
        # isolates exactly one change; the blackout gets its own row and is included in v2.
        runs.append((name, replace(base, **overrides), None if name.startswith("v1") else calendar, excluded))
        if name.startswith("v1 (") and calendar:
            runs.append(("v1 + earnings blackout", replace(base, **overrides), calendar, ()))
    if calendar:
        runs.append(("v2 without earnings blackout", base, None, ()))
    for name, cfg, cal, excluded in runs:
        subset = {s: df for s, df in data.items() if s not in excluded}
        portfolio.append(_variant_row(name, Backtester(cfg, RiskConfig(), bt_cfg, exec_cfg, cal).run(subset, start)))

    # Paired exits on identical v1 entries: differences per signal cancel most of the noise.
    v1 = replace(base, **V1_SPEC)
    outcomes = {name: Backtester(replace(v1, **overrides), RiskConfig(), bt_cfg, exec_cfg, calendar)
                .signal_outcomes(data, start) for name, overrides in EXIT_VARIANTS.items()}
    baseline = outcomes["ema_cross (v1)"]
    exits = []
    for name, out in outcomes.items():
        paired = out.merge(baseline[["symbol", "entry_time", "r_multiple"]], on=["symbol", "entry_time"],
                           suffixes=("", "_v1"), validate="one_to_one")
        if len(paired) != len(baseline):
            raise RuntimeError(f"paired exit study: '{name}' saw different entry signals than the baseline")
        delta = (paired["r_multiple"] - paired["r_multiple_v1"]).to_numpy()
        clusters = week_clusters(paired["entry_time"]) if len(paired) else None
        lo, hi = bootstrap_ci(delta, clusters=clusters) if name != "ema_cross (v1)" else (float("nan"),) * 2
        mix = out["exit_reason"].str.replace("_GAP", "").value_counts(normalize=True)
        exits.append({"exit": name, "signals": len(out), "expectancy_r": out["r_multiple"].mean(),
                      "delta_r": delta.mean() if name != "ema_cross (v1)" else float("nan"), "delta_lo": lo,
                      "delta_hi": hi, "nights": out["nights"].mean(),
                      "mix": ", ".join(f"{k} {v:.0%}" for k, v in mix.head(4).items())})

    # Entry rules vs the v1 signals, all under the v2 default exit. A pure filter only removes
    # signals; a replacement rule (e.g. the anchor check) can also add some, reported separately.
    def outcomes_for(overrides: dict) -> pd.DataFrame:
        cfg = replace(v1, exit_mode=base.exit_mode, **overrides)
        return Backtester(cfg, RiskConfig(), bt_cfg, exec_cfg, calendar).signal_outcomes(data, start)

    reference = outcomes_for({})
    key = ["symbol", "entry_time"]
    filters = []
    for name, overrides in FILTER_VARIANTS.items():
        variant = outcomes_for(overrides)
        merged = reference.merge(variant[key].assign(_in=True), on=key, how="left")
        kept_mask = merged["_in"].notna().to_numpy()
        added = variant.merge(reference[key].assign(_in=True), on=key, how="left")
        added = added[added["_in"].isna()]
        r_all = reference["r_multiple"].to_numpy(dtype=float)
        kept_r, removed_r = r_all[kept_mask], r_all[~kept_mask]
        k = len(kept_r)
        pure = len(added) == 0
        if pure and 0 < k < len(r_all):
            percentile = cluster_random_drop_percentile(r_all, week_clusters(reference["entry_time"]),
                                                        kept_r.mean(), k)
        else:
            percentile = float("nan")
        filters.append({"filter": name, "kept": k, "kept_r": kept_r.mean() if k else float("nan"),
                        "removed": int((~kept_mask).sum()),
                        "removed_r": removed_r.mean() if len(removed_r) else float("nan"),
                        "added": len(added), "added_r": added["r_multiple"].mean() if len(added) else float("nan"),
                        "variant_r": variant["r_multiple"].mean() if len(variant) else float("nan"),
                        "random_drop_pct": percentile, "pure_filter": pure})
    return {"portfolio": pd.DataFrame(portfolio), "exits": pd.DataFrame(exits), "filters": pd.DataFrame(filters)}


def format_comparison(tables: dict[str, pd.DataFrame], prior_runs: int = 0) -> str:
    width = 118
    lines = ["=" * width, " VARIANT COMPARISON (identical bars and costs; CIs are robust to same-week clustering)", "=" * width]
    if prior_runs:
        lines += [f" WARNING: this window was already evaluated in {prior_runs} earlier run(s) (see the ledger).",
                  " Every look spends it: it is no longer clean out-of-sample data for choosing a rule.", "-" * width]
    lines.append(f" {'portfolio variant':<33} {'trades':>6} {'/yr':>5} {'exp R':>7} {'95% CI':>15} {'gross R':>7} "
                 f"{'BE bp':>6} {'PF':>5} {'maxDD':>7} {'w/o best':>15} {'nights':>6}")
    for _, r in tables["portfolio"].iterrows():
        ci = f"[{_num(r['ci_lo'], '+.2f')},{_num(r['ci_hi'], '+.2f')}]"
        best = f"{_num(r['exp_without_best'], '+.2f')} ({r['best_symbol'] or '-'})"
        lines.append(f" {r['variant']:<33} {r['trades']:>6} {r['per_year']:>5.0f} {_num(r['expectancy_r'], '+.3f'):>7} "
                     f"{ci:>15} {_num(r['gross_r'], '+.3f'):>7} {_num(r['breakeven_bps'], '.1f'):>6} "
                     f"{_num(r['profit_factor']):>5} {_pct(r['max_drawdown']):>7} {best:>15} {_num(r['nights'], '.1f'):>6}")
    lines += ["-" * width, " Paired exit study: the same v1 entry signals, each exit simulated on its own (delta vs v1 exit)",
              f" {'exit':<29} {'signals':>7} {'exp R':>7} {'delta R':>8} {'95% CI of delta':>17} {'nights':>6}  exit mix"]
    for _, r in tables["exits"].iterrows():
        ci = f"[{_num(r['delta_lo'], '+.3f')},{_num(r['delta_hi'], '+.3f')}]" if r["delta_r"] == r["delta_r"] else "-"
        lines.append(f" {r['exit']:<29} {r['signals']:>7} {_num(r['expectancy_r'], '+.3f'):>7} "
                     f"{_num(r['delta_r'], '+.3f'):>8} {ci:>17} {_num(r['nights'], '.1f'):>6}  {r['mix']}")
    lines += ["-" * width, " Entry rules vs the v1 signals (R under the v2 default exit; 'added' = signals v1 did not have)",
              f" {'rule':<20} {'kept':>5} {'kept R':>8} {'removed':>8} {'removed R':>10} {'added':>6} {'added R':>8} "
              f"{'rule R':>7} {'vs random drop':>16}"]
    for _, r in tables["filters"].iterrows():
        if r["random_drop_pct"] == r["random_drop_pct"]:
            pct = f"{r['random_drop_pct']:.0%} pctile"
        else:
            pct = "n/a (not a pure filter)" if not r["pure_filter"] else "-"
        lines.append(f" {r['filter']:<20} {r['kept']:>5} {_num(r['kept_r'], '+.3f'):>8} {r['removed']:>8} "
                     f"{_num(r['removed_r'], '+.3f'):>10} {r['added']:>6} {_num(r['added_r'], '+.3f'):>8} "
                     f"{_num(r['variant_r'], '+.3f'):>7} {pct:>16}")
    lines += ["-" * width,
              " How to read this:",
              "  * Every v2 change was proposed after looking at this window, so gains here are in-sample. Confirm on",
              "    a period nobody has looked at (e.g. --start 2019-01-01 --end 2023-12-31) before relying on one.",
              "  * A filter is only evidence if 'vs random drop' is above ~95% (randomly dropping whole weeks of",
              "    signals rarely does as well) and 'w/o best' stays >= 0 (it is not one stock's rally). Rules that",
              "    also add signals (anchor) cannot be judged that way: compare 'rule R' with v1 and check folds.",
              "  * 'BE bp' is the per-side cost (first-order, about +/-10%) at which gross expectancy is used up.",
              "    Blank = no gross edge at all.",
              "  * v1 rows have no earnings blackout (each isolates one change); 'v1 + earnings blackout' isolates it.",
              "  * Fewer than ~30 trades per cell means the CI is wider than any plausible edge.",
              "=" * width]
    return "\n".join(lines)


def append_ledger(path: str, window: tuple, symbols: list[str], rows: list[dict]) -> int:
    """Append runs to the trials ledger; return how many earlier runs overlapped this window."""
    start, end = (pd.Timestamp(w).tz_convert(ET) if w is not None else None for w in window)
    prior = 0
    if os.path.exists(path):
        ledger = pd.read_csv(path)
        if len(ledger):
            s0 = pd.to_datetime(ledger["window_start"], utc=True, format="ISO8601")
            e0 = pd.to_datetime(ledger["window_end"], utc=True, format="ISO8601")
            overlap = (s0 <= (end if end is not None else s0.max())) & (e0 >= (start if start is not None else e0.min()))
            prior = int(ledger.loc[overlap, "run_at"].nunique())
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    out = pd.DataFrame([{"run_at": stamp, "window_start": start.isoformat() if start is not None else "",
                         "window_end": end.isoformat() if end is not None else "", "symbols": " ".join(symbols),
                         **row} for row in rows])
    out.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return prior


# ----------------------------------------------------------------------------- bar files
BAR_FILE_COLUMNS = ["open", "high", "low", "close", "volume", "bar_end", "n_source", "n_expected"]


def _is_single_file(path: str) -> bool:
    return path.lower().endswith((".csv", ".csv.gz", ".gz"))


def save_bars(data: dict[str, pd.DataFrame], path: str) -> str:
    """Write session bars as one long CSV (path ends in .csv / .csv.gz, gzip is ~4x smaller)
    or as <path>/<SYMBOL>.csv files. Returns a short description of what was written."""
    if _is_single_file(path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        frames = [df.reindex(columns=[c for c in BAR_FILE_COLUMNS if c in df]).assign(symbol=symbol)
                  for symbol, df in data.items()]
        combined = pd.concat(frames).rename_axis("timestamp").reset_index()
        combined = combined[["symbol", "timestamp"] + [c for c in BAR_FILE_COLUMNS if c in combined]]
        combined.to_csv(path, index=False)
        return f"{len(combined):,} bars for {len(data)} symbols -> {path}"
    os.makedirs(path, exist_ok=True)
    for symbol, df in data.items():
        df.to_csv(os.path.join(path, f"{symbol}.csv"))
    return f"{sum(len(df) for df in data.values()):,} bars -> {path}/<SYMBOL>.csv"


def _normalise_bars(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index, utc=True, format="ISO8601").tz_convert(ET)
    df.index.name = "timestamp"
    if "bar_end" in df:
        df["bar_end"] = pd.to_datetime(df["bar_end"], utc=True, format="ISO8601").dt.tz_convert(ET)
    return df.sort_index()


def load_bars(path: str, symbols: Optional[list[str]] = None) -> dict[str, pd.DataFrame]:
    """Inverse of save_bars. With `symbols`, every requested symbol must be present."""
    if _is_single_file(path):
        combined = pd.read_csv(path, dtype={"symbol": str})
        found = {sym: _normalise_bars(grp.drop(columns="symbol").set_index("timestamp"))
                 for sym, grp in combined.groupby("symbol")}
    else:
        files = sorted(f for f in os.listdir(path) if f.endswith(".csv"))
        found = {f[:-4]: _normalise_bars(pd.read_csv(os.path.join(path, f), index_col=0)) for f in files}
    if symbols is None:
        return found
    missing = [s for s in symbols if s not in found]
    if missing:
        raise FileNotFoundError(f"{path} has no bars for {', '.join(missing)} (found: {', '.join(found) or 'none'})")
    return {s: found[s] for s in symbols}


# ----------------------------------------------------------------------------- CLI
def _day(value: Optional[str], end_of_day: bool = False) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.Timestamp(value)
    ts = ts.tz_localize(ET) if ts.tzinfo is None else ts.tz_convert(ET)
    return ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) if end_of_day else ts


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", default=config.SYMBOLS)
    parser.add_argument("--years", type=float, default=BacktestConfig.years,
                        help="evaluation length when --start is not given")
    parser.add_argument("--start", help="first evaluation day, YYYY-MM-DD (earlier bars only warm up indicators)")
    parser.add_argument("--end", help="last evaluation day, YYYY-MM-DD (default: today / last bar on file)")
    parser.add_argument("--feed", default=None, help="iex (default, matches live) or sip")
    parser.add_argument("--equity", type=float, default=BacktestConfig.initial_equity)
    parser.add_argument("--slippage-bps", type=float, default=None,
                        help="per-side cost on market/stop fills (default 5 = config.SLIPPAGE_PER_SIDE); "
                             "rerun with 2 and 5 to see what depends on the cost assumption")
    parser.add_argument("--trades-out", help="write the trade list to this CSV")
    parser.add_argument("--save-bars", metavar="PATH",
                        help="save the hourly bars: bars.csv.gz (one file) or a directory of <SYMBOL>.csv")
    parser.add_argument("--from-bars", "--load-bars", dest="from_bars", metavar="PATH",
                        help="backtest saved bars from --save-bars instead of calling the API")
    parser.add_argument("--earnings", default=config.EARNINGS_CALENDAR_PATH, metavar="CSV",
                        help="earnings calendar (symbol,date,timing) for the blackout; see src/events.py")
    parser.add_argument("--compare", action="store_true",
                        help="v1 vs each v2 change vs v2 defaults, a paired exit study and a filter audit")
    parser.add_argument("--ledger", default=config.BACKTEST_LEDGER_PATH,
                        help="CSV log of every run, used to warn when a window has been looked at before")
    args = parser.parse_args(argv)

    bt_cfg = BacktestConfig(years=args.years, initial_equity=args.equity)
    strategy_cfg = StrategyConfig()
    if args.slippage_bps is not None:  # one cost assumption: fills, cost floor and breakeven move together
        bt_cfg = replace(bt_cfg, slippage_pct=args.slippage_bps / 1e4)
        strategy_cfg = replace(strategy_cfg, round_trip_cost=2 * args.slippage_bps / 1e4)
    exec_cfg = ExecutionConfig(data_feed=args.feed) if args.feed else ExecutionConfig()
    start, end = _day(args.start), _day(args.end, end_of_day=True)
    span = pd.Timedelta(days=int(365.25 * bt_cfg.years))

    if args.from_bars:
        try:
            data = load_bars(args.from_bars, args.symbols)
        except (FileNotFoundError, KeyError) as exc:
            print(exc)
            return 1
    else:
        from src.alpaca_client import AlpacaClient

        try:
            key, secret, paper = config.get_credentials()
        except RuntimeError as exc:
            print(exc)
            return 2
        client = AlpacaClient(key, secret, paper, exec_cfg)
        latest = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)  # SIP must be >15 min old on free plans
        fetch_end = min(end, latest) if end is not None else latest
        fetch_start = (start if start is not None else fetch_end - span) - pd.Timedelta(days=bt_cfg.warmup_days)
        print(f"Fetching {exec_cfg.data_feed.upper()} 30-min bars {fetch_start:%Y-%m-%d} -> {fetch_end:%Y-%m-%d} "
              f"for {', '.join(args.symbols)} ...")
        data = client.fetch_session_bars(args.symbols, fetch_start.to_pydatetime(), fetch_end.to_pydatetime())
    if args.save_bars:
        print(f"Saved {save_bars(data, args.save_bars)}")

    if end is not None:
        data = {s: df.loc[df.index <= end] for s, df in data.items()}
    empty = [s for s, df in data.items() if df.empty]
    if empty:
        print(f"No bars for: {', '.join(empty)}")
        data = {s: df for s, df in data.items() if not df.empty}
        if not data:
            return 1

    # Everything before `start` only warms up indicators; no trades or equity are recorded there.
    if start is None:
        start = max(df.index.max() for df in data.values()) - span
    calendar = load_earnings_calendar(args.earnings)
    last_bar = max(df.index.max() for df in data.values())
    thin = {s: events_per_year(s, calendar, start, last_bar) for s in data if s not in config.ETF_SYMBOLS}
    thin = {s: n for s, n in thin.items() if n < 3.5}
    if thin:
        print("!" * 78)
        print(f" EARNINGS BLACKOUT INCOMPLETE ({args.earnings}): " +
              ", ".join(f"{s} {n:.1f}/yr" for s, n in thin.items()) + " (expected ~4/yr).")
        print(" These names can be held through earnings gaps in this run; results understate gap risk.")
        print("!" * 78)
    window = (start, max(df.index.max() for df in data.values()))
    if args.compare:
        tables = compare_variants(data, start, bt_cfg, exec_cfg, calendar, strategy_cfg)
        prior = append_ledger(args.ledger, window, list(data), [
            {"variant": r["variant"], "trades": r["trades"], "expectancy_r": r["expectancy_r"]}
            for _, r in tables["portfolio"].iterrows()])
        print(format_comparison(tables, prior))
        return 0
    result = Backtester(strategy_cfg, RiskConfig(), bt_cfg, exec_cfg, calendar).run(data, start=start)
    append_ledger(args.ledger, window, list(data), [{"variant": "v2 (defaults)", "trades": result.metrics["n_trades"],
                                                    "expectancy_r": result.metrics["expectancy_r"]}])
    print(format_tear_sheet(result))
    if args.trades_out:
        result.trades.to_csv(args.trades_out, index=False)
        print(f"Trades written to {args.trades_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
