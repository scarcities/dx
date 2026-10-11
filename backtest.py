"""Portfolio-level, event-driven walk-forward backtester.

Uses the exact strategy, sizing and kill-switch code the live bot uses:
  - decisions come from `generate_signals` (shift(1): bar t only sees bars <= t-1)
  - entries/exits fill at the open of bar t, only when bar t starts inside 09:45-15:45 ET
    (the live bot acts right after the previous bar closes, inside the same window)
  - bracket stop/target are computed from Close[t-1] and ATR[t-1], like the live order
  - 0.05% slippage is charged on every fill, entries and exits alike
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
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config
from config import BacktestConfig, ExecutionConfig, RiskConfig, StrategyConfig
from src.risk_manager import RiskManager, r_multiple
from src.sessions import ET, in_trading_window
from src.strategy import generate_signals

TRADING_DAYS = 252


@dataclass
class Position:
    symbol: str
    qty: int
    entry_time: pd.Timestamp
    entry_raw: float
    entry_fill: float
    entry_ref: float
    stop: float
    take_profit: float
    planned_risk: float
    risk_pct: float
    notional_pct: float
    binding_constraint: str
    exit_armed: bool
    bars_held: int = 0


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
                 bt_cfg: BacktestConfig = BacktestConfig(), exec_cfg: ExecutionConfig = ExecutionConfig()):
        self.scfg, self.rcfg, self.bcfg, self.ecfg = strategy_cfg, risk_cfg, bt_cfg, exec_cfg

    def _fill(self, price: float, side: str) -> float:
        slip = self.bcfg.slippage_pct
        return price * (1.0 + slip) if side == "buy" else price * (1.0 - slip)

    def run(self, data: dict[str, pd.DataFrame], start: Optional[pd.Timestamp] = None) -> BacktestResult:
        """`data`: symbol -> session-aligned hourly OHLCV (tz-aware index = bar start).
        Bars before `start` only warm up indicators; no trades and no equity are recorded there."""
        risk = RiskManager(self.rcfg, self.scfg)
        frames = {}
        for symbol, df in data.items():
            df = df.sort_index()
            sig = generate_signals(df[["open", "high", "low", "close"]], self.scfg).add_prefix("sig_")
            frames[symbol] = df[["open", "high", "low", "close"]].join(sig)
        timeline = sorted(set().union(*[f.index for f in frames.values()]))
        if start is not None:
            start = pd.Timestamp(start)
        cash = self.bcfg.initial_equity
        positions: dict[str, Position] = {}
        last_close: dict[str, float] = {}
        trades, events, curve, reasons = [], [], [], []
        equity = cash
        day_start_equity, current_day = equity, None

        def close_trade(pos: Position, ts, raw_price: float, reason: str) -> None:
            nonlocal cash
            fill = self._fill(raw_price, "sell")
            cash += pos.qty * fill
            pnl = pos.qty * (fill - pos.entry_fill)
            trades.append({
                "symbol": pos.symbol, "entry_time": pos.entry_time, "exit_time": ts, "qty": pos.qty,
                "entry_ref": pos.entry_ref, "entry_raw": pos.entry_raw, "entry_fill": pos.entry_fill,
                "stop": pos.stop, "take_profit": pos.take_profit, "exit_raw": raw_price, "exit_fill": fill,
                "exit_reason": reason, "r_multiple": r_multiple(pos.entry_fill, fill, pos.stop, pos.planned_risk),
                "pnl": pnl, "bars_held": pos.bars_held, "risk_pct": pos.risk_pct,
                "notional_pct": pos.notional_pct, "binding_constraint": pos.binding_constraint,
                "slippage_cost": pos.qty * (pos.entry_fill - pos.entry_raw + raw_price - fill),
            })
            del positions[pos.symbol]

        for ts in timeline:
            active = start is None or ts >= start
            day = ts.tz_convert(ET).date()
            if day != current_day:
                current_day, day_start_equity = day, equity
            window_ok = in_trading_window(ts, self.ecfg.trade_window_start, self.ecfg.trade_window_end)

            for symbol, frame in frames.items():
                if ts not in frame.index:
                    continue
                row = frame.loc[ts]
                pos = positions.get(symbol)

                # 1) Decisions made at the open of bar t from bar t-1 information.
                if pos is not None:
                    pos.exit_armed |= bool(row.sig_above_fast) or not self.scfg.ema_exit_requires_arming
                    if window_ok and bool(row.sig_exit) and pos.exit_armed:
                        close_trade(pos, ts, row.open, "EMA_EXIT")
                        pos = None
                elif active:
                    reasons.append(row.sig_reason_code)
                    if (window_ok and bool(row.sig_entry) and not risk.is_halted(day)
                            and len(positions) < self.rcfg.max_open_positions):
                        size = risk.size_position(equity, float(row.sig_close), float(row.sig_atr), buying_power=cash)
                        if size.shares >= 1:
                            fill = self._fill(row.open, "buy")
                            cash -= size.shares * fill
                            pos = positions[symbol] = Position(
                                symbol, size.shares, ts, float(row.open), fill, size.entry_ref, size.stop_price,
                                size.take_profit, size.entry_ref - size.stop_price, size.risk_pct,
                                size.notional_pct, size.binding_constraint,
                                exit_armed=bool(row.sig_above_fast) or not self.scfg.ema_exit_requires_arming)

                # 2) Bracket legs during bar t. Gaps fill at the open; stop wins ties.
                if pos is not None:
                    if row.open <= pos.stop:
                        close_trade(pos, ts, row.open, "STOP_LOSS_GAP")
                    elif row.open >= pos.take_profit:
                        close_trade(pos, ts, row.open, "TAKE_PROFIT_GAP")
                    elif row.low <= pos.stop:
                        close_trade(pos, ts, pos.stop, "STOP_LOSS")
                    elif row.high >= pos.take_profit:
                        close_trade(pos, ts, pos.take_profit, "TAKE_PROFIT")
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

    # ------------------------------------------------------------------------- report
    def _summarise(self, trades: pd.DataFrame, equity: pd.Series, reasons: pd.Series, events: list,
                   symbols: list[str]) -> BacktestResult:
        if trades.empty:
            trades = pd.DataFrame(columns=["symbol", "entry_time", "exit_time", "r_multiple", "pnl", "exit_reason",
                                           "bars_held", "risk_pct", "notional_pct", "binding_constraint",
                                           "slippage_cost"])
        stats = trade_stats(trades["r_multiple"], trades["pnl"])
        metrics = {**equity_stats(equity), **stats}
        metrics["initial_equity"] = float(self.bcfg.initial_equity)
        metrics["final_equity"] = float(equity.iloc[-1]) if len(equity) else float(self.bcfg.initial_equity)
        metrics["avg_bars_held"] = float(trades["bars_held"].mean()) if len(trades) else float("nan")
        metrics["avg_risk_pct"] = float(trades["risk_pct"].mean()) if len(trades) else float("nan")
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
                         f"{_num(f['profit_factor']):>6} {_num(f['expectancy_r'], '+.3f'):>7}")
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


# ----------------------------------------------------------------------------- CLI
def load_bars_csv(directory: str, symbols: list[str]) -> dict[str, pd.DataFrame]:
    data = {}
    for symbol in symbols:
        df = pd.read_csv(os.path.join(directory, f"{symbol}.csv"), index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
        data[symbol] = df
    return data


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", default=config.SYMBOLS)
    parser.add_argument("--years", type=float, default=BacktestConfig.years)
    parser.add_argument("--feed", default=None, help="iex (default, matches live) or sip")
    parser.add_argument("--equity", type=float, default=BacktestConfig.initial_equity)
    parser.add_argument("--trades-out", help="write the trade list to this CSV")
    parser.add_argument("--save-bars", help="directory to save the fetched hourly bars as CSV")
    parser.add_argument("--load-bars", help="directory of <SYMBOL>.csv hourly bars (skips the API)")
    args = parser.parse_args(argv)

    bt_cfg = BacktestConfig(years=args.years, initial_equity=args.equity)
    exec_cfg = ExecutionConfig(data_feed=args.feed) if args.feed else ExecutionConfig()
    span = pd.Timedelta(days=int(365.25 * bt_cfg.years))

    if args.load_bars:
        data = load_bars_csv(args.load_bars, args.symbols)
    else:
        from src.alpaca_client import AlpacaClient

        try:
            key, secret, paper = config.get_credentials()
        except RuntimeError as exc:
            print(exc)
            return 2
        client = AlpacaClient(key, secret, paper, exec_cfg)
        end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)  # SIP must be >15 min old on free plans
        fetch_start = end - span - pd.Timedelta(days=bt_cfg.warmup_days)
        print(f"Fetching {exec_cfg.data_feed.upper()} 30-min bars {fetch_start:%Y-%m-%d} -> {end:%Y-%m-%d} "
              f"for {', '.join(args.symbols)} ...")
        data = client.fetch_session_bars(args.symbols, fetch_start.to_pydatetime(), end.to_pydatetime())
    if args.save_bars:
        os.makedirs(args.save_bars, exist_ok=True)
        for symbol, df in data.items():
            df.to_csv(os.path.join(args.save_bars, f"{symbol}.csv"))

    empty = [s for s, df in data.items() if df.empty]
    if empty:
        print(f"No bars for: {', '.join(empty)}")
        data = {s: df for s, df in data.items() if not df.empty}
        if not data:
            return 1

    # The evaluation window ends at the last bar; everything before `start` only warms up indicators.
    start = max(df.index.max() for df in data.values()) - span
    result = Backtester(StrategyConfig(), RiskConfig(), bt_cfg, exec_cfg).run(data, start=start)
    print(format_tear_sheet(result))
    if args.trades_out:
        result.trades.to_csv(args.trades_out, index=False)
        print(f"Trades written to {args.trades_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
