"""Live paper-trading loop for Alpaca (run locally, e.g. via ./start_mac.sh).

Every heartbeat (30s while the market is open) the bot:
  1. checks the 3% daily drawdown kill-switch against Alpaca's previous-close equity
  2. reconciles fills: late entry fills, bracket stop/target exits, unknown positions
  3. after each session-aligned hourly bar closes (10:30, 11:30, ... ET) and only inside
     09:45-15:45 ET, evaluates every symbol once and journals the decision with its reason

The broker is the source of truth. Order ids are deterministic per (symbol, bar), so a
restart or a retried request can never double an entry.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import logging.handlers
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

import config
from config import ExecutionConfig, RiskConfig, StrategyConfig
from src import journal as J
from src.events import EarningsCalendar, entry_blackout, forced_exit, has_upcoming_event, load_earnings_calendar
from src.journal import TradeJournal, slippage_bps
from src.risk_manager import KillSwitchStatus, RiskManager, client_order_id, r_multiple
from src.sessions import in_trading_window, latest_bar_end, to_et
from src.strategy import (EXIT, LONG, R_CALENDAR, R_EARNINGS, R_EXIT, R_TIME, R_TRAIL, WAIT, decide_latest, describe,
                          ratchet_stop, resolve_state, stop_too_close, trailing_exit_armed)

STRATEGY_TAG = "tpb2"  # v2 rules; keeps client_order_ids and journals apart from v1
EXIT_LABELS = {R_EXIT: "EMA_EXIT", R_EARNINGS: "EARNINGS_EXIT", R_TRAIL: "TRAIL_STOP", R_TIME: "TIME_EXIT"}
OPEN_STATUSES = {"new", "held", "accepted", "pending_new", "accepted_for_bidding", "partially_filled"}
MAX_RATCHET_ATTEMPTS = 3          # per bar, one per heartbeat
UNPROTECTED_GRACE_SECONDS = 60    # a position with no working stop for this long gets a new one
EXIT_RETRY_MAX_BACKOFF = 600      # failed exits are retried after 30s, 60s, 120s ... capped at 10 min
FILLED = "filled"
DEAD_ORDER = {"canceled", "expired", "rejected", "done_for_day", "stopped", "suspended"}

log = logging.getLogger("bot")


def _status(order) -> str:
    return str(getattr(order.status, "value", order.status)).lower()


def _kind(order) -> str:
    kind = getattr(order, "order_type", None) or getattr(order, "type", None)
    return str(getattr(kind, "value", kind)).lower()


def _float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


@dataclass
class OpenTrade:
    symbol: str
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    stop_price: Optional[float]           # the stop currently working at the broker
    take_profit: Optional[float]
    ref_bar: Optional[pd.Timestamp]       # signal bar (t-1 at entry); later bars feed the ratchet
    planned_risk: Optional[float] = None
    client_order_id: str = ""
    initial_stop: Optional[float] = None
    entry_ref: Optional[float] = None     # Close[t-1] at the signal; the ratchet's R is measured from it

    @property
    def trailed(self) -> bool:
        return (self.stop_price is not None and self.initial_stop is not None
                and self.stop_price > self.initial_stop + 1e-9)


class TradingBot:
    def __init__(self, client, journal: TradeJournal, symbols: list[str],
                 strategy_cfg: StrategyConfig = StrategyConfig(), risk_cfg: RiskConfig = RiskConfig(),
                 exec_cfg: ExecutionConfig = ExecutionConfig(), dry_run: bool = False,
                 now_fn: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
                 sleep: Callable[[float], None] = time.sleep, calendar: Optional[EarningsCalendar] = None):
        self.client = client
        self.journal = journal
        self.symbols = list(symbols)
        self.scfg, self.rcfg, self.ecfg = strategy_cfg, risk_cfg, exec_cfg
        self.risk = RiskManager(risk_cfg, strategy_cfg)
        self.dry_run = dry_run
        self.calendar = calendar or {}
        self.now_fn = now_fn
        self.sleep = sleep
        self.trades: dict[str, OpenTrade] = {}
        self.pending: dict[str, dict] = {}
        self.exit_pending: dict[str, dict] = {}       # exits that did not complete; retried every heartbeat
        self.desired_stop: dict[str, dict] = {}       # ratchets that did not complete; retried every heartbeat
        self.unprotected_since: dict[str, pd.Timestamp] = {}
        self._sessions: tuple = (None, None, None)    # (ET date fetched, session dates, last decision times)
        self.submitted_ids: set[str] = set()
        self.last_decision_bar: dict[str, pd.Timestamp] = {}
        self.last_processed_end: Optional[pd.Timestamp] = None
        self.kill_switch_flattening = False
        self.last_clock = None
        self.running = True

    # ------------------------------------------------------------------ startup
    def bootstrap(self) -> None:
        now = self.now_fn()
        journaled = self.journal.open_entries()
        positions = self.client.get_positions()
        for symbol in self.symbols:
            entry = journaled.get(symbol)
            pos = positions.get(symbol)
            if entry is None and pos is None:
                continue
            stop, target = self.client.bracket_levels(symbol) if pos is not None else (None, None)
            if entry is not None:
                ref, j_stop = _float(entry.get("ref_price")), _float(entry.get("stop_price"))
                ref_bar = entry.get("ref_bar_et")
                last_stop = _float(entry.get("last_journaled_stop")) or j_stop   # ratchets / re-protections
                trade = OpenTrade(symbol, _float(entry.get("qty")) or 0.0, _float(entry.get("fill_price")) or 0.0,
                                  pd.Timestamp(entry["ts_utc"]), stop if stop is not None else last_stop,
                                  _float(entry.get("take_profit")), ref_bar if pd.notna(ref_bar) else None,
                                  ref - j_stop if ref is not None and j_stop is not None else None,
                                  str(entry.get("client_order_id") or ""), j_stop, ref)
            else:
                trade = OpenTrade(symbol, float(pos.qty), float(pos.avg_entry_price), now, stop, target, None,
                                  initial_stop=stop, entry_ref=float(pos.avg_entry_price))
                log.warning("Adopting %s position not found in the journal (qty %s)", symbol, pos.qty)
            self.trades[symbol] = trade
            if pos is not None and stop is None:
                self.journal.log(J.ERROR, now=now, symbol=symbol, message="Open position has no working stop order")
        if self.journal.kill_switch_tripped_on(to_et(now).date()):
            self.risk.latch_halt(to_et(now).date())          # a restart must not re-open trading today
            self.kill_switch_flattening = True
            log.warning("Kill-switch already tripped today (journal): no new entries until tomorrow")
        for symbol in self.symbols:
            if symbol not in config.ETF_SYMBOLS and not has_upcoming_event(symbol, now, self.calendar):
                log.warning("%s: no earnings date in the next 100 days in the calendar%s", symbol,
                            "; new entries are blocked (fail closed)" if self.rcfg.earnings_fail_closed else "")
        account = self.client.get_account()
        self.journal.log(J.STARTUP, now=now, equity=float(account.equity),
                         day_start_equity=float(account.last_equity),
                         message=f"symbols={','.join(self.symbols)} paper={getattr(self.client, 'paper', True)} "
                                 f"dry_run={self.dry_run} tracked={','.join(self.trades) or '-'}")
        log.info("Bootstrap complete; tracking %s", list(self.trades) or "no positions")

    # ------------------------------------------------------------------ main step
    def run_once(self, now: Optional[pd.Timestamp] = None) -> str:
        now = pd.Timestamp(now) if now is not None else self.now_fn()
        clock = self.last_clock = self.client.get_clock()
        if not clock.is_open:
            return "MARKET_CLOSED"

        account = self.client.get_account()
        equity, day_start = float(account.equity), float(account.last_equity)
        today = to_et(now).date()
        status = self.risk.check_daily_drawdown(today, day_start, equity)
        if status.newly_tripped:
            self._trip_kill_switch(status, now)

        positions = self.client.get_positions()
        self._reconcile(positions, now, status)
        self._service_positions(positions, now, status)

        if not in_trading_window(now, self.ecfg.trade_window_start, self.ecfg.trade_window_end):
            return "OUTSIDE_WINDOW"
        due = latest_bar_end(now, self.ecfg.bar_settle_seconds)
        if due is None or (self.last_processed_end is not None and due <= self.last_processed_end):
            return "IDLE"
        stale = (now - due).total_seconds() > self.ecfg.max_signal_staleness_seconds

        start = now - pd.Timedelta(days=self.ecfg.live_lookback_days)
        bars = self.client.fetch_session_bars(self.symbols, start.to_pydatetime(), now.to_pydatetime(), now=now)
        positions = self.client.get_positions()
        self._reconcile(positions, now, status)  # a bracket leg may have filled since the first check
        open_order_symbols = {o.symbol for o in self.client.get_open_orders(self.symbols)}
        done = [self._evaluate_symbol(symbol, bars.get(symbol), due, now, positions, open_order_symbols,
                                      account, status, stale) for symbol in self.symbols]
        if all(done) or stale:
            self.last_processed_end = due
        return "EVALUATED" if all(done) else "WAITING_FOR_BARS"

    # ------------------------------------------------------------------ per symbol
    def _evaluate_symbol(self, symbol: str, bars: Optional[pd.DataFrame], due: pd.Timestamp, now: pd.Timestamp,
                         positions: dict, open_order_symbols: set, account, ks: KillSwitchStatus,
                         stale: bool) -> bool:
        """Returns False when the bar that just closed is not available yet (retry next heartbeat)."""
        if bars is None or bars.empty or bars["bar_end"].iloc[-1] < due:
            if stale:
                self.journal.log(J.ERROR, now=now, symbol=symbol, message=f"No closed bar ending {due} received")
                return True
            return False
        decision_bar = bars["bar_end"].iloc[-1]
        if self.last_decision_bar.get(symbol) == decision_bar:
            return True

        sig = decide_latest(bars, self.scfg)
        in_position = symbol in positions
        armed = self._exit_armed(symbol, bars)
        sessions = self._session_days(now)
        event = (forced_exit(symbol, decision_bar, self.calendar, sessions, self.scfg.earnings_exit_time,
                             self._sessions[2]) if in_position
                 else entry_blackout(symbol, decision_bar, self.calendar, sessions))
        state, reason_code = resolve_state(sig, in_position, self.scfg.exit_mode == "ema_cross" and armed,
                                           blackout=event is not None)
        detail = describe(sig, self.scfg, in_position, armed)
        if event is not None:
            detail = f"Earnings {event} inside its blackout window | {detail}"
        elif (state == LONG and self.rcfg.earnings_fail_closed and symbol not in config.ETF_SYMBOLS
              and not has_upcoming_event(symbol, decision_bar, self.calendar)):
            state, reason_code = WAIT, R_CALENDAR
            detail = f"{R_CALENDAR}: no {symbol} earnings date in the next 100 days in the calendar | {detail}"
        new_stop = None
        trade = self.trades.get(symbol)
        if in_position and state != EXIT and trade is not None and trade.ref_bar is not None:
            held = int((bars.index > pd.Timestamp(trade.ref_bar)).sum())
            if self.scfg.max_hold_bars and held >= self.scfg.max_hold_bars:
                state, reason_code = EXIT, R_TIME
                detail = f"{held} bars in the trade >= max_hold_bars {self.scfg.max_hold_bars}"
            else:
                new_stop = self._ratchet(symbol, bars, sig)
                if new_stop is not None and stop_too_close(new_stop, float(sig["close"]), self.scfg):
                    state, reason_code = EXIT, R_TRAIL
                    detail = (f"Ratcheted stop {new_stop:.2f} within {self.scfg.stop_buffer_pct:.1%} of "
                              f"Close[t-1] {sig['close']:.2f}: exit at market")
                    new_stop = None
                elif new_stop is not None:
                    detail += f"; stop {trade.stop_price:.2f} -> {new_stop:.2f}"
        account_fields = {"equity": ks.equity, "day_start_equity": ks.day_start_equity,
                          "daily_pnl_pct": ks.daily_pnl_pct}

        action, size, cid = ("HOLD" if in_position else "NONE"), None, ""
        if stale and state == LONG:          # exits and stop ratchets reduce risk, so they still run
            action = "SKIPPED_STALE"
        elif state == LONG:
            cid = client_order_id(STRATEGY_TAG, symbol, decision_bar)
            block = self.risk.entry_block_reason(symbol, to_et(now).date(), set(positions) | set(self.pending),
                                                 open_order_symbols | set(self.exit_pending), self.submitted_ids, cid)
            if block is None:
                size = self.risk.size_position(ks.equity, float(sig["close"]), float(sig["atr"]),
                                               buying_power=_float(account.buying_power))
                block = "SIZE_ZERO" if size.shares < 1 else None
            action = f"BLOCKED_{block}" if block else ("DRY_RUN_ENTER" if self.dry_run else "ENTER")
        elif state == EXIT and symbol in self.exit_pending:
            action = "EXIT_PENDING"          # already being worked by the heartbeat retry loop
        elif state == EXIT:
            action = "DRY_RUN_EXIT" if self.dry_run else "EXIT"
        elif new_stop is not None:
            action = "DRY_RUN_RATCHET_STOP" if self.dry_run else "RATCHET_STOP"

        sizing = {}
        if size is not None:
            sizing = {"qty": size.shares, "ref_price": size.entry_ref, "stop_price": size.stop_price,
                      "take_profit": size.take_profit, "risk_pct": size.risk_pct,
                      "notional_pct": size.notional_pct, "binding_constraint": size.binding_constraint}
        self.journal.log_bar_check(symbol, sig, decision_bar, state, reason_code, detail, action, in_position,
                                   now=now, client_order_id=cid, **sizing, **account_fields)
        self.last_decision_bar[symbol] = decision_bar
        log.info("%s %s %s (%s) -> %s | %s", symbol, decision_bar.strftime("%m-%d %H:%M"), state, reason_code,
                 action, detail)

        if action == "ENTER":
            self._submit_entry(symbol, sig, size, cid, now, ks)
        elif action == "EXIT":
            self._submit_exit(symbol, float(sig["close"]), now, ks, EXIT_LABELS.get(reason_code, "MARKET_EXIT"))
        elif action == "RATCHET_STOP":
            self.desired_stop[symbol] = {"stop": new_stop, "attempts": 0}
            self._try_ratchet(symbol, now)
        return True

    def _session_days(self, now: pd.Timestamp):
        """Exchange sessions around today from the broker calendar, refreshed once a day. On
        failure the event windows fall back to weekday arithmetic with an extra day of margin."""
        today = to_et(now).date()
        if self._sessions[0] != today:
            try:
                closes = self.client.get_session_closes(today - dt.timedelta(days=10), today + dt.timedelta(days=130))
                last_decision = {d: (dt.datetime.combine(d, dt.time.fromisoformat(c)) - dt.timedelta(minutes=30))
                                 .strftime("%H:%M") for d, c in closes.items()}
                self._sessions = (today, sorted(closes), last_decision)
            except Exception as exc:  # noqa: BLE001
                log.warning("Exchange calendar unavailable (%s); using weekday arithmetic", exc)
                self._sessions = (today, None, None)
        return self._sessions[1]

    def _ratchet(self, symbol: str, bars: pd.DataFrame, sig: pd.Series) -> Optional[float]:
        """New (higher) stop for an open position, or None. Same inputs as the backtester: closes
        of the bars after the signal bar, ATR[t-1], the entry fill, and the stop working at the
        broker (re-read every bar, so a restart or a half-failed replace heals itself). Adopted
        positions of unknown provenance are never ratcheted."""
        trade = self.trades.get(symbol)
        if self.scfg.exit_mode != "ratchet" or trade is None or trade.ref_bar is None:
            return None
        working_stop, _ = self.client.bracket_levels(symbol)
        if working_stop is not None:
            trade.stop_price = working_stop
        if None in (trade.stop_price, trade.initial_stop, trade.entry_ref):
            return None
        closes = bars.loc[bars.index > pd.Timestamp(trade.ref_bar), "close"]
        highest = float(closes.max()) if len(closes) else float("-inf")
        new_stop = ratchet_stop(trade.stop_price, trade.entry_price, trade.entry_ref, trade.initial_stop, highest,
                                float(sig["atr"]), self.scfg)
        return new_stop if new_stop > trade.stop_price + 1e-9 else None

    def _try_ratchet(self, symbol: str, now: pd.Timestamp) -> None:
        """One replace attempt for the desired stop. The PATCH is not retried blindly (it is not
        idempotent); after an error the broker's stop is re-read and treated as the truth."""
        want = self.desired_stop.get(symbol)
        trade = self.trades.get(symbol)
        if want is None or trade is None:
            self.desired_stop.pop(symbol, None)
            return
        want["attempts"] += 1
        old, new_stop = trade.stop_price, want["stop"]
        try:
            order = self.client.replace_stop(symbol, new_stop)
            confirmed = True
        except Exception as exc:  # noqa: BLE001 - the old stop stays in force, so this fails safe
            order, confirmed = None, False
            working, _ = self.client.bracket_levels(symbol)
            if working is not None and working >= new_stop - 1e-9:
                confirmed = True                     # the replace went through despite the error
            else:
                give_up = want["attempts"] >= MAX_RATCHET_ATTEMPTS
                self.journal.log(J.ERROR, now=now, symbol=symbol, stop_price=working if working is not None else old,
                                 message=(f"Stop ratchet {old} -> {new_stop} failed (attempt {want['attempts']}"
                                          f"{', giving up for this bar' if give_up else ''}); old stop kept: {exc}")[:500])
                log.error("Stop ratchet %s %s -> %s failed: %s", symbol, old, new_stop, exc)
                if give_up:
                    self.desired_stop.pop(symbol, None)
                return
        if confirmed:
            trade.stop_price = new_stop
            self.desired_stop.pop(symbol, None)
            self.journal.log(J.STOP_RATCHET, now=now, symbol=symbol, stop_price=new_stop,
                             entry_price=trade.entry_price, order_id=str(getattr(order, "id", "")),
                             message=f"stop {old:.2f} -> {new_stop:.2f}")
            log.info("STOP RATCHET %s %.2f -> %.2f", symbol, old, new_stop)

    def _exit_armed(self, symbol: str, bars: pd.DataFrame) -> bool:
        trade = self.trades.get(symbol)
        if trade is None or trade.ref_bar is None:
            return True
        return trailing_exit_armed(bars, pd.Timestamp(trade.ref_bar), self.scfg)

    # ------------------------------------------------------------------ execution
    def _submit_entry(self, symbol: str, sig: pd.Series, size, cid: str, now: pd.Timestamp,
                      ks: KillSwitchStatus) -> None:
        started = time.monotonic()
        order_fields = {"symbol": symbol, "side": "buy", "qty": size.shares, "ref_price": size.entry_ref,
                        "stop_price": size.stop_price, "take_profit": size.take_profit, "risk_pct": size.risk_pct,
                        "notional_pct": size.notional_pct, "binding_constraint": size.binding_constraint,
                        "client_order_id": cid, "equity": ks.equity}
        try:
            order = self.client.submit_bracket_order(symbol, size.shares, size.stop_price, size.take_profit, cid)
        except Exception as exc:  # noqa: BLE001 - journal every rejection
            self.journal.log(J.ORDER_REJECTED, now=now, message=str(exc)[:500], **order_fields)
            log.error("Entry %s rejected: %s", cid, exc)
            return
        self.submitted_ids.add(cid)
        self.journal.log(J.ORDER_SUBMITTED, now=now, order_id=str(order.id), order_status=_status(order),
                         **order_fields)
        order = self.client.wait_for_terminal(order.id)
        latency_ms = (time.monotonic() - started) * 1000
        pending = {"cid": cid, "ref_price": size.entry_ref, "stop": size.stop_price, "target": size.take_profit,
                   "ref_bar": sig["ref_bar_time"], "planned_risk": size.entry_ref - size.stop_price,
                   "submitted": now}
        if _status(order) == FILLED:
            self._record_entry_fill(symbol, order, pending, now, ks.equity, latency_ms)
        elif _status(order) in DEAD_ORDER:
            self.journal.log(J.ORDER_REJECTED, now=now, order_id=str(order.id), order_status=_status(order),
                             latency_ms=latency_ms, message="entry order ended without a fill", **order_fields)
        else:
            self.pending[symbol] = pending
            self.journal.log(J.ORDER_PENDING, now=now, order_id=str(order.id), order_status=_status(order),
                             latency_ms=latency_ms, **order_fields)

    def _record_entry_fill(self, symbol: str, order, pending: dict, now: pd.Timestamp, equity: float,
                           latency_ms: Optional[float] = None) -> None:
        fill = float(order.filled_avg_price)
        qty = float(order.filled_qty)
        risk_pct = qty * (fill - pending["stop"]) / equity if equity else float("nan")
        self.trades[symbol] = OpenTrade(symbol, qty, fill, pd.Timestamp(order.filled_at or now), pending["stop"],
                                        pending["target"], pending["ref_bar"], pending["planned_risk"], pending["cid"],
                                        initial_stop=pending["stop"], entry_ref=pending["ref_price"])
        self.journal.log(J.ENTRY_FILL, now=now, symbol=symbol, side="buy", qty=qty, ref_price=pending["ref_price"],
                         fill_price=fill, slippage_bps=slippage_bps("buy", pending["ref_price"], fill),
                         stop_price=pending["stop"], take_profit=pending["target"], entry_price=fill,
                         risk_pct=risk_pct, notional_pct=qty * fill / equity if equity else None,
                         ref_bar_et=pending["ref_bar"], order_id=str(order.id), client_order_id=pending["cid"],
                         order_status=_status(order), latency_ms=latency_ms, equity=equity)
        log.info("ENTRY FILL %s %s @ %.2f (ref %.2f, stop %.2f, target %.2f)", symbol, qty, fill,
                 pending["ref_price"], pending["stop"], pending["target"])

    def _submit_exit(self, symbol: str, ref_price: Optional[float], now: pd.Timestamp, ks: KillSwitchStatus,
                     reason: str) -> None:
        """Market exit. If it does not complete, the symbol goes on the exit_pending list: it is
        re-protected with a stop if needed and the exit is retried on every heartbeat."""
        started = time.monotonic()
        self.exit_pending.setdefault(symbol, {"reason": reason, "ref": ref_price, "attempts": 0, "order_id": None})
        self.exit_pending[symbol]["attempts"] += 1
        self.exit_pending[symbol]["last_try"] = now
        try:
            order = self.client.close_position(symbol)
        except Exception as exc:  # noqa: BLE001
            self._exit_failed(symbol, now, f"close_position failed: {exc}")
            return
        if order is None:  # no position at the broker (a leg or an earlier sell filled): reconcile journals
            return         # it and consumes exit_pending, so the reason and reference price survive
        order = self.client.wait_for_terminal(order.id)
        status = _status(order)
        if status == FILLED:
            self.exit_pending.pop(symbol, None)
            self._record_exit_fill(symbol, order, ref_price, reason, now, ks.equity, (time.monotonic() - started) * 1000)
        elif status in DEAD_ORDER:
            self._exit_failed(symbol, now, f"close order {order.id} ended {status}")
        else:
            self.exit_pending[symbol]["order_id"] = order.id   # still working; checked every heartbeat

    def _exit_failed(self, symbol: str, now: pd.Timestamp, message: str) -> None:
        self.journal.log(J.ERROR, now=now, symbol=symbol, message=f"EXIT_FAILED: {message}; retrying"[:500])
        log.error("Exit %s failed: %s", symbol, message)
        self.exit_pending[symbol]["order_id"] = None
        self._ensure_protection(symbol, now, immediate=True)

    def _ensure_protection(self, symbol: str, now: pd.Timestamp, immediate: bool = False) -> None:
        """Every open position must have a working sell stop. If it has had none for longer than
        the grace period (or right now, after a failed exit), place one at the tracked stop."""
        trade = self.trades.get(symbol)
        if trade is None or self.dry_run:
            return
        working = self.client.get_open_orders([symbol])
        if any(_kind(o) == "market" and str(getattr(o.side, "value", o.side)).lower() == "sell" for o in working):
            return  # a liquidation is in flight; re-protecting would cancel it
        stop, _ = self.client.bracket_levels(symbol)
        if stop is not None:
            self.unprotected_since.pop(symbol, None)
            return
        first_seen = self.unprotected_since.setdefault(symbol, now)
        if not immediate and (now - first_seen).total_seconds() < UNPROTECTED_GRACE_SECONDS:
            return
        level = trade.stop_price if trade.stop_price is not None else trade.initial_stop
        if level is None or self.kill_switch_flattening:
            why = "kill-switch active" if self.kill_switch_flattening else "no known stop level (adopted position)"
            self.journal.log(J.ERROR, now=now, symbol=symbol, message=f"UNPROTECTED: {why}; closing at market")
            self.exit_pending.setdefault(symbol, {"reason": "KILL_SWITCH" if self.kill_switch_flattening
                                                  else "UNPROTECTED_EXIT", "ref": None, "attempts": 0,
                                                  "order_id": None})
            return
        # Size from the broker, never from memory: protective sells on a position that has already
        # been sold would open a short if they triggered.
        held = self.client.get_positions().get(symbol)
        qty = abs(float(held.qty)) if held is not None else 0.0
        if qty <= 0:
            self.unprotected_since.pop(symbol, None)
            return
        try:
            kind = self.client.restore_protection(symbol, qty, level, trade.take_profit)
            self.unprotected_since.pop(symbol, None)
            self.journal.log(J.PROTECTIVE_STOP, now=now, symbol=symbol, qty=qty, stop_price=level,
                             take_profit=trade.take_profit if kind == "oco" else None,
                             message=f"no working stop found; protection restored as {kind.upper()}")
        except Exception as exc:  # noqa: BLE001
            self.journal.log(J.ERROR, now=now, symbol=symbol, stop_price=level,
                             message=f"UNPROTECTED: protective stop rejected ({exc}); closing at market"[:500])
            if symbol not in self.exit_pending:
                self.exit_pending[symbol] = {"reason": "UNPROTECTED_EXIT", "ref": None, "attempts": 0,
                                             "order_id": None}

    def _service_positions(self, positions: dict, now: pd.Timestamp, ks: KillSwitchStatus) -> None:
        """Heartbeat duties: earnings exits (from the calendar alone, so a market-data outage cannot
        delay them), finish pending exits, retry pending ratchets, enforce stop coverage."""
        if self.calendar and not self.dry_run:
            sessions = self._session_days(now)
            for symbol in list(self.trades):
                if symbol in positions and symbol not in self.exit_pending:
                    event = forced_exit(symbol, now, self.calendar, sessions, self.scfg.earnings_exit_time,
                                        self._sessions[2])
                    if event is not None:
                        self.journal.log(J.BAR_CHECK, now=now, symbol=symbol, state=EXIT, reason_code=R_EARNINGS,
                                         action="EXIT", in_position=True,
                                         reason_detail=f"Earnings {event}: last hold session, exit time reached")
                        self._submit_exit(symbol, None, now, ks, "EARNINGS_EXIT")
        for symbol in list(self.exit_pending):
            if symbol not in positions:
                continue  # reconcile has journaled (or will journal) the exit fill
            pending = self.exit_pending[symbol]
            if pending["order_id"] is not None:
                order = self.client.get_order(pending["order_id"])
                if _status(order) in OPEN_STATUSES or _status(order) in (FILLED, "calculated"):
                    continue          # still working, or done and about to show up in reconcile
                pending["order_id"] = None
            wait = min(self.ecfg.heartbeat_seconds * 2 ** max(pending["attempts"] - 1, 0), EXIT_RETRY_MAX_BACKOFF)
            last_try = pending.get("last_try")
            if last_try is not None and (now - last_try).total_seconds() < wait - 1:
                continue
            if not self.dry_run:
                self._submit_exit(symbol, pending["ref"], now, ks, pending["reason"])
        for symbol in list(self.desired_stop):
            if symbol in positions and symbol not in self.exit_pending:
                self._try_ratchet(symbol, now)
            else:
                self.desired_stop.pop(symbol, None)
        for symbol in list(self.trades):
            if symbol in positions and symbol not in self.exit_pending:
                self._ensure_protection(symbol, now)

    def _forget(self, symbol: str) -> Optional[OpenTrade]:
        """A trade has left the book: drop every piece of per-symbol state with it, so nothing
        (a pending exit, a ratchet, a protection timer) can act on the next position."""
        self.exit_pending.pop(symbol, None)
        self.desired_stop.pop(symbol, None)
        self.unprotected_since.pop(symbol, None)
        return self.trades.pop(symbol, None)

    def _record_exit_fill(self, symbol: str, order, ref_price: Optional[float], reason: str, now: pd.Timestamp,
                          equity: Optional[float], latency_ms: Optional[float] = None) -> None:
        trade = self._forget(symbol)
        fill = float(order.filled_avg_price)
        qty = float(order.filled_qty)
        fields = {}
        if trade is not None:
            initial = trade.initial_stop if trade.initial_stop is not None else trade.stop_price
            r_value = r_multiple(trade.entry_price, fill, initial, trade.planned_risk) if initial is not None else None
            fields = {"entry_price": trade.entry_price, "r_multiple": r_value,
                      "pnl": qty * (fill - trade.entry_price), "stop_price": trade.stop_price,
                      "take_profit": trade.take_profit}
        self.journal.log(J.EXIT_FILL, now=now, symbol=symbol, side="sell", qty=qty, ref_price=ref_price,
                         fill_price=fill, exit_price=fill, exit_reason=reason,
                         slippage_bps=slippage_bps("sell", ref_price, fill) if ref_price else None,
                         order_id=str(order.id), order_status=_status(order), latency_ms=latency_ms,
                         equity=equity, **fields)
        log.info("EXIT FILL %s %s @ %.2f (%s) R=%s", symbol, qty, fill, reason, fields.get("r_multiple"))

    # ------------------------------------------------------------------ reconciliation
    def _reconcile(self, positions: dict, now: pd.Timestamp, ks: KillSwitchStatus) -> None:
        for symbol, pending in list(self.pending.items()):
            order = self.client.get_order_by_client_id(pending["cid"])
            if _status(order) == FILLED:
                self.pending.pop(symbol)
                self._record_entry_fill(symbol, order, pending, now, ks.equity)
            elif _status(order) in DEAD_ORDER:
                self.pending.pop(symbol)
                self.journal.log(J.ORDER_REJECTED, now=now, symbol=symbol, client_order_id=pending["cid"],
                                 order_status=_status(order), message="pending entry ended without a fill")

        for symbol, trade in list(self.trades.items()):
            if symbol in positions:
                continue
            since = (trade.entry_time - pd.Timedelta(minutes=2)).to_pydatetime()
            fill = self.client.find_latest_fill(symbol, "sell", since)
            if fill is None:
                self._forget(symbol)
                self.journal.log(J.ERROR, now=now, symbol=symbol, message="Position closed but no sell fill found")
                continue
            kind = _kind(fill)
            pending_exit = self.exit_pending.pop(symbol, None)
            self.desired_stop.pop(symbol, None)
            self.unprotected_since.pop(symbol, None)
            if self.kill_switch_flattening and kind == "market":
                reason, ref = "KILL_SWITCH", None
            elif pending_exit is not None and kind == "market":
                reason, ref = pending_exit["reason"], pending_exit["ref"]
            elif kind in ("stop", "stop_limit"):
                reason, ref = ("TRAIL_STOP" if trade.trailed else "STOP_LOSS"), _float(fill.stop_price)
            elif kind == "limit":
                reason, ref = "TAKE_PROFIT", _float(fill.limit_price)
            else:
                reason, ref = "MARKET_EXIT", None
            self._record_exit_fill(symbol, fill, ref, reason, now, ks.equity)

        for symbol, pos in positions.items():
            if symbol in self.symbols and symbol not in self.trades and symbol not in self.pending:
                stop, target = self.client.bracket_levels(symbol)
                self.trades[symbol] = OpenTrade(symbol, float(pos.qty), float(pos.avg_entry_price), now, stop,
                                                target, None)
                self.journal.log(J.ERROR, now=now, symbol=symbol, qty=float(pos.qty),
                                 entry_price=float(pos.avg_entry_price), stop_price=stop, take_profit=target,
                                 message="Untracked position adopted (manual trade or missed fill)")

    def _trip_kill_switch(self, status: KillSwitchStatus, now: pd.Timestamp) -> None:
        log.critical("KILL-SWITCH: daily P&L %.2f%% <= -%.2f%%", status.daily_pnl_pct * 100,
                     self.rcfg.max_daily_drawdown * 100)
        self.journal.log(J.KILL_SWITCH, now=now, equity=status.equity, day_start_equity=status.day_start_equity,
                         daily_pnl_pct=status.daily_pnl_pct,
                         message=f"flatten={self.rcfg.kill_switch_flatten and not self.dry_run}")
        if self.rcfg.kill_switch_flatten and not self.dry_run:
            self.kill_switch_flattening = True
            self.pending.clear()
            self.desired_stop.clear()
            try:
                self.client.flatten_all()
            except Exception as exc:  # noqa: BLE001 - the retry loop below takes over
                self.journal.log(J.ERROR, now=now, message=f"flatten_all failed: {exc}"[:500])
            # Whatever is still held is closed by the heartbeat retry loop until it is gone.
            for symbol in self.client.get_positions():
                if symbol in self.trades:
                    self.exit_pending.setdefault(symbol, {"reason": "KILL_SWITCH", "ref": None, "attempts": 1,
                                                          "order_id": None, "last_try": now})

    # ------------------------------------------------------------------ loop
    def seconds_until_next_check(self) -> float:
        clock = self.last_clock
        if clock is None or clock.is_open:
            return float(self.ecfg.heartbeat_seconds)
        until_open = (pd.Timestamp(clock.next_open) - self.now_fn()).total_seconds()
        return float(min(max(until_open - 60, self.ecfg.heartbeat_seconds), 900))

    def run_forever(self) -> None:
        self.bootstrap()
        while self.running:
            if self.kill_switch_flattening and not self.risk.is_halted(to_et(self.now_fn()).date()):
                self.kill_switch_flattening = False  # new day
            try:
                result = self.run_once()
                log.debug("heartbeat: %s", result)
            except Exception as exc:  # noqa: BLE001 - never let one bad cycle kill the process
                log.exception("Cycle failed")
                try:
                    self.journal.log(J.ERROR, message=f"{type(exc).__name__}: {exc}"[:500])
                except OSError:
                    pass
            deadline = time.monotonic() + self.seconds_until_next_check()
            while self.running and time.monotonic() < deadline:  # wakes promptly on Ctrl-C
                self.sleep(min(1.0, deadline - time.monotonic()))
        self.journal.log(J.SHUTDOWN, message="clean shutdown; open positions keep their GTC bracket legs")

    def stop(self, *_args) -> None:
        log.info("Shutdown requested")
        self.running = False


def setup_logging(level: str = "INFO") -> None:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(os.path.join(config.LOG_DIR, "bot.log"),
                                                        maxBytes=5_000_000, backupCount=5)
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers[:] = [file_handler, console]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Alpaca paper-trading bot (trend pullback, 1h)")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="evaluate and journal, but send no orders")
    parser.add_argument("--journal", default=config.JOURNAL_PATH)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    from src.alpaca_client import AlpacaClient

    try:
        key, secret, paper = config.get_credentials()
    except RuntimeError as exc:
        log.critical(str(exc))
        return 2
    if not paper:
        log.critical("ALPACA_PAPER is not True. This bot is in a paper validation phase and refuses live trading.")
        return 2
    exec_cfg = ExecutionConfig()
    calendar = load_earnings_calendar(config.EARNINGS_CALENDAR_PATH)
    single_names = [s for s in config.SYMBOLS if s not in ("SPY", "QQQ")]
    if not calendar:
        log.warning("No earnings calendar at %s: %s can be held through earnings gaps. See src/events.py.",
                    config.EARNINGS_CALENDAR_PATH, ", ".join(single_names))
    elif missing := [s for s in single_names if s not in calendar]:
        log.warning("Earnings calendar has no dates for %s", ", ".join(missing))
    client = AlpacaClient(key, secret, paper, exec_cfg)
    account = client.get_account()
    if account.trading_blocked or account.account_blocked:
        log.critical("Account is blocked for trading (status=%s)", account.status)
        return 2
    log.info("Connected to Alpaca PAPER account %s | equity $%s | data feed %s", account.account_number,
             account.equity, exec_cfg.data_feed)

    bot = TradingBot(client, TradeJournal(args.journal), config.SYMBOLS, StrategyConfig(), RiskConfig(), exec_cfg,
                     dry_run=args.dry_run, calendar=calendar)
    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    if args.once:
        bot.bootstrap()
        log.info("Cycle result: %s", bot.run_once())
        return 0
    bot.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
