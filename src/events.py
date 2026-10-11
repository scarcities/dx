"""Scheduled-event blackout (earnings).

Hourly 2-ATR stops are 0.5-2.5% wide; an earnings gap is routinely 5-10%. A stop cannot
protect a position across that gap, so the strategy must not hold one.

alpaca-py has no earnings calendar, so dates come from a CSV you maintain
(earnings_calendar.example.csv shows the format):

    symbol,date,timing
    AAPL,2026-10-29,AMC
    NVDA,2026-11-19,AMC

timing: AMC (after the close), BMO (before the open) or blank (unknown, treated as BMO for
the exit and AMC for the entry block, i.e. the safe side of both).

With the exchange's session list (Alpaca calendar live, the bar dates in a backtest):
    L    = last session a position may be held: D for AMC, the session before D otherwise
    gap  = the session whose open carries the news: the session after D for AMC, D otherwise
    exit : at `exit_time` (default 14:30 ET) on L, or at L's last decision bar if L closes early
           (half days), and at any time after that until the gap. The live bot checks this every
           heartbeat from the calendar alone, so a market-data outage cannot delay it.
    no new entries from the session before L through the gap session
An event dated on a non-session day (e.g. a Saturday) is moved to the next session as BMO.
Without a session list, weekday arithmetic is used with one extra day of margin for holidays.
"""
from __future__ import annotations

import bisect
import datetime as dt
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from src.sessions import to_et

EarningsCalendar = dict[str, list[tuple[dt.date, str]]]


@dataclass(frozen=True)
class EventWindow:
    event: dt.date
    timing: str
    block_from: dt.date     # first session with no new entries
    last_hold: dt.date      # L: the position must be gone by this session's close
    gap: dt.date            # session that opens with the news (entries resume the session after)


def load_earnings_calendar(path: Optional[str]) -> EarningsCalendar:
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path, comment="#", dtype=str, skipinitialspace=True).fillna("")
    missing = {"symbol", "date"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}; expected symbol,date[,timing]")
    calendar: EarningsCalendar = {}
    for row in df.itertuples(index=False):
        timing = str(getattr(row, "timing", "") or "").strip().upper()
        if timing not in ("", "BMO", "AMC"):
            raise ValueError(f"{path}: timing must be BMO, AMC or blank, got {timing!r} for {row.symbol}")
        date = pd.Timestamp(row.date.strip()).date()
        calendar.setdefault(row.symbol.strip().upper(), []).append((date, timing))
    return calendar


def _neighbours(day: dt.date, sessions: Optional[Sequence[dt.date]]) -> tuple[dt.date, dt.date, dt.date]:
    """(session before `day`, two sessions before, session after) from the exchange list, or
    weekday arithmetic with an extra day of margin when no list is available."""
    if sessions:
        i = bisect.bisect_left(sessions, day)
        before = sessions[i - 1] if i >= 1 else (pd.Timestamp(day) - pd.offsets.BDay(1)).date()
        before2 = sessions[i - 2] if i >= 2 else (pd.Timestamp(before) - pd.offsets.BDay(1)).date()
        j = bisect.bisect_right(sessions, day)
        after = sessions[j] if j < len(sessions) else (pd.Timestamp(day) + pd.offsets.BDay(1)).date()
        return before, before2, after
    bday = pd.offsets.BDay
    return ((pd.Timestamp(day) - bday(2)).date(), (pd.Timestamp(day) - bday(3)).date(),
            (pd.Timestamp(day) + bday(1)).date())


def event_window(date: dt.date, timing: str, sessions: Optional[Sequence[dt.date]] = None) -> EventWindow:
    if sessions and date not in sessions:      # weekend/holiday report: the next open carries it
        i = bisect.bisect_left(sessions, date)
        if i < len(sessions):
            date, timing = sessions[i], "BMO"
    before, before2, after = _neighbours(date, sessions)
    if timing == "AMC":
        last_hold, gap, block_from = date, after, before
    else:  # BMO, or unknown: exit as if BMO, block entries as if AMC
        last_hold, block_from = before, before2
        gap = date if timing == "BMO" else after
    return EventWindow(date, timing, block_from, last_hold, gap)


def _windows(symbol: str, calendar: EarningsCalendar, sessions) -> list[EventWindow]:
    return [event_window(date, timing, sessions) for date, timing in calendar.get(symbol.upper(), [])]


def entry_blackout(symbol: str, ts, calendar: EarningsCalendar, sessions=None) -> Optional[dt.date]:
    """Event date if new entries are blocked at `ts`, else None."""
    day = to_et(ts).date()
    for w in _windows(symbol, calendar, sessions):
        if w.block_from <= day <= w.gap:
            return w.event
    return None


def forced_exit(symbol: str, ts, calendar: EarningsCalendar, sessions=None, exit_time: str = "14:30",
                last_decision: Optional[dict] = None) -> Optional[dt.date]:
    """Event date if an open position must be closed at `ts`, else None. `last_decision` maps a
    session date to its last decision time ("HH:MM") for early closes (e.g. "12:30" on a 13:00
    close); the exit time on that day becomes the earlier of the two."""
    local = to_et(ts)
    day = local.date()
    cutoff = exit_time
    if last_decision and day in last_decision:
        cutoff = min(exit_time, last_decision[day])
    hours, minutes = (int(x) for x in cutoff.split(":"))
    for w in _windows(symbol, calendar, sessions):
        if (day == w.last_hold and (local.hour, local.minute) >= (hours, minutes)) or w.last_hold < day <= w.gap:
            return w.event
    return None


def has_upcoming_event(symbol: str, ts, calendar: EarningsCalendar, horizon_days: int = 100) -> bool:
    """True if the calendar lists an event for `symbol` within `horizon_days` of `ts`. Earnings
    are quarterly (~91 days apart), so a calendar without one is stale for that symbol."""
    day = to_et(ts).date()
    return any(day <= date <= day + dt.timedelta(days=horizon_days) for date, _ in calendar.get(symbol.upper(), []))


def events_per_year(symbol: str, calendar: EarningsCalendar, start, end) -> float:
    first, last = to_et(start).date(), to_et(end).date()
    years = max((last - first).days / 365.25, 1e-9)
    return sum(first <= date <= last for date, _ in calendar.get(symbol.upper(), [])) / years
