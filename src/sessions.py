"""US equity session helpers: session-aligned hourly bars and the trading window.

Pure functions (no network) so the live bot, the backtester and the tests share them.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from config import SOURCE_BAR_MINUTES, TIMEZONE

ET = pytz.timezone(TIMEZONE)
SESSION_OPEN_MIN = 9 * 60 + 30
SESSION_CLOSE_MIN = 16 * 60
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "bar_end", "n_source", "n_expected"]


def _minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def to_et(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"Naive timestamp {ts!r}: pass timezone-aware datetimes")
    return ts.tz_convert(TIMEZONE)


def in_trading_window(ts, start: str = "09:45", end: str = "15:45") -> bool:
    """True on weekdays between start and end ET (inclusive). Holidays are handled by the
    broker clock in live trading and by the absence of bars in backtests."""
    local = to_et(ts)
    if local.weekday() >= 5:
        return False
    minute_of_day = local.hour * 60 + local.minute + local.second / 60.0
    return _minutes(start) <= minute_of_day <= _minutes(end)


def to_session_bars(source: pd.DataFrame, source_minutes: int = SOURCE_BAR_MINUTES) -> pd.DataFrame:
    """Aggregate intraday bars (index = bar start, tz-aware) into session-aligned hourly
    bars: 09:30, 10:30, ..., 14:30 (60 min) and 15:30 (30 min). Extended-hours bars are
    dropped. Output index is the hourly bar start in ET."""
    if source.empty:
        return pd.DataFrame(columns=BAR_COLUMNS, index=pd.DatetimeIndex([], tz=TIMEZONE, name="timestamp"))

    local = source.index.tz_convert(TIMEZONE)
    minute_of_day = np.asarray(local.hour * 60 + local.minute)
    rth = (minute_of_day >= SESSION_OPEN_MIN) & (minute_of_day < SESSION_CLOSE_MIN)
    src = source.loc[rth]
    local = local[rth]
    bucket_min = ((minute_of_day[rth] - SESSION_OPEN_MIN) // 60) * 60 + SESSION_OPEN_MIN
    key = local.normalize() + pd.to_timedelta(bucket_min, unit="min")

    grouped = src.groupby(key)
    out = pd.DataFrame({
        "open": grouped["open"].first(),
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "close": grouped["close"].last(),
        "volume": grouped["volume"].sum() if "volume" in src else 0.0,
        "n_source": grouped.size(),
    })
    start_min = np.asarray(out.index.hour * 60 + out.index.minute)
    end_min = np.minimum(start_min + 60, SESSION_CLOSE_MIN)
    out["bar_end"] = out.index.normalize() + pd.to_timedelta(end_min, unit="min")
    out["n_expected"] = (end_min - start_min) // source_minutes
    out.index.name = "timestamp"
    return out[BAR_COLUMNS]


def complete_bars(bars: pd.DataFrame, now, grace_seconds: int = 120) -> pd.DataFrame:
    """Keep bars that have closed and contain all source bars, or are past the grace period
    (a thin IEX interval can legitimately miss a source bar)."""
    if bars.empty:
        return bars
    now = to_et(now)
    ended = bars["bar_end"] <= now
    full = bars["n_source"] >= bars["n_expected"]
    late = bars["bar_end"] + pd.Timedelta(seconds=grace_seconds) <= now
    return bars.loc[ended & (full | late)]


def scheduled_bar_ends(day: dt.date) -> list[pd.Timestamp]:
    """Hourly bar close times for a regular session (early closes are not modelled)."""
    midnight = pd.Timestamp(ET.localize(dt.datetime(day.year, day.month, day.day)))
    starts = range(SESSION_OPEN_MIN, SESSION_CLOSE_MIN, 60)
    return [midnight + pd.Timedelta(minutes=min(m + 60, SESSION_CLOSE_MIN)) for m in starts]


def latest_bar_end(now, settle_seconds: int = 0) -> Optional[pd.Timestamp]:
    """Most recent scheduled bar close at or before `now - settle_seconds`, today only."""
    cutoff = to_et(now) - pd.Timedelta(seconds=settle_seconds)
    ends = [end for end in scheduled_bar_ends(cutoff.date()) if end <= cutoff]
    return ends[-1] if ends else None
