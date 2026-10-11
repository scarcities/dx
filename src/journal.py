"""Append-only CSV journal: one row per bar check, order, fill and safety event.

The schema is fixed. If an existing file has a different header it is rotated aside
instead of being appended to with misaligned columns.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import os
from typing import Any, Optional

import pandas as pd

from src.sessions import ET

# Events
BAR_CHECK = "BAR_CHECK"
ORDER_SUBMITTED = "ORDER_SUBMITTED"
ORDER_REJECTED = "ORDER_REJECTED"
ORDER_PENDING = "ORDER_PENDING"
ENTRY_FILL = "ENTRY_FILL"
EXIT_FILL = "EXIT_FILL"
STOP_RATCHET = "STOP_RATCHET"
PROTECTIVE_STOP = "PROTECTIVE_STOP"
KILL_SWITCH = "KILL_SWITCH"
ERROR = "ERROR"
STARTUP = "STARTUP"
SHUTDOWN = "SHUTDOWN"

COLUMNS = [
    "ts_utc", "ts_et", "event", "symbol",
    # Decision context: the bar being traded (t) and the closed bar it is based on (t-1)
    "decision_bar_et", "ref_bar_et",
    "open", "high", "low", "close",
    "ema_fast", "ema_slow", "rsi", "rsi_prev", "rsi_min", "prev_high", "adx", "plus_di", "minus_di", "atr", "natr",
    "natr_lo", "natr_hi", "cost_r",
    "state", "reason_code", "reason_detail", "action", "in_position",
    # Orders / fills
    "side", "qty", "ref_price", "fill_price", "slippage_bps",
    "stop_price", "take_profit", "risk_pct", "notional_pct", "binding_constraint",
    "entry_price", "exit_price", "exit_reason", "r_multiple", "pnl",
    # Account
    "equity", "day_start_equity", "daily_pnl_pct",
    "order_id", "client_order_id", "order_status", "latency_ms", "message",
]

SNAPSHOT_FIELDS = ["open", "high", "low", "close", "ema_fast", "ema_slow", "rsi", "rsi_prev", "rsi_min",
                   "prev_high", "adx", "plus_di", "minus_di", "atr", "natr", "natr_lo", "natr_hi", "cost_r"]


def slippage_bps(side: str, ref_price: float, fill_price: float) -> float:
    """Positive = adverse. Buy: paid above reference. Sell: received below reference."""
    if not ref_price or ref_price <= 0 or fill_price is None:
        return float("nan")
    raw = (fill_price - ref_price) / ref_price * 1e4
    return raw if side.lower() == "buy" else -raw


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.10g}"
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return pd.Timestamp(value).isoformat()
    return value


class TradeJournal:
    def __init__(self, path: str):
        self.path = path
        self._ensure_header()

    def _ensure_header(self) -> None:
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, newline="") as fh:
                header = next(csv.reader(fh), [])
            if header == COLUMNS:
                return
            stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
            root, ext = os.path.splitext(self.path)
            os.replace(self.path, f"{root}.schema-{stamp}{ext}")
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", newline="") as fh:
            csv.writer(fh).writerow(COLUMNS)

    def log(self, event: str, now: Optional[dt.datetime] = None, **fields: Any) -> dict:
        unknown = set(fields) - set(COLUMNS)
        if unknown:
            raise ValueError(f"Unknown journal fields: {sorted(unknown)}")
        ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        row = {col: "" for col in COLUMNS}
        row.update({"ts_utc": ts.tz_convert("UTC").isoformat(), "ts_et": ts.tz_convert(ET).isoformat(),
                    "event": event})
        row.update({key: _fmt(value) for key, value in fields.items()})
        with open(self.path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLUMNS).writerow(row)
            fh.flush()
            os.fsync(fh.fileno())
        return row

    def log_bar_check(self, symbol: str, sig: pd.Series, decision_bar, state: str, reason_code: str,
                      reason_detail: str, action: str, in_position: bool, now=None, **extra: Any) -> dict:
        snapshot = {name: sig.get(name) for name in SNAPSHOT_FIELDS}
        return self.log(
            BAR_CHECK, now=now, symbol=symbol, decision_bar_et=decision_bar,
            ref_bar_et=sig.get("ref_bar_time"), state=state, reason_code=reason_code,
            reason_detail=reason_detail, action=action, in_position=in_position, **snapshot, **extra,
        )

    def read(self) -> pd.DataFrame:
        return read_journal(self.path)

    def open_entries(self) -> dict[str, dict]:
        """Last ENTRY_FILL per symbol that has no later EXIT_FILL (survives restarts)."""
        df = self.read()
        if df.empty:
            return {}
        open_trades: dict[str, dict] = {}
        for _, row in df[df["event"].isin([ENTRY_FILL, EXIT_FILL, STOP_RATCHET, PROTECTIVE_STOP])].iterrows():
            symbol = row["symbol"]
            if row["event"] == ENTRY_FILL:
                open_trades[symbol] = row.to_dict()
                open_trades[symbol]["last_journaled_stop"] = row["stop_price"]
            elif row["event"] == EXIT_FILL:
                open_trades.pop(symbol, None)
            elif symbol in open_trades and pd.notna(row["stop_price"]):
                prev = open_trades[symbol]["last_journaled_stop"]
                open_trades[symbol]["last_journaled_stop"] = max(row["stop_price"], prev) if pd.notna(prev) \
                    else row["stop_price"]
        return open_trades

    def kill_switch_tripped_on(self, day) -> bool:
        df = self.read()
        if df.empty:
            return False
        trips = df[df["event"] == KILL_SWITCH]
        return bool((trips["ts_utc"].dt.tz_convert(ET).dt.date == day).any())


def read_journal(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"symbol": str, "client_order_id": str, "order_id": str})
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601", errors="coerce")
    for col in ("decision_bar_et", "ref_bar_et"):
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601", errors="coerce").dt.tz_convert(ET)
    return df
