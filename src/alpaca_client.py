"""Defensive wrapper around alpaca-py's TradingClient and StockHistoricalDataClient.

- Exponential backoff with jitter on transient failures (connection errors, 429, 5xx).
  4xx business errors (rejections, insufficient qty) are raised immediately.
- Session-aligned hourly bars built from 30-minute source bars, regular hours only.
- Bracket entries with deterministic client_order_id, so a retried or repeated
  submission can never create a second position.
"""
from __future__ import annotations

import datetime as dt
import logging
import random
import time
from typing import Callable, Iterable, Optional, TypeVar

import pandas as pd
import requests
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (GetCalendarRequest, GetOrdersRequest, LimitOrderRequest, MarketOrderRequest,
                                     ReplaceOrderRequest, StopLossRequest, StopOrderRequest, TakeProfitRequest)

from config import SOURCE_BAR_MINUTES, ExecutionConfig
from src.risk_manager import round_price
from src.sessions import complete_bars, in_trading_window, to_session_bars

log = logging.getLogger(__name__)
T = TypeVar("T")

TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "done_for_day", "stopped", "suspended", "replaced"}
WORKING_STATUSES = {"new", "held", "accepted", "pending_new", "accepted_for_bidding", "partially_filled"}


class ExitBlocked(RuntimeError):
    """The bracket legs could not be cancelled in time, so a market sell would be rejected."""


def enum_value(value) -> str:
    return str(getattr(value, "value", value) or "").lower()


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, APIError):
        status = exc.status_code
        return status is not None and (status == 429 or status >= 500)
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                            ConnectionError, TimeoutError))


def with_retries(fn: Callable[[], T], description: str, max_retries: int = 5, base_delay: float = 1.0,
                 max_delay: float = 30.0, sleep: Callable[[float], None] = time.sleep,
                 retryable: Callable[[BaseException], bool] = is_retryable) -> T:
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified below
            if attempt >= max_retries or not retryable(exc):
                raise
            delay = min(max_delay, base_delay * 2 ** attempt) * (0.5 + random.random() / 2)
            log.warning("%s failed (%s: %s); retry %d/%d in %.1fs", description, type(exc).__name__,
                        exc, attempt + 1, max_retries, delay)
            sleep(delay)
            attempt += 1


def build_bracket_order(symbol: str, qty: int, stop_price: float, take_profit: float,
                        client_order_id: str) -> MarketOrderRequest:
    # GTC, not DAY: DAY bracket legs expire at the close and leave the position unprotected overnight.
    return MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit),
        stop_loss=StopLossRequest(stop_price=stop_price),
        client_order_id=client_order_id,
    )


def bars_frame_by_symbol(raw: pd.DataFrame, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Split alpaca-py's (symbol, timestamp) MultiIndex frame into one frame per symbol."""
    result = {}
    for symbol in symbols:
        if raw.empty or symbol not in raw.index.get_level_values(0):
            result[symbol] = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            continue
        frame = raw.xs(symbol, level=0)[["open", "high", "low", "close", "volume"]].astype(float)
        frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
        result[symbol] = frame.sort_index()
    return result


class AlpacaClient:
    def __init__(self, api_key: str, secret_key: str, paper: bool, exec_cfg: ExecutionConfig,
                 trading: Optional[TradingClient] = None, data: Optional[StockHistoricalDataClient] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg = exec_cfg
        self.paper = paper
        self.trading = trading or TradingClient(api_key, secret_key, paper=paper)
        self.data = data or StockHistoricalDataClient(api_key, secret_key)
        self._sleep = sleep

    def _call(self, description: str, fn: Callable[..., T], *args, **kwargs) -> T:
        return with_retries(lambda: fn(*args, **kwargs), description, self.cfg.max_retries,
                            self.cfg.backoff_base_seconds, self.cfg.backoff_max_seconds, self._sleep)

    # ------------------------------------------------------------------ account / clock
    def get_account(self):
        return self._call("get_account", self.trading.get_account)

    def get_clock(self):
        return self._call("get_clock", self.trading.get_clock)

    def is_trading_window(self, now: Optional[dt.datetime] = None) -> bool:
        """Market open per the broker clock (handles holidays) AND inside 09:45-15:45 ET."""
        clock = self.get_clock()
        now = now or clock.timestamp
        return bool(clock.is_open) and in_trading_window(now, self.cfg.trade_window_start,
                                                         self.cfg.trade_window_end)

    def get_sessions(self, start: dt.date, end: dt.date) -> list:
        """Exchange trading days (holidays excluded) from Alpaca's calendar."""
        request = GetCalendarRequest(start=start, end=end)
        return [c.date for c in self._call("get_calendar", self.trading.get_calendar, request)]

    def get_positions(self) -> dict:
        return {p.symbol: p for p in self._call("get_all_positions", self.trading.get_all_positions)}

    def get_open_orders(self, symbols: Optional[list[str]] = None) -> list:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=symbols, limit=500)
        return list(self._call("get_orders(open)", self.trading.get_orders, filter=request))

    # ------------------------------------------------------------------ market data
    def fetch_source_bars(self, symbols: list[str], start: dt.datetime, end: dt.datetime,
                          minutes: int = SOURCE_BAR_MINUTES) -> dict[str, pd.DataFrame]:
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
            start=start,
            end=end,
            adjustment=Adjustment.ALL,
            feed=DataFeed(self.cfg.data_feed),
        )
        barset = self._call("get_stock_bars", self.data.get_stock_bars, request)
        return bars_frame_by_symbol(barset.df, symbols)

    def fetch_session_bars(self, symbols: list[str], start: dt.datetime, end: dt.datetime,
                           now: Optional[dt.datetime] = None) -> dict[str, pd.DataFrame]:
        """Session-aligned hourly bars. With `now`, only fully closed bars are returned."""
        result = {}
        for symbol, source in self.fetch_source_bars(symbols, start, end).items():
            bars = to_session_bars(source)
            if now is not None:
                bars = complete_bars(bars, now, self.cfg.incomplete_bar_grace_seconds)
            result[symbol] = bars
        return result

    # ------------------------------------------------------------------ orders
    def submit_bracket_order(self, symbol: str, qty: int, stop_price: float, take_profit: float,
                             client_order_id: str):
        request = build_bracket_order(symbol, qty, stop_price, take_profit, client_order_id)
        try:
            return self._call(f"submit_order({client_order_id})", self.trading.submit_order, request)
        except APIError as exc:
            if exc.status_code == 422 and "client_order_id" in str(exc).lower():
                log.warning("Order %s already exists at the broker; reusing it", client_order_id)
                return self.get_order_by_client_id(client_order_id)
            raise

    def get_order(self, order_id):
        return self._call("get_order_by_id", self.trading.get_order_by_id, order_id)

    def get_order_by_client_id(self, client_order_id: str):
        return self._call("get_order_by_client_id", self.trading.get_order_by_client_id, client_order_id)

    def wait_for_terminal(self, order_id, timeout: Optional[float] = None, poll: float = 0.5):
        """Poll an order until it is filled/cancelled/rejected or the timeout expires."""
        deadline = time.monotonic() + (timeout if timeout is not None else self.cfg.fill_poll_timeout_seconds)
        order = self.get_order(order_id)
        while enum_value(order.status) not in TERMINAL_STATUSES and time.monotonic() < deadline:
            self._sleep(poll)
            order = self.get_order(order_id)
        return order

    def cancel_symbol_orders(self, symbol: str) -> int:
        orders = self.get_open_orders([symbol])
        for order in orders:
            try:
                self._call("cancel_order", self.trading.cancel_order_by_id, order.id)
            except APIError as exc:
                if exc.status_code not in (404, 422):  # already filled / already cancelled
                    raise
        deadline = time.monotonic() + self.cfg.cancel_poll_timeout_seconds
        while self.get_open_orders([symbol]) and time.monotonic() < deadline:
            self._sleep(0.5)
        return len(orders)

    def close_position(self, symbol: str):
        """Cancel the bracket legs (they reserve the shares), then market-sell the position.
        Returns the closing order, or None only if the broker has no such position (404).
        Raises ExitBlocked if legs are still open after the cancel wait: selling then would be
        rejected for held shares and must not be mistaken for "already closed"."""
        self.cancel_symbol_orders(symbol)
        if self.get_open_orders([symbol]):
            raise ExitBlocked(f"{symbol}: bracket legs still open after cancel; exit not sent")
        try:
            return self._call(f"close_position({symbol})", self.trading.close_position, symbol)
        except APIError as exc:
            if exc.status_code == 404:
                log.warning("close_position(%s): no position (%s)", symbol, exc)
                return None
            raise

    def flatten_all(self):
        return self._call("close_all_positions", self.trading.close_all_positions, cancel_orders=True)

    def working_stop_legs(self, symbol: str) -> list:
        return [o for o in self.get_open_orders([symbol])
                if enum_value(o.side) == "sell" and enum_value(o.order_type or o.type) in ("stop", "stop_limit")
                and enum_value(o.status) in WORKING_STATUSES]

    def replace_stop(self, symbol: str, stop_price: float):
        """Raise the working stop leg of a bracket with ONE attempt: a replace is not idempotent,
        so a timed-out-but-applied PATCH must not be resent. Raises on any problem; the caller
        re-reads the broker's stop and keeps the old one, which is still working."""
        legs = self.working_stop_legs(symbol)
        if len(legs) != 1:
            raise RuntimeError(f"{symbol}: expected exactly one working stop leg, found {len(legs)}")
        request = ReplaceOrderRequest(stop_price=round_price(stop_price))
        return self.trading.replace_order_by_id(legs[0].id, request)

    def restore_protection(self, symbol: str, qty: float, stop_price: float, take_profit: Optional[float]) -> str:
        """Re-protect a position that has lost its stop leg. Any remaining leg (e.g. an orphaned
        take-profit) holds the shares and would get a new stop rejected, so everything working on
        the symbol is cancelled first; then an OCO (target + stop) is placed, falling back to a
        plain GTC stop. Returns "oco" or "stop"; raises if nothing could be placed."""
        self.cancel_symbol_orders(symbol)
        if self.get_open_orders([symbol]):
            raise ExitBlocked(f"{symbol}: orders still open after cancel; cannot re-protect yet")
        stop = StopLossRequest(stop_price=round_price(stop_price))
        if take_profit is not None:
            target = round_price(take_profit)
            oco = LimitOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                                    order_class=OrderClass.OCO, limit_price=target,
                                    take_profit=TakeProfitRequest(limit_price=target), stop_loss=stop)
            try:
                self._call(f"protect_oco({symbol})", self.trading.submit_order, oco)
                return "oco"
            except APIError as exc:
                log.warning("OCO re-protection for %s rejected (%s); placing a plain stop", symbol, exc)
        plain = StopOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                                 stop_price=stop.stop_price)
        self._call(f"protect_stop({symbol})", self.trading.submit_order, plain)
        return "stop"

    def bracket_levels(self, symbol: str) -> tuple[Optional[float], Optional[float]]:
        """(stop_price, take_profit) of the working sell legs for a symbol, if any."""
        stop = target = None
        for order in self.get_open_orders([symbol]):
            if enum_value(order.side) != "sell":
                continue
            kind = enum_value(order.order_type or order.type)
            if kind in ("stop", "stop_limit") and order.stop_price is not None:
                stop = float(order.stop_price)
            elif kind == "limit" and order.limit_price is not None:
                target = float(order.limit_price)
        return stop, target

    def find_latest_fill(self, symbol: str, side: str, since: dt.datetime):
        """Most recent filled order (legs included) for a symbol and side since a time."""
        request = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol], nested=True,
                                   after=since - dt.timedelta(minutes=5), limit=200)
        candidates = []
        for order in self._call("get_orders(closed)", self.trading.get_orders, filter=request):
            for item in [order, *(order.legs or [])]:
                if (enum_value(item.side) == side and enum_value(item.status) == "filled"
                        and item.filled_at is not None and item.filled_at >= since):
                    candidates.append(item)
        return max(candidates, key=lambda o: o.filled_at) if candidates else None
