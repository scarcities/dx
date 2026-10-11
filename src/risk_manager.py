"""Position sizing, daily drawdown kill-switch and order idempotency gates."""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Iterable, Optional

from config import RiskConfig, StrategyConfig


@dataclass(frozen=True)
class PositionSize:
    shares: int
    entry_ref: float
    stop_price: float
    take_profit: float
    stop_distance: float
    risk_dollars: float
    risk_pct: float
    notional: float
    notional_pct: float
    binding_constraint: str  # RISK | NOTIONAL_CAP | BUYING_POWER | NONE


@dataclass(frozen=True)
class KillSwitchStatus:
    tripped: bool
    newly_tripped: bool
    daily_pnl_pct: float
    day_start_equity: float
    equity: float


def round_price(price: float) -> float:
    """Alpaca rejects sub-penny prices >= $1.00 and allows 4 decimals below."""
    return round(price, 2) if price >= 1.0 else round(price, 4)


def r_multiple(entry: float, exit_: float, stop: float, planned_risk: Optional[float] = None) -> float:
    """Realized R of a long trade. Uses fill-based risk; falls back to planned risk
    if the entry filled at or below the stop (gap through the stop)."""
    risk = entry - stop
    if risk <= 1e-9:
        risk = planned_risk if planned_risk and planned_risk > 0 else float("nan")
    return (exit_ - entry) / risk


def client_order_id(strategy_tag: str, symbol: str, decision_bar: "dt.datetime") -> str:
    """Deterministic id per (symbol, decision bar). Alpaca rejects duplicates, which makes
    order submission idempotent across retries and restarts."""
    return f"{strategy_tag}-{symbol}-{decision_bar:%Y%m%dT%H%M}"


class RiskManager:
    def __init__(self, risk_cfg: RiskConfig, strategy_cfg: StrategyConfig):
        self.cfg = risk_cfg
        self.strategy_cfg = strategy_cfg
        self._halted_day: Optional[dt.date] = None

    # ------------------------------------------------------------------ sizing
    def size_position(
        self, equity: float, entry_ref: float, atr: float, buying_power: Optional[float] = None
    ) -> PositionSize:
        stop_distance = self.strategy_cfg.stop_atr_mult * atr
        stop = round_price(entry_ref - stop_distance)
        target = round_price(entry_ref + self.strategy_cfg.target_atr_mult * atr)
        if equity <= 0 or entry_ref <= 0 or not math.isfinite(atr) or stop_distance <= 0:
            return PositionSize(0, entry_ref, stop, target, stop_distance, 0.0, 0.0, 0.0, 0.0, "NONE")

        candidates = {
            "RISK": math.floor(equity * self.cfg.risk_per_trade / stop_distance),
            "NOTIONAL_CAP": math.floor(equity * self.cfg.max_position_notional / entry_ref),
        }
        if buying_power is not None:
            candidates["BUYING_POWER"] = math.floor(max(buying_power, 0.0) / entry_ref)
        binding = min(candidates, key=candidates.get)
        shares = max(candidates[binding], 0)

        risk_dollars = shares * stop_distance
        notional = shares * entry_ref
        return PositionSize(
            shares=shares,
            entry_ref=entry_ref,
            stop_price=stop,
            take_profit=target,
            stop_distance=stop_distance,
            risk_dollars=risk_dollars,
            risk_pct=risk_dollars / equity,
            notional=notional,
            notional_pct=notional / equity,
            binding_constraint=binding if shares > 0 else "NONE",
        )

    # ------------------------------------------------------------------ kill-switch
    def check_daily_drawdown(self, day: dt.date, day_start_equity: float, equity: float) -> KillSwitchStatus:
        """Trips when equity <= day_start_equity * (1 - max_daily_drawdown). Latched for the day."""
        pnl_pct = (equity - day_start_equity) / day_start_equity if day_start_equity > 0 else 0.0
        already = self._halted_day == day
        breached = pnl_pct <= -self.cfg.max_daily_drawdown + 1e-12
        if breached and not already:
            self._halted_day = day
        tripped = already or breached
        return KillSwitchStatus(tripped, tripped and not already, pnl_pct, day_start_equity, equity)

    def is_halted(self, day: dt.date) -> bool:
        return self._halted_day == day

    def latch_halt(self, day: dt.date) -> None:
        """Restore a kill-switch trip recorded earlier today (e.g. after a restart)."""
        self._halted_day = day

    # ------------------------------------------------------------------ idempotency
    def entry_block_reason(
        self,
        symbol: str,
        day: dt.date,
        open_position_symbols: Iterable[str],
        open_order_symbols: Iterable[str],
        submitted_client_ids: Iterable[str],
        client_id: str,
    ) -> Optional[str]:
        """Return why a new entry must not be sent, or None if it is allowed."""
        positions = set(open_position_symbols)
        if self.is_halted(day):
            return "KILL_SWITCH_ACTIVE"
        if symbol in positions:
            return "POSITION_ALREADY_OPEN"
        if symbol in set(open_order_symbols):
            return "OPEN_ORDER_EXISTS"
        if client_id in set(submitted_client_ids):
            return "DUPLICATE_CLIENT_ORDER_ID"
        if len(positions) >= self.cfg.max_open_positions:
            return "MAX_OPEN_POSITIONS"
        return None
