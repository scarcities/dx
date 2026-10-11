"""Vectorized technical indicators.

Every function is causal: the value at row k depends only on rows <= k, so the
output on a prefix of the data equals the prefix of the output on the full data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import StrategyConfig


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA): seeded with the SMA of the first `period` valid values,
    then avg[k] = avg[k-1] + (x[k] - avg[k-1]) / period.

    Expects NaNs only at the start of the series (as produced by diff/shift).
    """
    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(values))
    if valid.size == 0:
        return pd.Series(out, index=series.index)
    first = valid[0]
    seed_idx = first + period - 1
    if seed_idx >= values.size:
        return pd.Series(out, index=series.index)

    seeded = np.full(values.shape, np.nan)
    seeded[seed_idx] = values[first:seed_idx + 1].mean()
    seeded[seed_idx + 1:] = values[seed_idx + 1:]
    smoothed = pd.Series(seeded, index=series.index).ewm(alpha=1.0 / period, adjust=False).mean()
    smoothed.iloc[:seed_idx] = np.nan
    return smoothed


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = wilder_smooth(delta.clip(lower=0.0), period)
    avg_loss = wilder_smooth((-delta).clip(lower=0.0), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
    return out.where(avg_gain.notna() & avg_loss.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1, skipna=True)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return wilder_smooth(true_range(high, low, close), period)


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's ADX. Returns (adx, +DI, -DI)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan

    tr = true_range(high, low, close)
    tr.iloc[0] = np.nan  # DM starts at bar 1, so TR must too

    smooth_tr = wilder_smooth(tr, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * wilder_smooth(plus_dm, period) / smooth_tr
        minus_di = 100.0 * wilder_smooth(minus_dm, period) / smooth_tr
        di_sum = plus_di + minus_di
        dx = (100.0 * (plus_di - minus_di).abs() / di_sum).where(di_sum != 0.0, 0.0)
    dx = dx.where(plus_di.notna() & minus_di.notna())
    return wilder_smooth(dx, period), plus_di, minus_di


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with every indicator the strategy uses."""
    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]
    out["ema_fast"] = ema(close, cfg.ema_fast)
    out["ema_slow"] = ema(close, cfg.ema_slow)
    out["rsi"] = rsi(close, cfg.rsi_period)
    out["atr"] = atr(high, low, close, cfg.atr_period)
    out["adx"], out["plus_di"], out["minus_di"] = adx(high, low, close, cfg.adx_period)
    out["natr"] = out["atr"] / close
    window = out["natr"].rolling(cfg.natr_window, min_periods=cfg.natr_window)
    out["natr_lo"] = window.quantile(cfg.natr_low_pct)
    out["natr_hi"] = window.quantile(cfg.natr_high_pct)
    return out
