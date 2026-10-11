# Alpaca 1h Trend-Pullback Bot (paper validation)

A long-only, regime-filtered trend-pullback strategy for US equities (`SPY QQQ AAPL MSFT NVDA`) on
session-aligned hourly bars, built on `alpaca-py`. The backtester, the live bot and the trust
evaluator all run the same strategy, sizing, exit and kill-switch code.

## Quick start (macOS)

```bash
git clone <this repo> && cd dx
cp .env.example .env                                   # add your Alpaca PAPER keys
cp earnings_calendar.example.csv earnings_calendar.csv # add AAPL/MSFT/NVDA report dates (see below)
./start_mac.sh --dry-run      # first day: decisions are journaled, no orders sent
./start_mac.sh                # paper trading (Ctrl-C to stop)
```

`start_mac.sh` creates `.venv`, installs `requirements.txt` when it changes, runs the quick offline
tests (and refuses to start if they fail), then runs `bot.py` under `caffeinate -i -s`. That blocks
idle sleep, and system sleep while on AC power. **A MacBook with its lid closed still sleeps**, so keep
it open and plugged in from 09:30 to 16:00 ET.

| Command | What it does |
| --- | --- |
| `python backtest.py --feed sip --save-bars bars.csv.gz` | 2 years of bars from Alpaca, tear sheet, and a copy of the bars (one ~1 MB file) |
| `python backtest.py --from-bars bars.csv.gz --compare` | v1 vs each v2 change vs v2 defaults, paired exit study, filter audit (no API calls) |
| `python backtest.py --from-bars bars.csv.gz --slippage-bps 2` | Same bars at a different cost assumption |
| `python backtest.py --feed sip --start 2019-01-01 --end 2023-12-31 --compare` | The held-out test (see below) |
| `python bot.py --once` | One bot cycle, for a smoke test |
| `python evaluate_trust.py --replay` | After 2 to 3 weeks: the go/no-go report on `paper_trading_journal.csv` (it will not judge edge before 30 trades over 20 distinct weeks) |
| `pytest` | Full offline suite (about a minute); `pytest -m "not slow"` is the quick one |

## v2: what changed after the first real backtest, and why

The 2024-10 to 2026-10 SIP run lost money: 81 trades, -0.182R expectancy (95% CI -0.43 to +0.06),
profit factor 0.76. The trade list shows:

- **There was no edge before costs either.** Gross expectancy (zero slippage) was -0.093R, profit
  factor 0.90. Slippage made it worse, but did not cause it.
- **One earnings gap was 66% of the loss.** AAPL bought 2026-07-30 at 14:30 gapped -8.6% at the next
  open: -4.94R, -$1,694. Without it the net loss is -$860.
- **The result is statistically indistinguishable from zero.** So is NVDA's +0.39R (13 trades, CI
  -0.15 to +0.94): it is not evidence that NVDA is the "right" symbol.

What v2 changes, and the status of each change:

| Change | Default | Status |
| --- | --- | --- |
| Earnings blackout: no entries around a report, exit at 14:30 on the last session before the gap | **on** | Risk control, justified without data: a 0.6-2.5% stop cannot protect a 5-10% gap. Needs `earnings_calendar.csv`; the live bot refuses new entries in a stock with no date in the next 100 days |
| `exit_mode="bracket"`: the EMA50 exit is removed, trades end at the stop or target | **on** | Removes the noise-triggered exit with zero new parameters. Expect longer holds (more nights, so more gap exposure) and roughly the same expectancy: exits cannot add edge to entries that have none |
| `exit_mode="ratchet"`: breakeven at +1R, then trail 2 ATR under the highest close | off | With a 1.5R target this can lock at most +0.5R; effectively a breakeven stop. Kept for the paired test |
| `max_hold_bars`: time stop | off | Alternative to the EMA exit for trades that go nowhere |
| Cost floor `max_cost_r=0.10` | off | At 5 bp/side it removes SPY (and QQQ most of the time), so it is a universe change driven by an unmeasured cost. Measure slippage, set `SLIPPAGE_PER_SIDE`, then decide |
| `directional_check="anchor"`: ADX >= 20 and +DI > -DI measured at the last bar with RSI >= 50 before the dip, instead of on the signal bar | off | Your diagnosis #3 is right in mechanism (on the signal bar ADX mostly measures the pullback). Hypothesis-grade until the held-out test |
| `directional_check="signal_bar"`: +DI > -DI on the signal bar | off | Do not use: right after RSI recovers from < 42, -DI still leads ~90% of the time, so this deletes almost every entry |
| Backtest fills: take-profit limits fill at the limit with no slippage, and only on a 1-cent trade-through | on | A sell limit cannot fill below its price, and a touch is not a fill |
| Live: a failed exit is re-protected at once (OCO target + stop, sized from the broker's position) and retried with backoff (30 s doubling to 10 min); any position without a working stop is re-protected within 60 s, or closed if that is refused; stop replacement is never blindly retried | on | Fixes a v1 bug: if cancelling the bracket legs was still pending, the market sell was rejected and the bot assumed the position was gone, leaving it with no stop |

### How to decide whether any of this works

Every v2 change was proposed after looking at the same 81 trades it would be judged on, so an
improvement on that window proves nothing. The honest sequence:

1. `python backtest.py --feed sip --save-bars bars_2y.csv.gz --compare` on the last 2 years. Use it
   for mechanism checks only: does the blackout remove the AAPL gap, what share of signals each filter
   removes, what the paired exit study says.
2. Write down which variant you would pick and what result would change your mind **before** step 3.
3. Run the same `--compare` once on a period nobody has looked at, e.g. `--start 2019-01-01 --end
   2023-12-31` (it contains the 2020 crash, the 2022 bear market and 2023). Accept a variant only
   if it is positive there, holds up at `--slippage-bps 5` as well as 2, does not depend on one
   symbol (the `w/o best` column), and a filter beats a random drop of the same size (`vs random
   drop` above ~95%). The backtest keeps a ledger and warns when a window has been used before.
4. If nothing passes, stop iterating on these 5 symbols and 2 years: every further look spends the
   held-out data. Paper trading is still useful for checking operations and measuring real costs.

Power check: with R's standard deviation around 1, detecting a +0.10R edge takes roughly 600
trades. These five correlated symbols produce about 40 a year, so plan on a broader universe for
evaluation (`--symbols` accepts any list, e.g. sector ETFs) before trusting any edge.

## Strategy rules (v2 defaults)

All decisions for bar `t` use only closed bars `<= t-1` (`generate_signals` applies `shift(1)`).

1. **Trend**: `Close > EMA200` and `EMA50 > EMA200`.
2. **WAIT filters**: `CHOP_ADX_LOW` if `ADX14 < 20`; `VOLATILITY_DEAD` / `VOLATILITY_SPIKE` if
   `ATR14/Close` is below the 15th / above the 92nd percentile of the last 100 bars;
   `EARNINGS_BLACKOUT` inside an event window.
3. **Entry**: RSI14 crossed back above 42 on `t-1` after dipping below it, and `Close[t-1] > High[t-2]`.
   (The "dipped within 3 bars" clause adds nothing: a cross above 42 already implies the bar before
   was at or below 42.)
4. **Exit**: GTC bracket, stop `Close[t-1] - 2 ATR`, target `Close[t-1] + 3 ATR` (1.5R).
5. **Risk**: 1% of equity per trade, at most 20% of equity per position, whole shares. A 3% daily
   loss vs the previous close's equity trips the kill-switch: everything is flattened and entries
   stop until the next session.
6. **When**: right after an hourly bar closes inside 09:45 to 15:45 ET (10:30, 11:30, ... 15:30),
   only on days the broker clock says the market is open.

Every decision is journaled with indicator values and the numeric reason, e.g.
`CHOP_ADX_LOW: ADX[t-1] 14.82 < 20.00 | TREND_FILTER: EMA50 401.20 <= EMA200 405.11`.

## Earnings calendar

alpaca-py has no earnings calendar, so `earnings_calendar.csv` is maintained by hand (format in
`earnings_calendar.example.csv`): `symbol,date,timing` with timing `AMC`, `BMO` or blank. Get dates
from each company's investor-relations page or the SEC 8-K filing time. For a report on day D the
bot stops opening new positions from the session before the last hold day, and closes open ones at
the 14:30 decision on the last hold day (D for after-close reports, the session before D otherwise),
retrying every 30 s. Backtests need past dates too (~4 per year per stock) or they print a warning.

## Other design notes

- **Hourly bars are rebuilt from 30-minute bars** (09:30, 10:30, ... 15:30; regular hours only).
  Alpaca's `1Hour` bars are clock-aligned, so the 09:00 bar would mix pre-market prints into the open.
- **Brackets are GTC**: DAY legs expire at the close and leave positions unprotected overnight.
- **Costs**: one setting, `SLIPPAGE_PER_SIDE` (5 bp, the original brief), drives both backtest fills
  and the optional cost floor. Real market-order costs for these names are probably 1-3 bp; Alpaca's
  paper fills are optimistic, so treat paper slippage as a lower bound.
- **The 1% risk target almost never applies**: the 20% notional cap binds first, so real risk per
  trade is about 0.1-0.5% of equity.
- **SIP vs IEX**: the free plan only gives real-time bars from IEX, so the live bot uses IEX. IEX
  bars are thinner than SIP; compare a backtest on both feeds before relying on absolute thresholds.
- **The offline tests use synthetic data.** They prove the plumbing (no look-ahead, filter logic,
  sizing, kill-switch, earnings windows, failed-exit handling, live/backtest parity trade by trade),
  not that the strategy makes money.

## Layout

```
config.py              universe, costs, strategy / risk / execution / backtest parameters
src/indicators.py      EMA, Wilder RSI / ATR / ADX, normalized ATR percentiles (vectorized, causal)
src/strategy.py        rules, shift(1) decisions, reason codes, stop ratchet
src/events.py          earnings calendar and blackout windows
src/risk_manager.py    sizing, daily kill-switch, idempotency gates, deterministic client_order_id
src/sessions.py        session-aligned bars, ET trading window, bar schedule
src/alpaca_client.py   alpaca-py wrapper: retries with backoff, data, orders, exchange calendar
src/journal.py         append-only CSV journal (fixed schema, fsync per row)
bot.py                 live loop: kill-switch, reconciliation, stop coverage, per-bar decisions
backtest.py            portfolio backtester, tear sheet, --compare harness, bar export/import
evaluate_trust.py      go/no-go report from the journal
start_mac.sh           macOS launcher
tests/                 offline pytest suite with the synthetic multi-regime generator
```

Safety rails in the bot: it refuses to start unless `ALPACA_PAPER=True`; entry order ids are
deterministic per symbol and bar (`tpb2-...`), so retries and restarts cannot double an entry; the
broker is the source of truth on restart (positions, working stops, fills while the bot was off).
The journal schema changed in v2, so an existing v1 journal is renamed aside, not appended to.
