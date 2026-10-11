# Alpaca 1h Trend-Pullback Bot (paper validation)

A long-only, regime-filtered trend-pullback strategy for US equities (`SPY QQQ AAPL MSFT NVDA`) on
session-aligned hourly bars, built on `alpaca-py`. The backtester, the live bot and the trust
evaluator all run the same strategy, sizing and kill-switch code.

## Quick start (macOS)

```bash
git clone <this repo> && cd dx
cp .env.example .env          # add your Alpaca PAPER keys
./start_mac.sh --dry-run      # first day: decisions are journaled, no orders sent
./start_mac.sh                # paper trading (Ctrl-C to stop)
```

`start_mac.sh` creates `.venv`, installs `requirements.txt` when it changes, runs the offline test
suite (and refuses to start if it fails), then runs `bot.py` under `caffeinate -i -s`. That blocks
idle sleep, and system sleep while on AC power. **A MacBook with its lid closed still sleeps**, so keep
it open and plugged in from 09:30 to 16:00 ET.

Other entry points:

| Command | What it does |
| --- | --- |
| `python backtest.py` | 2 years of 30-min bars from Alpaca, rebuilt into session hours, portfolio backtest and tear sheet |
| `python backtest.py --feed sip --trades-out trades.csv` | Same on SIP data (allowed for history older than 15 min on the free plan) |
| `python bot.py --once` | One bot cycle, for a smoke test |
| `python evaluate_trust.py --replay --backtest-expectancy 0.15` | After 2 to 3 weeks: the go/no-go report on `paper_trading_journal.csv` |
| `pytest` | 59 offline tests, no network needed |

## Strategy rules

All decisions for bar `t` use only closed bars `<= t-1` (`generate_signals` applies `shift(1)`).

1. **Trend**: `Close > EMA200` and `EMA50 > EMA200`.
2. **WAIT filters**: `CHOP_ADX_LOW` if `ADX14 < 20`; `VOLATILITY_DEAD` if `ATR14/Close` is below the
   15th percentile of the last 100 bars; `VOLATILITY_SPIKE` if above the 92nd.
3. **Entry**: RSI14 dipped below 42 within the last 3 closed bars, crossed back above 42 on `t-1`,
   and `Close[t-1] > High[t-2]`.
4. **Exit**: GTC bracket with stop `Close[t-1] - 2 ATR` and target `Close[t-1] + 3 ATR` (1.5R), plus
   a trailing exit when a bar closes below EMA50.
5. **Risk**: 1% of equity per trade, at most 20% of equity per position, whole shares. A 3% daily
   loss vs the previous close's equity trips the kill-switch: everything is flattened and entries
   stop until the next session.
6. **When**: decisions are made only right after an hourly bar closes inside 09:45 to 15:45 ET
   (10:30, 11:30, ... 15:30), and only on days the broker clock says the market is open.

Every decision is written to `paper_trading_journal.csv` with the indicator values and the
numeric reason, e.g. `CHOP_ADX_LOW: ADX[t-1] 14.82 < 20.00 | TREND_FILTER: EMA50 401.20 <= EMA200 405.11`.

## Where this departs from the brief, and why

- **Hourly bars are rebuilt from 30-minute bars.** Alpaca's `1Hour` bars are clock-aligned, so the
  09:00 bar mixes pre-market prints into the open. The bot uses 09:30, 10:30, ... 15:30 bars
  (the last one is 30 minutes), regular hours only.
- **The EMA50 exit arms only after a close above EMA50 since entry.** Pullback entries often fire
  while price is still under EMA50, so the literal rule would exit one bar after entering. Set
  `ema_exit_requires_arming = False` in `config.py` for the literal rule.
- **The bracket is GTC, not DAY.** DAY legs expire at the close and leave the position unprotected
  overnight.
- **"Walk-forward" means a strictly causal simulation reported per chronological fold.** The
  parameters are fixed, so nothing is optimized in-sample; the folds show whether results are
  stable over time or come from one period.

## Things you should know before trusting the results

- **The 1% risk target almost never applies.** A 2x hourly-ATR stop is roughly 0.5 to 2.5% of price,
  so 1% risk would need 40 to 200% of equity in one position. The 20% notional cap binds first, and
  the real risk per trade is roughly 0.1 to 0.5% of equity. The journal records the binding
  constraint on every order.
- **Three weeks will not prove the edge.** These filters trade rarely. For a real estimate, divide the
  backtest's trade count by ~35 (the number of 3-week windows in 2 years); expect single digits to
  low teens. `evaluate_trust.py` therefore answers two questions separately:
  operational trust (coverage, latency, duplicates, slippage, risk limits, replay parity), which
  3 weeks can establish, and statistical edge, which needs about 30+ closed trades and will
  usually say `INCONCLUSIVE`. It only says `READY` when both pass.
- **The universe is one bet.** SPY, QQQ, AAPL, MSFT and NVDA are highly correlated; five open
  positions behave more like one position of 5 times the size.
- **IEX data is the default** because the free plan only serves real-time bars from IEX. IEX carries
  a few percent of volume, so its bars differ slightly from consolidated (SIP) bars. Backtest on the
  same feed you trade on (the default), or compare both with `--feed sip`.
- **Paper fills are optimistic.** Alpaca's simulator does not model queue position or impact, and
  gaps through a stop fill worse than the stop in real markets. The backtest charges 0.05% per side
  on every fill and fills gapped stops at the open.
- **Early-close days** (for example the day after Thanksgiving) are not modelled: the bot simply
  does not trade the shortened afternoon.
- **The offline tests use synthetic data.** They prove the plumbing (no look-ahead, filter logic,
  sizing, kill-switch, live/backtest parity, order idempotency), not that the strategy makes money.
  Run `backtest.py` on real data before reading anything into performance.

## Layout

```
config.py              universe, timeframe, strategy / risk / execution / backtest parameters
src/indicators.py      EMA, Wilder RSI / ATR / ADX, normalized ATR percentiles (vectorized, causal)
src/strategy.py        rules, shift(1) decisions, reason codes and text
src/risk_manager.py    sizing, daily kill-switch, idempotency gates, deterministic client_order_id
src/sessions.py        session-aligned bars, ET trading window, bar schedule
src/alpaca_client.py   alpaca-py wrapper: retries with backoff, data, bracket orders, fill lookup
src/journal.py         append-only CSV journal (fixed schema, fsync per row)
bot.py                 live loop: kill-switch, reconciliation, per-bar decisions, orders
backtest.py            portfolio backtester and tear sheet
evaluate_trust.py      go/no-go report from the journal
start_mac.sh           macOS launcher
tests/                 offline pytest suite with the synthetic multi-regime generator
```

Safety rails in the bot: it refuses to start unless `ALPACA_PAPER=True`; entry order ids are
deterministic per symbol and bar, so retries and restarts cannot double an entry; the broker is
the source of truth on restart (positions, working stop legs, fills while the bot was off).
