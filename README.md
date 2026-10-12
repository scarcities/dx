# Alpaca 1h Trend-Pullback Bot (paper validation)

A long-only, regime-filtered trend-pullback strategy for US equities on session-aligned hourly
bars, built on `alpaca-py`. The live universe is `SPY QQQ AAPL MSFT NVDA` (`core5`); a 12-symbol
`expanded12` universe exists for backtests only. The backtester, the live bot and the trust
evaluator run the same strategy, ranking, sizing, exit and kill-switch code.

## Quick start (macOS)

```bash
git clone <this repo> && cd dx
cp .env.example .env                                   # add your Alpaca PAPER keys
./start_mac.sh --dry-run      # first day: decisions are journaled, no orders sent
./start_mac.sh                # paper trading (Ctrl-C to stop)
```

`earnings_calendar.csv` is now part of the repository. If you already have your own untracked
copy, `git pull` will refuse to overwrite it: move yours aside first, pull, then merge any dates
you had that the committed file lacks.

`start_mac.sh` creates `.venv`, installs `requirements.txt` when it changes, runs the quick offline
tests (and refuses to start if they fail), then runs `bot.py` under `caffeinate -i -s`. That blocks
idle sleep, and system sleep while on AC power. **A MacBook with its lid closed still sleeps**, so keep
it open and plugged in from 09:30 to 16:00 ET.

| Command | What it does |
| --- | --- |
| `python backtest.py --feed sip --save-bars bars.csv.gz` | 2 years of bars from Alpaca, tear sheet, and a copy of the bars |
| `python backtest.py --from-bars bars.csv.gz --compare` | All rule variants, heat study, paired exit study, filter audit (no API calls) |
| `python backtest.py --universe expanded12 --feed sip --save-bars bars_x12.csv.gz --compare` | The same on 12 symbols, plus a core-universe-only row |
| `python backtest.py --from-bars bars.csv.gz --cash-yield 0.045` | Idle cash earns 4.5%; Sharpe becomes excess of it |
| `python backtest.py --from-bars bars.csv.gz --slippage-bps 5` | Same bars at a different cost assumption |
| `python tools/build_earnings_calendar.py --user-agent "Name you@example.com"` | Checks `earnings_calendar.csv` against SEC 8-K filings (see below) |
| `python bot.py --once` | One bot cycle, for a smoke test |
| `python evaluate_trust.py --replay` | After 2 to 3 weeks: the go/no-go report (it will not judge edge before 30 trades over 20 distinct weeks) |
| `pytest` | Full offline suite (a few minutes); `pytest -m "not slow"` is the quick one |

## v3: what changed and why

Defaults now match what you ran in the 4-year study: `exit_mode="ratchet"`,
`max_position_notional=0.33`, `SLIPPAGE_PER_SIDE=2 bp`. No indicator period or threshold changed.

| Change | Default | Status |
| --- | --- | --- |
| **Cross-sectional ranking**: when several symbols signal on the same bar, they are funded in order of `signal_rank_by` on `t-1` (`"natr"`, `"adx"`, `"rs_ema"` = Close/EMA200 - 1, or `None` = symbol list order); ties go by symbol name | `"natr"` | Deterministic and identical in backtest and bot. It only matters on bars where signals compete for capital: check the `unfunded` column first |
| **Gross-exposure cap** `max_gross_exposure`: entries are funded from `min(cash or buying power, equity x cap - open exposure)` | `1.0` | Fixes a live/backtest mismatch: the backtest was limited to cash while the bot could use Alpaca margin (5 x 33% = 165% gross) |
| **Two-pass live allocation**: exits and stop ratchets first, then entries ranked and funded with buying power re-read after the exits; entries wait up to `rank_wait_seconds` (120 s) for every symbol's bar | on | The bot now allocates exactly like the backtest (tested trade by trade on simultaneous signals) |
| **RS filter** `rs_filter`: enter only if Close/EMA200 - 1 on `t-1` is `>=` the benchmark's (`"vs_benchmark"`, SPY) or `>=` the universe median (`"vs_median"`) | off | Hypothesis. In `--compare` as portfolio rows and as pure filters in the entry-rules audit (random-drop test). Needs SPY in the bars; the median depends on which symbols are loaded |
| **Benchmarks**: SPY buy and hold over the same bars, cash at `--cash-yield`, Sharpe in excess of cash | on | Context the tear sheet lacked |
| **Heat study** in `--compare`: notional cap 20% / 33% / 50% x ranking on / off, with average gross exposure, share of trades cut by capital, and unfunded signals | on | Sizing evidence, not a rule change |
| **Universes**: `--universe core5` (live `SYMBOLS`) or `expanded12` (adds META AMZN GOOGL AMD AVGO NFLX JPM) | `core5` | Backtest only. The live list is unchanged until a held-out test supports it |

### Read this before acting on v3 results

- **2022-2024 is no longer out of sample.** The ratchet exit and the 33% cap were chosen after
  seeing it, and so will anything else you pick from it. Decide v3 settings on a window nobody has
  looked at: `--start 2019-10-01 --end 2022-09-30` (the calendar covers it), write down the rule you
  will accept before running it, and run it once.
- **Beat cash before beating v2.** A strategy that is in the market part of the time earns T-bill
  interest on the rest only if you model it (`--cash-yield`). T-bills paid roughly 4-5% a year
  through most of 2023-2025, so a 3-4% CAGR is below cash, and far below SPY buy and hold in a bull
  market. The tear sheet now prints both lines; compare against them, not against v1.
- **The evidence is still thin.** About 185 pooled trades with an expectancy CI that straddles
  zero. Detecting a +0.10R edge takes roughly 600 trades. `expanded12` adds seven stocks, so expect
  very roughly twice the trades, but they are correlated large caps, so it buys less independent
  evidence than the count suggests.
- **Ranking cannot create edge.** It decides which signal gets the money when two arrive together.
  `natr` favours the most volatile names (NVDA in 2023-2025): if it wins, check `w/o best` before
  believing it is the ranking and not one stock's rally.
- **Partial positions count as full trades.** When capital runs short the next candidate gets
  what is left (sometimes a few shares), and trade-level statistics (expectancy R, win rate) weigh
  it like a full position. The heat study's `BP-bound` column shows how often that happens; prefer
  portfolio return and Sharpe when they disagree with R.

## History: v1 and v2

The 2024-10 to 2026-10 SIP run of v1 lost money (81 trades, -0.182R, gross expectancy -0.093R);
one AAPL earnings gap was 66% of the loss. v2 added the earnings blackout (on), the bracket and
ratchet exits, an optional time stop, and two entry filters that later failed out of sample (cost
floor `max_cost_r`, ADX/DI at the pre-dip anchor `directional_check="anchor"`; both stay off). The
v2 live-safety work stays in place: failed exits are re-protected and retried with backoff, every
position gets a working stop within 60 s, stop replacement is never blindly retried.

## Strategy rules (v3 defaults)

All decisions for bar `t` use only closed bars `<= t-1` (`generate_signals` applies `shift(1)`).

1. **Trend**: `Close > EMA200` and `EMA50 > EMA200`.
2. **WAIT filters**: `CHOP_ADX_LOW` if `ADX14 < 20`; `VOLATILITY_DEAD` / `VOLATILITY_SPIKE` if
   `ATR14/Close` is below the 15th / above the 92nd percentile of the last 100 bars;
   `EARNINGS_BLACKOUT` inside an event window; `RS_LAGGARD` if the RS filter is on.
3. **Entry**: RSI14 crossed back above 42 on `t-1` after dipping below it, and `Close[t-1] > High[t-2]`.
4. **Exit**: GTC bracket, stop `Close[t-1] - 2 ATR`, target `Close[t-1] + 3 ATR` (1.5R). Ratchet:
   at +1R the stop moves to breakeven plus costs, then trails 2 ATR under the highest close.
5. **Allocation**: candidates on the same bar are ranked (NATR on `t-1`, highest first) and funded
   in that order. Each gets the smallest of 1% risk, 33% of equity, and the capital left under the
   100% gross cap; at most 5 positions.
6. **Kill switch**: a 3% daily loss vs the previous close's equity flattens everything and stops
   entries until the next session.
7. **When**: right after an hourly bar closes inside 09:45 to 15:45 ET (10:30, 11:30, ... 15:30),
   only on days the broker clock says the market is open.

Every decision is journaled with indicator values and the numeric reason, e.g.
`rank 1/2 by natr 0.0061, capital left $41,200 | ENTRY_PULLBACK_RECLAIM: ...`.

## Earnings calendar

alpaca-py has no earnings calendar, so `earnings_calendar.csv` holds one row per report:
`symbol,date,timing[,period,source]` with timing `AMC`, `BMO` or blank. For a report on day D the
bot stops opening new positions from the session before the last hold day, and closes open ones at
14:30 on the last hold day (D for after-close reports, the session before D otherwise). The live
bot refuses new entries in a stock with no date in the next 100 days, so **add each upcoming date
when the company announces it** (usually 2 to 4 weeks ahead).

`tools/build_earnings_calendar.py` checks the file against SEC EDGAR: every earnings release is
filed as an 8-K under Item 2.02, and the filing's acceptance time gives the date and BMO/AMC. It
prints every disagreement; `--write` appends past dates the file lacks. SEC requires a User-Agent
with your contact address (sent only to sec.gov). Late 8-K filings and preliminary-results 8-Ks
show up as differences: resolve them by hand.

## Other design notes

- **Hourly bars are rebuilt from 30-minute bars** (09:30, 10:30, ... 15:30; regular hours only).
- **Brackets are GTC**: DAY legs expire at the close and leave positions unprotected overnight.
- **Costs**: `SLIPPAGE_PER_SIDE` (2 bp) drives backtest fills and the optional cost floor. Measure
  it from the paper journal (`evaluate_trust.py`); Alpaca's paper fills are optimistic.
- **SIP vs IEX**: the free plan only gives real-time bars from IEX, so the live bot uses IEX.
- **Tests never read config defaults.** Every test pins its configs, and a guard test rejects bare
  `StrategyConfig()` / `RiskConfig()` calls, so changing a default in `config.py` cannot make
  `start_mac.sh` refuse to start. The offline tests prove plumbing (no look-ahead, filters, sizing,
  ranking parity, kill switch, earnings windows, failed-exit handling), not that the strategy makes money.

## Layout

```
config.py              universes, costs, strategy / risk / execution / backtest parameters
src/indicators.py      EMA, Wilder RSI / ATR / ADX, normalized ATR percentiles (vectorized, causal)
src/strategy.py        rules, shift(1) decisions, reason codes, stop ratchet, ranking, RS filter
src/events.py          earnings calendar and blackout windows
src/risk_manager.py    sizing, daily kill-switch, idempotency gates, deterministic client_order_id
src/sessions.py        session-aligned bars, ET trading window, bar schedule
src/alpaca_client.py   alpaca-py wrapper: retries with backoff, data, orders, exchange calendar
src/journal.py         append-only CSV journal (fixed schema, fsync per row)
bot.py                 live loop: kill-switch, reconciliation, stop coverage, ranked allocation
backtest.py            portfolio backtester, tear sheet, --compare harness, bar export/import
evaluate_trust.py      go/no-go report from the journal
tools/                 build_earnings_calendar.py (SEC EDGAR check)
start_mac.sh           macOS launcher
tests/                 offline pytest suite with the synthetic multi-regime generator
```

Safety rails in the bot: it refuses to start unless `ALPACA_PAPER=True`; entry order ids are
deterministic per symbol and bar (`tpb2-...`), so retries and restarts cannot double an entry; the
broker is the source of truth on restart (positions, working stops, fills while the bot was off).
