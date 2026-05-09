# krypt-sim

A historical simulation funnel for crypto trading strategies. Built to honestly
filter strategy ideas before they reach paper trading on live ticks.

This is the upstream funnel for the krypt-agent project. Strategies that
survive here become candidates for live paper validation; strategies that
fail here die cheaply.

## Design principles

1. **Brutal honesty over flattering numbers.** The engine subtracts realistic
   fees (0.10% per side, MEXC taker default) and slippage (5 bps default) on
   every fill. A strategy that looks great with zero costs but loses with
   realistic costs is dead.

2. **No look-ahead.** Signals computed at bar close execute at next bar's open.
   The engine handles the shift; strategies cannot peek.

3. **Survivors must work in every regime, not on average.** The grid search
   ranks by worst-case performance across (symbol, year) slices, not by
   average. A config that hits +200% in 2021 but -50% in 2022 is overfit.

4. **Backtest results are candidates, not winners.** Anything that survives
   here must still go through live paper validation on real ticks.
   Candle-data backtests are not the truth — they're a sieve.

5. **Realistic targets, not impossible ones.** A strategy that produces
   0.1–0.3% per day net of fees on a single instrument is exceptional.
   Portfolio-level returns come from stacking uncorrelated strategies.

## Layout

```
krypt-sim/
├── src/
│   ├── data/binance_vision.py    # data loader, S3 cache as parquet
│   ├── engine/backtest.py        # vectorized backtest with realistic costs
│   ├── strategies/
│   │   ├── base.py               # Strategy ABC
│   │   ├── donchian.py           # channel breakout (momentum)
│   │   ├── zscore_rev.py         # mean reversion on z-score extremes
│   │   ├── squeeze.py            # volatility expansion (range squeeze breakout)
│   │   └── volume_anomaly.py     # volume spike directional follow-through
│   ├── runner/
│   │   ├── grid_search.py        # parameter sweep w/ brutal-filter ranking
│   │   └── walk_forward.py       # out-of-sample rolling validation
│   └── reports/                  # (placeholder for leaderboard tooling)
├── tests/
│   ├── sanity.py                 # engine validation (always-flat, B&H, perfect-foresight)
│   ├── end_to_end.py             # one-strategy full-pipeline smoke test
│   └── family_comparison.py      # all 4 families on 3 pairs
├── results/                      # CSV outputs (gitignored)
└── requirements.txt
```

## Setup

```bash
git clone <this-repo> krypt-sim
cd krypt-sim
python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

### 1. Verify the engine is sane

```bash
PYTHONPATH=src python3 tests/sanity.py
```

This downloads 6 months of BTCUSDT 1h, then runs:
- always-flat strategy (must produce zero return, zero trades)
- always-long strategy (must approximately match buy-and-hold)
- random strategy (proves the engine isn't accidentally generous)
- perfect-foresight strategy (upper bound; proves the engine can detect edge)

If any of these look wrong, do not trust downstream results.

### 2. Run all 4 families on 3 pairs

```bash
PYTHONPATH=src python3 tests/family_comparison.py
```

Loads BTCUSDT, ETHUSDT, SOLUSDT (1h, 2024). Runs Donchian, ZScoreRev,
Squeeze, and VolumeAnomaly with reasonable default params. Prints a grid.

### 3. Grid search a single family

```bash
PYTHONPATH=src python3 -m runner.grid_search \
    --family donchian \
    --pairs BTCUSDT ETHUSDT SOLUSDT AVAXUSDT LINKUSDT \
    --years 2021 2022 2023 2024 \
    --interval 1h \
    --workers 16 \
    --output results/donchian_grid.csv
```

The search is parallelized across `--workers` processes. Output CSV has one
row per (param-combo, symbol, year). The end-of-run summary ranks combos by
**worst-case Sharpe across slices** — this is the brutal filter.

Param grids are defined in `src/runner/grid_search.py:GRIDS`. Edit there to
narrow or widen the search.

### 4. Walk-forward validation on a survivor

After grid search, take a top survivor's params and validate out-of-sample:

```bash
PYTHONPATH=src python3 -m runner.walk_forward \
    --family donchian \
    --params '{"entry_n":100,"exit_n":50,"long_only":true,"regime_ma":0}' \
    --pairs BTCUSDT ETHUSDT SOLUSDT \
    --interval 1h \
    --start 2021-01-01 \
    --end 2025-01-01 \
    --test-months 3 \
    --step-months 3 \
    --output results/donchian_wf.csv
```

A config that survives here is a real candidate for paper trading.

## Data

`src/data/binance_vision.py` pulls from Binance's public S3 bucket
(no API key, no rate limits, deepest free history). First call for any
(symbol, interval, month) downloads the zip and caches as parquet under
`~/.krypt-sim/binance_vision/`. Subsequent reads are local.

Storage budget: ~30 pairs × 5 years × 12 months × 1h interval ≈ 0.5 GB.
At 1m interval, ≈ 30 GB. Plan accordingly.

## Strategy families and what they exploit

| Family            | Edge premise                                         | Wins in  | Loses in       |
|-------------------|------------------------------------------------------|----------|----------------|
| Donchian          | Trend continuation after channel breakout            | Trend    | Chop, reversals|
| ZScoreRev         | Price extremes mean-revert toward rolling mean       | Chop     | Strong trends  |
| Squeeze breakout  | Compressed vol expands directionally                 | Regime shifts | Persistent vol |
| Volume anomaly    | Volume spikes signal information arrival, follow it  | Event-driven moves | Quiet markets |

These are deliberately orthogonal. The portfolio thesis is that a stack of
uncorrelated strategies should produce smoother equity than any single one.

## Engine cost model

- **Fees:** `fee_per_side` (default 0.001 = 0.10% MEXC taker) charged on every
  position change, scaled by turnover. A 0→1 entry pays 1×, a 1→-1 flip pays
  2×, a 1→0 exit pays 1×.
- **Slippage:** `slippage_bps` (default 5 bps = 0.05%) per side, additive to fees.

To experiment with a maker-only model, pass `--fee 0` (and accept that this
assumes a more complex order-routing system you'd still need to build).

## What this is *not*

- **Not a live trading system.** It only simulates against historical candles.
- **Not authoritative.** Candle-close backtests are a useful sieve, not a verdict.
- **Not a guarantee.** A strategy passing every filter here can still fail
  live for reasons (slippage realism, exchange depth, latency, regime change)
  the simulator can't fully model.

## What's intentionally missing

- **No optimizer.** Grid search is exhaustive; we don't fit per slice. That
  prevents the worst kind of overfitting (per-window optimization).
- **No automatic position sizing / Kelly / risk parity.** Position size is
  a fixed fraction of equity. Portfolio construction is downstream of this tool.
- **No multi-asset cross-correlation.** Each backtest is single-asset.

## Roadmap (your call)

- More strategy families (funding rate skew overlay, time-of-day,
  cross-asset momentum)
- Portfolio simulator (combine N validated strategies, simulate equity together)
- Integration with krypt-agent's paper trading pipeline (export survivors as
  configs the live engine can consume)
