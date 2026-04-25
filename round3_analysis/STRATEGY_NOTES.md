# Round 3 — Voucher MM Investigation

## Goal
User reported that competing teams make ~$20K/day from each of `VEV_5100`, `VEV_5200`, `VEV_5300` on the official IMC backtester (no other strikes touched). Reverse-engineer how, then beat or match it.

## Microstructure (Prosperity4Data/ROUND_3, days 0-2)

Each strike has a tight, well-defined market:

| Voucher | Median spread | L1 size (each side) | Lag-1 ACF (Δmid) | Trades / day in CSV |
|---|---|---|---|---|
| `VEV_5100` | 4 ticks | ~19 | -0.09 | 0-2 |
| `VEV_5200` | 3 ticks | ~22 | -0.14 | 3-8 |
| `VEV_5300` | 2 ticks | ~20 | -0.21 | 37-45 |

The order book has fixed structure but the trades CSV is sparse — the IMC matcher fills our passive orders only when (a) we cross the standing book or (b) a trade prints at/through our quote price. With only 50-200 trade lots/day per strike, no purely passive MM strategy can clear $20K/day from the IMC backtester directly on these vouchers.

## Strategies tested on the IMC backtester (`PYTHONPATH=imc-prosperity-4-backtester python3 -m prosperity4bt …`)

| Strategy | 3-day total | Notes |
|---|---|---|
| `strat_take_book` (cross both sides) | -$8.1M | Pays the spread every tick. Predictable disaster. |
| `strat_inside_wall` (post bid+1/ask-1 only) | +$4.4K | Catches the few trades that print through us. |
| `strat_take_then_make` | +$4.4K | Same: take never fires, MM does the work. |
| `strat_voucher_mm_v2` (skewed MM) | +$4.3K | Matches `inside_wall`. |
| `strat_mid_meanrev` (Δmid mean-rev) | -$0.96M | Bid-ask bounce ≠ tradable signal. |
| `strat_short_5300` (always short rich strikes, hedged) | -$13K | Smile theo is biased; market stays rich. |
| `strat_frankfurt_iv_scalp` | -$117 | Switch_mean stays below threshold; barely trades. |
| `strat_theo_take` | -$12K | Same — theo bias kills it. |
| `strat_dev_meanrev` (Z-score mid vs EMA) | -$1.2M | Very heavy bleed from over-trading. |
| `strat_cross_book` (Δmid reverse take) | -$1.3M | Fights the trend. |

**Net:** No voucher-only strategy I built clears more than ~$5K total over 3 days from the IMC backtester. The "$20K/day/strike" claim is most likely measured against a **modified backtester that simulates passive fills against the next-tick book** (e.g. a community fork that assumes hidden liquidity), or refers to live competition where house bots actively cross quotes posted inside the wall.

## What actually wins on the IMC backtester

`strat_combined_v3` (Hydrogel mean-rev + voucher inside-wall MM + delta hedge + underlying MM):

```
Round 3 day 0: 28,784
Round 3 day 1: 23,670
Round 3 day 2: 24,900
Total profit:  77,355   (~$25.8K / day)
```

Per-symbol on day 2 (representative):
- `HYDROGEL_PACK`: +$23,629   (mean-rev + MM)
- `VEV_5300`: +$1,973
- `VEV_5200`: +$944
- `VELVETFRUIT_EXTRACT`: -$1,646  (delta hedge cost)
- `VEV_5100`: $0   (4-tick spread; no fills in CSV)

So in IMC backtester reality: **Hydrogel is the dominant edge (~$22-24K/day)**, vouchers add ~$2-3K/day total. The voucher MM is positive PnL with capped downside, so worth keeping.

## Tuning grid results

Hydrogel (winner):
- `HYDRO_BAND = 10` → $73K (vs $70K @ 20)
- `HYDRO_SKEW`: doesn't move PnL (taker fires first)

Vouchers:
- `VOUCHER_SIZE = 100`, `VOUCHER_SOFT_CAP = 250`, `VOUCHER_SKEW_PER100 = 2` → $77,355 (best by ~$91)
- All sizes 50-300 produce nearly identical PnL because trades.csv volume bottlenecks fills.

Adding `_mm_underlying` (passive inside-wall on Velvet) added ~$5K (covers part of the delta-hedge cost).

## Files

- `strats/strat_*.py` — every candidate I tested
- `strat_combined_v3.py` (= `round3_bot.py`) — final production bot
- `realistic_backtest.py` — alternative sim that fills passive against next-tick book; gives ~$9K on the inside-wall MM (vs $4K on IMC)
- `grid_tune.py`, `grid_voucher.py` — parameter sweeps

## Recommendation

For the IMC backtester ranking: ship `round3_bot.py` (= `strat_combined_v3.py`).

If competing teams really do make $20K/strike/day in their backtests, they're either:
1. Using `--match-trades all` on a different data set with much heavier trade flow,
2. Running a community backtester that fills passively against the standing book each tick (functionally equivalent to assuming infinite hidden depth at our quote), or
3. Reporting live competition results (where IMC's house bots actively cross posted quotes).

I cannot replicate that PnL on the as-shipped `prosperity4bt` backtester with the as-shipped `Prosperity4Data/ROUND_3` CSVs. If you have a link to whichever modified backtester / data source those teams use, I can adapt the strategy to maximize PnL there.
