# IMC Prosperity 4 — Round 3 Analysis & Strategy

Analysis of `Prosperity4Data/ROUND_3` (CSV days 0-2) reproducing the conversation
between the user and Claude, plus a stdlib-only production bot and a backtest harness.

## Products

| Product | Pos limit | Notes |
|---|---|---|
| `HYDROGEL_PACK` | 200 | Mean-reverting around 9990, std 32. **Independent** asset. |
| `VELVETFRUIT_EXTRACT` (S) | 200 | Underlying for the 10 vouchers. |
| `VEV_4000` … `VEV_6500` (10 strikes) | 300 each | European calls, DTE 5 at start of round 3. |

## Headline findings (reproduced)

1. **Magritte hint pays off.** `HYDROGEL_PACK` looks grouped with the option products
   but corr(d HYDROGEL, d VELVETFRUIT) = **+0.0059** (n=29,999). It's a separate book.
2. **Smile is stable.** Quadratic IV smile in m_t = log(K/S)/sqrt(T):
   - day 0: a=0.150, b=-0.0076, c (ATM) = 0.234
   - day 1: a=0.146, b=-0.0049, c = 0.235
   - day 2: a=0.136, b=+0.0041, c = 0.237
3. **Per-strike residuals are persistent across days** (matches Claude's table):

   | K | resid_iv (vol pts) | side suggested | $ edge / contract @ S=5255, T~6d |
   |---|---|---|---|
   | 5000 | -0.014 | BUY | $0.94 |
   | 5100 | -0.003 | BUY | $0.54 |
   | 5200 | +0.006 | SELL | $1.47 |
   | 5300 | +0.009 | SELL | $2.30 |
   | 5400 | -0.012 | BUY | $2.13 |
   | 5500 | -0.003 | BUY | $0.28 |
   | 4000, 4500 | n/a | intrinsic-pinned | use intrinsic-band logic |
   | 6000, 6500 | n/a | $0.5 floor (vega ≈ 0) | skip |

4. **Realized vs implied vol — variance ratio:** lag-1 realized vol comes out to 41 %
   (bid-ask bounce, lag-1 ACF = -0.16). At lag 1000 it converges to **24.3 %**, matching
   the ATM IV of ~24 %. The headline IV level is **fair**; only the per-strike skew is mispriced.

   ```
      lag   ann_vol   lag1_acf
        1   0.4116    -0.16
        5   0.3559     0.73
       10   0.3481     0.86
       50   0.3456     0.97
      100   0.3332     0.98
      500   0.2843     1.00
     1000   0.2431     1.00
   ```

5. **Hydrogel mean-reversion is enormous.**
   `HYDROGEL_PACK` mean=9990.81, std=31.94, lag-1 ACF on changes = -0.13.
   Buy-when-mid <= mu-20, hold 500 ticks → avg gain **$+19.96** per fill, n=8,306.

6. **Deep-ITM (4000, 4500) trade essentially at intrinsic** (mean extrinsic ≈ +$0.01,
   std ~$0.8). Pure intrinsic-band arb fits.

## Strategy (round3_bot.py)

Two completely independent books:

### Book 1 — `HYDROGEL_PACK`
- Frankfurt-style passive market-make inside the wall (Bot A spread is ±8, Bot B ±10/11).
- Inventory skew: passive fair shifts down by `0.04 * pos`.
- Aggressive **mean-reversion taker**: when wall_mid is ≥20 below the long-term mean,
  fire `max_buy` at a price walking toward the mean; symmetric for the short side.
- Above all, a Frankfurt-style "take any ask ≤ wm-1, hit any bid ≥ wm+1" loop runs first.

### Book 2 — `VELVETFRUIT_EXTRACT` + 10 VEV
- Quadratic IV smile constants frozen from days 0-2.
- For each strike: theo = BS(S, K, T, smile_iv(S,K,T)).
- **Deep ITM (4000, 4500):** intrinsic-band arb, 1.5-tick threshold, capped at 30 lots.
- **Deep OTM (6000, 6500):** untradeable (pinned at $0.5 floor); skipped.
- **Near-money (5000-5500):** smile-relative speculative trades are **disabled by default**
  (`VOUCHER_TAKES_ENABLED = False`). Backtest showed the smile-residual edge of
  ~$0.5–2/contract is too small to overcome holding-period drift; every lit strike bled
  $300–1,500 over 3 days. Code path is preserved for live tuning if the live spread
  is wider than historical.
- **Underlying market-making:** narrow passive quotes one tick inside the Bot A wall,
  with a 2 % inventory skew. Cross-the-wall takes only fire if a quote pierces the
  full Bot B wall.
- **Delta hedge:** when net |voucher delta| > 8, sweep underlying at the touch.

## Backtest result (Prosperity4 ROUND_3, days 0-2 sequential)

| Sleeve | PnL |
|---|---|
| HYDROGEL_PACK | **+$49,491** |
| VELVETFRUIT_EXTRACT | $0 (passive only; 0 fills in the simple sim) |
| Vouchers (intrinsic arb only) | $0 (never triggered — already fairly priced) |
| **TOTAL** | **+$49,491** |

The simple harness only fills crossing orders, so passive Velvet quotes never get hit
even though they would in production. The Hydrogel result is from real crossing fills
(taker leg + mean-reversion taker).

## Files

- `01_smile_analysis.py` — Brent IV per (day,K,t), quadratic fits, per-strike residuals.
- `02_underlying_dynamics.py` — Hydrogel/Velvet correlation, mean-rev, variance-ratio sweep, deep-ITM check.
- `03_edge_sizing.py` — converts vol residuals to $/contract and sizes Hydrogel mean-rev.
- `round3_bot.py` — production-ready single-file bot, **stdlib only** (math.erf for normal CDF).
- `backtest.py` — minimal harness that walks the CSVs and feeds the bot.
- `ablation.py` — per-piece PnL attribution (use with caution: module reload is finicky).

## Tunables in round3_bot.py

| Constant | Default | What it does |
|---|---|---|
| `HYDRO_MEAN` | 9990 | Long-term mean for Hydrogel mean-rev taker |
| `HYDRO_BAND` | 20 | Distance from mean before we fire the layered taker |
| `VOUCHER_TAKES_ENABLED` | `False` | Flip to `True` to trade the smile-residual mispricings |
| `TAKE_DOLLAR_EDGE` | 2.50 | Required $ edge per contract above fair |
| `VOUCHER_HARD_LIMIT` | 60 | Per-strike position cap (vs hard 300 limit) |
| `INTRINSIC_BAND` | 1.5 | Deep-ITM intrinsic arb tolerance |
| `DELTA_HEDGE_TOL` | 8 | Net |delta| trigger for hedging |
