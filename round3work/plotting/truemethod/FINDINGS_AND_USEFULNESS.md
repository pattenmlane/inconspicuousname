# True FV vs tape mid — findings (Round 3)

## What we did

- **`truemethod/wind_down/`** — IV smile, residuals, and summaries using **`true_fv`** from
  `round3work/fairs/**/*_true_fv_day39.csv`, merged on `timestamp`, with **DTE = 5 days** at
  session open (Round 3 final sim per `round3description.txt`) and **intraday winding**
  (same fractional session decay as the historical Frankfurt convention).
- **`truemethod/no_wind_down/`** — same pipeline with **T = 5/365** for the whole session
  (no intraday decay).
- **Original method split** — see `plotting/original_method/README_WINDING_LAYOUT.txt`:
  **`wind_down/`** symlinks to the existing `combined_analysis` + `pervoucher_analysis`
  (those paths **do** wind). **`no_wind_down/`** is a patched copy where `t_years_effective`
  ignores timestamp for T.

## Data caveat (important)

Hold-1 probes are **day 39** sandbox exports, **not** historical CSV days 0–2. Each voucher
(and extract) was a **separate** upload; timestamps **do** align (1000 common ticks in this
repo). That makes a clean cross-section for smiles, but it is **one session slice**, not the
full three-day historical tape.

## Is true FV useful?

**Yes, for some questions; limited for others.**

| Useful | Less useful |
|--------|-------------|
| Checking whether **IV / smile fits** are driven by **spread noise in mids** vs the engine’s mark | Replacing historical **days 0–2** analysis entirely — different calendar slice |
| Calibrating **PnL-consistent** “fair” for sim-style bots | Live **liquidity-taking** models that must fill against **actual** bid/ask |
| Spotting **systematic mid − fv** gaps by strike (inventory / fee / model effects) | Assuming all 10 probes saw **bit-identical** underlying paths (we only verified timestamp equality) |

### Mid vs true FV (mean |fv − mid| over the session)

See `fv_vs_mid_mean_abs_gap.csv` in each branch’s `combined_analysis/`. On this bundle,
gaps are **~0.13–0.30** on most vouchers (extract ~0.27); deep OTM / wings can read **~0.5**
where the book is wide and `mid_price` is a poor summary of economic value.

### Wind vs no-wind on **true FV** (global quadratic in \(m_t=\log(K/S)/\sqrt{T}\))

From `true_fv_iv_meta.json`:

| Mode | Global RMSE (pooled \(m_t\) fit) | Notes |
|------|-----------------------------------|--------|
| wind_down | ~0.01923 | Coeffs shift slightly vs no-wind |
| no_wind_down | ~0.01917 | Marginally lower RMSE here (T a bit larger on average → smoother IV surface) |

Coefficients differ only at the **1e-3** level between the two modes on this tape — **not**
a dominant effect compared with switching **mid → fv** or changing the calendar slice.

## Should our approach change?

- **Smile / IV research on historical 0–2** — keep mids (or walls) if the goal is **market
  microstructure**. Add **parallel FV-based** curves when the goal is **agreement with the
  sim’s internal mark** (e.g. explaining inventory PnL, or tuning theo bands).
- **Frankfurt-style IV scalping** — the exchange still matches on **visible** prices; FV is
  best used as a **diagnostic** (is `wall_mid − theo` chasing noise?) not as the only price
  input unless you confirm fills against fv in the engine (they do not).
- **Voucher work / global fit** — refitting coeffs on **fv** would align theo with **marks**
  you care about for sim PnL; refitting on **mids** aligns with **tape** you can trade.

**Bottom line:** true FV is **useful** for reconciliation and for **second opinions** on IV
surfaces; it does **not** obsolete mid-based work for execution-focused modeling. Wind vs
no-wind matters more for **absolute IV level** at long horizons; here (5d) the wind vs
no-wind split is **second-order** relative to the fv vs mid choice.
