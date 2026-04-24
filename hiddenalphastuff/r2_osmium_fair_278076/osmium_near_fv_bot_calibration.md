# Osmium Near-FV Bot Calibration — ASH_COATED_OSMIUM Round 2

## Method

Same as inner / wall osmium bots. **Identification** matches tomato Bot 3: any book level with **continuous** offset **|price − true_FV| ≤ 4** (inside the inner MM’s ±8 rungs in tick space, but excluding those far levels by construction: MM sits near **±8…11**, so this band only sees **residual** depth).

### Inputs

- `prices_round_*_day_*.csv` + `osmium_true_fv.csv` in each session folder (semicolon CSV).

### Analysis

```bash
python3 analyze_osmium_near_fv.py --all-sessions   # → per-session txt + osmium_near_fv_analysis_MULTISESSION.txt
python3 validate_osmium_near_fv.py --all-sessions
python3 run_all_osmium_session_checks.py
```

| Script | Role |
|--------|------|
| `analyze_osmium_near_fv.py --all-sessions` | Same metrics per session; **combined** report in **`osmium_near_fv_analysis_MULTISESSION.txt`**. |
| `validate_osmium_near_fv.py --all-sessions` | Per-session validation + **pooled** ring χ² and side **z** on merged events. |
| `osmium_near_fv_exact_rule.py --all-sessions` | δ = `price − round(FV)` breakdown, **fv_frac × δ** count grid, crossing×δ, pooled ring χ² (same spirit as `bot1_exact_rule.py`). |

### Key findings (vs tomato Bot 3)

Tomato Bot 3 (tutorial JSON, `bot3_calibration.md`):

- Rare (~**6.3%** of timestamps), almost always **1 tick** runs (~92% length-1).
- **Never both sides** same timestamp.
- Price: **`round(FV) + uniform{-2,-1,0,+1}`** (≈25% each).
- **Volume | crossing**: crossing larger **U(5,12)** mean ~8.1; passive **U(2,6)** mean ~4.2.
- Side ~**50/50** (not significant dev. from fair coin).

**Osmium** (two 1000-tick fair probes: **278076**, **248329**):

| Property | 278076 | 248329 | Pooled read |
|----------|--------|--------|----------------|
| Timesteps with ≥1 near-FV level | 79/1000 (**7.90%**) | 77/1000 (**7.70%**) | ~**7.8%** — same order as tomato |
| Total level-events | 80 | 79 | 159 |
| Timesteps with **both** bid and ask near-FV | **0** | **0** | Same as tomato: **single-sided** at snapshot level |
| Events per ts | 78×1 + 1×2 | 75×1 + 2×2 | Almost always **one** extra level |
| ON-run length 1 | **93.2%** | **93.1%** | Matches tomato “ephemeral” |

**Price law — different from tomato**

- Tomato’s **{-2,-1,0,+1}** uniform is **rejected** on osmium (χ² ≫ 7.815): almost no mass at **`round(FV)`** (δ=0) and little at ±1 compared to **±2, ±3**.
- Restricting to the **ring** **δ ∈ {-3, -2, +1, +2}** (where almost all mass lives), **uniform on those four offsets is not rejected**:
  - 278076: χ² = **4.36**, df = 3, n = 78  
  - 248329: χ² = **3.03**, df = 3, n = 78  
  - **Pooled** merged events: χ² = **2.82**, df = 3, n = **156** (`validate_osmium_near_fv.py --all-sessions`)

So osmium is **not** “pick one of four ticks symmetric about round(FV)” in the tomato way; it is closer to **“pick one of four ticks in a skirt around FV, avoiding at-the-money”** (δ = 0 almost absent).

**(Side, δ) is not independent**

Example **278076** joint counts (each row is one book level):

| (side, δ) | n |
|-----------|---|
| (bid, −3) | 15 |
| (bid, −2) | 9 |
| (bid, +1) | 3 |
| (bid, +2) | 5 |
| (ask, −3) | 5 |
| (ask, −2) | 17 |
| (ask, +1) | 10 |
| (ask, +2) | 14 |
| rare (δ ∈ {0,±3} on wrong tail) | 2 |

**Asks with negative δ** are common (e.g. ask at `round(FV)−3`): that is **not** explainable as “side fair coin × uniform δ” independence. For simulation, **`near_fv_quote_empirical`** resamples **(side, δ)** from this joint and reproduces the **crossing rate** (~37–40%) whereas an **independent** ring + side model overshoots crossing to **~49%** on the same FV path (see `validate_osmium_near_fv.py`).

### Key discovery: volume still depends on crossing (tomato parallel)

Tomato **Bot 3** markdown splits **events** into **crossing** (aggressive side of FV, larger vol) vs **passive** (smaller vol). Observed tutorial counts are **52 crossing / 73 passive** (~**42% / 58%** by event — sometimes described loosely as “~⅓ vs ~⅔” by **volume law** or eyeball, not a hardcoded third/two-thirds engine). Their **reference `bot3.py`** uses a **different** structural draw (`random.random() > 0.063` then side, etc.); treat **MD = measured behavior**, **code = one simulator**.

| Pool | Crossing n | Crossing vol mean [min,max] | Passive n | Passive vol mean [min,max] |
|------|--------------|-----------------------------|-------------|----------------------------|
| 278076 | 30 | **7.37** [4, 10] | 50 | **4.24** [2, 21] |
| 248329 | 32 | **7.41** [4, 10] | 47 | **3.64** [1, 13] |
| **Pooled** | **62** | **7.39** [4, 10] | **97** | **3.95** [1, 21] |

**Pooled crossing fraction (by event):** **62 / 159 ≈ 39.0%** — same qualitative **minority-crossing / majority-passive** split as tomato, not distinguishable from tomato’s ~42% without more data (binomial noise).

Crossing cluster matches **tight** band **[4, 10]** (tomato used **[5, 12]**). Passive is mostly **[2, 6]** but there are **outliers** (passive vol **> 6**: 2/50 and 1/47 on the two tapes) — same story as any thin “U(2,6)” law: rare fat tail or mis-tagged level.

**Volume | (side, crossing)** (278076): bid cross mean **7.62**, bid passive **3.75**; ask cross **7.27**, ask passive **4.69** — same qualitative split as tomato.

### Multiple “participant” bots?

On these slices **n ≈ 80** events per 1k ticks — too thin to separate **two** latent Markov participants with confidence. What we can say:

1. **Joint (side, δ)** is structured (**asks** use **negative δ** often) → not i.i.d. “coin × ring”.
2. **Pearson χ²** on coarse **sign(δ) × crossing** is **5.17** (df = 2) on 278076 — mild association; **0.88** on 248329 — no signal. Pooling more days is required before claiming **two** rule-based agents vs **one** richer rule.
3. **Presence vs FV fraction** (0.05 bins) fluctuates ~5–22% across bins with small **n per bin** — no stable time-of-day signal on 1k rows.

So: **one** empirical resampler for **(side, δ)** + **crossing-conditioned volume** is the honest first simulator; splitting into “two random bots” is **not** identified here.

---

## Interpretation

Same high-level read as tomato: a **rare, single-sided, short-lived** limit order **inside** the structured MM layers, with **larger size when price crosses FV** (aggressive-looking) and **smaller when passive**. Quantitatively, osmium’s **offset law differs** (skirt ring, not tomato’s ±2 box around `round(FV)`), and **side/offset are coupled**.

For strategy: still mostly **noise** at ~8% of timestamps and **one** level — same “don’t rely on it for fills” takeaway, but **do not** import tomato’s `{-2,-1,0,+1}` uniform into an osmium sim.

---

## Result

### Factorized model (marginals; crossing rate biased high — use for rough order-of-magnitude only)

```python
def near_fv_quote(fv, rng, p_tick=0.078, p_bid=0.45):
    if rng.random() > p_tick:
        return None
    side = "bid" if rng.random() < p_bid else "ask"
    delta = rng.choice([-3, -2, 1, 2])
    price = round(fv) + delta
    crossing = (side == "bid" and price > fv) or (side == "ask" and price < fv)
    vol = rng.randint(4, 10) if crossing else rng.randint(2, 6)
    return side, price, vol
```

### Recommended: empirical joint (matches crossing rate on replay)

```python
from collections import Counter
from osmium_near_fv_bot import near_fv_quote_empirical

# joint = Counter((e.side, e.delta) for e in events_from_reference_log)
# p_tick = n_timesteps_with_event / n_timesteps
# each tick: near_fv_quote_empirical(fv, joint, p_tick=p_tick)
```

Source: `osmium_near_fv_bot.py` (`near_fv_quote`, `near_fv_quote_empirical`, `fit_params_from_counts`).

---

### Validation (`validate_osmium_near_fv.py`)

| Metric | 278076 actual | Note |
|--------|----------------|------|
| Presence | 7.90% | Fitted `p_tick` matches |
| Side bid % among events | 40.0% | vs 50%: **z = −1.79, p ≈ 0.07** (marginal asymmetry **this day only**) |
| Ring δ uniform on {-3,-2,+1,+2} | χ² = 4.36, df = 3 | **Pass** vs uniform |
| Crossing rate | 37.5% | Empirical-joint sim mean **37.5%**; indep-ring sim **~49%** (reject indep model) |
| Crossing vol | mean 7.37, [4, 10] | Matches U(4,10) |
| Passive vol | mean 4.24, [2, 21] | Mostly [2,6]; **2** outliers >6 |

| Metric | 248329 actual | Note |
|--------|----------------|------|
| Presence | 7.70% | — |
| Side bid % | 50.6% | vs 50%: **z = 0.11, p ≈ 0.91** |
| Ring χ² | 3.03 | **Pass** |
| Crossing | 40.5% | Empirical-joint sim tracks actual |

**Pooled** (159 events): bid **45.3%** — **z ≈ −1.21, p ≈ 0.23** vs 50/50; consistent with a **symmetric** participant once **n** grows.

---

### Statistical tests (same style as tomato Bot 3 doc)

- **Side 50/50 (278076):** borderline **p ≈ 0.07** — do not hardcode 40/60; use **50/50** or **empirical joint**.
- **Tomato δ uniform {-2,-1,0,+1}:** **Rejected** (χ² ≈ 44.6 on 278076).
- **Osmium ring δ uniform {-3,-2,+1,+2}:** **Not rejected** on either slice.
- **Crossing:** independence of **side** and **δ ring** fails for crossing rate; use **joint** resampling.
- **Crossing rate vs a toy “⅓ crossing” null:** pooled **62** crossing of **159** events — not useful to nail to **⅓** vs **½** at this **n** (wide binomial CIs); same caveat as tomato’s **z = −1.63, p = 0.10** vs their simulator’s ~49% crossing.

---

### Near-FV bot properties

- **Rare** — ~7.8% of timesteps (two-session average) (tomato ~6.3% parallel).
- **Ephemeral** — ~93% of presence runs length **1** in file order (tomato ~92%).
- **Single-sided** — **0** timestamps with both a near-FV bid and ask on these exports (tomato: never both).
- **Near FV** — |cont.| ≤ 4; discrete offsets concentrate on **−3, −2, +1, +2** vs `round(FV)` (tomato used **{-2,…,+1}** uniform).
- **Event mix** — **~39% crossing / ~61% passive** pooled (tomato tutorial **~42% / ~58%**); both show **larger vol when crossing** (tomato parallel).
- **Volume** — crossing mean **~7.4** on **[4, 10]**; passive mean **~3.9**, mostly **[2, 6]** with rare high outliers (tomato crossing **[5,12]**, passive **[2,6]**).
- **No memory / weak patterns** at this **n** — same “noise participant” read as tomato Bot 3 properties.
- **Not tomato’s exact price rule** — different δ support; **empirical (side, δ)** for faithful replay.

---

### Files

- `osmium_near_fv_events.py` — event iterator, optional MM-offset exclusion.
- `osmium_near_fv_bot.py` — simulators.
- `analyze_osmium_near_fv.py` — writes `osmium_near_fv_analysis.txt`; **`--all-sessions`** → **`osmium_near_fv_analysis_MULTISESSION.txt`** too.
- `validate_osmium_near_fv.py` — **`--all-sessions`** for both folders + pooled stats.
- `osmium_near_fv_exact_rule.py` — **`--all-sessions`** per session + pooled δ / crossing summary.

Run from `hiddenalphastuff/r2_osmium_fair_278076/`:

```bash
python3 analyze_osmium_near_fv.py --all-sessions
python3 validate_osmium_near_fv.py --all-sessions
python3 osmium_near_fv_exact_rule.py --all-sessions
python3 run_all_osmium_session_checks.py   # inner + wall + near-FV + quote-rule bundle
```
