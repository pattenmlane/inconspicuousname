# Osmium Wall Bot (DEEP / ±10 & ±11) — ASH_COATED_OSMIUM Round 2

## Method

### Inputs required

Same as inner bot:

1. **True FV** — `true_fv(t) = profit_and_loss(t) + buy_price` on osmium lines from a one-share hold export.
2. **Order book** — Three bid / three ask slots per timestamp.

### Extraction

- `export_osmium_fair_log.py` → `prices_round_*_day_*.csv`, `osmium_true_fv.csv`.
- `discover_book_behaviors.py` — persistence and volume tables.

### Bot identification

| Side | Wall rung | Tag |
|------|-----------|-----|
| Bid | “inner” wall | `round(price − true_fv) = −10` |
| Bid | “outer” wall | `round(price − true_fv) = −11` |
| Ask | “inner” wall | `round(price − true_fv) = +10` |
| Ask | “outer” wall | `round(price − true_fv) = +11` |

Tie-breaking: among prices in the same offset bin, bid = **max** price, ask = **min** price.

---

## Analysis (both fair sessions)

Run everything on **both** exports (278076 + 248329) for confidence:

```bash
cd hiddenalphastuff/r2_osmium_fair_278076
python3 run_all_osmium_session_checks.py
```

Or individually:

| Script | Role |
|--------|------|
| `analyze_osmium_quote_rules.py --all-sessions` | Brute force per rung; writes `osmium_quote_rule_search.txt` in **each** session dir + **`osmium_quote_rule_search_MULTISESSION.txt`** here. |
| `validate_osmium_wall.py --all-sessions` | Price + volume χ² per session; **pooled** price counts + **merged** volume χ². |
| **`osmium_wall_exact_rule.py --all-sessions`** | **R = implied anchor** per rung vs `round(FV)`; **presence vs FV fractional part** (pooled + per session). |

---

## Key question: “−10 vs −11” — two price rules or one anchor + ladder?

Tomato **Bot 1** and **Bot 2** docs explain **price** via **FV fractional part** (which way `floor`/`ceil`/`round` snap). Here the situation is different:

### A) **Price** (when a rung is actually posted)

Define implied integer anchor from each posted price:

- Bid −10: **R₁₀ = bid_m10 + 10**
- Bid −11: **R₁₁ = bid_m11 + 11**
- Ask +10: **R₁₀ₐ = ask_p10 − 10**
- Ask +11: **R₁₁ₐ = ask_p11 − 11**

On **both** sessions, for **every** tick where that rung exists:

- **R = round(true_FV)** matches with **zero** misses (same as inner bot’s anchor style).
- On snapshots where **both** bid wall rungs exist, **R₁₀ = R₁₁ = round(FV)** (too rare on a single 1k slice to count; **pooled** still supports one anchor — see `osmium_wall_exact_rule.py` when both legs co-occur).

So **−10 and −11 are not “pick floor for one and ceil for the other”** at the price level; they are **two fixed tick offsets from the same `round(FV)`**, exactly like writing two lines on a ladder from one mid.

### B) **Which rungs appear** (presence / partial ladder) — **does** depend on FV fraction

`osmium_wall_exact_rule.py` tabulates, per **FV fractional bin** (0.05 steps), the fraction of ticks where each rung exists. **Pooled 2000 timesteps** show a strong **regime** pattern (not independent coin flips):

| FV frac bin (pooled) | n ticks | bid −10 | bid −11 | ask +10 | ask +11 |
|----------------------|---------|---------|---------|---------|---------|
| 0.00 | 114 | 44.7% | 35.1% | 30.7% | 47.4% |
| 0.05 – 0.45 | ~100 each | **~78–85%** | **~0%** | **~0%** | **~73–86%** |
| 0.50 | 91 | 41.8% | 42.9% | 42.9% | 27.5% |
| 0.55 | 76 | **0%** | **76.3%** | **77.6%** | **0%** |

**Reading:** in the middle fractional band, the book often shows **bid −10** and **ask +11** (one “outer” on each side) **without** the complementary **bid −11** / **ask +10** rungs. Near **frac ~ 0.55**, the pattern **flips** (bid −11 and ask +10 dominate; −10/+11 sparse). That is **which depth slots the ladder fills**, driven by **FV region** — analogous in spirit to tomato’s “fractional part → which rounding branch”, but here it governs **posting / ladder state**, not two different **price** anchors.

**Caveat:** counts per 0.05 bin are modest (~76–114); treat as **hypothesis** until a longer tape or second product confirms.

### Continuous `price − true_FV` when a rung posts (tomato “offset mean/std” parallel)

Tomato **Bot 1** documents **which rungs exist** via validation counts, but “Bot 1 Properties” still gives **mean/std of offset from FV** on those levels (~±7.75, ~0.45). Here, **presence** is the binned/regime story above; **price** is still four fixed ticks from **R**. On **pooled** fair exports, among ticks where each wall slot is non-empty, the **continuous** offset from `true_fv` is tightly clustered (FV drifts inside the tick):

| Rung | n | mean(`price − FV`) | stdev |
|------|---|---------------------|-------|
| Bid −10 | 815 | **−10.24** | **0.15** |
| Bid −11 | 772 | **−10.76** | **0.14** |
| Ask +10 | 783 | **+10.25** | **0.14** |
| Ask +11 | 801 | **+10.76** | **0.14** |

So: **bins** for *which ladder slots fire*; **tight continuous offset** (mean near −10/−11/+10/+11 with ~0.14 stdev) for *where inside the tick the continuous FV sits when that slot is filled*.

---

### Transfer function — **price** (same as tomato-style single R)

Let **R = round(true_FV)** (Python `round`; same as inner bot on these logs).

| Rung | Price (always, when that rung is posted) |
|------|------------------------------------------|
| Bid −11 | **R − 11** |
| Bid −10 | **R − 10** |
| Ask +10 | **R + 10** |
| Ask +11 | **R + 11** |

There is **no** separate “use 10 vs 11” **price** rule beyond “that slot exists”; both use the **same** **R**.

### Transfer function — **presence** (empirical; for simulation)

Use measured **P(rung | FV frac bin)** from pooled `osmium_wall_exact_rule.py` output, or a simple **two-regime** Markov model fit to data — **not** a second `round`/`floor` formula for price.

---

## Result

```python
R = round(true_fv)
bid_m10 = R - 10   # when −10 rung is posted
bid_m11 = R - 11   # when −11 rung is posted
ask_p10 = R + 10   # when +10 rung is posted
ask_p11 = R + 11   # when +11 rung is posted
vol_slot = randint(20, 30)   # per posted slot, ~uniform on these tapes
```

Simulator: `osmium_wall_bot.py` — `wall_prices(fv)` returns all four **theoretical** prices; a **book builder** should subset by your presence model.

---

### Validation (`validate_osmium_wall.py --all-sessions`)

**Per-session:** 100% price match on every leg where present (unchanged from single-day runs).

**Pooled (both sessions)**

| Leg | Price match | Merged vol χ² U[20,30] (df=10) |
|-----|-------------|--------------------------------|
| bid −10 | 815/815 (100%) | 8.49 — pass |
| bid −11 | 772/772 (100%) | 7.80 — pass |
| ask +10 | 783/783 (100%) | 3.79 — pass |
| ask +11 | 801/801 (100%) | 1.97 — pass |

---

### Misses

**Price:** none vs `round(FV)+k` on pooled wall-tagged levels.

**Presence model:** not yet reduced to a closed 3-line formula — use empirical bins or fit a small state machine from longer logs.

---

### Wall bot properties

- **One integer anchor R = round(FV)** for all four rung **prices** (tomato Bot 1 parallel).
- **−10 vs −11** is primarily **which ladder slots are visible**, strongly **FV-fraction–dependent** in pooled probes — not two different rounding maps for **price**.
- **Offsets from true FV** on posted rungs match ladder design (see continuous table above); **symmetric** bid/ask ladder.
- **Volumes** 20–30, χ² vs uniform not rejected (pooled and per-session).
- **No second price engine inferred** from these tapes — same “no memory” read as tomato once **R** is fixed at the tick.

---

### Caveats

1. **Single-day CSVs are short** — presence-vs-frac should be re-checked on more timesteps.
2. **Process count** (1 ladder MM vs 2 bots) still not ID’d from prices alone; the **presence** pattern supports **one coordinated ladder** more than two independent price engines.
