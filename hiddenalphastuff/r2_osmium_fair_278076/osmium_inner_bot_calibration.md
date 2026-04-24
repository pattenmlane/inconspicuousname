# Osmium Inner Bot (TOUCH / ±8) — ASH_COATED_OSMIUM Round 2

## Method

### Inputs required

1. **True FV stream** — From a **one-share hold** on the competition site: buy 1 `ASH_COATED_OSMIUM` at the best ask at `t=0`, hold. Each `activitiesLog` row PnL is marked against server fair value, so on osmium lines  
   `true_fv(t) = profit_and_loss(t) + buy_price`.  
   Buy price is taken from `tradeHistory`: first `buyer == SUBMISSION` fill for osmium (e.g. **10016.0** for session 278076).

2. **Order book** — Same export: `activitiesLog` rows for `ASH_COATED_OSMIUM`, up to three bid and three ask price/volume slots per timestamp.

### Extraction

- `export_osmium_fair_log.py` — parses the JSON activity log → `prices_round_*_day_*.csv`, `osmium_true_fv.csv` (semicolon-separated).
- `discover_book_behaviors.py` — builds per-level observations with `offset_int = round(price - true_fv)` for discovery and volume tables.

### Key discovery

Server fair value for osmium is a **continuous** random walk (same spirit as tomato FV): not the displayed mid. After anchoring levels to **true** FV, the **inner** cluster sits at integer offsets **−8** (bid) and **+8** (ask) in `round(price - true_fv)` space.

Brute-force search (`analyze_osmium_quote_rules.py`, same grid as `analyze_bot1.py`) finds many equivalent `floor(FV+shift)+k` parameterizations on these slices; the **parsimonious** closed form matches tomato **Bot 1–style** symmetric rounding: **`round(FV) ± 8`**.

### Bot identification

Assign the **inner** process to book levels where:

- `side == bid` and `round(price - true_fv) == -8`, or  
- `side == ask` and `round(price - true_fv) == +8`.

When multiple bids share the same offset bin, take the **highest** price; for asks, the **lowest** price (best quote at that offset).

---

## Analysis

Re-run on **both** fair exports (278076 + 248329) whenever you add a new session folder to `osmium_sessions.py`:

```bash
python3 validate_osmium_inner.py --all-sessions
python3 osmium_inner_exact_rule.py --all-sessions
# or everything at once:
python3 run_all_osmium_session_checks.py
```

| Script | Role |
|--------|------|
| `analyze_osmium_quote_rules.py --all-sessions` | Brute-force inner + wall; writes per-session `osmium_quote_rule_search.txt` + **`osmium_quote_rule_search_MULTISESSION.txt`**. |
| `osmium_inner_exact_rule.py --all-sessions` | R vs `floor(FV)` bins and misses — **each session + both**. |
| `validate_osmium_inner.py --all-sessions` | Validation per folder + **pooled** summary below. |

---

### Transfer function: FV → Inner quotes

| FV fractional part (binned) | R_bid = bid+8 vs floor(FV) | R_ask = ask−8 vs floor(FV) | Inner bid | Inner ask |
|------------------------------|----------------------------|-----------------------------|-----------|-----------|
| [0.05, 0.45] | 0 | 0 | `round(FV) − 8` | `round(FV) + 8` |
| ~0.50 (mixed in sample) | 0 or 1 | 0 or 1 | `round(FV) − 8` | `round(FV) + 8` |
| [0.55, 0.95] | 1 | 1 | `round(FV) − 8` | `round(FV) + 8` |

On fair-probe exports below, **every** observed inner leg satisfied `price = round(FV) ± 8` with **zero** misses, so the table collapses to the single-line rule in practice for these tapes.

### Bins vs distribution (tomato parallel)

Tomato **Bot 2** uses a **fractional-part transfer table** to explain *which* rounding branch fires, and separately reports **continuous offset from FV** in “Bot 2 Properties” (mean bid offset −6.73, etc.). The bins here are the same kind of **rounding diagnostic**, not a separate “binned price process”: the **law** is still the single map **`round(FV) ± 8`**.

For **distribution** on our two fair exports (**pooled**, `validate_osmium_inner.py --all-sessions` sample paths), among ticks where that inner leg exists:

| Leg | n | mean(`price − true_FV`) | stdev |
|-----|---|---------------------------|-------|
| Bid | 1559 | **−8.00** | **0.28** |
| Ask | 1617 | **+7.99** | **0.28** |

The small stdev is expected: `true_FV` is continuous inside the tick while quotes sit on discrete offsets from `round(FV)`—same spirit as tomato Bot 1’s “±7.75 mean, ±0.45 std” on a ±8 grid.

---

## Result

```python
bid = round(FV) - 8
ask = round(FV) + 8
vol = randint(10, 15)    # same value for bid and ask on each tick (when both post)
```

`round()` is Python 3 semantics (ties to even at exact halves); FV is effectively non-degenerate on these logs.

---

### Validation (`validate_osmium_inner.py`)

**Session 278076** (`prices_round_2_day_1.csv`, 1000 timesteps)

| Metric | Score |
|--------|--------|
| Bid price match | 801/801 (100.0%) — ticks with inner bid |
| Ask price match | 811/811 (100.0%) |
| Both match | 641/641 (100.0%) — ticks with both inner legs |
| Spread match | 641/641 (100.0%) |
| Spread value | 16 on all 641 ticks (model 16) |
| Volume distribution (bid @ match) | Uniform {10,…,15}, χ² = 1.60 (df = 5, critical 11.1 → passes) |
| Volume distribution (ask @ match) | χ² = 1.56 |
| Bid vol = ask vol same tick | 641/641 (100.0%) when both inner legs present and prices match |

**Session 248329** (`prices_round_2_day_0.csv`, 1000 timesteps)

| Metric | Score |
|--------|--------|
| Bid price match | 758/758 (100.0%) |
| Ask price match | 806/806 (100.0%) |
| Both match | 618/618 (100.0%) |
| Spread match | 618/618 (100.0%) |
| Bid vol = ask vol | 618/618 (100.0%) |
| Volume χ² bid @ match | 1.67 (passes) |

**Pooled (`validate_osmium_inner.py --all-sessions`)** — 2000 timesteps total

| Metric | Pooled |
|--------|--------|
| Bid price match | **1559/1559 (100.0%)** |
| Ask price match | **1617/1617 (100.0%)** |
| Both match | **1259/1259 (100.0%)** |

---

### Misses (`osmium_inner_exact_rule.py`)

| Session | Bid misses vs `round(FV)−8` | Ask misses vs `round(FV)+8` |
|---------|------------------------------|------------------------------|
| 278076 | 0 | 0 |
| 248329 | 0 | 0 |

Unlike tomato Bot 1 on the tutorial JSON (~3% misses near `X.5`), these two osmium slices show **no** residual ±1 price error at inner-tagged levels. If longer tapes show misses, expect the same half-integer / float story as tomato.

---

### Inner bot properties

- **Often present, not guaranteed** — e.g. ~80% of timesteps per side on 278076 (sparse book / only 1–2 levels many steps).
- **Symmetric** — bid / ask offsets from true FV are mirror images (pooled means **≈ −8.00** / **≈ +7.99**, stdev **≈ 0.28** when the leg exists).
- **Spread** — 16 ticks when both legs post (same as tomato-style ±8 wall in tick space).
- **Volume** — integer uniform **{10,…,15}**, **identical** on bid and ask inner slots on every tick where both are present and prices match the model.
- **No memory in price rule** — quotes depend only on current `true_fv` in the fitted model (tomato Bot 2 parallel).

---

### Simulator

`osmium_inner_bot.py` — `inner_quote(fv)` → `(bid, ask, vol)`; `inner_prices(fv)` for deterministic price checks.

---

### Caveats / next steps

1. **Presence** — Simulator must allow missing inner on a side when the real book is thin.
2. **Multi-day** — Re-run validation on additional fair logs; watch for `X.5` boundary misses if sample size grows.
3. **Naming** — “Inner bot” is our label; IMC does not publish process names.
