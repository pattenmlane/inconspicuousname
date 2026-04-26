# Round 4 agent follow-up bodies (loaded by `automation/ping.py`)

Separators `<<<PHASE1>>>`, `<<<PHASE2>>>`, `<<<PHASE3>>>` split sections.

<<<PHASE1>>>

ROUND 4 — **PHASE 1 ONLY** (exhaust this before Phase 2)

**Authoritative repo docs (read first):**
- `round4work/round4description.txt` — counterparty fields, products, limits, TTE note.
- `round4work/suggested direction.txt` — **Phase 1** block (verbatim plan you must execute).
- `round4work/outputs/r4_initial_analysis_summary.txt` — who the Marks are, pair structure, bursts (starting point, not gospel).

**Data:** `Prosperity4Data/ROUND_4/` trade + price CSVs (tape days present in tree; use all available days). Trades include **buyer** and **seller** names.

**Stop / deprioritize:** Treat old “Round 3 only / vouchers_final_strategy-only” iteration as **background**. Your **primary** deliverables now are **Round 4** counterparty-aware analysis and **legitimate tradeable edges** (signals you can state, test on tape, and defend out-of-sample).

---

### Phase 1 — be extremely thorough; find **legitimate** edges

Work through **every** Phase 1 bullet in `suggested direction.txt` with **quantitative** outputs (tables, CSVs, plots, or summary txt under your strategy folder).

**1) Participant-level alpha / predictiveness**

For **each** distinct name `U` appearing as buyer or seller:

- Tag every print where `buyer==U` **or** `seller==U` (separate **aggressive buy** vs **aggressive sell** if you can infer from price vs concurrent BBO; else analyze **both** sides separately).
- For horizons **K ∈ {5, 20, 100}** (ticks or timestamps — **define** and document): compute **forward mid change** of the **same symbol** and, where relevant, **VELVETFRUIT_EXTRACT** and **HYDROGEL_PACK** (cross-asset).
- **Stratify** by: product, hour/session bucket, spread quantile (wide vs tight), and optionally “burst vs isolated print.”
- Report **mean, median, t-stat / bootstrap CI**, fraction positive, and **n** per cell. Flag only patterns with **sensible n** and stability across **multiple tape days**.

**Edge identification:** Any `(U, side, product, regime)` with **repeated** positive mean markout and tight CI is a **candidate signal** (informed-style or bot-schedule exploitable).

**2) Deviation from “bot baseline”**

- **Cluster** participants into roles using **pair counts** and **volume balance** (e.g. dominant buyer→seller pairs, net buy/sell imbalance per name). Reference the initial summary: Mark 01→Mark 22, 14↔38 on hydro, etc.
- Fit a **simple baseline**: expected forward move as function of `(buyer, seller, symbol, regime)` (e.g. linear regression, cell means, or gradient boosting with heavy regularization).
- Compute **residuals** per print or per aggregated bucket. **Large stable residuals** → either mis-modeled bot (fade / lean) or a **second-order** edge.

**Edge identification:** Systematic **positive residuals** after baseline = **tradeable anomaly** (document the rule: when to act, size, and hold horizon).

**3) Sequence / graph methods**

- Build **directed graph** buyer → seller (weights: count, notional). Report **top pairs**, **reciprocity**, and any **hub** (who sits on many edges).
- Search **motifs**: 2-hop chains (A→B→C) that precede large moves in extract or a wing strike; time-align with **lead–lag** (Granger optional; at least **lagged correlation** of signed flow).

**Edge identification:** Recurring short paths that **precede** measurable mid movement = **structural** edge (scheduler / basket), even if not “insider.”

**4) Burst structure**

- Detect **same (day, timestamp)** with **multiple** trade rows. Attribute **orchestrator** (common buyer / seller across symbols).
- Event study: **forward** extract / core VEV mids after bursts vs after random matched-time controls.

**Edge identification:** If bursts **predict** direction or volatility regime, that is a **tradeable event** (gate entries/exits/size).

**5) Adverse selection vs inventory**

- If you have **sim** access to full book: proxy **passive adverse selection** when `U` hits the book. Otherwise approximate with **mid moves** after prints where `U` is aggressor.
- Relate to **your** hypothetical inventory only if you simulate MM; else describe **population-level** “who hurts passive liquidity.”

**Edge identification:** Pairs/names with **worst** markout for passive side → **do not quote into them** or **fade** after trigger.

---

### Phase 1 completion gate (required before you claim Phase 1 done)

In **`analysis.json`** (append one object titled `round4_phase1_complete`) list:

1. Every bullet above with **file path(s)** to outputs (csv/png/txt) and **one sentence conclusion** (edge or null).
2. **Ranked list** of **top 5 tradeable edges** you believe are **legitimate** (with effect size, n, and **which days** validated).
3. Explicit **“no edge”** findings worth publishing (negative result = saves team time).

**Do not** start Phase 2 work in depth until Phase 1 gate is satisfied **or** your lead instructs otherwise.

<<<PHASE2>>>

ROUND 4 — **PHASE 2 ONLY** (after Phase 1 gate is done)

Re-read `round4work/suggested direction.txt` **Phase 2** block.

**Prerequisite:** Your `analysis.json` must contain `round4_phase1_complete`. If Phase 1 was thin, briefly state what was left unproven and still proceed with Phase 2 **orthogonal** edges.

---

### Phase 2 — be extremely thorough; find **legitimate** edges

**1) Named-bot exploitation**

- Condition **all** Phase-1-style stats on `(buyer, seller)` **and** on **burst signatures** (e.g. within ±W ms of a Mark 01→Mark 22 multi-VEV burst).
- Separate **mean-revert** vs **trend** follow-through (define hold horizon; use **worse**-fill robustness if you simulate trades).

**Edge:** Rules like “after burst type B, fade 5300 touch” only if **out-of-sample** day holds.

**2) Microstructure**

- Queue / touch / **microprice** vs mid, spread regimes, trade-through rate. **Per Mark** if possible.

**Edge:** Regimes where **spread compression** predicts short-horizon **vol** or **direction**.

**3) Cross-instrument lead–lag**

- Signed flow or returns: extract vs each VEV vs hydro; distributed lags.

**Edge:** **Lead** instrument + lag window + direction rule.

**4) Regime splits**

- Tight vs wide book, high vs low depth, session buckets. Same signal **only** where it survives.

**5) Vol / smile**

- Reuse R3-style IV/smile residual logic **if** you have conventions; tie residuals to **which Marks** print.

**6) Execution / adverse selection**

- When **not** to provide liquidity vs named counterparty (from markouts).

**7) Inventory-aware MM**

- Skew rules when **repeated** seller/buyer pressure from a Mark aligns with extract mid drift.

---

### Phase 2 completion gate

Append `round4_phase2_complete` to **`analysis.json`**: top **5** tradeable edges (orthogonal to Phase 1 if possible), file paths, and **explicit** interaction with Phase 1 (confirm, refine, or falsify Phase-1 edges).

<<<PHASE3>>>

ROUND 4 — **PHASE 3 ONLY** (after Phase 2 gate)

**Read again:**

- `round4work/suggested direction.txt` **Phase 3** (inclineGod / Sonic / STRATEGY synthesis + verbatim excerpts).
- `previous round work/round3work/vouchers_final_strategy/ORIGINAL_DISCORD_QUOTES.txt`
- `previous round work/round3work/vouchers_final_strategy/STRATEGY.txt` (“How the messages fit together”).

**What to do**

1. **Re-run** the **highest-value** Phase 1 and Phase 2 analyses **conditional on Sonic’s joint gate**: require **VEV_5200** and **VEV_5300** top-of-book spreads **both ≤ 2** (same timestamp / merge-asof — **match** the R3 script convention in `analyze_vev_5200_5300_tight_gate_r3.py` logic where applicable).

2. **inclineGod:** add **spread–spread** and **spread vs price** panels for Round **4** tape (not only mid–mid correlation).

3. **Synthesize:** Does counterparty identity **interact** with the joint gate (e.g. Mark 01→Mark 22 only informative when gate tight)? That is a **primary** Round-4 thesis candidate.

4. **Legitimate edges:** Any **three-way** interaction `(Mark pattern, joint gate, product)` with stable markout / PnL in sim is **Tier-A** output.

Append `round4_phase3_complete` to **`analysis.json`** with ranked edges, paths to plots/tables, and **explicit** comparison to Phase 1/2 without the gate (does the gate **clean** the signal as Sonic claimed?).
