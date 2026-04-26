#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned forward mids (tape-only).

Horizon K: number of price **ticks** (timestamp index steps of 100) forward on the
aligned grid shared by all products (verified identical ts sets per day).

Outputs under manual_traders/R4/r3v_inventory_vega_rail_18/analysis_outputs/:
  - r4_phase1_pair_forward_summary.csv, r4_phase1_pair_forward_by_day.csv
  - r4_phase1_mark_forward_summary.csv (7 Marks, fixed)
  - r4_phase1_participant_U_forward.csv — **each distinct name** U, roles, same-symbol + extract + hydro
  - r4_phase1_participant_U_spread_tert.csv — trade-symbol spread tertile
  - r4_phase1_participant_U_session_terc.csv — **session** = early/mid/late third of price grid (tick index)
  - r4_phase1_participant_U_burst_iso.csv — same tick burst vs isolated (multi-symbol same ts)
  - r4_phase1_participant_flow.csv — per-name qty buy/sell/net by symbol (clustering / imbalance)
  - r4_phase1_key_pairs_stability_by_day.csv — top-|t| (buyer,seller,symbol) cells × day
  - r4_phase1_mark67_extract_buyer_by_day.csv — Mark 67 buyer on extract, mean fwd by day
  - r4_phase1_burst_extract_fwd20_control.json — burst mean vs **random** isolated-timestamp null (pooled + per day)
  - r4_phase1_graph_degree_by_mark.csv — out/in print counts + distinct counterparties
  - r4_phase1_graph_top_edges_reciprocity.csv — top directed edges vs reverse (A→B vs B→A)
  - r4_phase1_motif_secondleg_extract_fwd.csv — fwd extract after **second leg** of top same-symbol 2-hop motifs
  - r4_phase1_pair_residuals.csv, r4_phase1_graph_edges.csv, r4_phase1_bursts.csv
  - r4_phase1_adverse_extract_after_01_22_5300.csv, r4_phase1_summary.txt
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BOOTSTRAP_B = 400
RNG_SEED = 1

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = REPO / "manual_traders/R4/r3v_inventory_vega_rail_18" / "analysis_outputs"
PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]
K_HORIZONS = (5, 20, 100)
CROSS = ("VELVETFRUIT_EXTRACT", "HYDROGEL_PACK")


def load_price_grid(csv_day: int) -> tuple[list[int], dict[str, list[float]], dict[str, list[int]]]:
    """Return (timestamps sorted), mids[product], spreads[product] aligned on grid."""
    path = DATA / f"prices_round_4_day_{csv_day}.csv"
    by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open() as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            ts = int(row["timestamp"])
            prod = row["product"]
            bb = row.get("bid_price_1")
            ba = row.get("ask_price_1")
            if not bb or not ba:
                continue
            bid = int(bb)
            ask = int(ba)
            if ask < bid:
                continue
            mid = 0.5 * (bid + ask)
            by_ts[ts][prod] = {"mid": mid, "spr": ask - bid, "bid": bid, "ask": ask}
    tss = sorted(by_ts)
    mids: dict[str, list[float]] = {p: [] for p in PRODUCTS}
    sprs: dict[str, list[int]] = {p: [] for p in PRODUCTS}
    for ts in tss:
        d = by_ts[ts]
        for p in PRODUCTS:
            if p not in d:
                mids[p].append(float("nan"))
                sprs[p].append(-1)
            else:
                mids[p].append(float(d[p]["mid"]))
                sprs[p].append(int(d[p]["spr"]))
    return tss, mids, sprs


def t_stat(xs: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    m = statistics.mean(xs)
    try:
        s = statistics.stdev(xs)
    except statistics.StatisticsError:
        return None
    if s < 1e-12:
        return None
    return m / (s / math.sqrt(len(xs)))


def spread_tertiles(sprs_sym: list[int]) -> tuple[int, int] | None:
    """Tertile cutoffs on non-negative BBO spreads for one symbol-day grid."""
    vals = sorted(s for s in sprs_sym if s >= 0)
    n = len(vals)
    if n < 9:
        return None
    return (vals[n // 3], vals[2 * n // 3])


def spr_bucket(spr: int, cuts: tuple[int, int] | None) -> str:
    if spr < 0 or cuts is None:
        return "miss"
    t1, t2 = cuts
    if spr <= t1:
        return "T1_tightest_third"
    if spr <= t2:
        return "T2_mid_third"
    return "T3_widest_third"


def session_tercile_grid(j: int, n_grid: int) -> str:
    """Early / mid / late by **tick index** on the day's price grid (wall-clock proxy)."""
    if n_grid < 3:
        return "miss"
    t1 = n_grid // 3
    t2 = 2 * n_grid // 3
    if j < t1:
        return "S1_early_third_ticks"
    if j < t2:
        return "S2_mid_third_ticks"
    return "S3_late_third_ticks"


def bootstrap_mean_ci(vals: list[float], b: int = BOOTSTRAP_B, seed: int = RNG_SEED) -> tuple[float | None, float | None]:
    """Percentile CI on resampled means (with replacement). Returns (lo, hi) or (None, None) if n<5."""
    n = len(vals)
    if n < 5:
        return None, None
    rng = random.Random(seed + n)
    means: list[float] = []
    for _ in range(b):
        sm = sum(rng.choice(vals) for _ in range(n)) / n
        means.append(sm)
    means.sort()
    lo_i = int(0.025 * b)
    hi_i = int(0.975 * b) - 1
    hi_i = max(hi_i, lo_i)
    return means[lo_i], means[min(hi_i, b - 1)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_days = [1, 2, 3]
    grids: dict[int, tuple[list[int], dict[str, list[float]], dict[str, list[int]]]] = {}
    for d in csv_days:
        grids[d] = load_price_grid(d)

    # --- trades with forward returns ---
    rows_out: list[dict] = []
    pair_fwd: dict[tuple, list[float]] = defaultdict(list)  # (buy,sell,sym,k) -> fwd same sym
    pair_fwd_day: dict[tuple, list[float]] = defaultdict(list)  # (buy,sell,sym,k,day)
    burst_groups: list[tuple[int, int, int, str, int]] = []  # day, ts, n, orchestrator, n_sym
    all_names: set[str] = set()

    for csv_day in csv_days:
        tss, mids, sprs = grids[csv_day]
        idx_of = {ts: i for i, ts in enumerate(tss)}
        cuts_by_sym: dict[str, tuple[int, int] | None] = {p: spread_tertiles(sprs[p]) for p in PRODUCTS}
        path = DATA / f"trades_round_4_day_{csv_day}.csv"
        trades_raw: list[dict] = []
        with path.open() as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                trades_raw.append(dict(row))
        # bursts: (day, ts) -> list
        by_burst: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for row in trades_raw:
            ts = int(row["timestamp"])
            by_burst[(csv_day, ts)].append(row)
        burst_multi_sym: set[tuple[int, int]] = set()
        for (d0, ts0), lst in by_burst.items():
            if len(lst) <= 1:
                continue
            buyers = [x["buyer"] for x in lst]
            orch = Counter(buyers).most_common(1)[0][0]
            syms = len({x["symbol"] for x in lst})
            burst_groups.append((d0, ts0, len(lst), orch, syms))
            if syms >= 2:
                burst_multi_sym.add((d0, ts0))

        n_grid = len(tss)
        for row in trades_raw:
            sym = row["symbol"]
            if sym not in mids:
                continue
            ts = int(row["timestamp"])
            if ts not in idx_of:
                continue
            j = idx_of[ts]
            buyer = row.get("buyer") or ""
            seller = row.get("seller") or ""
            price = float(row["price"])
            qty = float(row.get("quantity") or 0)
            i = j
            bid = None
            ask = None
            if j < len(tss) and tss[j] == ts:
                # recover bid/ask from grid via mids row — need bid ask; stored in by_ts only in load — re-get from spr
                pass
            # Re-read bid/ask from price file row is heavy; approximate aggressor from mid
            mid0 = mids[sym][j]
            if math.isnan(mid0):
                continue
            # find spread at j
            spr0 = sprs[sym][j]
            # approximate bid/ask from mid and spread (integers)
            half = spr0 / 2.0 if spr0 >= 0 else 0.0
            bid_ap = int(round(mid0 - half))
            ask_ap = int(round(mid0 + half))
            if price >= ask_ap - 1e-9:
                aggr_side = "buyer"
            elif price <= bid_ap + 1e-9:
                aggr_side = "seller"
            else:
                aggr_side = "passive_mid"

            all_names.add(buyer)
            all_names.add(seller)

            sess = session_tercile_grid(j, n_grid)
            bkey = (csv_day, ts)
            burst_iso = (
                "burst_multi_sym"
                if bkey in burst_multi_sym
                else ("burst_single_sym" if len(by_burst.get(bkey, [])) > 1 else "isolated")
            )

            rec: dict = {
                "csv_day": csv_day,
                "timestamp": ts,
                "grid_index": j,
                "session_tercile_ticks": sess,
                "burst_iso": burst_iso,
                "symbol": sym,
                "buyer": buyer,
                "seller": seller,
                "price": price,
                "quantity": qty,
                "mid_t": mid0,
                "spread_t": spr0,
                "aggressor": aggr_side,
                "spr_bucket_sym": spr_bucket(spr0, cuts_by_sym.get(sym)),
            }
            for K in K_HORIZONS:
                if j + K >= len(tss):
                    rec[f"fwd_{K}_{sym}"] = None
                    continue
                m1 = mids[sym][j + K]
                if math.isnan(m1):
                    rec[f"fwd_{K}_{sym}"] = None
                else:
                    dv = float(m1 - mid0)
                    rec[f"fwd_{K}_{sym}"] = dv
                    pair_fwd[(buyer, seller, sym, K)].append(dv)
                    pair_fwd_day[(buyer, seller, sym, K, csv_day)].append(dv)
                for cx in CROSS:
                    if cx == sym:
                        continue
                    m0c = mids[cx][j]
                    m1c = mids[cx][j + K]
                    if math.isnan(m0c) or math.isnan(m1c):
                        rec[f"fwd_{K}_{cx}"] = None
                    else:
                        rec[f"fwd_{K}_{cx}"] = float(m1c - m0c)
            rows_out.append(rec)

    all_names.discard("")

    # Write forward-by-trade sample (can be large — write pair aggregates instead mostly)
    pair_rows: list[dict] = []
    for (buy, sell, sym, K), vals in pair_fwd.items():
        if len(vals) < 5:
            continue
        pair_rows.append(
            {
                "buyer": buy,
                "seller": sell,
                "symbol": sym,
                "K": K,
                "n": len(vals),
                "mean_fwd": statistics.mean(vals),
                "median_fwd": statistics.median(vals),
                "frac_pos": sum(1 for x in vals if x > 0) / len(vals),
                "t_stat": t_stat(vals),
            }
        )
    pair_rows.sort(key=lambda x: (-(x["t_stat"] or 0), -abs(x["mean_fwd"]), -x["n"]))

    # -------------------------------------------------------------------------
    # Participant U — roles, same + cross-asset; spread tertile, session tercile,
    # burst vs isolated; bootstrap CI on cell mean (B=BOOTSTRAP_B)
    # -------------------------------------------------------------------------
    u_cells: dict[tuple, list[float]] = defaultdict(list)
    u_cells_tert: dict[tuple, list[float]] = defaultdict(list)
    u_cells_sess: dict[tuple, list[float]] = defaultdict(list)
    u_cells_burst: dict[tuple, list[float]] = defaultdict(list)
    flow: dict[tuple[str, str], float] = defaultdict(float)  # (U, symbol) -> signed qty

    for rec in rows_out:
        sym = rec["symbol"]
        ag = rec["aggressor"]
        sb = rec.get("spr_bucket_sym", "miss")
        sess = rec.get("session_tercile_ticks", "miss")
        biso = rec.get("burst_iso", "isolated")
        b = rec.get("buyer") or ""
        s = rec.get("seller") or ""
        q = float(rec.get("quantity") or 0)
        if b:
            flow[(b, sym)] += q
        if s:
            flow[(s, sym)] -= q

        for K in K_HORIZONS:
            f_same = rec.get(f"fwd_{K}_{sym}")
            f_ex = rec.get(f"fwd_{K}_VELVETFRUIT_EXTRACT")
            f_hy = rec.get(f"fwd_{K}_HYDROGEL_PACK")

            def push(
                cellmap: dict[tuple, list[float]],
                key_extra: tuple,
                fv: float | None,
            ) -> None:
                if fv is None:
                    return
                cellmap[key_extra].append(float(fv))

            for U, role_tag in ((b, "as_buyer"), (s, "as_seller")):
                if not U:
                    continue
                if role_tag == "as_buyer":
                    push(u_cells, (U, "as_buyer", "same", sym, K), f_same)
                    push(u_cells, (U, "as_buyer", "to_extract", "x", K), f_ex)
                    push(u_cells, (U, "as_buyer", "to_hydro", "x", K), f_hy)
                    if ag == "buyer":
                        push(u_cells, (U, "aggr_buy", "same", sym, K), f_same)
                    if sb in ("T1_tightest_third", "T2_mid_third", "T3_widest_third"):
                        push(u_cells_tert, (U, "as_buyer", sb, "same", sym, K), f_same)
                        push(u_cells_tert, (U, "as_buyer", sb, "to_extract", "x", K), f_ex)
                    if sess.startswith("S"):
                        push(u_cells_sess, (U, "as_buyer", sess, "same", sym, K), f_same)
                        push(u_cells_sess, (U, "as_buyer", sess, "to_extract", "x", K), f_ex)
                    push(u_cells_burst, (U, "as_buyer", biso, "same", sym, K), f_same)
                    push(u_cells_burst, (U, "as_buyer", biso, "to_extract", "x", K), f_ex)
                else:
                    push(u_cells, (U, "as_seller", "same", sym, K), f_same)
                    push(u_cells, (U, "as_seller", "to_extract", "x", K), f_ex)
                    push(u_cells, (U, "as_seller", "to_hydro", "x", K), f_hy)
                    if ag == "seller":
                        push(u_cells, (U, "aggr_sell", "same", sym, K), f_same)
                    if sb in ("T1_tightest_third", "T2_mid_third", "T3_widest_third"):
                        push(u_cells_tert, (U, "as_seller", sb, "same", sym, K), f_same)
                        push(u_cells_tert, (U, "as_seller", sb, "to_extract", "x", K), f_ex)
                    if sess.startswith("S"):
                        push(u_cells_sess, (U, "as_seller", sess, "same", sym, K), f_same)
                        push(u_cells_sess, (U, "as_seller", sess, "to_extract", "x", K), f_ex)
                    push(u_cells_burst, (U, "as_seller", biso, "same", sym, K), f_same)
                    push(u_cells_burst, (U, "as_seller", biso, "to_extract", "x", K), f_ex)

    def summarize_u_cells(cells: dict[tuple, list[float]], n_min: int) -> list[dict]:
        out: list[dict] = []
        for key, vals in cells.items():
            if len(vals) < n_min:
                continue
            m = statistics.mean(vals)
            med = statistics.median(vals)
            t = t_stat(vals)
            lo, hi = bootstrap_mean_ci(vals)
            row = {
                "mark": key[0],
                "role": key[1],
                "fwd_target": key[2],
                "out_symbol": key[3],
                "K": key[4],
                "n": len(vals),
                "mean": m,
                "median": med,
                "frac_pos": sum(1 for x in vals if x > 0) / len(vals),
                "t_stat": t,
                "boot_mean_lo": lo,
                "boot_mean_hi": hi,
            }
            out.append(row)
        return out

    def summarize_u_strat(
        cells: dict[tuple, list[float]], n_min: int, strat_name: str, strat_idx: int
    ) -> list[dict]:
        """Keys are always (U, role, strat_value, fwd_target, out_symbol, K)."""
        out: list[dict] = []
        for key, vals in cells.items():
            if len(vals) < n_min:
                continue
            if len(key) < 6:
                continue
            m = statistics.mean(vals)
            med = statistics.median(vals)
            t = t_stat(vals)
            lo, hi = bootstrap_mean_ci(vals)
            row = {
                "mark": key[0],
                "role": key[1],
                strat_name: key[strat_idx],
                "fwd_target": key[3],
                "out_symbol": key[4],
                "K": key[5],
                "n": len(vals),
                "mean": m,
                "median": med,
                "frac_pos": sum(1 for x in vals if x > 0) / len(vals),
                "t_stat": t,
                "boot_mean_lo": lo,
                "boot_mean_hi": hi,
            }
            out.append(row)
        return out

    u_rows = summarize_u_cells(u_cells, n_min=5)
    u_rows.sort(key=lambda x: (-(abs(x["t_stat"] or 0)), -x["n"]))

    u_tert_rows = summarize_u_strat(u_cells_tert, 5, "spread_tertile_trade_symbol", 2)
    u_tert_rows.sort(key=lambda x: (-(abs(x["t_stat"] or 0)), -x["n"]))

    u_sess_rows = summarize_u_strat(u_cells_sess, 5, "session_tercile_ticks", 2)
    u_sess_rows.sort(key=lambda x: (-(abs(x["t_stat"] or 0)), -x["n"]))

    u_burst_rows = summarize_u_strat(u_cells_burst, 5, "burst_iso", 2)
    u_burst_rows.sort(key=lambda x: (-(abs(x["t_stat"] or 0)), -x["n"]))

    u_cols = [
        "mark",
        "role",
        "fwd_target",
        "out_symbol",
        "K",
        "n",
        "mean",
        "median",
        "frac_pos",
        "t_stat",
        "boot_mean_lo",
        "boot_mean_hi",
    ]
    with (OUT / "r4_phase1_participant_U_forward.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=u_cols, extrasaction="ignore")
        w.writeheader()
        for row in u_rows:
            w.writerow({c: ("" if row.get(c) is None else row[c]) for c in u_cols})
    with (OUT / "r4_phase1_participant_U_spread_tert.csv").open("w", newline="") as f:
        cols = ["mark", "role", "spread_tertile_trade_symbol"] + u_cols[3:]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in u_tert_rows:
            w.writerow({c: ("" if row.get(c) is None else row[c]) for c in cols})
    with (OUT / "r4_phase1_participant_U_session_terc.csv").open("w", newline="") as f:
        cols = ["mark", "role", "session_tercile_ticks"] + u_cols[3:]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in u_sess_rows:
            w.writerow({c: ("" if row.get(c) is None else row[c]) for c in cols})
    with (OUT / "r4_phase1_participant_U_burst_iso.csv").open("w", newline="") as f:
        cols = ["mark", "role", "burst_iso"] + u_cols[3:]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in u_burst_rows:
            w.writerow({c: ("" if row.get(c) is None else row[c]) for c in cols})

    flow_rows: list[dict] = []
    for (U, sym), net in sorted(flow.items(), key=lambda x: -abs(x[1])):
        flow_rows.append({"participant": U, "symbol": sym, "signed_qty_net": net})
    with (OUT / "r4_phase1_participant_flow.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["participant", "symbol", "signed_qty_net"])
        w.writeheader()
        for row in flow_rows:
            w.writerow(row)

    n_participants = len(all_names)
    n_participant_u_cells = len(u_rows)
    n_participant_u_tert_cells = len(u_tert_rows)
    n_participant_u_sess_cells = len(u_sess_rows)
    n_participant_u_burst_cells = len(u_burst_rows)

    day_rows: list[dict] = []
    for key, vals in pair_fwd_day.items():
        buy, sell, sym, K, dday = key
        if len(vals) < 3:
            continue
        day_rows.append(
            {
                "buyer": buy,
                "seller": sell,
                "symbol": sym,
                "K": K,
                "csv_day": dday,
                "n": len(vals),
                "mean_fwd": statistics.mean(vals),
                "t_stat": t_stat(vals),
            }
        )
    with (OUT / "r4_phase1_pair_forward_by_day.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["buyer", "seller", "symbol", "K", "csv_day", "n", "mean_fwd", "t_stat"],
        )
        w.writeheader()
        for row in day_rows:
            w.writerow(row)

    # Burst vs non-burst: forward extract mid K=20 at trade timestamps
    burst_ts: set[tuple[int, int]] = {(d0, ts0) for d0, ts0, _, _, _ in burst_groups}
    fwd20_burst: list[float] = []
    fwd20_iso: list[float] = []
    for rec in rows_out:
        if rec["symbol"] != "VELVETFRUIT_EXTRACT":
            continue
        v = rec.get("fwd_20_VELVETFRUIT_EXTRACT")
        if v is None:
            continue
        key = (rec["csv_day"], rec["timestamp"])
        if key in burst_ts:
            fwd20_burst.append(float(v))
        else:
            fwd20_iso.append(float(v))
    burst_summary = {
        "n_burst_trades_extract": len(fwd20_burst),
        "mean_fwd20_extract_burst": statistics.mean(fwd20_burst) if fwd20_burst else None,
        "n_isolated_extract_trades": len(fwd20_iso),
        "mean_fwd20_extract_isolated": statistics.mean(fwd20_iso) if fwd20_iso else None,
    }
    (OUT / "r4_phase1_burst_vs_isolated_extract_fwd20.json").write_text(
        json.dumps(burst_summary, indent=2), encoding="utf-8"
    )

    # --- Burst vs **random** isolated-timestamp null (same n as burst, resample isolated pool)
    iso_by_day: dict[int, list[float]] = defaultdict(list)
    for rec in rows_out:
        if rec["symbol"] != "VELVETFRUIT_EXTRACT":
            continue
        v = rec.get("fwd_20_VELVETFRUIT_EXTRACT")
        if v is None:
            continue
        key = (rec["csv_day"], rec["timestamp"])
        if key not in burst_ts:
            iso_by_day[rec["csv_day"]].append(float(v))
    rng_ctrl = random.Random(RNG_SEED + 901)

    def null_burst_enrichment(burst_vals: list[float], iso_vals: list[float], trials: int = 2000) -> dict:
        nb = len(burst_vals)
        ni = len(iso_vals)
        if nb < 1 or ni < 1:
            return {"n_burst": nb, "n_iso_pool": ni, "burst_mean": None, "null_note": "insufficient"}
        obs = statistics.mean(burst_vals)
        null_means: list[float] = []
        for _ in range(trials):
            null_means.append(statistics.mean(rng_ctrl.choices(iso_vals, k=nb)))
        null_means.sort()
        frac_ge = sum(1 for x in null_means if x >= obs) / trials
        frac_le = sum(1 for x in null_means if x <= obs) / trials
        return {
            "n_burst": nb,
            "n_iso_pool": ni,
            "burst_mean_fwd20": obs,
            "isolated_pool_mean_fwd20": statistics.mean(iso_vals),
            "null_resample_trials": trials,
            "null_mean_of_means": statistics.mean(null_means),
            "null_p05_mean": null_means[int(0.05 * trials)],
            "null_p95_mean": null_means[int(0.95 * trials) - 1],
            "one_sided_p_burst_higher": frac_ge,
            "one_sided_p_burst_lower": frac_le,
        }

    burst_by_day: dict[int, list[float]] = defaultdict(list)
    for rec in rows_out:
        if rec["symbol"] != "VELVETFRUIT_EXTRACT":
            continue
        v = rec.get("fwd_20_VELVETFRUIT_EXTRACT")
        if v is None:
            continue
        if (rec["csv_day"], rec["timestamp"]) in burst_ts:
            burst_by_day[rec["csv_day"]].append(float(v))

    control_payload = {
        "pooled": null_burst_enrichment(fwd20_burst, fwd20_iso),
        "per_csv_day": {},
    }
    for d in csv_days:
        bdv = burst_by_day.get(d, [])
        idv = iso_by_day.get(d, [])
        control_payload["per_csv_day"][str(d)] = null_burst_enrichment(bdv, idv)
    (OUT / "r4_phase1_burst_extract_fwd20_control.json").write_text(
        json.dumps(control_payload, indent=2), encoding="utf-8"
    )

    ctrl_pool = control_payload.get("pooled") or {}
    burst_ctrl_note = (
        f"Burst fwd20 vs random-isolated null (pooled): burst_mean={ctrl_pool.get('burst_mean_fwd20')} "
        f"one_sided_p_burst_higher={ctrl_pool.get('one_sided_p_burst_higher')}"
    )

    # --- Top pair cells × day (stability table for Phase-1 narrative)
    top_pairs: list[tuple[str, str, str]] = []
    seen_tp: set[tuple[str, str, str]] = set()
    for r in sorted(pair_rows, key=lambda x: -(abs(x["t_stat"] or 0))):
        if r["n"] < 30 or r["t_stat"] is None:
            continue
        key3 = (r["buyer"], r["seller"], r["symbol"])
        if key3 in seen_tp:
            continue
        seen_tp.add(key3)
        top_pairs.append(key3)
        if len(top_pairs) >= 12:
            break
    stab_rows: list[dict] = []
    for buy, sell, sym in top_pairs:
        for K in K_HORIZONS:
            for d in csv_days:
                vals = pair_fwd_day.get((buy, sell, sym, K, d), [])
                stab_rows.append(
                    {
                        "buyer": buy,
                        "seller": sell,
                        "symbol": sym,
                        "K": K,
                        "csv_day": d,
                        "n": len(vals),
                        "mean_fwd": statistics.mean(vals) if vals else "",
                        "t_stat": t_stat(vals) if vals else "",
                    }
                )
    with (OUT / "r4_phase1_key_pairs_stability_by_day.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["buyer", "seller", "symbol", "K", "csv_day", "n", "mean_fwd", "t_stat"],
            extrasaction="ignore",
        )
        w.writeheader()
        for row in stab_rows:
            w.writerow(row)

    # --- Mark 67 as **buyer** on extract: per-day mean fwd (all K)
    m67_rows: list[dict] = []
    m67_by_day_k: dict[tuple[int, int], list[float]] = defaultdict(list)
    for rec in rows_out:
        if rec.get("buyer") != "Mark 67" or rec.get("symbol") != "VELVETFRUIT_EXTRACT":
            continue
        d = rec["csv_day"]
        sym = rec["symbol"]
        for K in K_HORIZONS:
            v = rec.get(f"fwd_{K}_{sym}")
            if v is not None:
                m67_by_day_k[(d, K)].append(float(v))
    for (d, K), vals in sorted(m67_by_day_k.items()):
        if len(vals) < 2:
            continue
        m67_rows.append(
            {
                "csv_day": d,
                "K": K,
                "n": len(vals),
                "mean_fwd_same": statistics.mean(vals),
                "median": statistics.median(vals),
                "frac_pos": sum(1 for x in vals if x > 0) / len(vals),
                "t_stat": t_stat(vals),
            }
        )
    with (OUT / "r4_phase1_mark67_extract_buyer_by_day.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["csv_day", "K", "n", "mean_fwd_same", "median", "frac_pos", "t_stat"],
        )
        w.writeheader()
        for row in m67_rows:
            w.writerow(row)

    # Two-hop chains on same symbol: consecutive prints in time order (per day, per symbol)
    hop2: Counter[tuple[str, str, str, str, str]] = Counter()
    for csv_day in csv_days:
        path = DATA / f"trades_round_4_day_{csv_day}.csv"
        rows_sym: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        with path.open() as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                sym = row["symbol"]
                ts = int(row["timestamp"])
                rows_sym[sym].append((ts, row.get("buyer") or "", row.get("seller") or ""))
        for sym, seq in rows_sym.items():
            seq.sort(key=lambda x: x[0])
            for i in range(len(seq) - 1):
                _, b1, s1 = seq[i]
                _, b2, s2 = seq[i + 1]
                hop2[(b1, s1, b2, s2, sym)] += 1
    with (OUT / "r4_phase1_twohop_same_symbol.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["buyer1", "seller1", "buyer2", "seller2", "symbol", "count"])
        for (b1, s1, b2, s2, sym), c in hop2.most_common(80):
            w.writerow([b1, s1, b2, s2, sym, c])

    # --- Motif-conditioned **extract** forward at the **second** same-symbol leg (top chains)
    top_motif_keys = [t[0] for t in hop2.most_common(25)]
    motif_set = frozenset(top_motif_keys)
    motif_fwd: dict[tuple, list[float]] = defaultdict(list)
    by_day_sym: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for rec in rows_out:
        by_day_sym[(rec["csv_day"], rec["symbol"])].append(rec)
    for key, lst in by_day_sym.items():
        lst.sort(key=lambda r: (r["timestamp"], r.get("buyer"), r.get("seller"), r.get("price")))
        for i in range(len(lst) - 1):
            r1, r2 = lst[i], lst[i + 1]
            sym = r1["symbol"]
            m = (r1["buyer"], r1["seller"], r2["buyer"], r2["seller"], sym)
            if m not in motif_set:
                continue
            for K in K_HORIZONS:
                v = r2.get(f"fwd_{K}_VELVETFRUIT_EXTRACT")
                if v is not None:
                    motif_fwd[(m, K)].append(float(v))
    motif_rows: list[dict] = []
    for (m, K), vals in motif_fwd.items():
        b1, s1, b2, s2, sym = m
        if len(vals) < 5:
            continue
        motif_rows.append(
            {
                "buyer1": b1,
                "seller1": s1,
                "buyer2": b2,
                "seller2": s2,
                "symbol": sym,
                "K": K,
                "n_second_leg": len(vals),
                "mean_fwd_extract": statistics.mean(vals),
                "median": statistics.median(vals),
                "frac_pos": sum(1 for x in vals if x > 0) / len(vals),
                "t_stat": t_stat(vals),
                "chain_count_hop2_table": hop2.get(m, 0),
            }
        )
    motif_rows.sort(key=lambda x: (-(abs(x["t_stat"] or 0)), -x["n_second_leg"]))
    with (OUT / "r4_phase1_motif_secondleg_extract_fwd.csv").open("w", newline="") as f:
        cols = [
            "buyer1",
            "seller1",
            "buyer2",
            "seller2",
            "symbol",
            "K",
            "n_second_leg",
            "mean_fwd_extract",
            "median",
            "frac_pos",
            "t_stat",
            "chain_count_hop2_table",
        ]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in motif_rows:
            w.writerow({c: ("" if row.get(c) is None else row[c]) for c in cols})

    with (OUT / "r4_phase1_pair_forward_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["buyer", "seller", "symbol", "K", "n", "mean_fwd", "median_fwd", "frac_pos", "t_stat"],
        )
        w.writeheader()
        for row in pair_rows[:5000]:
            w.writerow(row)

    # Per-Mark U: aggregate when U is buyer vs seller (aggressive buys: buyer==U and aggr buyer)
    mark_stats: dict[tuple, list[float]] = defaultdict(list)
    for rec in rows_out:
        sym = rec["symbol"]
        ag = rec["aggressor"]
        for U in sorted(all_names):
            for K in K_HORIZONS:
                key = rec.get(f"fwd_{K}_{sym}")
                if key is None:
                    continue
                if rec["buyer"] == U:
                    mark_stats[(U, "as_buyer", sym, K)].append(float(key))
                if rec["seller"] == U:
                    mark_stats[(U, "as_seller", sym, K)].append(float(key))
                if rec["buyer"] == U and ag == "buyer":
                    mark_stats[(U, "buyer_aggr", sym, K)].append(float(key))
                if rec["seller"] == U and ag == "seller":
                    mark_stats[(U, "seller_aggr", sym, K)].append(float(key))

    mark_out: list[dict] = []
    for (U, role, sym, K), vals in mark_stats.items():
        if len(vals) < 15:
            continue
        mark_out.append(
            {
                "mark": U,
                "role": role,
                "symbol": sym,
                "K": K,
                "n": len(vals),
                "mean_fwd": statistics.mean(vals),
                "t_stat": t_stat(vals),
                "frac_pos": sum(1 for x in vals if x > 0) / len(vals),
            }
        )
    mark_out.sort(key=lambda x: (-abs(x["mean_fwd"]) if x["t_stat"] else 0, -x["n"]))
    with (OUT / "r4_phase1_mark_forward_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["mark", "role", "symbol", "K", "n", "mean_fwd", "t_stat", "frac_pos"],
        )
        w.writeheader()
        for row in mark_out:
            w.writerow({k: ("" if row[k] is None else row[k]) for k in row})

    # Graph edges
    edge_c = Counter()
    edge_q = Counter()
    for rec in rows_out:
        b, s = rec["buyer"], rec["seller"]
        edge_c[(b, s)] += 1
        edge_q[(b, s)] += int(rec["quantity"])
    with (OUT / "r4_phase1_graph_edges.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["buyer", "seller", "count", "qty"])
        for (b, s), c in edge_c.most_common(200):
            w.writerow([b, s, c, edge_q[(b, s)]])

    # --- Graph hubs + reciprocity (directed buyer→seller)
    out_cnt: Counter[str] = Counter()
    in_cnt: Counter[str] = Counter()
    out_qty: Counter[str] = Counter()
    in_qty: Counter[str] = Counter()
    neigh_out: dict[str, set[str]] = defaultdict(set)
    neigh_in: dict[str, set[str]] = defaultdict(set)
    for (b, s), c in edge_c.items():
        out_cnt[b] += c
        in_cnt[s] += c
        out_qty[b] += edge_q[(b, s)]
        in_qty[s] += edge_q[(b, s)]
        neigh_out[b].add(s)
        neigh_in[s].add(b)
    deg_rows: list[dict] = []
    for U in sorted(all_names):
        deg_rows.append(
            {
                "mark": U,
                "prints_as_buyer": out_cnt.get(U, 0),
                "prints_as_seller": in_cnt.get(U, 0),
                "qty_as_buyer": int(out_qty.get(U, 0)),
                "qty_as_seller": int(in_qty.get(U, 0)),
                "n_distinct_sellers_bought_from": len(neigh_out.get(U, ())),
                "n_distinct_buyers_sold_to": len(neigh_in.get(U, ())),
            }
        )
    with (OUT / "r4_phase1_graph_degree_by_mark.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "mark",
                "prints_as_buyer",
                "prints_as_seller",
                "qty_as_buyer",
                "qty_as_seller",
                "n_distinct_sellers_bought_from",
                "n_distinct_buyers_sold_to",
            ],
        )
        w.writeheader()
        for row in deg_rows:
            w.writerow(row)

    recip_rows: list[dict] = []
    seen_e: set[tuple[str, str]] = set()
    for (b, s), c_ab in edge_c.most_common(80):
        if not b or not s:
            continue
        if (b, s) in seen_e:
            continue
        seen_e.add((b, s))
        seen_e.add((s, b))
        c_ba = edge_c.get((s, b), 0)
        q_ab = edge_q.get((b, s), 0)
        q_ba = edge_q.get((s, b), 0)
        recip_rows.append(
            {
                "buyer_a": b,
                "seller_a": s,
                "count_a_to_b": c_ab,
                "qty_a_to_b": int(q_ab),
                "count_b_to_a": int(c_ba),
                "qty_b_to_a": int(q_ba),
                "reciprocity_count_ratio": (round(c_ba / c_ab, 4) if c_ab else ""),
            }
        )
    with (OUT / "r4_phase1_graph_top_edges_reciprocity.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "buyer_a",
                "seller_a",
                "count_a_to_b",
                "qty_a_to_b",
                "count_b_to_a",
                "qty_b_to_a",
                "reciprocity_count_ratio",
            ],
        )
        w.writeheader()
        for row in recip_rows:
            w.writerow(row)

    # Bursts summary
    with (OUT / "r4_phase1_bursts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["csv_day", "timestamp", "n_prints", "top_buyer", "n_distinct_symbols"])
        for row in burst_groups:
            w.writerow(row)

    # Baseline: mean fwd per (buyer,seller,symbol,K) already in pair_rows — residual vs global mean per symbol
    glob_mean: dict[tuple[str, int], float] = {}
    for sym in PRODUCTS:
        for K in K_HORIZONS:
            acc: list[float] = []
            for rec in rows_out:
                if rec["symbol"] != sym:
                    continue
                v = rec.get(f"fwd_{K}_{sym}")
                if v is not None:
                    acc.append(float(v))
            glob_mean[(sym, K)] = statistics.mean(acc) if acc else 0.0

    resid_rows: list[dict] = []
    for row in pair_rows:
        gm = glob_mean.get((row["symbol"], row["K"]), 0.0)
        resid_rows.append(
            {
                **row,
                "global_mean_fwd": gm,
                "residual_mean": row["mean_fwd"] - gm,
            }
        )
    resid_rows.sort(key=lambda x: -abs(x["residual_mean"]))
    with (OUT / "r4_phase1_pair_residuals.csv").open("w", newline="") as f:
        cols = [
            "buyer",
            "seller",
            "symbol",
            "K",
            "n",
            "mean_fwd",
            "global_mean_fwd",
            "residual_mean",
            "t_stat",
        ]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in resid_rows[:2000]:
            w.writerow({c: row[c] for c in cols})

    # Adverse: Mark 01→22 on VEV_5300 aggressive buy fwd
    adv_lines = []
    for rec in rows_out:
        if rec["buyer"] == "Mark 01" and rec["seller"] == "Mark 22" and rec["symbol"] == "VEV_5300":
            for K in K_HORIZONS:
                v = rec.get(f"fwd_{K}_VELVETFRUIT_EXTRACT")
                if v is not None:
                    adv_lines.append((K, rec["aggressor"], v))
    with (OUT / "r4_phase1_adverse_extract_after_01_22_5300.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "aggressor", "fwd_extract"])
        for line in adv_lines:
            w.writerow(line)

    # Text summary + top edges
    lines = [
        "Round 4 Phase 1 — automated summary",
        "====================================",
        f"Trade rows analyzed: {len(rows_out)}",
        f"Distinct participant names (buyer ∪ seller, non-empty): {n_participants}",
        f"Aggregated (U, role, fwd_target, out_symbol, K) cells n>=5: {n_participant_u_cells} (see r4_phase1_participant_U_forward.csv; boot_mean_lo/hi = {BOOTSTRAP_B} resamples)",
        f"Stratified by trade-symbol spread tertile (per day grid), n>=5: {n_participant_u_tert_cells} (r4_phase1_participant_U_spread_tert.csv)",
        f"Stratified by **session** = early/mid/late third of **tick index** per csv day, n>=5: {n_participant_u_sess_cells} (r4_phase1_participant_U_session_terc.csv)",
        f"Stratified burst_iso = isolated | burst_single_sym | burst_multi_sym, n>=5: {n_participant_u_burst_cells} (r4_phase1_participant_U_burst_iso.csv)",
        f"Participant signed qty balance (buy qty minus sell qty) per symbol: r4_phase1_participant_flow.csv",
        f"Key pair **stability by csv day** (top-|t| pairs × all K): r4_phase1_key_pairs_stability_by_day.csv",
        f"Mark 67 buyer on extract, per day × K: r4_phase1_mark67_extract_buyer_by_day.csv",
        burst_ctrl_note,
        f"Unique (buyer,seller,symbol,K) pair cells (n>=5): {len(pair_rows)}",
        f"Same-timestamp multi-print bursts: {len(burst_groups)}",
        "",
        "Graph (see graph_degree_by_mark.csv, top_edges_reciprocity.csv):",
        "  Mark 01 prints_as_buyer / seller and distinct_sellers (from degree CSV): check file for exact counts.",
        "  Motif second-leg extract fwd: see r4_phase1_motif_secondleg_extract_fwd.csv (top chains only).",
        "",
        "Top 10 participant-U cells (|t_stat|) with n>=30:",
    ]
    u_top = [r for r in u_rows if r["n"] >= 30 and r.get("t_stat") is not None]
    u_top.sort(key=lambda x: -abs(x["t_stat"] or 0))
    for r in u_top[:10]:
        lines.append(
            f"  {r['mark']} {r['role']} {r['fwd_target']} {r['out_symbol']} K={r['K']} n={r['n']} mean={r['mean']:.4f} t={r['t_stat']:.3f}"
        )
    lines += [
        "",
        "Top 15 (buyer,seller,symbol,K) by |t_stat| among n>=30:",
    ]
    top = [r for r in pair_rows if r["n"] >= 30 and r["t_stat"] is not None]
    top.sort(key=lambda x: -abs(x["t_stat"]))
    for r in top[:15]:
        lines.append(
            f"  {r['buyer']}->{r['seller']} {r['symbol']} K={r['K']} n={r['n']} mean={r['mean_fwd']:.4f} t={r['t_stat']:.3f}"
        )
    lines += ["", "Mark-conditioned (n>=15) top 10 by |mean_fwd|:"]
    for r in mark_out[:10]:
        lines.append(
            f"  {r['mark']} {r['role']} {r['symbol']} K={r['K']} n={r['n']} mean={r['mean_fwd']:.4f} t={r['t_stat']}"
        )
    (OUT / "r4_phase1_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # Minimal analysis.json phase1 gate (filled in file next)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
