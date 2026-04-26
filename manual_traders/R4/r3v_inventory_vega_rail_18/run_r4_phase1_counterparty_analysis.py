#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned forward mids (tape-only).

Horizon K: number of price **ticks** (timestamp index steps of 100) forward on the
aligned grid shared by all products (verified identical ts sets per day).

Outputs under manual_traders/R4/r3v_inventory_vega_rail_18/analysis_outputs/:
  - r4_phase1_forward_by_mark.csv
  - r4_phase1_pair_baseline_residuals.csv
  - r4_phase1_graph_edges.csv
  - r4_phase1_bursts.csv
  - r4_phase1_adverse_aggressor.csv
  - r4_phase1_summary.txt
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

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

    for csv_day in csv_days:
        tss, mids, sprs = grids[csv_day]
        idx_of = {ts: i for i, ts in enumerate(tss)}
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
        for (d0, ts0), lst in by_burst.items():
            if len(lst) <= 1:
                continue
            buyers = [x["buyer"] for x in lst]
            orch = Counter(buyers).most_common(1)[0][0]
            syms = len({x["symbol"] for x in lst})
            burst_groups.append((d0, ts0, len(lst), orch, syms))

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

            rec: dict = {
                "csv_day": csv_day,
                "timestamp": ts,
                "symbol": sym,
                "buyer": buyer,
                "seller": seller,
                "price": price,
                "quantity": qty,
                "mid_t": mid0,
                "spread_t": spr0,
                "aggressor": aggr_side,
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
        for U in ["Mark 01", "Mark 14", "Mark 22", "Mark 38", "Mark 49", "Mark 55", "Mark 67"]:
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
        f"Unique (buyer,seller,symbol,K) pair cells (n>=5): {len(pair_rows)}",
        f"Same-timestamp multi-print bursts: {len(burst_groups)}",
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
