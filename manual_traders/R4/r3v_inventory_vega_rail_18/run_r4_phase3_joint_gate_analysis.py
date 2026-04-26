#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate (s5200<=TH, s5300<=TH) on R4 tape + inclineGod
spread–spread / spread–price + three-way (Mark pair × gate × product).

Convention matches round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py:
inner-join timestamps where VEV_5200, VEV_5300, VELVETFRUIT_EXTRACT exist; tight = both spreads <= TH.

Outputs: manual_traders/R4/r3v_inventory_vega_rail_18/analysis_outputs/r4_phase3_*
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = REPO / "manual_traders/R4/r3v_inventory_vega_rail_18" / "analysis_outputs"

TH = 2
K_MAIN = 20  # match R3 script default for forward extract
K_LIST = (5, 20, 100)
PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]
FOCUS_PAIRS = [
    ("Mark 67", "Mark 22", "VELVETFRUIT_EXTRACT"),
    ("Mark 67", "Mark 49", "VELVETFRUIT_EXTRACT"),
    ("Mark 01", "Mark 22", "VEV_5300"),
    ("Mark 01", "Mark 22", "VELVETFRUIT_EXTRACT"),
    ("Mark 55", "Mark 14", "VELVETFRUIT_EXTRACT"),
]


def load_grid(csv_day: int):
    path = DATA / f"prices_round_4_day_{csv_day}.csv"
    by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open() as f:
        for row in csv.DictReader(f, delimiter=";"):
            ts = int(row["timestamp"])
            prod = row["product"]
            bb, ba = row.get("bid_price_1"), row.get("ask_price_1")
            if not bb or not ba:
                continue
            bid, ask = int(bb), int(ba)
            if ask < bid:
                continue
            by_ts[ts][prod] = {
                "mid": 0.5 * (bid + ask),
                "spr": int(ask - bid),
            }
    tss = sorted(by_ts)
    n = len(tss)
    idx = {tss[i]: i for i in range(n)}
    mids = {p: [float("nan")] * n for p in PRODUCTS}
    sprs = {p: [-1] * n for p in PRODUCTS}
    for i, ts in enumerate(tss):
        d = by_ts[ts]
        for p in PRODUCTS:
            if p not in d:
                continue
            mids[p][i] = float(d[p]["mid"])
            sprs[p][i] = int(d[p]["spr"])
    tight = [False] * n
    for i in range(n):
        s52, s53 = sprs["VEV_5200"][i], sprs["VEV_5300"][i]
        tight[i] = s52 >= 0 and s53 >= 0 and s52 <= TH and s53 <= TH
    return tss, idx, n, mids, sprs, tight


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


def welch_simple(a: list[float], b: list[float]) -> dict:
    """Return means and rough t (not full Welch without scipy); use difference of means / pooled stderr approx."""
    a = [x for x in a if math.isfinite(x)]
    b = [x for x in b if math.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return {"na": True}
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa = statistics.stdev(a) if len(a) > 1 else 0.0
    sb = statistics.stdev(b) if len(b) > 1 else 0.0
    se = math.sqrt((sa * sa) / len(a) + (sb * sb) / len(b) + 1e-18)
    return {
        "mean_a": ma,
        "mean_b": mb,
        "diff": ma - mb,
        "t_approx": (ma - mb) / se,
        "n_a": len(a),
        "n_b": len(b),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 30:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    num = sum(dx[i] * dy[i] for i in range(len(xs)))
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy) + 1e-18)
    if den < 1e-12:
        return None
    return float(num / den)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    days = [1, 2, 3]
    grids = {d: load_grid(d) for d in days}

    # --- Pooled arrays for spread correlations (all ticks with valid triple) ---
    s52_all: list[float] = []
    s53_all: list[float] = []
    sext_all: list[float] = []
    s5100_all: list[float] = []
    dm_ext: list[float] = []  # same-tick change vs next (need align i and i+1)
    s52_for_dm: list[float] = []
    s53_for_dm: list[float] = []

    for d in days:
        _, _, n, mids, sprs, tight = grids[d]
        for i in range(n):
            s52, s53 = sprs["VEV_5200"][i], sprs["VEV_5300"][i]
            se = sprs["VELVETFRUIT_EXTRACT"][i]
            if s52 < 0 or s53 < 0 or se < 0:
                continue
            s52_all.append(float(s52))
            s53_all.append(float(s53))
            sext_all.append(float(se))
            s5100_all.append(float(sprs["VEV_5100"][i]) if sprs["VEV_5100"][i] >= 0 else float("nan"))
            if i + 1 < n:
                m0, m1 = mids["VELVETFRUIT_EXTRACT"][i], mids["VELVETFRUIT_EXTRACT"][i + 1]
                if not math.isnan(m0) and not math.isnan(m1):
                    dm_ext.append(float(m1 - m0))
                    s52_for_dm.append(float(s52))
                    s53_for_dm.append(float(s53))

    spread_corr = {
        "pearson_s5200_s5300": pearson(s52_all, s53_all),
        "pearson_s5200_s_extract": pearson(s52_all, sext_all),
        "pearson_s5300_s_extract": pearson(s53_all, sext_all),
        "n_rows": len(s52_all),
    }
    s5100_clean = [(a, b) for a, b in zip(s52_all, s5100_all) if math.isfinite(b)]
    if len(s5100_clean) > 30:
        xa, xb = zip(*s5100_clean)
        spread_corr["pearson_s5200_s5100"] = pearson(list(xa), list(xb))
    spread_corr["pearson_s5200_dm_extract_next_tick"] = pearson(s52_for_dm, dm_ext)
    spread_corr["pearson_s5300_dm_extract_next_tick"] = pearson(s53_for_dm, dm_ext)

    (OUT / "r4_phase3_spread_correlations.json").write_text(json.dumps(spread_corr, indent=2), encoding="utf-8")

    # --- Joint gate: forward extract K=20 tight vs not (R3 primary stat, R4 tape) ---
    fwd20_t, fwd20_n = [], []
    for d in days:
        _, _, n, mids, sprs, tight = grids[d]
        for i in range(n - K_MAIN):
            m0, m1 = mids["VELVETFRUIT_EXTRACT"][i], mids["VELVETFRUIT_EXTRACT"][i + K_MAIN]
            if math.isnan(m0) or math.isnan(m1):
                continue
            dv = float(m1 - m0)
            if tight[i]:
                fwd20_t.append(dv)
            else:
                fwd20_n.append(dv)
    gate_extract = {
        "K": K_MAIN,
        "n_tight": len(fwd20_t),
        "n_not_tight": len(fwd20_n),
        "mean_fwd_extract_tight": statistics.mean(fwd20_t) if fwd20_t else None,
        "mean_fwd_extract_not_tight": statistics.mean(fwd20_n) if fwd20_n else None,
        "welch_style": welch_simple(fwd20_t, fwd20_n),
    }
    (OUT / "r4_phase3_gate_forward_extract_k20.json").write_text(json.dumps(gate_extract, indent=2), encoding="utf-8")

    # --- Trades: three-way (buyer, seller, symbol) × tight × fwd K ---
    three_way: dict[tuple, dict] = defaultdict(lambda: {"tight": [], "wide": []})
    for d in days:
        tss, idx, n, mids, sprs, tight = grids[d]
        path = DATA / f"trades_round_4_day_{d}.csv"
        for row in csv.DictReader(path.open(), delimiter=";"):
            sym = row["symbol"]
            if sym not in mids:
                continue
            ts = int(row["timestamp"])
            if ts not in idx:
                continue
            j = idx[ts]
            b, s = row.get("buyer") or "", row.get("seller") or ""
            for K in K_LIST:
                if j + K >= n:
                    continue
                m0, m1 = mids[sym][j], mids[sym][j + K]
                if math.isnan(m0) or math.isnan(m1):
                    continue
                fv = float(m1 - m0)
                key = (b, s, sym, K)
                if tight[j]:
                    three_way[key]["tight"].append(fv)
                else:
                    three_way[key]["wide"].append(fv)

    rows_3 = []
    for (b, s, sym, K), v in three_way.items():
        nt, nw = len(v["tight"]), len(v["wide"])
        if nt < 8 or nw < 15:
            continue
        mt, mw = statistics.mean(v["tight"]), statistics.mean(v["wide"])
        rows_3.append(
            {
                "buyer": b,
                "seller": s,
                "symbol": sym,
                "K": K,
                "n_tight": nt,
                "mean_tight": mt,
                "n_wide": nw,
                "mean_wide": mw,
                "diff_tight_minus_wide": mt - mw,
                "t_tight": t_stat(v["tight"]),
                "t_wide": t_stat(v["wide"]),
            }
        )
    rows_3.sort(key=lambda x: -abs(x["diff_tight_minus_wide"]))
    with (OUT / "r4_phase3_threeway_pair_gate_fwd.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "buyer",
                "seller",
                "symbol",
                "K",
                "n_tight",
                "mean_tight",
                "n_wide",
                "mean_wide",
                "diff_tight_minus_wide",
                "t_tight",
                "t_wide",
            ],
        )
        w.writeheader()
        for row in rows_3[:150]:
            w.writerow(row)

    # --- Focus pairs: table tight vs wide for each K ---
    focus_out = []
    for b, s, sym in FOCUS_PAIRS:
        for K in K_LIST:
            key = (b, s, sym, K)
            if key not in three_way:
                continue
            v = three_way[key]
            nt, nw = len(v["tight"]), len(v["wide"])
            if nt < 3 and nw < 3:
                continue
            mt = statistics.mean(v["tight"]) if v["tight"] else None
            mw = statistics.mean(v["wide"]) if v["wide"] else None
            focus_out.append(
                {
                    "buyer": b,
                    "seller": s,
                    "symbol": sym,
                    "K": K,
                    "n_tight": nt,
                    "mean_tight": mt,
                    "n_wide": nw,
                    "mean_wide": mw,
                }
            )
    (OUT / "r4_phase3_focus_pairs_gate.json").write_text(json.dumps(focus_out, indent=2), encoding="utf-8")

    # --- Mark 01->22 VEV_5300: same-symbol fwd in tight only vs wide only (explicit Sonic interaction) ---
    m01_22 = [r for r in rows_3 if r["buyer"] == "Mark 01" and r["seller"] == "Mark 22" and r["symbol"] == "VEV_5300"]
    (OUT / "r4_phase3_mark01_mark22_5300_gate_slice.json").write_text(json.dumps(m01_22, indent=2), encoding="utf-8")

    lines = [
        "Round 4 Phase 3 — joint gate + spread panels",
        "===========================================",
        f"Pooled pearson s5200,s5300: {spread_corr.get('pearson_s5200_s5300')}",
        f"Forward extract K={K_MAIN} mean tight vs not: {gate_extract.get('mean_fwd_extract_tight')} vs {gate_extract.get('mean_fwd_extract_not_tight')}",
        f"Three-way rows (n_tight>=8, n_wide>=15): {len(rows_3)}",
        "See r4_phase3_*.csv/json",
    ]
    (OUT / "r4_phase3_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
