#!/usr/bin/env python3
"""
Round 4 Phase 2 — orthogonal edges (tape): burst windows, microprice/spreads,
lead–lag signed flow, Sonic joint-gate stratification, smile residual at prints,
adverse summary.

Reads Prosperity4Data/ROUND_4 days 1–3. Writes analysis_outputs/r4_phase2_*.
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

COEFF = [0.14215151147708086, -0.0016298611395181932, 0.23576325646627055]
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    *[f"VEV_{k}" for k in STRIKES],
]
# Historical tape day index 1,2,3 -> open DTE 4,3,2 (analogous to R3 8,7,6 for days 0,1,2)
def dte_open(csv_day: int) -> float:
    return max(float(5 - int(csv_day)), 1.0)
W_TICKS = 5  # ±5 grid steps (500 timestamp units) around burst center


def load_grid(csv_day: int):
    path = DATA / f"prices_round_4_day_{csv_day}.csv"
    by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open() as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            ts = int(row["timestamp"])
            prod = row["product"]
            bb, ba = row.get("bid_price_1"), row.get("ask_price_1")
            if not bb or not ba:
                continue
            bid, ask = int(bb), int(ba)
            if ask < bid:
                continue
            bv = int(row.get("bid_volume_1") or 0)
            av = int(row.get("ask_volume_1") or 0)
            by_ts[ts][prod] = {
                "mid": 0.5 * (bid + ask),
                "spr": ask - bid,
                "bid": bid,
                "ask": ask,
                "bv": bv,
                "av": av,
            }
    tss = sorted(by_ts)
    n = len(tss)
    idx = {tss[i]: i for i in range(n)}

    mids = {p: [float("nan")] * n for p in PRODUCTS}
    sprs = {p: [-1] * n for p in PRODUCTS}
    micro = {p: [float("nan")] * n for p in PRODUCTS}
    for i, ts in enumerate(tss):
        d = by_ts[ts]
        for p in PRODUCTS:
            if p not in d:
                continue
            x = d[p]
            mids[p][i] = float(x["mid"])
            sprs[p][i] = int(x["spr"])
            den = x["bv"] + x["av"]
            if den > 0:
                micro[p][i] = (x["bid"] * x["av"] + x["ask"] * x["bv"]) / den
            else:
                micro[p][i] = float(x["mid"])
    return tss, idx, n, mids, sprs, micro, by_ts


def dte_years(csv_day: int, ts: int) -> float:
    d0 = dte_open(csv_day)
    prog = (ts // 100) / 10_000.0
    return max(d0 - prog, 1e-6) / 365.0


def iv_smile(S: float, K: float, T: float) -> float:
    m = math.log(K / S) / math.sqrt(T)
    c0, c1, c2 = COEFF
    return float(c0 + c1 * m + c2 * m * m)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def joint_tight(s52: int, s53: int) -> bool:
    return s52 >= 0 and s53 >= 0 and s52 <= 2 and s53 <= 2


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

    # Burst centers: (day, ts) with >=2 trades same ts AND exists Mark01->Mark22 in batch
    burst_01_22: set[tuple[int, int]] = set()
    all_burst_ts: set[tuple[int, int]] = set()
    trades_by_day: dict[int, list[dict]] = {}

    for d in csv_days:
        path = DATA / f"trades_round_4_day_{d}.csv"
        rows = list(csv.DictReader(path.open(), delimiter=";"))
        trades_by_day[d] = rows
        by_ts: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_ts[int(row["timestamp"])].append(row)
        for ts, lst in by_ts.items():
            if len(lst) <= 1:
                continue
            all_burst_ts.add((d, ts))
            if any(
                x.get("buyer") == "Mark 01" and x.get("seller") == "Mark 22" for x in lst
            ):
                burst_01_22.add((d, ts))

    grids = {d: load_grid(d) for d in csv_days}

    # --- 1) Pair forward in ±W window around Mark01->Mark22 burst centers ---
    window_fwd: dict[tuple, list[float]] = defaultdict(list)
    baseline_fwd: dict[tuple, list[float]] = defaultdict(list)

    for d in csv_days:
        tss, idx, n, mids, sprs, micro, _ = grids[d]
        for row in trades_by_day[d]:
            sym = row["symbol"]
            if sym not in mids:
                continue
            ts = int(row["timestamp"])
            if ts not in idx:
                continue
            j = idx[ts]
            buyer, seller = row.get("buyer") or "", row.get("seller") or ""
            key = (buyer, seller, sym, 5)
            if j + 5 >= n:
                continue
            m0, m1 = mids[sym][j], mids[sym][j + 5]
            if math.isnan(m0) or math.isnan(m1):
                continue
            fwd = float(m1 - m0)
            baseline_fwd[key].append(fwd)
            near_burst = False
            for (bd, bts) in burst_01_22:
                if bd != d:
                    continue
                if abs(bts - ts) <= W_TICKS * 100:
                    near_burst = True
                    break
            if near_burst:
                window_fwd[key].append(fwd)

    burst_rows = []
    for key in sorted(set(baseline_fwd) | set(window_fwd)):
        b, s, sym, K = key
        ba = baseline_fwd.get(key, [])
        wn = window_fwd.get(key, [])
        if len(ba) < 5 and len(wn) < 3:
            continue
        burst_rows.append(
            {
                "buyer": b,
                "seller": s,
                "symbol": sym,
                "K": K,
                "n_baseline": len(ba),
                "mean_baseline": statistics.mean(ba) if ba else None,
                "n_burst_window": len(wn),
                "mean_burst_window": statistics.mean(wn) if wn else None,
            }
        )
    with (OUT / "r4_phase2_burst_window_fwd_k5.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "buyer",
                "seller",
                "symbol",
                "K",
                "n_baseline",
                "mean_baseline",
                "n_burst_window",
                "mean_burst_window",
            ],
        )
        w.writeheader()
        for row in sorted(burst_rows, key=lambda x: -abs((x["mean_burst_window"] or 0) - (x["mean_baseline"] or 0)))[:200]:
            w.writerow(row)

    # --- 2) Microprice: spread quantile vs forward |Δmid| VEV_5300 ---
    spr_u5300: list[tuple[int, int, int, float, float]] = []  # day, i, spr5300, fwd_abs, spr_u
    for d in csv_days:
        tss, _, n, mids, sprs, micro, _ = grids[d]
        for i in range(n - 5):
            s530 = sprs["VEV_5300"][i]
            if s530 < 0:
                continue
            m0, m1 = mids["VEV_5300"][i], mids["VEV_5300"][i + 5]
            if math.isnan(m0) or math.isnan(m1):
                continue
            su = sprs["VELVETFRUIT_EXTRACT"][i]
            spr_u5300.append((d, i, s530, abs(float(m1 - m0)), float(su) if su >= 0 else -1.0))
    spr_vals = [x[2] for x in spr_u5300]
    q33 = sorted(spr_vals)[len(spr_vals) // 3] if spr_vals else 0
    q66 = sorted(spr_vals)[2 * len(spr_vals) // 3] if spr_vals else 0
    reg_fwd: dict[str, list[float]] = defaultdict(list)
    for d, i, s530, fa, _su in spr_u5300:
        if s530 <= q33:
            reg_fwd["tight_s5300"].append(fa)
        elif s530 >= q66:
            reg_fwd["wide_s5300"].append(fa)
        else:
            reg_fwd["mid_s5300"].append(fa)
    micro_out = {
        "quantiles_5300_spread": {"q33": q33, "q66": q66},
        "mean_abs_fwd5_VEV_5300_mid": {
            k: float(statistics.mean(v)) if v else None for k, v in reg_fwd.items()
        },
        "n": {k: len(v) for k, v in reg_fwd.items()},
    }
    (OUT / "r4_phase2_microprice_spread_regimes.json").write_text(
        json.dumps(micro_out, indent=2), encoding="utf-8"
    )

    # --- 3) Lead–lag signed flow: correlate extract signed flow with VEV_5300 at lag L ---
    max_lag = 15
    corrs = []
    for d in csv_days:
        tss, idx, n, mids, sprs, micro, by_ts = grids[d]
        fu = [0.0] * n
        f5 = [0.0] * n
        for row in trades_by_day[d]:
            sym = row["symbol"]
            if sym not in ("VELVETFRUIT_EXTRACT", "VEV_5300"):
                continue
            ts = int(row["timestamp"])
            if ts not in idx:
                continue
            j = idx[ts]
            price = float(row["price"])
            qty = float(row.get("quantity") or 0)
            buyer, seller = row.get("buyer") or "", row.get("seller") or ""
            dd = by_ts[ts].get(sym)
            if not dd:
                continue
            mid = dd["mid"]
            spr = dd["spr"]
            half = spr / 2.0 if spr > 0 else 0.0
            bid_ap = int(round(mid - half))
            ask_ap = int(round(mid + half))
            sgn = 0.0
            if buyer and seller:
                if price >= ask_ap - 1e-9:
                    sgn = qty
                elif price <= bid_ap + 1e-9:
                    sgn = -qty
            if sym == "VELVETFRUIT_EXTRACT":
                fu[j] += sgn
            else:
                f5[j] += sgn
        def pearson(a: list[float], b: list[float]) -> float | None:
            if len(a) != len(b) or len(a) < 50:
                return None
            ma, mb = statistics.mean(a), statistics.mean(b)
            da = [x - ma for x in a]
            db = [x - mb for x in b]
            num = sum(da[i] * db[i] for i in range(len(a)))
            den = math.sqrt(sum(x * x for x in da) * sum(x * x for x in db) + 1e-18)
            if den < 1e-12:
                return None
            return float(num / den)

        for lag in range(0, max_lag + 1):
            xs = [fu[i] for i in range(n - lag)]
            ys = [f5[i + lag] for i in range(n - lag)]
            pr = pearson(xs, ys)
            if pr is None:
                continue
            corrs.append({"csv_day": d, "lag_ticks": lag, "pearson_extract_vs_5300_flow": pr, "n": len(xs)})

    with (OUT / "r4_phase2_leadlag_signed_flow.json").open("w") as f:
        json.dump(corrs, f, indent=2)

    # --- 4) Joint gate stratify: Mark67->22 extract fwd K=5 tight vs wide ---
    tight_fwd: list[float] = []
    wide_fwd: list[float] = []
    for d in csv_days:
        tss, idx, n, mids, sprs, _, _ = grids[d]
        for row in trades_by_day[d]:
            if row.get("buyer") != "Mark 67" or row.get("seller") != "Mark 22":
                continue
            if row["symbol"] != "VELVETFRUIT_EXTRACT":
                continue
            ts = int(row["timestamp"])
            if ts not in idx:
                continue
            j = idx[ts]
            s52, s53 = sprs["VEV_5200"][j], sprs["VEV_5300"][j]
            if j + 5 >= n:
                continue
            m0, m1 = mids["VELVETFRUIT_EXTRACT"][j], mids["VELVETFRUIT_EXTRACT"][j + 5]
            if math.isnan(m0) or math.isnan(m1):
                continue
            fwd = float(m1 - m0)
            if joint_tight(s52, s53):
                tight_fwd.append(fwd)
            else:
                wide_fwd.append(fwd)
    gate_out = {
        "Mark67_to_Mark22_VELVETFRUIT_EXTRACT_fwd5": {
            "n_joint_tight_s5200_s5300_le2": len(tight_fwd),
            "mean_fwd_tight": statistics.mean(tight_fwd) if tight_fwd else None,
            "n_wide_gate": len(wide_fwd),
            "mean_fwd_wide": statistics.mean(wide_fwd) if wide_fwd else None,
            "t_tight": t_stat(tight_fwd),
            "t_wide": t_stat(wide_fwd),
        }
    }
    (OUT / "r4_phase2_joint_gate_stratify.json").write_text(json.dumps(gate_out, indent=2), encoding="utf-8")

    # --- 4b) Session bucket: early vs late tape (by timestamp tercile) same gate stat ---
    early_t, late_t = [], []
    for d in csv_days:
        tss, idx, n, mids, sprs, _, _ = grids[d]
        if not tss:
            continue
        cut = int(tss[0] + (tss[-1] - tss[0]) / 3.0)
        for row in trades_by_day[d]:
            if row.get("buyer") != "Mark 67" or row.get("seller") != "Mark 22":
                continue
            if row["symbol"] != "VELVETFRUIT_EXTRACT":
                continue
            ts = int(row["timestamp"])
            if ts not in idx:
                continue
            j = idx[ts]
            if j + 5 >= n:
                continue
            m0, m1 = mids["VELVETFRUIT_EXTRACT"][j], mids["VELVETFRUIT_EXTRACT"][j + 5]
            if math.isnan(m0) or math.isnan(m1):
                continue
            fwd = float(m1 - m0)
            s52, s53 = sprs["VEV_5200"][j], sprs["VEV_5300"][j]
            if not joint_tight(s52, s53):
                continue
            if ts < cut:
                early_t.append(fwd)
            else:
                late_t.append(fwd)
    (OUT / "r4_phase2_session_tercile_gate.json").write_text(
        json.dumps(
            {
                "joint_tight_Mark67_Mark22_extract_fwd5": {
                    "early_tercile_n": len(early_t),
                    "early_mean": statistics.mean(early_t) if early_t else None,
                    "late_n": len(late_t),
                    "late_mean": statistics.mean(late_t) if late_t else None,
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- 5) Smile residual magnitude when Mark01->Mark22 on any VEV ---
    residuals_at_print: list[float] = []
    for d in csv_days:
        tss, idx, n, mids, sprs, _, _ = grids[d]
        for row in trades_by_day[d]:
            if row.get("buyer") != "Mark 01" or row.get("seller") != "Mark 22":
                continue
            sym = row["symbol"]
            if not sym.startswith("VEV_"):
                continue
            ts = int(row["timestamp"])
            if ts not in idx:
                continue
            j = idx[ts]
            K = int(sym.split("_")[1])
            S = mids["VELVETFRUIT_EXTRACT"][j]
            vm = mids[sym][j]
            if math.isnan(S) or math.isnan(vm):
                continue
            T = dte_years(d, ts)
            sig = iv_smile(S, K, T)
            theo = bs_call(S, K, T, sig)
            residuals_at_print.append(abs(float(vm) - theo))

    smile_out = {
        "n_prints": len(residuals_at_print),
        "mean_abs_mid_minus_bs": float(statistics.mean(residuals_at_print))
        if residuals_at_print
        else None,
    }
    (OUT / "r4_phase2_smile_residual_01_22_vev.json").write_text(json.dumps(smile_out, indent=2), encoding="utf-8")

    # --- 6) Worst (buyer,seller,symbol) mean fwd K=5 for passive seller story ---
    pair_acc: dict[tuple, list[float]] = defaultdict(list)
    for d in csv_days:
        tss, idx, n, mids, _, _, _ = grids[d]
        for row in trades_by_day[d]:
            sym = row["symbol"]
            if sym not in mids:
                continue
            ts = int(row["timestamp"])
            if ts not in idx or idx[ts] + 5 >= n:
                continue
            j = idx[ts]
            m0, m1 = mids[sym][j], mids[sym][j + 5]
            if math.isnan(m0) or math.isnan(m1):
                continue
            b, s = row.get("buyer") or "", row.get("seller") or ""
            pair_acc[(b, s, sym)].append(float(m1 - m0))
    worst = []
    for (b, s, sym), vals in pair_acc.items():
        if len(vals) < 10:
            continue
        worst.append((statistics.mean(vals), len(vals), b, s, sym))
    worst.sort()
    with (OUT / "r4_phase2_worst_mean_fwd_k5_pairs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mean_fwd5", "n", "buyer", "seller", "symbol"])
        for row in worst[:40]:
            w.writerow(row)

    lines = [
        "Round 4 Phase 2 summary",
        "=======================",
        f"Mark01->Mark22 burst centers (multi-print timestamps): {len(burst_01_22)}",
        f"All multi-print burst timestamps: {len(all_burst_ts)}",
        f"Joint-gate stratify Mark67->22 extract fwd5: tight n={len(tight_fwd)} mean={statistics.mean(tight_fwd) if tight_fwd else 'n/a'} | wide n={len(wide_fwd)} mean={statistics.mean(wide_fwd) if wide_fwd else 'n/a'}",
        f"Mean |mid-BS| at Mark01->Mark22 VEV prints: {smile_out.get('mean_abs_mid_minus_bs')}",
        "See r4_phase2_*.csv/json for full tables.",
    ]
    (OUT / "r4_phase2_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
