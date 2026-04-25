#!/usr/bin/env python3
"""
Round 3 tapes: lead/lag between underlying Δmid and voucher Δmid (lags 0/1/2 ticks),
and correlation of spread change with |ΔS| for core strikes.

DTE mapping for context only: DTE_open = 8 - csv_day (round3description).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "underlying_leadlag_spread_elasticity.json"

U = "VELVETFRUIT_EXTRACT"
CORE = [5000, 5100, 5200, 5300, 5400, 5500]
VOUS = {k: f"VEV_{k}" for k in CORE}


def bba(row: dict) -> tuple[float | None, float | None]:
    bids, asks = [], []
    for i in (1, 2, 3):
        bp, bv = row.get(f"bid_price_{i}"), row.get(f"bid_volume_{i}")
        ap, av = row.get(f"ask_price_{i}"), row.get(f"ask_volume_{i}")
        if bp and bv and int(float(bv)) > 0:
            bids.append(float(bp))
        if ap and av and int(float(av)) > 0:
            asks.append(float(ap))
    if not bids or not asks:
        return None, None
    return max(bids), min(asks)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return float("nan")
    a, b = x[m], y[m]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def series_for_day(day: int) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
    path = DATA / f"prices_round_3_day_{day}.csv"
    with path.open() as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            p = row["product"]
            if p == U or p in VOUS.values():
                by_ts[int(row["timestamp"])][p] = row
    tss = np.array(sorted(by_ts.keys()), dtype=np.int64)
    n = len(tss)
    u = np.full(n, np.nan)
    vm = {k: np.full(n, np.nan) for k in CORE}
    sp = {k: np.full(n, np.nan) for k in CORE}
    for i, ts in enumerate(tss):
        s = by_ts[int(ts)]
        if U in s:
            bb, ba = bba(s[U])
            if bb is not None and ba > bb:
                u[i] = 0.5 * (bb + ba)
        for k in CORE:
            v = VOUS[k]
            if v not in s:
                continue
            b, a = bba(s[v])
            if b is None or a <= b:
                continue
            vm[k][i] = 0.5 * (b + a)
            sp[k][i] = 0.5 * (a - b)
    return tss, u, vm, sp


def main() -> None:
    out: dict = {"by_day": {}, "summary_core": {}}
    acc = {k: {"l0": [], "l1": [], "l2": [], "el": []} for k in CORE}

    for day in (0, 1, 2):
        _tss, u, vm, sp = series_for_day(day)
        dU = np.diff(u)
        leadlag: dict = {}
        spread_el: dict = {}
        for k in CORE:
            dV = np.diff(vm[k])
            if len(dU) < 10 or len(dV) < 10:
                leadlag[str(k)] = {"corr_dU_dV_lag0": float("nan"), "corr_dU_dV_lag1": float("nan"), "corr_dU_dV_lag2": float("nan")}
                spread_el[str(k)] = {"corr_absdU_dspread": float("nan")}
                continue
            c0 = safe_corr(dU, dV)
            c1 = safe_corr(dU[:-1], dV[1:]) if len(dU) > 2 else float("nan")
            c2 = safe_corr(dU[:-2], dV[2:]) if len(dU) > 3 else float("nan")
            leadlag[str(k)] = {"corr_dU_dV_lag0": c0, "corr_dU_dV_lag1": c1, "corr_dU_dV_lag2": c2}
            acc[k]["l0"].append(c0)
            acc[k]["l1"].append(c1)
            acc[k]["l2"].append(c2)

            d_sp = np.diff(sp[k])
            if len(d_sp) != len(dU):
                d_sp = np.diff(sp[k], prepend=np.nan)[1:]
            if len(d_sp) != len(dU):
                spread_el[str(k)] = {"corr_absdU_dspread": float("nan")}
                acc[k]["el"].append(float("nan"))
                continue
            adu = np.abs(dU)
            msk = np.isfinite(adu) & np.isfinite(d_sp)
            if msk.sum() > 200:
                el = float(np.corrcoef(adu[msk], d_sp[msk])[0, 1])
            else:
                el = float("nan")
            spread_el[str(k)] = {"corr_absdU_dspread": el}
            acc[k]["el"].append(el)

        out["by_day"][str(day)] = {"n_ticks": int(len(_tss)), "leadlag": leadlag, "spread": spread_el}

    for k in CORE:
        out["summary_core"][str(k)] = {
            "mean_corr_lag0": float(np.nanmean(acc[k]["l0"])),
            "mean_corr_lag1": float(np.nanmean(acc[k]["l1"])),
            "mean_corr_lag2": float(np.nanmean(acc[k]["l2"])),
            "mean_elasticity_absdU_dspread": float(np.nanmean(acc[k]["el"])),
        }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
