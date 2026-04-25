"""
Lead/lag: corr(du_t, do_{t+k}) for k=0..3; shock spread: E[Δ spread | |du| top decile] vs rest.
Vectorized per day; spread lookup O(1) via (timestamp, product) map.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
U = "VELVETFRUIT_EXTRACT"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOU = [f"VEV_{k}" for k in STRIKES]
LAGS = (0, 1, 2, 3)


def top_spread_row(row: pd.Series) -> float | None:
    bps, aps = [], []
    for i in (1, 2, 3):
        bp, ap = row.get(f"bid_price_{i}"), row.get(f"ask_price_{i}")
        bv, av = row.get(f"bid_volume_{i}"), row.get(f"ask_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and int(bv) > 0:
            bps.append(float(bp))
        if pd.notna(ap) and pd.notna(av) and int(av) > 0:
            aps.append(float(ap))
    if bps and aps:
        return min(aps) - max(bps)
    return None


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return None
    x, y = a[m], b[m]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lead_stores: dict[str, dict[int, list[float]]] = {v: {k: [] for k in LAGS} for v in VOU}
    shock_store: dict[str, list[tuple[bool, float]]] = {v: [] for v in VOU}

    for day in (0, 1, 2):
        raw = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        spread_map: dict[tuple[int, str], float] = {}
        for _, row in raw.iterrows():
            ts = int(row["timestamp"])
            p = str(row["product"])
            sp = top_spread_row(row)
            if sp is not None:
                spread_map[(ts, p)] = float(sp)
        piv = raw.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first").sort_index()
        if U not in piv.columns:
            continue
        idx = np.array(list(piv.index), dtype=np.int64)
        n = len(idx)
        s = piv[U].to_numpy(dtype=float)
        dus: list[float] = []
        for i in range(n - 1):
            a, b = float(s[i]), float(s[i + 1])
            dus.append(math.log(b / a) if a > 0 and b > 0 else float("nan"))
        dus = np.array(dus, dtype=float)
        q9 = float(np.nanquantile(np.abs(dus), 0.9)) if len(dus) > 20 else 0.0
        for v in VOU:
            if v not in piv.columns:
                continue
            o = np.clip(piv[v].to_numpy(dtype=float), 1e-9, None)
            dolog = np.diff(np.log(o))
            mlen = min(len(dus), len(dolog))
            if mlen < 5:
                continue
            dus2, do2 = dus[:mlen], dolog[:mlen]
            for k in LAGS:
                if mlen <= k:
                    continue
                a = dus2[: mlen - k]
                b = do2[k:mlen]
                c = safe_corr(a, b)
                if c is not None and np.isfinite(c):
                    lead_stores[v][k].append(c)
            for j in range(mlen - 1):
                ts0, ts1 = int(idx[j]), int(idx[j + 1])
                sp0 = spread_map.get((ts0, v))
                sp1 = spread_map.get((ts1, v))
                if sp0 is None or sp1 is None:
                    continue
                is_shock = bool(q9 > 0 and abs(float(dus2[j])) >= q9)
                shock_store[v].append((is_shock, float(sp1 - sp0)))

    lead_rows = []
    for v in VOU:
        for k in LAGS:
            xs = lead_stores[v][k]
            lead_rows.append(
                {
                    "voucher": v,
                    "lag": k,
                    "mean_corr_du_do": float(np.mean(xs)) if xs else None,
                    "n_day_estimates": int(len(xs)),
                }
            )
    lead_df = pd.DataFrame(lead_rows)
    lead_p = OUT / "leadlag_du_do_by_voucher.csv"
    lead_df.to_csv(lead_p, index=False)

    shock_rows = []
    for v in VOU:
        xs = shock_store[v]
        if not xs:
            continue
        sh = [b for a, b in xs if a]
        ns = [b for a, b in xs if not a]
        shock_rows.append(
            {
                "voucher": v,
                "mean_dspread_shock": float(np.mean(sh)) if sh else None,
                "mean_dspread_calm": float(np.mean(ns)) if ns else None,
                "n_shock": int(len(sh)),
                "n_calm": int(len(ns)),
            }
        )
    shock_df = pd.DataFrame(shock_rows)
    shock_p = OUT / "shock_spread_response_by_voucher.csv"
    shock_df.to_csv(shock_p, index=False)

    def get_mean(v: str, lag: int) -> float | None:
        r = lead_df[(lead_df["voucher"] == v) & (lead_df["lag"] == lag)]
        if r.empty or pd.isna(r["mean_corr_du_do"].iloc[0]):
            return None
        return float(r["mean_corr_du_do"].iloc[0])

    def shock_diff(v: str) -> float | None:
        r = shock_df[shock_df["voucher"] == v]
        if r.empty or pd.isna(r["mean_dspread_shock"].iloc[0]) or pd.isna(r["mean_dspread_calm"].iloc[0]):
            return None
        return float(r["mean_dspread_shock"].iloc[0] - r["mean_dspread_calm"].iloc[0])

    summ = {
        "method": "Fast path: (timestamp,product)->spread; du/do over consecutive timestamps; per-day corr averaged across 3 days.",
        "focal_5200_corr_lag0_lag1": [get_mean("VEV_5200", 0), get_mean("VEV_5200", 1)],
        "focal_5000_corr_lag0_lag1": [get_mean("VEV_5000", 0), get_mean("VEV_5000", 1)],
        "vev5200_shock_minus_calm_dspread": shock_diff("VEV_5200"),
        "vev5000_shock_minus_calm_dspread": shock_diff("VEV_5000"),
    }
    (OUT / "leadlag_shock_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("Wrote", lead_p, shock_p, summ)


if __name__ == "__main__":
    main()
