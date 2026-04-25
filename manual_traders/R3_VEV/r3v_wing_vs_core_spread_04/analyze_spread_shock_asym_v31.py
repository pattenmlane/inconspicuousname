"""
Round 3: per-voucher spread asymmetry on large extract moves (up vs down).
Pooled by csv day: shock if |Δextract| >= day 90p of |dS|.
TTE mapping consistent with traders (8-dte day0 open with intraday wind).
Reads Prosperity4Data/ROUND_3; writes spread_shock_asym_v31.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = (
    REPO
    / "manual_traders"
    / "R3_VEV"
    / "r3v_wing_vs_core_spread_04"
    / "spread_shock_asym_v31.json"
)

U = "VELVETFRUIT_EXTRACT"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOU = [f"VEV_{k}" for k in STRIKES]


def spread_of_row(row: pd.Series) -> int | None:
    bps = [row[f"bid_price_{i}"] for i in (1, 2, 3) if pd.notna(row.get(f"bid_price_{i}"))]
    aps = [row[f"ask_price_{i}"] for i in (1, 2, 3) if pd.notna(row.get(f"ask_price_{i}"))]
    if not bps or not aps:
        return None
    return int(min(aps) - max(bps))


def main() -> None:
    all_abs: list[float] = []
    pre_thr: list[tuple[int, int, float]] = []  # (day, ts, abs_dS)
    for day in (0, 1, 2):
        df0 = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        p = df0.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        if U not in p.columns:
            continue
        p = p.sort_index()
        prev: float | None = None
        for ts in p.index:
            S = float(p.loc[ts, U])
            if prev is not None:
                ad = abs(S - float(prev))
                all_abs.append(float(ad))
                pre_thr.append((int(day), int(ts), ad))
            prev = S

    thr: dict[str, float] = {}
    for d in (0, 1, 2):
        sub = [x for (day, _, x) in pre_thr if day == d]
        thr[str(d)] = float(np.percentile(sub, 90)) if sub else 0.0

    rows: list[dict] = []
    for day in (0, 1, 2):
        df0 = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        p = df0.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        p = p.sort_index()
        thr_d = thr[str(day)]
        prev_s: float | None = None
        for ts in p.index:
            S = float(p.loc[ts, U])
            dS = 0.0 if prev_s is None else S - float(prev_s)
            prev_s = S
            shock = abs(dS) >= thr_d
            for v in VOU:
                if v not in df0["product"].values:
                    continue
                r = df0[(df0["day"] == day) & (df0["timestamp"] == ts) & (df0["product"] == v)]
                if r.empty:
                    continue
                sp = spread_of_row(r.iloc[0])
                if sp is None:
                    continue
                rows.append(
                    {
                        "day": day,
                        "ts": int(ts),
                        "K": int(v.split("_")[1]),
                        "spread": sp,
                        "dS": float(dS),
                        "shock": shock,
                    }
                )

    dfp = pd.DataFrame(rows)
    out_strike = []
    for K in STRIKES:
        sub = dfp[dfp["K"] == K]
        sh = sub[sub["shock"]]
        up = sh[sh["dS"] > 0]
        dn = sh[sh["dS"] < 0]
        up_m = float(up["spread"].mean()) if len(up) else 0.0
        dn_m = float(dn["spread"].mean()) if len(dn) else 0.0
        out_strike.append(
            {
                "K": K,
                "voucher": f"VEV_{K}",
                "spread_all_mean": float(sub["spread"].mean()) if len(sub) else 0.0,
                "spread_shock_mean": float(sh["spread"].mean()) if len(sh) else 0.0,
                "spread_shock_up_mean": up_m,
                "spread_shock_dn_mean": dn_m,
                "asym_up_minus_dn": float(up_m - dn_m) if (len(up) and len(dn)) else 0.0,
                "n_shock": int(len(sh)),
                "n_shock_up": int(len(up)),
                "n_shock_dn": int(len(dn)),
            }
        )

    doc = {
        "method": "From pivot mid: dS=extract_t - extract_{t-1}; shock if |dS|>=day 90p(|dS|). Spread=ask1-bid1 width.",
        "shock_thresholds_by_csv_day_90p_abs_dS": thr,
        "by_strike": out_strike,
    }
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
