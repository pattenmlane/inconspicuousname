"""Tape: HYDROGEL_PACK K=20 forward mid vs Sonic joint gate (same join as Phase 3 extract panel)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "r4_hydro_fwd_vs_sonic_gate.json"

TH = 2
K = 20
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
EX = "VELVETFRUIT_EXTRACT"
HY = "HYDROGEL_PACK"


def days():
    return sorted(int(p.stem.split("_")[-1]) for p in DATA.glob("prices_round_4_day_*.csv"))


def one(df, prod):
    v = df[df["product"] == prod].drop_duplicates("timestamp").sort_values("timestamp")
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return v.assign(s=(ask - bid).astype(float), m=mid)[["day", "timestamp", "s", "m"]]


def main():
    parts = []
    for d in days():
        raw = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        raw["day"] = d
        a = one(raw, VEV_5200).rename(columns={"s": "s5200", "m": "m5200"})
        b = one(raw, VEV_5300).rename(columns={"s": "s5300", "m": "m5300"})
        e = one(raw, EX).rename(columns={"s": "s_ext", "m": "m_ext"})
        h = one(raw, HY).rename(columns={"s": "s_hy", "m": "m_hy"})
        m = a.merge(b, on=["day", "timestamp"]).merge(e[["day", "timestamp", "m_ext"]], on=["day", "timestamp"]).merge(
            h[["day", "timestamp", "m_hy"]], on=["day", "timestamp"]
        )
        m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
        m["m_hy_f"] = m.groupby("day")["m_hy"].shift(-K)
        m["fwd_hy"] = m["m_hy_f"] - m["m_hy"]
        parts.append(m)
    pan = pd.concat(parts, ignore_index=True)
    pan = pan[np.isfinite(pan["fwd_hy"])]
    t = pan[pan["tight"]]["fwd_hy"].to_numpy()
    l = pan[~pan["tight"]]["fwd_hy"].to_numpy()
    tt = stats.ttest_ind(t, l, equal_var=False) if len(t) > 20 and len(l) > 20 else None
    out = {
        "K": K,
        "n_tight": int(len(t)),
        "n_loose": int(len(l)),
        "mean_fwd_hy_tight": float(np.mean(t)) if len(t) else None,
        "mean_fwd_hy_loose": float(np.mean(l)) if len(l) else None,
        "welch_t": float(tt.statistic) if tt else None,
        "p": float(tt.pvalue) if tt else None,
        "by_day": {},
    }
    for d in days():
        sub = pan[pan["day"] == d]
        tt2 = sub[sub["tight"]]["fwd_hy"]
        ll2 = sub[~sub["tight"]]["fwd_hy"]
        out["by_day"][str(d)] = {
            "mean_tight": float(tt2.mean()) if len(tt2) else None,
            "mean_loose": float(ll2.mean()) if len(ll2) else None,
            "n_tight": int(len(tt2)),
            "n_loose": int(len(ll2)),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
