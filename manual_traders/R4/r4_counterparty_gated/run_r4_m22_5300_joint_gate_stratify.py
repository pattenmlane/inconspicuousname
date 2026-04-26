"""
Stratify Mark 22 aggressive sells on VEV_5300 by Sonic joint gate (VEV_5200 & VEV_5300
BBO spread both <= 2 at same day,timestamp as the trade). Uses price tape from ROUND_4
(same convention as run_r4_phase1_analysis: forward K = K rows on same (day, product)).

Outputs:
  analysis_outputs/r4_m22_aggr_sell_5300_by_joint_gate_by_day.csv
  analysis_outputs/r4_m22_aggr_sell_5300_joint_gate_summary.json

Rerun: python3 run_r4_m22_5300_joint_gate_stratify.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
TH = 2.0
K = 5


def load_prices() -> pd.DataFrame:
    frames = []
    for p in sorted(DATA.glob("prices_round_4_day_*.csv")):
        day = int(p.stem.replace("prices_round_4_day_", ""))
        df = pd.read_csv(p, sep=";")
        df["day"] = day
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def price_spreads(px: pd.DataFrame) -> pd.DataFrame:
    bp = pd.to_numeric(px["bid_price_1"], errors="coerce")
    ap = pd.to_numeric(px["ask_price_1"], errors="coerce")
    out = px[["day", "timestamp", "product"]].copy()
    out["spread"] = ap - bp
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev_path = OUT / "r4_phase1_trade_events.csv"
    if not ev_path.is_file():
        raise SystemExit(f"Missing {ev_path}; run run_r4_phase1_analysis.py first")

    px = price_spreads(load_prices())
    p520 = px[px["product"] == "VEV_5200"][["day", "timestamp", "spread"]].rename(
        columns={"spread": "spr5200"}
    )
    p530 = px[px["product"] == "VEV_5300"][["day", "timestamp", "spread"]].rename(
        columns={"spread": "spr5300"}
    )
    gate = p520.merge(p530, on=["day", "timestamp"], how="inner")
    gate["joint_tight"] = (gate["spr5200"] <= TH) & (gate["spr5300"] <= TH)

    ev = pd.read_csv(ev_path)
    sub = ev[
        (ev["seller"] == "Mark 22")
        & (ev["aggression"] == "aggr_sell")
        & (ev["symbol"] == "VEV_5300")
    ].copy()
    m = sub.merge(gate[["day", "timestamp", "joint_tight"]], on=["day", "timestamp"], how="left")
    m["joint_tight"] = m["joint_tight"].fillna(False).astype(bool)

    col = f"fwd_mid_{K}"
    rows = []
    for (jt, d), g in m.groupby(["joint_tight", "day"]):
        x = g[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        n = len(x)
        if n == 0:
            continue
        rows.append(
            {
                "joint_tight": bool(jt),
                "day": int(d),
                "n": n,
                "mean_fwd": float(np.mean(x)),
                "median_fwd": float(np.median(x)),
                "frac_pos": float(np.mean(x > 0)),
            }
        )
    by_day = pd.DataFrame(rows).sort_values(["joint_tight", "day"])
    by_day.to_csv(OUT / "r4_m22_aggr_sell_5300_by_joint_gate_by_day.csv", index=False)

    pooled = []
    for jt, g in m.groupby("joint_tight"):
        x = g[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        pooled.append(
            {
                "joint_tight": bool(jt),
                "n": len(x),
                "mean_fwd": float(np.mean(x)) if len(x) else float("nan"),
                "median_fwd": float(np.median(x)) if len(x) else float("nan"),
                "frac_pos": float(np.mean(x > 0)) if len(x) else float("nan"),
            }
        )
    summary = {
        "description": "Mark 22 aggressive sell on VEV_5300; fwd_mid_K = K tape rows on VEV_5300",
        "K": K,
        "spread_threshold_each_leg": TH,
        "n_events_total": int(len(m)),
        "pooled_by_joint_gate": pooled,
        "by_day_csv": str(
            OUT / "r4_m22_aggr_sell_5300_by_joint_gate_by_day.csv"
        ).replace(str(REPO) + "/", ""),
    }
    (OUT / "r4_m22_aggr_sell_5300_joint_gate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
