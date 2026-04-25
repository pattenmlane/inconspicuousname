"""
Round 3 tapes: on extract shocks (|dS|>=2 vs prior timestamp), summarize |dm| for core vs wing VEV mids,
split by up-move vs down-move of extract.

DTE not needed for this spread/mid propagation view; uses mid_price from long-format CSV.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent.parent
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "extract_shock_core_mid_asymmetry.json"

U = "VELVETFRUIT_EXTRACT"
CORE = ("VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300")
WING_NEAR = ("VEV_5400", "VEV_5500")
WING_DEEP = ("VEV_4000", "VEV_4500", "VEV_6000", "VEV_6500")
TS_STEP = 5  # subsample timestamps for speed


def main() -> None:
    up_core: list[float] = []
    down_core: list[float] = []
    up_near: list[float] = []
    down_near: list[float] = []
    up_deep: list[float] = []
    down_deep: list[float] = []

    for csv_day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{csv_day}.csv", sep=";")
        ts_list = sorted(df["timestamp"].unique())[::TS_STEP]
        prev_s: float | None = None
        prev_mid: dict[str, float] = {}
        for ts in ts_list:
            sub = df.loc[df["timestamp"] == ts]
            ex = sub[sub["product"] == U]
            if ex.empty:
                continue
            S = float(ex.iloc[0]["mid_price"])
            if prev_s is None:
                prev_s = S
                for v in CORE + WING_NEAR + WING_DEEP:
                    r = sub[sub["product"] == v]
                    if not r.empty:
                        prev_mid[v] = float(r.iloc[0]["mid_price"])
                continue
            dS = S - prev_s
            prev_s = S
            if abs(dS) < 2.0:
                for v in CORE + WING_NEAR + WING_DEEP:
                    r = sub[sub["product"] == v]
                    if not r.empty:
                        prev_mid[v] = float(r.iloc[0]["mid_price"])
                continue
            bucket = up_core if dS > 0 else down_core
            bucket_near = up_near if dS > 0 else down_near
            bucket_deep = up_deep if dS > 0 else down_deep
            for v in CORE:
                r = sub[sub["product"] == v]
                if v in prev_mid and not r.empty:
                    m = float(r.iloc[0]["mid_price"])
                    bucket.append(abs(m - prev_mid[v]))
                if not r.empty:
                    prev_mid[v] = float(r.iloc[0]["mid_price"])
            for v in WING_NEAR:
                r = sub[sub["product"] == v]
                if v in prev_mid and not r.empty:
                    m = float(r.iloc[0]["mid_price"])
                    bucket_near.append(abs(m - prev_mid[v]))
                if not r.empty:
                    prev_mid[v] = float(r.iloc[0]["mid_price"])
            for v in WING_DEEP:
                r = sub[sub["product"] == v]
                if v in prev_mid and not r.empty:
                    m = float(r.iloc[0]["mid_price"])
                    bucket_deep.append(abs(m - prev_mid[v]))
                if not r.empty:
                    prev_mid[v] = float(r.iloc[0]["mid_price"])

    def pack(a: list[float]) -> dict[str, float | None]:
        if not a:
            return {"n": 0.0, "median_abs_dm": None}
        arr = np.asarray(a, float)
        return {"n": float(len(arr)), "median_abs_dm": float(np.median(arr))}

    payload = {
        "timestamp_step": TS_STEP,
        "shock_def": "|dS|>=2 vs prior subsampled timestamp",
        "core_5000_5300": {"up": pack(up_core), "down": pack(down_core)},
        "near_wing_5400_5500": {"up": pack(up_near), "down": pack(down_near)},
        "deep_wings": {"up": pack(up_deep), "down": pack(down_deep)},
        "interpretation": "If core |dm| on shocks is larger and more symmetric than near-wings, a core-only passive size bump on |dS|>=2 targets liquidity when co-movement is strongest without touching illiquid deep wings.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
