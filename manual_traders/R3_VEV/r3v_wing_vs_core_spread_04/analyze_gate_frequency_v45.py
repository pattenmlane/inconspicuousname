"""P(joint BBO spread tight) day 0 from CSV (no optional tipworkflow). TH=2 like vouchers_final."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
VEV_5200, VEV_5300 = "VEV_5200", "VEV_5300"
TH = 2


def main() -> None:
    p = _ROOT / "Prosperity4Data" / "ROUND_3" / "prices_round_3_day_0.csv"
    df = pd.read_csv(p, sep=";")
    df = df[df["product"].isin((VEV_5200, VEV_5300))].copy()
    bid = pd.to_numeric(df["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(df["ask_price_1"], errors="coerce")
    df["s"] = (ask - bid).astype(float)
    a = (
        df[df["product"] == VEV_5200]
        .drop_duplicates(subset=["day", "timestamp"], keep="first")[
            ["day", "timestamp", "s"]
        ]
        .rename(columns={"s": "s5200"})
    )
    b = (
        df[df["product"] == VEV_5300]
        .drop_duplicates(subset=["day", "timestamp"], keep="first")[
            ["day", "timestamp", "s"]
        ]
        .rename(columns={"s": "s5300"})
    )
    m = a.merge(b, on=["day", "timestamp"], how="inner")
    tight = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    out = (
        _ROOT
        / "manual_traders"
        / "R3_VEV"
        / "r3v_wing_vs_core_spread_04"
        / "gate_p_joint_tight_day0_v45.json"
    )
    out.write_text(
        json.dumps(
            {
                "source": "Prosperity4Data/ROUND_3/prices_round_3_day_0.csv",
                "TH": TH,
                "n": int(len(m)),
                "P_joint_tight": float(tight.mean()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out, float(tight.mean()), len(m))


if __name__ == "__main__":
    main()
