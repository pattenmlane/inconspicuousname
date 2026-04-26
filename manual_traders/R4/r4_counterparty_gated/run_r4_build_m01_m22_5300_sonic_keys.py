"""Keys (day:timestamp) where tape has Mark01->Mark22 on VEV_5300 AND Sonic joint_tight (5200&5300 spread<=2)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
TH = 2


def one_spread(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    return v.assign(s=(ask - bid).astype(float))[["timestamp", "s"]].rename(columns={"s": f"s_{product}"})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys: set[str] = set()
    for p in sorted(DATA.glob("trades_round_4_day_*.csv")):
        day = int(p.stem.replace("trades_round_4_day_", ""))
        tr = pd.read_csv(p, sep=";")
        tr = tr[(tr["buyer"] == "Mark 01") & (tr["seller"] == "Mark 22") & (tr["symbol"] == "VEV_5300")]
        if tr.empty:
            continue
        px = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
        a = one_spread(px, "VEV_5200")
        b = one_spread(px, "VEV_5300")
        g = a.merge(b, on="timestamp", how="inner")
        g["joint_tight"] = (g["s_VEV_5200"] <= TH) & (g["s_VEV_5300"] <= TH)
        gt = g.loc[g["joint_tight"], "timestamp"].astype(int)
        ts_set = set(gt.tolist())
        for t in tr["timestamp"].astype(int):
            if int(t) in ts_set:
                keys.add(f"{day}:{int(t)}")
    out = {"keys": sorted(keys), "n": len(keys), "description": "Mark01->Mark22 VEV_5300 print AND joint Sonic gate on tape"}
    (OUT / "signals_m01_m22_5300_sonic_tight.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("n keys", len(keys), "->", OUT / "signals_m01_m22_5300_sonic_tight.json")


if __name__ == "__main__":
    main()
