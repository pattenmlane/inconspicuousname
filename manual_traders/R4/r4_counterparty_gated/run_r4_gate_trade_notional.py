"""Merge Sonic joint gate to each trade; report notional share by symbol when tight vs loose."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
TH = 2
VEV_5200, VEV_5300, EX = "VEV_5200", "VEV_5300", "VELVETFRUIT_EXTRACT"


def one_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    v = v.assign(spread=(ask - bid).astype(float))
    return v[["timestamp", "spread"]].copy()


def main() -> None:
    frames = []
    for p in sorted(DATA.glob("trades_round_4_day_*.csv")):
        day = int(p.stem.replace("trades_round_4_day_", ""))
        tr = pd.read_csv(p, sep=";")
        tr["day"] = day
        px = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
        a = one_product(px, VEV_5200).rename(columns={"spread": "s5200"})
        b = one_product(px, VEV_5300).rename(columns={"spread": "s5300"})
        g = a.merge(b, on="timestamp", how="inner")
        g["joint_tight"] = (g["s5200"] <= TH) & (g["s5300"] <= TH)
        g["day"] = day
        gm = g[["day", "timestamp", "joint_tight"]].drop_duplicates()
        tr = tr.merge(gm, on=["day", "timestamp"], how="left")
        tr["notional"] = pd.to_numeric(tr["price"], errors="coerce").abs() * tr["quantity"].abs()
        frames.append(tr)
    allt = pd.concat(frames, ignore_index=True)
    summ = (
        allt.groupby(["joint_tight", "symbol"])["notional"]
        .sum()
        .reset_index()
        .pivot(index="symbol", columns="joint_tight", values="notional")
        .fillna(0)
    )
    summ["pct_tight"] = summ.get(True, 0) / (summ.get(True, 0) + summ.get(False, 0)).replace(0, float("nan"))
    summ.to_csv(OUT / "r4_trade_notional_by_symbol_gate.csv")
    print(summ.sort_values(True if True in summ.columns else summ.columns[0], ascending=False).head(12))
    print("Wrote", OUT / "r4_trade_notional_by_symbol_gate.csv")


if __name__ == "__main__":
    main()
