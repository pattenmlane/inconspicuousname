#!/usr/bin/env python3
"""
Round 4 — **Mark 67 aggressive buy on extract** vs **Sonic joint gate** (same timestamp, inner-join panel).

Uses the same **inner join** (VEV_5200, VEV_5300, VELVETFRUIT_EXTRACT) and **TH=2** as Phase 3 / pair gate scripts.
Aggressive buy: extract trade price >= ask_1 at that (day, timestamp).

Outputs: analysis_outputs/r4_mark67_buy_gate_cooccur.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "r4_mark67_buy_gate_cooccur.json"

TH = 2
EX = "VELVETFRUIT_EXTRACT"
V520 = "VEV_5200"
V530 = "VEV_5300"
MARK = "Mark 67"


def days():
    return sorted(int(p.stem.split("_")[-1]) for p in DATA.glob("prices_round_4_day_*.csv"))


def one(df: pd.DataFrame, prod: str) -> pd.DataFrame:
    v = df[df["product"] == prod].drop_duplicates("timestamp").sort_values("timestamp")
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return v.assign(bb=bid, ba=ask, m=mid)[["day", "timestamp", "bb", "ba", "m"]]


def gate_panel() -> pd.DataFrame:
    parts = []
    for d in days():
        raw = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        raw["day"] = d
        a = one(raw, V520)[["day", "timestamp", "bb", "ba"]].rename(columns={"bb": "bb52", "ba": "ba52"})
        b = one(raw, V530)[["day", "timestamp", "bb", "ba"]].rename(columns={"bb": "bb53", "ba": "ba53"})
        e = one(raw, EX)[["day", "timestamp", "bb", "ba", "m"]].rename(columns={"bb": "bbu", "ba": "bau", "m": "m_ext"})
        m = a.merge(b, on=["day", "timestamp"], how="inner").merge(e, on=["day", "timestamp"], how="inner")
        s52 = (m["ba52"] - m["bb52"]).astype(float)
        s53 = (m["ba53"] - m["bb53"]).astype(float)
        m["tight"] = (s52 <= TH) & (s53 <= TH)
        parts.append(m[["day", "timestamp", "tight", "bbu", "bau"]])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    gate = gate_panel()
    n_panel = int(len(gate))
    n_tight_panel = int(gate["tight"].sum())

    m67_rows = []
    for d in days():
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        t = pd.read_csv(p, sep=";")
        t = t[t["symbol"] == EX].copy()
        t["day"] = int(d)
        m67_rows.append(t)
    tr = pd.concat(m67_rows, ignore_index=True)

    g2 = gate.set_index(["day", "timestamp"])
    hits = []
    for _, r in tr.iterrows():
        buyer = str(r["buyer"])
        if buyer != MARK:
            continue
        d, ts = int(r["day"]), int(r["timestamp"])
        if (d, ts) not in g2.index:
            continue
        row = g2.loc[(d, ts)]
        bau = float(row["bau"]) if isinstance(row, pd.Series) else float(row.iloc[0]["bau"])
        bbu = float(row["bbu"]) if isinstance(row, pd.Series) else float(row.iloc[0]["bbu"])
        tight = bool(row["tight"]) if isinstance(row, pd.Series) else bool(row.iloc[0]["tight"])
        pr = float(r["price"])
        if pr < bau:
            continue
        hits.append({"day": d, "timestamp": ts, "tight": tight})
    h = pd.DataFrame(hits)
    n_m67_buy_agg = int(len(h))
    n_m67_tight = int(h["tight"].sum()) if n_m67_buy_agg else 0
    pct_m67_in_tight = (n_m67_tight / n_m67_buy_agg) if n_m67_buy_agg else None

    by_day = []
    if n_m67_buy_agg:
        for d in sorted(h["day"].unique()):
            g = h[h["day"] == d]
            nt = int(g["tight"].sum())
            nn = int(len(g))
            by_day.append({"day": int(d), "n_m67_buy_agg": nn, "n_tight": nt, "pct_tight": nt / nn if nn else None})

    obj = {
        "gate": "inner_join_5200_5300_extract_TH_2",
        "panel_rows": n_panel,
        "panel_tight_rows": n_tight_panel,
        "panel_tight_frac": n_tight_panel / n_panel if n_panel else None,
        "extract_mark67_aggressive_buy_rows_in_panel": n_m67_buy_agg,
        "of_those_gate_tight": n_m67_tight,
        "pct_m67_buys_when_tight": pct_m67_in_tight,
        "by_day": by_day,
        "note": "M67 aggressive buy = buyer Mark 67 and trade price >= extract ask_1 at same (day,timestamp) in inner-join panel only.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(json.dumps(obj, indent=2))


if __name__ == "__main__":
    main()
