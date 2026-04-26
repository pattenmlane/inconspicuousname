#!/usr/bin/env python3
"""
Forward **VELVETFRUIT_EXTRACT** mid markouts for **Mark 67** **aggressive** buys
(price >= L1 ask), split by **Sonic joint tight** (5200+5300 spread<=2) at print time.

Horizon K = forward steps on each tape day's **sorted unique timestamps** (same
convention as Phase 1 `r4_phase1_counterparty_analysis.py`).

Outputs:
  analysis_outputs/r4_mark67_extract_fwdmid_joint_gate_by_day.csv
  analysis_outputs/r4_mark67_extract_fwdmid_joint_gate_pooled.csv
"""
from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = (1, 2, 3)
EXTRACT = "VELVETFRUIT_EXTRACT"
SURFACE = ("VEV_5200", "VEV_5300")
SPREAD_TH = 2
MARK67 = "Mark 67"
KS = (5, 20, 100)


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        df = pd.read_csv(p, sep=";")
        if "day" not in df.columns:
            df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def joint_tight_at(px_day: pd.DataFrame, ts: int) -> bool:
    sub = px_day[
        (px_day["timestamp"] == ts) & (px_day["product"].isin(SURFACE))
    ]
    if len(sub) < 2:
        return False
    rows = {r["product"]: r for _, r in sub.iterrows()}
    for sym in SURFACE:
        if sym not in rows:
            return False
        r = rows[sym]
        bb = pd.to_numeric(r["bid_price_1"], errors="coerce")
        ba = pd.to_numeric(r["ask_price_1"], errors="coerce")
        if pd.isna(bb) or pd.isna(ba):
            return False
        if float(ba - bb) > SPREAD_TH:
            return False
    return True


def fwd_ts(ts_sorted: np.ndarray, t: int, k: int) -> int | None:
    i = bisect.bisect_left(ts_sorted, t)
    if i >= len(ts_sorted) or ts_sorted[i] != t:
        return None
    j = i + k
    if j >= len(ts_sorted):
        return None
    return int(ts_sorted[j])


def main() -> None:
    px = load_prices()
    ex_ts: dict[int, np.ndarray] = {}
    mid_ex: dict[tuple[int, int], float] = {}
    ask_ex: dict[tuple[int, int], float] = {}
    for d in DAYS:
        sub = px[(px["day"] == d) & (px["product"] == EXTRACT)]
        tsu = np.sort(sub["timestamp"].unique())
        ex_ts[d] = tsu
        g = sub.groupby("timestamp").first()
        for ts, row in g.iterrows():
            t = int(ts)
            mid_ex[(d, t)] = float(pd.to_numeric(row["mid_price"], errors="coerce"))
            ask_ex[(d, t)] = float(pd.to_numeric(row["ask_price_1"], errors="coerce"))

    tr = []
    for d in DAYS:
        tdf = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        tdf["day"] = d
        tr.append(tdf)
    tr = pd.concat(tr, ignore_index=True)
    tr = tr[(tr["symbol"] == EXTRACT) & (tr["buyer"].fillna("").astype(str) == MARK67)].copy()
    tr["timestamp"] = tr["timestamp"].astype(int)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")

    by_day_rows = []
    events = []
    for d in DAYS:
        px_d = px[px["day"] == d]
        for _, r in tr[tr["day"] == d].iterrows():
            ts = int(r["timestamp"])
            pr = float(r["price"])
            ask = ask_ex.get((d, ts))
            if ask is None or np.isnan(ask) or pr < ask:
                continue
            if (d, ts) not in mid_ex:
                continue
            tight = joint_tight_at(px_d, ts)
            m0 = mid_ex[(d, ts)]
            tsu = ex_ts[d]
            row = {"day": d, "timestamp": ts, "joint_tight": tight, "m0": m0}
            for k in KS:
                fts = fwd_ts(tsu, ts, k)
                if fts is None:
                    row[f"d_mid_k{k}"] = np.nan
                else:
                    m1 = mid_ex.get((d, fts), np.nan)
                    row[f"d_mid_k{k}"] = (m1 - m0) if m1 == m1 and m0 == m0 else np.nan
            events.append(row)
            by_day_rows.append({**row})

    ev = pd.DataFrame(events)
    if ev.empty:
        raise SystemExit("No Mark67 aggressive extract events")

    def summarize(sub: pd.DataFrame, label: str) -> dict:
        out: dict[str, float | int | str] = {"cohort": label, "n": len(sub)}
        for k in KS:
            col = f"d_mid_k{k}"
            s = sub[col].dropna()
            out[f"n_k{k}"] = int(len(s))
            out[f"mean_k{k}"] = float(s.mean()) if len(s) else float("nan")
            out[f"median_k{k}"] = float(s.median()) if len(s) else float("nan")
        return out

    pooled = []
    for tight, name in [(True, "joint_tight"), (False, "not_joint_tight")]:
        pooled.append(summarize(ev[ev["joint_tight"] == tight], name))
    pooled.append(summarize(ev, "all"))
    pd.DataFrame(pooled).to_csv(OUT / "r4_mark67_extract_fwdmid_joint_gate_pooled.csv", index=False)

    day_rows = []
    for d in DAYS:
        sub = ev[ev["day"] == d]
        for tight, name in [(True, "joint_tight"), (False, "not_joint_tight")]:
            r = summarize(sub[sub["joint_tight"] == tight], f"day{d}_{name}")
            r["day"] = d
            day_rows.append(r)
        r = summarize(sub, f"day{d}_all")
        r["day"] = d
        day_rows.append(r)
    pd.DataFrame(day_rows).to_csv(OUT / "r4_mark67_extract_fwdmid_joint_gate_by_day.csv", index=False)

    print(pd.DataFrame(pooled).to_string(index=False))


if __name__ == "__main__":
    main()
