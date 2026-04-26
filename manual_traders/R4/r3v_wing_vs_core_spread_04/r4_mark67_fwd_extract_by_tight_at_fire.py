#!/usr/bin/env python3
"""
Mark67 buy-aggr extract (164): forward extract **mid** K=20 **tape rows** after print,
split by joint Sonic tight at **fire_ts = print_ts + 100** (from act_gate CSV).

K=20 means 20 index steps on the 100-tick grid (same as other R4 fwd scripts).

Outputs:
  outputs/phase3/mark67_fwd_extract_k20_by_tight_at_fire_summary.json
  outputs/phase3/mark67_fwd_extract_k20_by_tight_at_fire_rows.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "phase3"
SIG = Path(__file__).resolve().parent / "outputs" / "phase2" / "signals_mark67_buy_aggr_extract.json"
GATE_CSV = OUT / "act_gate_at_fire_mark67_buy_aggr_extract_all.csv"
K = 20


def mid_series(day: int) -> pd.Series:
    px = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    sub = px[px["product"] == "VELVETFRUIT_EXTRACT"].drop_duplicates("timestamp").sort_values("timestamp")
    return sub.set_index("timestamp")["mid_price"].astype(float)


def fwd20(s: pd.Series, ts: int) -> float:
    idx = s.index.to_numpy()
    pos = np.searchsorted(idx, ts)
    if pos >= len(idx) or int(idx[pos]) != ts:
        return float("nan")
    j = pos + K
    if j >= len(idx):
        return float("nan")
    a, b = float(s.iloc[pos]), float(s.iloc[j])
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    return b - a


def main() -> None:
    gate = pd.read_csv(GATE_CSV)
    tight_map = {
        (int(r.tape_day), int(r.print_ts)): bool(r.joint_tight_at_fire)
        for r in gate.itertuples()
    }
    raw = json.loads(SIG.read_text(encoding="utf-8"))
    mids = {d: mid_series(d) for d in (1, 2, 3)}

    rows = []
    for tape_day, print_ts in raw:
        d, ts = int(tape_day), int(print_ts)
        tkey = (d, ts)
        tight = tight_map.get(tkey)
        if tight is None:
            continue
        f = fwd20(mids[d], ts)
        rows.append(
            {
                "tape_day": d,
                "print_ts": ts,
                "joint_tight_at_fire": tight,
                "fwd_extract_k20": f,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mark67_fwd_extract_k20_by_tight_at_fire_rows.csv", index=False)

    summ = []
    for tight_lab, sub in [("tight_at_fire", df[df["joint_tight_at_fire"]]), ("wide_at_fire", df[~df["joint_tight_at_fire"]])]:
        x = sub["fwd_extract_k20"].dropna()
        summ.append(
            {
                "subset": tight_lab,
                "n": int(len(x)),
                "mean": float(x.mean()) if len(x) else float("nan"),
                "median": float(x.median()) if len(x) else float("nan"),
            }
        )
    (OUT / "mark67_fwd_extract_k20_by_tight_at_fire_summary.json").write_text(
        json.dumps(summ, indent=2), encoding="utf-8"
    )

    wdf = df.loc[~df["joint_tight_at_fire"], ["tape_day", "print_ts"]]
    wide = [[int(a), int(b)] for a, b in zip(wdf["tape_day"], wdf["print_ts"], strict=True)]
    (OUT / "signals_mark67_buy_aggr_extract_wide_at_fire.json").write_text(json.dumps(wide), encoding="utf-8")
    meta = {
        "n_total": len(raw),
        "n_wide_at_fire": len(wide),
        "summary": summ,
        "signals_wide": str(OUT / "signals_mark67_buy_aggr_extract_wide_at_fire.json"),
    }
    (OUT / "mark67_wide_at_fire_signals_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
