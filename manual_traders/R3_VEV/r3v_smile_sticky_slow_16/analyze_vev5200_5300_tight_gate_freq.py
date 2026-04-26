"""
Round 3 tapes: fraction of timestamps where VEV_5200 and VEV_5300 both have top-of-book spread <= 2.

Thesis: joint tight book as risk-on regime (see team STRATEGY when available).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent.parent
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "vev5200_5300_tight_gate_freq.json"

GATE_MAX = 2
V52 = "VEV_5200"
V53 = "VEV_5300"


def spread_bbo(sub: pd.DataFrame, prod: str) -> int | None:
    r = sub[sub["product"] == prod]
    if r.empty:
        return None
    row = r.iloc[0]
    bp, ap = row.get("bid_price_1"), row.get("ask_price_1")
    if pd.isna(bp) or pd.isna(ap):
        return None
    return int(ap) - int(bp)


def main() -> None:
    by_day2: dict[str, dict] = {}
    n_t_all = 0
    n_ok_all = 0
    for d in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{d}.csv", sep=";")
        n_t = 0
        n_ok = 0
        for ts in sorted(df["timestamp"].unique()):
            sub = df[df["timestamp"] == ts]
            s52 = spread_bbo(sub, V52)
            s53 = spread_bbo(sub, V53)
            if s52 is None or s53 is None:
                continue
            n_t += 1
            n_t_all += 1
            if s52 <= GATE_MAX and s53 <= GATE_MAX:
                n_ok += 1
                n_ok_all += 1
        by_day2[f"day_{d}"] = {
            "n_timestamps_both_bbo": n_t,
            "n_tight_gate": n_ok,
            "share": float(n_ok / max(n_t, 1)),
        }

    payload = {
        "gate": f"ask1-bid1 <= {GATE_MAX} for both {V52} and {V53} at same timestamp",
        "days_0_2_aggregated": {
            "n_timestamps": n_t_all,
            "n_tight": n_ok_all,
            "share_tight": float(n_ok_all / max(n_t_all, 1)),
        },
        "by_day": by_day2,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
