"""Tape: fraction of Mark67 aggressive-extract signal keys where Sonic joint tight (5200&5300 spread<=2)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
SIG = Path(__file__).resolve().parent / "analysis_outputs" / "signals_mark67_aggr_extract_buy.json"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "r4_m67_keys_sonic_joint_overlap.json"
TH = 2.0


def joint_tight(px: pd.DataFrame, day: int, ts: int) -> bool:
    sub = px[(px["day"] == day) & (px["timestamp"] == ts)]

    def sp(prod: str) -> float | None:
        r = sub[sub["product"] == prod]
        if r.empty:
            return None
        r = r.iloc[0]
        bp = pd.to_numeric(r["bid_price_1"], errors="coerce")
        ap = pd.to_numeric(r["ask_price_1"], errors="coerce")
        if pd.isna(bp) or pd.isna(ap):
            return None
        return float(ap) - float(bp)

    a, b = sp("VEV_5200"), sp("VEV_5300")
    return a is not None and b is not None and a <= TH and b <= TH


def main() -> None:
    keys = json.loads(SIG.read_text(encoding="utf-8")).get("keys", [])
    keys = [str(k) for k in keys]
    px = pd.concat(
        [pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";").assign(day=d) for d in (1, 2, 3)]
    )
    n = 0
    n_joint = 0
    by_day: dict[str, dict[str, int]] = {}
    for k in keys:
        if ":" not in k:
            continue
        d_s, t_s = k.split(":", 1)
        try:
            day, ts = int(d_s), int(t_s)
        except ValueError:
            continue
        n += 1
        j = joint_tight(px, day, ts)
        if j:
            n_joint += 1
        bd = by_day.setdefault(str(day), {"n": 0, "joint": 0})
        bd["n"] += 1
        if j:
            bd["joint"] += 1
    summ = {
        "n_mark67_signal_keys": n,
        "n_joint_tight_at_key": n_joint,
        "frac_joint_tight": float(n_joint / n) if n else 0.0,
        "by_day": by_day,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
