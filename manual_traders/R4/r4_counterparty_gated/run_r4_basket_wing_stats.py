"""Stats: at M01->M22 VEV_5300 + joint tight, which co-print symbols appear (tape)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
TH = 2


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
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for d in (1, 2, 3):
        tr = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        tr["day"] = d
        frames.append(tr)
    tr = pd.concat(frames, ignore_index=True)
    px = pd.concat(
        [pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";").assign(day=d) for d in (1, 2, 3)]
    )

    m = tr[(tr["buyer"] == "Mark 01") & (tr["seller"] == "Mark 22") & (tr["symbol"] == "VEV_5300")]
    from collections import Counter

    co_syms: Counter[str] = Counter()
    n = 0
    n_wing_syms = 0
    n_m01_m22_all_five = 0
    wing_syms = {"VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"}
    ladder = ("VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500")

    def m01_m22_row(day: int, ts: int, sym: str) -> bool:
        s = tr[(tr["day"] == day) & (tr["timestamp"] == ts) & (tr["symbol"] == sym)]
        return bool(((s["buyer"] == "Mark 01") & (s["seller"] == "Mark 22")).any())

    for _, r in m.iterrows():
        day, ts = int(r["day"]), int(r["timestamp"])
        if not joint_tight(px, day, ts):
            continue
        n += 1
        same = tr[(tr["day"] == day) & (tr["timestamp"] == ts)]
        syms = set(same["symbol"].astype(str))
        if wing_syms <= syms:
            n_wing_syms += 1
        if all(m01_m22_row(day, ts, sym) for sym in ladder):
            n_m01_m22_all_five += 1
        for s in syms:
            co_syms[s] += 1

    summ = {
        "n_m01_m22_5300_joint_tight_ticks": n,
        "n_tick_includes_symbols_5400_6500": n_wing_syms,
        "frac_tick_includes_wing_syms": float(n_wing_syms / n) if n else 0.0,
        "n_tick_m01_m22_on_each_of_5300_through_6500": n_m01_m22_all_five,
        "frac_m01_m22_full_ladder": float(n_m01_m22_all_five / n) if n else 0.0,
        "co_print_counts": dict(co_syms.most_common()),
    }
    (OUT / "r4_m01_m22_5300_joint_tight_coprint_summary.json").write_text(
        json.dumps(summ, indent=2), encoding="utf-8"
    )
    print(json.dumps(summ, indent=2)[:1200])


if __name__ == "__main__":
    main()
