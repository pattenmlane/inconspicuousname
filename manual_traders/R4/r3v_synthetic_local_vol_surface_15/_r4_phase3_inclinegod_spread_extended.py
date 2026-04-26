#!/usr/bin/env python3
"""
Round 4 Phase 3 extension — **inclineGod** spread panels (level + **Δ** on clock time).

Source: `r4_p3_joint_gate_panel_by_timestamp.csv` (inner join 5200/5300/extract).

**Deltas:** first differences along the **full** sorted timestamp grid (consecutive tape rows),
then regime subsets use rows where **both** t and t-1 satisfy the regime (tight→tight or loose→loose).

Output: `outputs_r4_phase3/r4_p13_inclinegod_spread_panels_by_day.csv`
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
PAN = BASE / "outputs_r4_phase3" / "r4_p3_joint_gate_panel_by_timestamp.csv"
OUT = BASE / "outputs_r4_phase3" / "r4_p13_inclinegod_spread_panels_by_day.csv"
MAN = BASE / "outputs_r4_phase3" / "r4_p3_manifest.json"


def _corr(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return float("nan")
    return float(a[m].astype(float).corr(b[m].astype(float)))


def main() -> None:
    pan = pd.read_csv(PAN)
    rows = []
    for d in sorted(pan["day"].unique()):
        sub = pan[pan["day"] == d].sort_values("timestamp").reset_index(drop=True)
        if len(sub) < 50:
            continue
        tight = sub["tight"].astype(bool)
        loose = ~tight
        stay_tight = tight & tight.shift(1).fillna(False)
        stay_loose = loose & loose.shift(1).fillna(False)

        d5200 = sub["s5200"].diff()
        d5300 = sub["s5300"].diff()
        dexts = sub["s_ext"].diff()
        dm = sub["m_ext"].diff()

        for label, lev_mask, d_mask in [
            ("all", pd.Series(True, index=sub.index), pd.Series(True, index=sub.index)),
            ("tight", tight, stay_tight),
            ("loose", loose, stay_loose),
        ]:
            lv = sub.loc[lev_mask]
            if len(lv) < 30:
                continue
            row = {
                "day": int(d),
                "subset": label,
                "n_level_rows": int(len(lv)),
                "lvl_corr_s5200_s5300": _corr(lv["s5200"], lv["s5300"]),
                "lvl_corr_s5200_s_ext": _corr(lv["s5200"], lv["s_ext"]),
                "lvl_corr_s5300_s_ext": _corr(lv["s5300"], lv["s_ext"]),
                "lvl_corr_s5200_m_ext": _corr(lv["s5200"], lv["m_ext"]),
                "lvl_corr_s5300_m_ext": _corr(lv["s5300"], lv["m_ext"]),
            }
            dd = sub.loc[d_mask]
            if len(dd) >= 30:
                row["n_delta_rows"] = int(d_mask.sum())
                row["d_mean_abs_s5200"] = float(d5200.loc[d_mask].abs().mean())
                row["d_mean_abs_s5300"] = float(d5300.loc[d_mask].abs().mean())
                row["d_mean_abs_s_ext"] = float(dexts.loc[d_mask].abs().mean())
                row["d_mean_abs_m_ext"] = float(dm.loc[d_mask].abs().mean())
                row["d_corr_s5200_s5300"] = _corr(d5200.loc[d_mask], d5300.loc[d_mask])
                row["d_corr_s5200_s_ext"] = _corr(d5200.loc[d_mask], dexts.loc[d_mask])
                row["d_corr_s5300_s_ext"] = _corr(d5300.loc[d_mask], dexts.loc[d_mask])
                row["d_corr_s5200_m_ext"] = _corr(d5200.loc[d_mask], dm.loc[d_mask])
                row["d_corr_s5300_m_ext"] = _corr(d5300.loc[d_mask], dm.loc[d_mask])
                row["d_corr_s_ext_m_ext"] = _corr(dexts.loc[d_mask], dm.loc[d_mask])
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))

    if MAN.is_file():
        m = json.loads(MAN.read_text(encoding="utf-8"))
        outs = set(m.get("outputs", []))
        outs.add(OUT.name)
        m["outputs"] = sorted(outs)
        MAN.write_text(json.dumps(m, indent=2), encoding="utf-8")

    summ = BASE / "outputs_r4_phase3" / "r4_p3_executive_summary.txt"
    if summ.is_file():
        txt = summ.read_text(encoding="utf-8")
        if "r4_p13_inclinegod_spread_panels" not in txt:
            tail = (
                "\n--- inclineGod extended (`r4_p13_inclinegod_spread_panels_by_day.csv`) ---\n"
                "Level and **clock-time Δ** correlations (5200/5300/extract spreads vs extract mid); "
                "Δ rows for tight/loose require **two consecutive** timestamps in that regime.\n"
            )
            summ.write_text(txt + tail, encoding="utf-8")

    print("Wrote", OUT)


if __name__ == "__main__":
    main()
