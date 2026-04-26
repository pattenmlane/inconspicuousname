"""
Round 3 days 0-2: align VELVETFRUIT_EXTRACT and VEV_4000 on (day, timestamp);
ret1 = diff(extract mid). Summarize VEV_4000 spread and |ret1| relationship.
Output: analysis_v10_extract_shock_v4000_brief.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA = ROOT / "Prosperity4Data" / "ROUND_3"
U = "VELVETFRUIT_EXTRACT"
V0 = "VEV_4000"
OUT = Path(__file__).resolve().parent / "analysis_v10_extract_shock_v4000_brief.json"


def main() -> None:
    rows: list[dict] = []
    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        sub = df[df["product"].isin([U, V0])].copy()
        sub["spr"] = sub["ask_price_1"] - sub["bid_price_1"]
        piv = sub.pivot_table(
            index="timestamp", columns="product", values=["mid_price", "spr"], aggfunc="first"
        )
        m_u = piv[("mid_price", U)]
        m0 = piv[("mid_price", V0)]
        s0 = piv[("spr", V0)]
        ok = m_u.notna() & m0.notna() & s0.notna()
        m_u, m0, s0 = m_u[ok], m0[ok], s0[ok]
        du = m_u.diff()
        d0 = m0.diff()
        adu = du.abs()
        s0p = s0.shift(1)
        j = pd.DataFrame({"adu": adu, "d0": d0, "s0": s0, "s0_prev": s0p})
        j = j.iloc[1:].dropna()
        hi = j["adu"] >= j["adu"].quantile(0.9)
        lo = j["adu"] <= j["adu"].quantile(0.5)
        rows.append(
            {
                "csv_day": int(day),
                "n": int(len(j)),
                "mean_s0": float(j["s0"].mean()),
                "mean_s0_when_adu_hi": float(j.loc[hi, "s0"].mean()),
                "mean_s0_when_adu_lo": float(j.loc[lo, "s0"].mean()),
                "mean_abs_d0": float(j["d0"].abs().mean()),
                "mean_abs_d0_when_adu_hi": float(j.loc[hi, "d0"].abs().mean()),
                "mean_abs_d0_when_adu_lo": float(j.loc[lo, "d0"].abs().mean()),
                "corr_adu_s0": float(j["adu"].corr(j["s0"])),
                "corr_adu_absd0": float(j["adu"].corr(j["d0"].abs())),
            }
        )
    out = {
        "method": "R3 d0-2, aligned extract + VEV_4000; adu=|du|; hi=adu>=p90, lo=adu<=p50 within day",
        "by_day": rows,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
