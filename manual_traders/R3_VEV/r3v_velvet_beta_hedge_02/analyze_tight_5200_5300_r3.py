"""
Round 3 tape: align VEV_5200, VEV_5300, VELVETFRUIT_EXTRACT on (day, timestamp);
spread = ask1 - bid1; "tight" = both <= 2. Summarize |extract mid step| and mean spread stats.
Output: analysis_v9_tight_5200_5300_brief.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA = ROOT / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_v9_tight_5200_5300_brief.json"


def main() -> None:
    rows: list[dict] = []
    for day in (0, 1, 2):
        p = DATA / f"prices_round_3_day_{day}.csv"
        df = pd.read_csv(p, sep=";")
        sub = df[df["product"].isin(["VEV_5200", "VEV_5300", "VELVETFRUIT_EXTRACT"])].copy()
        sub["spread"] = sub["ask_price_1"] - sub["bid_price_1"]
        piv = sub.pivot_table(
            index="timestamp", columns="product", values=["mid_price", "spread"], aggfunc="first"
        )
        if piv.empty:
            continue
        ext_mid = piv[("mid_price", "VELVETFRUIT_EXTRACT")]
        s5200 = piv[("spread", "VEV_5200")]
        s5300 = piv[("spread", "VEV_5300")]
        valid = ext_mid.notna() & s5200.notna() & s5300.notna()
        ext_mid = ext_mid[valid]
        s5200 = s5200[valid]
        s5300 = s5300[valid]
        du = ext_mid.diff().abs()
        tight = (s5200 <= 2) & (s5300 <= 2)
        n = int(len(ext_mid))
        n_t = int(tight.sum())
        summary = {
            "csv_day": int(day),
            "n_rows_aligned": n,
            "n_tight": n_t,
            "p_tight": float(n_t / n) if n else 0.0,
            "mean_abs_u_step_all": float(du[1:].mean()) if n > 1 else None,
            "mean_abs_u_step_tight": float(du[1:][tight[1:]].mean()) if n > 1 else None,
            "mean_abs_u_step_nottight": float(du[1:][~tight[1:]].mean()) if n > 1 else None,
            "mean_s5200": float(s5200.mean()),
            "mean_s5300": float(s5300.mean()),
        }
        # correlation 5200 vs 5300 spread (level)
        summary["corr_s5200_s5300"] = float(s5200.corr(s5300)) if n > 1 else None
        summary["corr_s5200_m_ext"] = float(s5200.corr(ext_mid)) if n > 1 else None
        rows.append(summary)

    out = {
        "method": "prices_round_3_day_0..2: pivot mid/spread; tight = (s_5200<=2 and s_5300<=2); |du| = abs diff extract mid",
        "by_day": rows,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
