#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate (VEV_5200 & VEV_5300 spread ≤ TH) on Round 4 tape.

Mirrors round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py:
  inner-join 5200, 5300, VELVETFRUIT_EXTRACT on timestamp; spread = ask1 - bid1;
  tight = (s5200 <= TH) & (s5300 <= TH).

Merges gate onto Phase-1 enriched trades (same day+timestamp) for counterparty × gate stats.
inclineGod-style: spread–spread and spread vs extract spread correlations by day.

Outputs: manual_traders/R4/r3v_synthetic_local_vol_surface_15/outputs_r4_phase3/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
P1 = Path(__file__).resolve().parent / "outputs_r4_phase1" / "r4_p1_trades_enriched.csv"
OUT = Path(__file__).resolve().parent / "outputs_r4_phase3"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
TH = 2
K_FWD = 20  # match R3 default K for extract forward in panel


def _one_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    v = v.assign(
        spread=(ask - bid).astype(float),
        mid=mid,
    )
    return v[["timestamp", "spread", "mid"]].copy()


def aligned_panel_day(day: int) -> pd.DataFrame:
    p = DATA / f"prices_round_4_day_{day}.csv"
    df = pd.read_csv(p, sep=";")
    a = _one_product(df, "VEV_5200").rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = _one_product(df, "VEV_5300").rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = _one_product(df, "VELVETFRUIT_EXTRACT").rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(
        e[["timestamp", "m_ext", "s_ext"]], on="timestamp", how="inner"
    )
    m = m.sort_values("timestamp").reset_index(drop=True)
    m["day"] = int(day)
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    m["m_ext_f"] = m["m_ext"].shift(-K_FWD)
    m["fwd_extract_k"] = m["m_ext_f"] - m["m_ext"]
    return m


def main() -> None:
    panels = [aligned_panel_day(d) for d in DAYS]
    pan = pd.concat(panels, ignore_index=True)
    pan.to_csv(OUT / "r4_p3_joint_gate_panel_by_timestamp.csv", index=False)

    # Per-day: P(tight), Welch fwd K, correlations (inclineGod)
    rows = []
    for d in DAYS:
        sub = pan[pan["day"] == d].copy()
        valid = sub["fwd_extract_k"].notna()
        s = sub.loc[valid]
        t_mask = s["tight"]
        f_t = s.loc[t_mask, "fwd_extract_k"].astype(float)
        f_n = s.loc[~t_mask, "fwd_extract_k"].astype(float)
        f_t = f_t[np.isfinite(f_t)]
        f_n = f_n[np.isfinite(f_n)]
        if len(f_t) > 1 and len(f_n) > 1:
            tt = stats.ttest_ind(f_t, f_n, equal_var=False)
            tstat, pval = float(tt.statistic), float(tt.pvalue)
        else:
            tstat, pval = float("nan"), float("nan")
        rows.append(
            {
                "day": d,
                "P_tight": float(t_mask.mean()) if len(s) else float("nan"),
                "n_rows_valid_fwd": int(len(s)),
                "mean_fwd_tight": float(f_t.mean()) if len(f_t) else float("nan"),
                "mean_fwd_loose": float(f_n.mean()) if len(f_n) else float("nan"),
                "welch_t": tstat,
                "welch_p": pval,
                "corr_s5200_s5300": float(s["s5200"].corr(s["s5300"])),
                "corr_s5200_s_ext": float(s["s5200"].corr(s["s_ext"])),
                "corr_s5300_s_ext": float(s["s5300"].corr(s["s_ext"])),
                "corr_s5200_m_ext": float(s["s5200"].corr(s["m_ext"])),
                "corr_s5300_m_ext": float(s["s5300"].corr(s["m_ext"])),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "r4_p3_gate_summary_by_day.csv", index=False)

    # Merge gate onto trades
    tr = pd.read_csv(P1)
    gcols = ["day", "timestamp", "tight", "s5200", "s5300", "s_ext", "m_ext"]
    mg = tr.merge(pan[gcols], on=["day", "timestamp"], how="left")
    mg["tight"] = mg["tight"].fillna(False).astype(bool)

    def _ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 8 or len(b) < 8:
            return float("nan"), float("nan")
        r = stats.ttest_ind(a, b, equal_var=False)
        return float(r.statistic), float(r.pvalue)

    # Mark 67 aggressive buy extract: fwd_mid_k20 tight vs not
    m67 = mg[
        (mg["symbol"] == "VELVETFRUIT_EXTRACT")
        & (mg["buyer"] == "Mark 67")
        & (mg["aggressor_bucket"] == "aggr_buy")
    ]
    x = pd.to_numeric(m67["fwd_mid_k20"], errors="coerce")
    t67_t = x[m67["tight"]].dropna().to_numpy()
    t67_f = x[~m67["tight"]].dropna().to_numpy()
    tstat67, p67 = _ttest(t67_t, t67_f)

    # Mark 01 -> 22 on VEV_5300
    m01 = mg[
        (mg["buyer"] == "Mark 01")
        & (mg["seller"] == "Mark 22")
        & (mg["symbol"] == "VEV_5300")
    ]
    x2 = pd.to_numeric(m01["fwd_mid_k20"], errors="coerce")
    t01_t = x2[m01["tight"]].dropna().to_numpy()
    t01_f = x2[~m01["tight"]].dropna().to_numpy()
    tstat01, p01 = _ttest(t01_t, t01_f)

    # Burst extract trades × gate
    be = mg[(mg["symbol"] == "VELVETFRUIT_EXTRACT") & (mg["burst"] == 1)]
    y = pd.to_numeric(be["fwd_mid_k20"], errors="coerce")
    bt_t = y[be["tight"]].dropna().to_numpy()
    bt_f = y[~be["tight"]].dropna().to_numpy()
    tstatb, pb = _ttest(bt_t, bt_f)

    inter = pd.DataFrame(
        [
            {
                "slice": "Mark67_aggr_buy_EXTRACT_fwd20",
                "n_tight": int(len(t67_t)),
                "mean_tight": float(np.mean(t67_t)) if len(t67_t) else float("nan"),
                "n_loose": int(len(t67_f)),
                "mean_loose": float(np.mean(t67_f)) if len(t67_f) else float("nan"),
                "welch_t": tstat67,
                "welch_p": p67,
            },
            {
                "slice": "Mark01_to_Mark22_VEV5300_fwd20",
                "n_tight": int(len(t01_t)),
                "mean_tight": float(np.mean(t01_t)) if len(t01_t) else float("nan"),
                "n_loose": int(len(t01_f)),
                "mean_loose": float(np.mean(t01_f)) if len(t01_f) else float("nan"),
                "welch_t": tstat01,
                "welch_p": p01,
            },
            {
                "slice": "EXTRACT_burst_fwd20",
                "n_tight": int(len(bt_t)),
                "mean_tight": float(np.mean(bt_t)) if len(bt_t) else float("nan"),
                "n_loose": int(len(bt_f)),
                "mean_loose": float(np.mean(bt_f)) if len(bt_f) else float("nan"),
                "welch_t": tstatb,
                "welch_p": pb,
            },
        ]
    )
    inter.to_csv(OUT / "r4_p3_counterparty_x_gate_welch_fwd20.csv", index=False)

    # Day-stability table for Mark67 within tight only
    stab = []
    for d in DAYS:
        sub = m67[m67["day"] == d]
        xt = pd.to_numeric(sub["fwd_mid_k20"], errors="coerce")
        stab.append(
            {
                "day": d,
                "n_tight": int(sub["tight"].sum()),
                "mean_fwd20_when_tight": float(xt[sub["tight"]].mean()) if sub["tight"].any() else float("nan"),
                "n_loose": int((~sub["tight"]).sum()),
                "mean_fwd20_when_loose": float(xt[~sub["tight"]].mean()) if (~sub["tight"]).any() else float("nan"),
            }
        )
    pd.DataFrame(stab).to_csv(OUT / "r4_p3_mark67_fwd20_tight_vs_loose_by_day.csv", index=False)

    # inclineGod spread matrix (mean correlation isn't meaningful across days — already in by_day)
    lines = [
        "Phase 3 Sonic gate (TH=2) on Round 4 days 1–3",
        f"Panel rows: {len(pan)} | merge onto trades: {len(mg)}",
        "",
        "Mark 67 aggr_buy EXTRACT fwd20: tight vs loose (Welch on merged trades)",
        inter.iloc[0].to_string(),
        "",
        "Interpretation: if mean_tight > mean_loose with low p, gate **amplifies** Phase-1 Mark67 signal;",
        "if p high, gate does not cleanly separate markouts for this slice on R4 days 1–3.",
    ]
    (OUT / "r4_p3_executive_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (OUT / "r4_p3_manifest.json").write_text(
        json.dumps({"outputs": sorted(p.name for p in OUT.glob("*"))}, indent=2),
        encoding="utf-8",
    )
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
