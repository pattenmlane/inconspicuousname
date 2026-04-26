#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate (R3 convention) on ROUND_4 prices + counterparty overlays.

Matches analyze_vev_5200_5300_tight_gate_r3.py: inner-join 5200/5300/extract on timestamp,
dedupe per product per ts (keep first), spread = ask1-bid1, tight = s5200<=TH & s5300<=TH,
fwd K = shift(-K) on extract mid minus current.

Outputs: gate stats, spread–spread correlations (inclineGod), Mark01→Mark22 VEV trades
split by tight vs not, burst×tight three-way table, Phase1-style Welch t-test tight vs loose
per csv day.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = Path("Prosperity4Data/ROUND_4")
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
EXTRACT = "VELVETFRUIT_EXTRACT"
TH = 2
K = 20


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


def aligned_panel(csv_day: int) -> pd.DataFrame:
    path = DATA / f"prices_round_4_day_{csv_day}.csv"
    df = pd.read_csv(path, sep=";")
    df = df[df["day"] == csv_day]
    a = _one_product(df, VEV_5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = _one_product(df, VEV_5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = _one_product(df, EXTRACT).rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(
        e[["timestamp", "m_ext", "s_ext"]], on="timestamp", how="inner"
    )
    m = m.sort_values("timestamp").reset_index(drop=True)
    m["csv_day"] = csv_day
    return m


def add_gate_fwd(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy()
    out["tight"] = (out["s5200"] <= TH) & (out["s5300"] <= TH)
    out["m_ext_f"] = out["m_ext"].shift(-K)
    out["fwd_k"] = out["m_ext_f"] - out["m_ext"]
    return out


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"n_a": len(a), "n_b": len(b)}
    t = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "t_stat": float(t.statistic),
        "p_value": float(t.pvalue),
    }


def load_trades(csv_day: int) -> pd.DataFrame:
    p = DATA / f"trades_round_4_day_{csv_day}.csv"
    t = pd.read_csv(p, sep=";")
    t["csv_day"] = csv_day
    return t


def main() -> None:
    burst_path = OUT / "r4_p2_m01_m22_burst_pairs.json"
    burst_set = set()
    if burst_path.exists():
        pairs = json.loads(burst_path.read_text())
        burst_set = {(int(a), int(b)) for a, b in pairs}

    per_day = {}
    all_panels = []

    for d in (1, 2, 3):
        p = add_gate_fwd(aligned_panel(d))
        all_panels.append(p)
        valid = p["fwd_k"].notna()
        pv = p.loc[valid]
        t_mask = pv["tight"]
        f_t = pv.loc[t_mask, "fwd_k"].values
        f_n = pv.loc[~t_mask, "fwd_k"].values
        per_day[str(d)] = {
            "p_tight": float(t_mask.mean()),
            "welch_tight_vs_loose_fwd20_extract": welch(f_t, f_n),
            "corr_s5200_s5300": float(pv["s5200"].corr(pv["s5300"])),
            "corr_s5200_s_ext": float(pv["s5200"].corr(pv["s_ext"])),
            "corr_s5300_s_ext": float(pv["s5300"].corr(pv["s_ext"])),
            "corr_s5200_m_ext": float(pv["s5200"].corr(pv["m_ext"])),
            "corr_s5300_m_ext": float(pv["s5300"].corr(pv["m_ext"])),
        }
        # correlations conditional on tight vs loose
        pt = pv.loc[t_mask]
        pn = pv.loc[~t_mask]
        per_day[str(d)]["corr_s5200_s5300_tight_only"] = (
            float(pt["s5200"].corr(pt["s5300"])) if len(pt) > 30 else None
        )
        per_day[str(d)]["corr_s5200_s5300_loose_only"] = (
            float(pn["s5200"].corr(pn["s5300"])) if len(pn) > 30 else None
        )

    (OUT / "r4_p3_joint_gate_per_day.json").write_text(json.dumps(per_day, indent=2))

    # Pooled panel (concat) — inclineGod spread–spread on full vs tight-only rows
    pool = pd.concat(all_panels, ignore_index=True)
    pv = pool[pool["fwd_k"].notna()]
    tight_rows = pv[pv["tight"]]
    loose_rows = pv[~pv["tight"]]
    spread_spread = {
        "pooled_full_corr_s5200_s5300": float(pv["s5200"].corr(pv["s5300"])),
        "pooled_tight_n": int(len(tight_rows)),
        "pooled_tight_corr_s5200_s5300": float(tight_rows["s5200"].corr(tight_rows["s5300"]))
        if len(tight_rows) > 20
        else None,
        "pooled_loose_n": int(len(loose_rows)),
        "pooled_loose_corr_s5200_s5300": float(loose_rows["s5200"].corr(loose_rows["s5300"]))
        if len(loose_rows) > 20
        else None,
        "pooled_welch_tight_vs_loose_fwd20": welch(
            tight_rows["fwd_k"].values, loose_rows["fwd_k"].values
        ),
    }
    (OUT / "r4_p3_spread_spread_inclinegod.json").write_text(
        json.dumps(spread_spread, indent=2)
    )

    # Merge trades onto panel for counterparty × gate
    rows = []
    for d in (1, 2, 3):
        tr = load_trades(d)
        p = add_gate_fwd(aligned_panel(d))
        key = p.set_index("timestamp")
        for _, r in tr.iterrows():
            ts = int(r["timestamp"])
            if ts not in key.index:
                continue
            pr = key.loc[ts]
            if isinstance(pr, pd.DataFrame):
                pr = pr.iloc[0]
            rows.append(
                {
                    "csv_day": d,
                    "timestamp": ts,
                    "buyer": str(r.get("buyer", "") or "").strip(),
                    "seller": str(r.get("seller", "") or "").strip(),
                    "symbol": str(r["symbol"]),
                    "tight": bool(pr["tight"]),
                    "fwd20_extract": float(pr["fwd_k"]) if pd.notna(pr["fwd_k"]) else None,
                    "burst": (d, ts) in burst_set,
                }
            )
    tdf = pd.DataFrame(rows)

    # Mark 01 -> Mark 22 on any VEV: tight vs not on extract fwd20 at trade time
    m01 = tdf[
        (tdf["buyer"] == "Mark 01")
        & (tdf["seller"] == "Mark 22")
        & (tdf["symbol"].str.startswith("VEV_"))
        & (tdf["fwd20_extract"].notna())
    ]
    gate_m01 = {
        "n_total": int(len(m01)),
        "n_tight": int(m01["tight"].sum()),
        "n_loose": int((~m01["tight"]).sum()),
        "mean_fwd20_extract_when_tight": float(m01.loc[m01["tight"], "fwd20_extract"].mean())
        if m01["tight"].any()
        else None,
        "mean_fwd20_extract_when_loose": float(m01.loc[~m01["tight"], "fwd20_extract"].mean())
        if (~m01["tight"]).any()
        else None,
        "welch": welch(
            m01.loc[m01["tight"], "fwd20_extract"].dropna().values,
            m01.loc[~m01["tight"], "fwd20_extract"].dropna().values,
        )
        if m01["tight"].sum() >= 2 and (~m01["tight"]).sum() >= 2
        else {},
    }
    (OUT / "r4_p3_m01_m22_vev_gate_extract_fwd20.json").write_text(
        json.dumps(gate_m01, indent=2)
    )

    # Three-way: burst × tight × M01-M22 VEV extract fwd20
    m01b = m01.assign(burst=m01.apply(lambda r: (r["csv_day"], r["timestamp"]) in burst_set, axis=1))
    three = {}
    for burst in (True, False):
        for tight in (True, False):
            sub = m01b[(m01b["burst"] == burst) & (m01b["tight"] == tight)]
            key = f"burst={burst}_tight={tight}"
            three[key] = {
                "n": int(len(sub)),
                "mean_fwd20_extract": float(sub["fwd20_extract"].mean()) if len(sub) else None,
            }
    (OUT / "r4_p3_three_way_burst_gate_pair.json").write_text(json.dumps(three, indent=2))

    # Overlap: how many joint-tight timestamps coincide with any >=3-trade burst?
    burst_ts_by_day: dict[int, set[int]] = {1: set(), 2: set(), 3: set()}
    for d in (1, 2, 3):
        tr = load_trades(d)
        vc = tr.groupby("timestamp").size()
        burst_ts_by_day[d] = set(int(ts) for ts, c in vc.items() if c >= 3)

    overlap = {}
    for d in (1, 2, 3):
        p = add_gate_fwd(aligned_panel(d))
        tight_ts = set(int(x) for x in p.loc[p["tight"], "timestamp"])
        bts = burst_ts_by_day[d]
        inter = tight_ts & bts
        overlap[str(d)] = {
            "n_tight_ts": len(tight_ts),
            "n_burst_ts": len(bts),
            "n_tight_and_burst": len(inter),
            "frac_tight_that_is_burst": len(inter) / len(tight_ts) if tight_ts else None,
        }
    (OUT / "r4_p3_tight_burst_overlap.json").write_text(json.dumps(overlap, indent=2))

    # Phase 1 style: unconditional vs gated — same M01-M22 VEV mean fwd20 all trades vs tight-only
    summary = {
        "phase3_vs_phase1": "Phase1 pooled Mark01->Mark22 on VEV_5300 unconditional mean ~ -0.08 (r4_focal_pair); Phase3 uses aligned extract fwd20 at trade timestamp (not VEV mid) — see r4_p3_m01_m22_vev_gate_extract_fwd20.json.",
        "phase3_vs_phase2_without_gate": "Phase2 pooled Welch tight vs loose on extract fwd20 (r4_p3_spread_spread): mean_tight ~+0.32 vs mean_loose ~-0.12 (t~7.9, p~2e-15) — Sonic gate strongly separates forward extract mid on R4 tape.",
        "per_day_joint_gate_ttest": per_day,
        "spread_spread": spread_spread,
        "tight_burst_overlap": overlap,
        "sonic_clean_hypothesis": "Day 1: corr(s5200,s5300) rises to ~0.51 in tight-only vs ~0.02 loose-only — spread co-movement visible when Sonic gate on (inclineGod).",
    }
    (OUT / "r4_p3_phase3_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", OUT / "r4_p3_phase3_summary.json")


if __name__ == "__main__":
    main()
