#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate (VEV_5200 & VEV_5300 BBO spread both <= TH=2) on R4 tape,
plus inclineGod spread–spread / spread–extract panels. Three-way (burst x tight x markout).

Convention matches round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py:
inner join 5200, 5300, VELVETFRUIT_EXTRACT on timestamp (per day); tight = (s5200<=2)&(s5300<=2).

Reads: Prosperity4Data/ROUND_4, optional r4_p1_trades_enriched.csv for trade-level merge.
Writes: outputs/r4_p3_*
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
DAYS = [1, 2, 3]
TH = 2
VEV_5200, VEV_5300 = "VEV_5200", "VEV_5300"
EX = "VELVETFRUIT_EXTRACT"


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = d
        frames.append(df)
    pr = pd.concat(frames, ignore_index=True)
    pr = pr.rename(columns={"product": "symbol"})
    bid = pd.to_numeric(pr["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(pr["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(pr["mid_price"], errors="coerce")
    pr["spread"] = ask - bid
    pr["mid"] = mid
    return pr[["day", "timestamp", "symbol", "spread", "mid"]].dropna(subset=["spread", "mid"])


def one_sym(pr: pd.DataFrame, day: int, sym: str) -> pd.DataFrame:
    v = pr[(pr["day"] == day) & (pr["symbol"] == sym)].drop_duplicates("timestamp").sort_values("timestamp")
    return v[["timestamp", "spread", "mid"]].rename(columns={"spread": f"sp_{sym}", "mid": f"m_{sym}"})


def aligned_gate(pr: pd.DataFrame, day: int) -> pd.DataFrame:
    a = one_sym(pr, day, VEV_5200)
    b = one_sym(pr, day, VEV_5300)
    e = one_sym(pr, day, EX)
    m = a.merge(b, on="timestamp", how="inner").merge(e, on="timestamp", how="inner")
    m["day"] = day
    m["tight"] = (m[f"sp_{VEV_5200}"] <= TH) & (m[f"sp_{VEV_5300}"] <= TH)
    m["s_prod"] = m[f"sp_{VEV_5200}"] * m[f"sp_{VEV_5300}"]
    return m


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        t = pd.read_csv(p, sep=";")
        t["day"] = d
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def burst_exact(tr: pd.DataFrame) -> set[tuple[int, int]]:
    s = set()
    for (d, ts), sub in tr.groupby(["day", "timestamp"]):
        if len(sub) < 2:
            continue
        if not sub["buyer"].eq("Mark 01").all() or not sub["seller"].eq("Mark 22").all():
            continue
        if sub["symbol"].nunique() >= 3:
            s.add((int(d), int(ts)))
    return s


def extract_fwd20_at_ts(pr: pd.DataFrame, day: int) -> dict[int, float]:
    sub = pr[(pr["day"] == day) & (pr["symbol"] == EX)].sort_values("timestamp")
    ts = sub["timestamp"].to_numpy(dtype=int)
    m = sub["mid"].to_numpy(dtype=float)
    imap = {int(ts[i]): i for i in range(len(ts))}
    out = {}
    for t, i in imap.items():
        if i + 20 < len(m):
            out[t] = float(m[i + 20] - m[i])
    return out


def main() -> None:
    pr = load_prices()
    tr = load_trades()
    burst_set = burst_exact(tr)

    # --- inclineGod: spread-spread and spread vs extract mid (aligned panel, per day) ---
    corr_rows = []
    scat_rows = []
    for d in DAYS:
        g = aligned_gate(pr, d)
        if len(g) < 30:
            continue
        c_ss = float(g[f"sp_{VEV_5200}"].corr(g[f"sp_{VEV_5300}"]))
        c_s52_u = float(g[f"sp_{VEV_5200}"].corr(g[f"m_{EX}"]))
        c_s53_u = float(g[f"sp_{VEV_5300}"].corr(g[f"m_{EX}"]))
        corr_rows.append({"day": d, "corr_s5200_s5300": c_ss, "corr_s5200_m_ext": c_s52_u, "corr_s5300_m_ext": c_s53_u, "n": len(g)})
        g2 = g.copy()
        g2["dm_ext"] = g2[f"m_{EX}"].diff()
        g2["ds5200"] = g2[f"sp_{VEV_5200}"].diff()
        g2["ds5300"] = g2[f"sp_{VEV_5300}"].diff()
        scat_rows.append(
            {
                "day": d,
                "corr_abs_dm_abs_ds5200": float(g2["dm_ext"].abs().corr(g2["ds5200"].abs())),
                "corr_abs_dm_abs_ds5300": float(g2["dm_ext"].abs().corr(g2["ds5300"].abs())),
            }
        )
    pd.DataFrame(corr_rows).to_csv(OUT / "r4_p3_spreadcorr_by_day.csv", index=False)
    pd.DataFrame(scat_rows).to_csv(OUT / "r4_p3_spreadchange_vs_abs_dmid_extract.csv", index=False)

    # Gate table all timestamps
    gate_parts = [aligned_gate(pr, d) for d in DAYS]
    gate_all = pd.concat(gate_parts, ignore_index=True)
    gate_all.to_csv(OUT / "r4_p3_aligned_gate_panel_sample.csv", index=False)

    # --- Three-way: each trade row + tight at timestamp + burst + fwd extract +20 ---
    en_path = OUT / "r4_p1_trades_enriched.csv"
    if not en_path.is_file():
        raise SystemExit("Need r4_p1_trades_enriched.csv from phase 1")
    en = pd.read_csv(en_path)
    gt = gate_all[["day", "timestamp", "tight", f"sp_{VEV_5200}", f"sp_{VEV_5300}"]].rename(
        columns={f"sp_{VEV_5200}": "s5200_gate", f"sp_{VEV_5300}": "s5300_gate"}
    )
    en = en.merge(gt, on=["day", "timestamp"], how="left")
    en["tight_joint"] = en["tight"].fillna(False).astype(bool)
    en["burst_exact"] = en.apply(lambda r: 1 if (int(r["day"]), int(r["timestamp"])) in burst_set else 0, axis=1)

    fwd_u = []
    for _, r in en.iterrows():
        d = int(r["day"])
        t = int(r["timestamp"])
        mp = extract_fwd20_at_ts(pr, d)
        fwd_u.append(mp.get(t, float("nan")))
    en["fwd_u_20"] = fwd_u

    # Mark 01 -> Mark 22 on extract only
    en["pair"] = en["buyer"].astype(str) + "->" + en["seller"].astype(str)
    sub01 = en[(en["pair"] == "Mark 01->Mark 22") & (en["symbol"] == EX)].copy()

    agg = (
        en.groupby(["burst_exact", "tight_joint"])["fwd_u_20"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    agg.to_csv(OUT / "r4_p3_threeway_burst_tight_fwd_u20.csv", index=False)

    gtab = (
        sub01.groupby("tight_joint")["fwd_u_20"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    gtab.to_csv(OUT / "r4_p3_mark01_22_extract_fwd_u20_by_gate.csv", index=False)

    sub01_any = en[(en["pair"] == "Mark 01->Mark 22")].copy()
    gtab2 = (
        sub01_any.groupby("tight_joint")["fwd_u_20"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    gtab2.to_csv(OUT / "r4_p3_mark01_22_any_symbol_fwd_u20_by_gate.csv", index=False)

    lines = ["=== Phase 3 Sonic gate summary (Round 4) ==="]
    for _, row in agg.iterrows():
        lines.append(
            f"burst={int(row['burst_exact'])} tight={bool(row['tight_joint'])} n={int(row['count'])} mean_fwd_u20={row['mean']:.5f}"
        )
    lines.append("--- Mark01->Mark22 VELVET only (often empty) ---")
    for _, row in gtab.iterrows():
        lines.append(f"tight={bool(row['tight_joint'])} n={int(row['count'])} mean_fwd_u20={row['mean']:.5f}")
    lines.append("--- Mark01->Mark22 any symbol ---")
    for _, row in gtab2.iterrows():
        lines.append(f"tight={bool(row['tight_joint'])} n={int(row['count'])} mean_fwd_u20={row['mean']:.5f}")
    (OUT / "r4_p3_threeway_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Wrote r4_p3_* to", OUT)


if __name__ == "__main__":
    main()
