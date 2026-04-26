#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate (VEV_5200 & VEV_5300 spread <= 2) on Round 4 tape,
inclineGod-style spread correlations, and counterparty × gate interaction tables.

Aligned panel: **inner join** on `timestamp` for VEV_5200, VEV_5300, VELVETFRUIT_EXTRACT
(same convention as `round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py`).

Outputs: manual_traders/R4/r3v_volume_weighted_residual_05/analysis_outputs/phase3/
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

VEV_5200, VEV_5300 = "VEV_5200", "VEV_5300"
U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"
TH = 2
K_FWD = 20


def _one(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return v.assign(
        spread=(ask - bid).astype(float),
        mid=mid,
    )[["timestamp", "spread", "mid"]].copy()


def aligned(day: int) -> pd.DataFrame:
    p = DATA / f"prices_round_4_day_{day}.csv"
    df = pd.read_csv(p, sep=";")
    a = _one(df, VEV_5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = _one(df, VEV_5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = _one(df, U).rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(
        e[["timestamp", "m_ext", "s_ext"]], on="timestamp", how="inner"
    )
    h = _one(df, H).rename(columns={"spread": "s_hydro", "mid": "mid_hydro"})
    m = m.merge(h[["timestamp", "s_hydro", "mid_hydro"]], on="timestamp", how="inner")
    m["day"] = day
    m = m.sort_values("timestamp").reset_index(drop=True)
    return m


def fwd_series(mid: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(mid), np.nan)
    for i in range(len(mid) - k):
        out[i] = float(mid[i + k] - mid[i])
    return out


def build_paths(pr_all: pd.DataFrame) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (day, sym), g in pr_all.groupby(["day", "product"]):
        g = g.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
        ts = g["timestamp"].to_numpy(np.int64)
        mid = pd.to_numeric(g["mid_price"], errors="coerce").to_numpy(np.float64)
        spr = (
            pd.to_numeric(g["ask_price_1"], errors="coerce")
            - pd.to_numeric(g["bid_price_1"], errors="coerce")
        ).to_numpy(np.float64)
        out[(int(day), str(sym))] = (ts, mid, spr)
    return out


def fwd_at(paths: dict, day: int, sym: str, t: int, k: int) -> float:
    key = (day, sym)
    if key not in paths:
        return float("nan")
    ts_a, mid_a, _ = paths[key]
    i = int(np.searchsorted(ts_a, t, side="left"))
    if i >= len(ts_a) or ts_a[i] != t:
        return float("nan")
    j = i + k
    if j >= len(mid_a):
        return float("nan")
    return float(mid_a[j] - mid_a[i])


def main() -> None:
    frames = []
    for d in (1, 2, 3):
        frames.append(aligned(d))
    panel = pd.concat(frames, ignore_index=True)
    panel["tight"] = (panel["s5200"] <= TH) & (panel["s5300"] <= TH)
    panel["fwd_u"] = panel.groupby("day")["m_ext"].transform(lambda s: pd.Series(fwd_series(s.to_numpy(), K_FWD), index=s.index))

    # --- Sonic replication: Welch on fwd_u tight vs not ---
    rows_tt = []
    for d in (1, 2, 3):
        g = panel[panel["day"] == d].dropna(subset=["fwd_u"])
        t = g.loc[g["tight"], "fwd_u"].to_numpy(float)
        n = g.loc[~g["tight"], "fwd_u"].to_numpy(float)
        t = t[np.isfinite(t)]
        n = n[np.isfinite(n)]
        if len(t) > 2 and len(n) > 2:
            a = stats.ttest_ind(t, n, equal_var=False, nan_policy="omit")
            rows_tt.append(
                {
                    "day": d,
                    "n_tight": len(t),
                    "n_loose": len(n),
                    "mean_tight": float(np.mean(t)),
                    "mean_loose": float(np.mean(n)),
                    "welch_t": float(a.statistic),
                    "p_value": float(a.pvalue),
                }
            )
    pd.DataFrame(rows_tt).to_csv(OUT / "r4_phase3_sonic_welch_extract_fwd20_by_day.csv", index=False)

    g_all = panel.dropna(subset=["fwd_u"])
    t = g_all.loc[g_all["tight"], "fwd_u"].to_numpy(float)
    n = g_all.loc[~g_all["tight"], "fwd_u"].to_numpy(float)
    t, n = t[np.isfinite(t)], n[np.isfinite(n)]
    if len(t) > 2 and len(n) > 2:
        a = stats.ttest_ind(t, n, equal_var=False, nan_policy="omit")
        pooled = {
            "n_tight": len(t),
            "n_loose": len(n),
            "mean_tight": float(np.mean(t)),
            "mean_loose": float(np.mean(n)),
            "welch_t": float(a.statistic),
            "p_value": float(a.pvalue),
            "P_tight": float(g_all["tight"].mean()),
        }
    else:
        pooled = {}
    (OUT / "r4_phase3_sonic_welch_extract_fwd20_pooled.txt").write_text(str(pooled) + "\n")

    # inclineGod: spread–spread and spread vs extract mid
    g2 = panel.dropna(subset=["s5200", "s5300", "s_ext", "m_ext"])
    corr_txt = []
    for a, b in [
        ("s5200", "s5300"),
        ("s5200", "s_ext"),
        ("s5300", "s_ext"),
        ("s5200", "s_hydro"),
        ("s5300", "s_hydro"),
        ("s_ext", "s_hydro"),
    ]:
        c = float(g2[a].corr(g2[b]))
        corr_txt.append(f"corr({a},{b})={c:.4f}")
    (OUT / "r4_phase3_spread_spread_correlations.txt").write_text("\n".join(corr_txt) + f"\nn_rows={len(g2)}\n")

    # Subsample for scatter CSV (5200 vs 5300 spread)
    samp = g2.sample(min(5000, len(g2)), random_state=0)[["day", "timestamp", "s5200", "s5300", "tight"]]
    samp.to_csv(OUT / "r4_phase3_scatter_s5200_vs_s5300_sample.csv", index=False)

    # Load full prices for trade merge
    pr_frames = []
    for d in (1, 2, 3):
        df = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        df["day"] = d
        pr_frames.append(df)
    pr_all = pd.concat(pr_frames, ignore_index=True)
    paths = build_paths(pr_all)

    gate_map = panel.set_index(["day", "timestamp"])[["tight", "s5200", "s5300"]]

    trs = []
    for d in (1, 2, 3):
        t = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        t["day"] = d
        trs.append(t)
    tr = pd.concat(trs, ignore_index=True)
    tr["timestamp"] = tr["timestamp"].astype(int)

    def gate_row(r) -> tuple[bool, float, float]:
        key = (int(r["day"]), int(r["timestamp"]))
        if key not in gate_map.index:
            return False, float("nan"), float("nan")
        row = gate_map.loc[key]
        return bool(row["tight"]), float(row["s5200"]), float(row["s5300"])

    tg, s5, s3 = [], [], []
    for _, r in tr.iterrows():
        t, a, b = gate_row(r)
        tg.append(t)
        s5.append(a)
        s3.append(b)
    tr["sonic_tight"] = tg
    tr["s5200_at"] = s5
    tr["s5300_at"] = s3

    tr["fwd5300_20"] = [
        fwd_at(paths, int(d), VEV_5300, int(ts), 20)
        for d, ts in zip(tr["day"], tr["timestamp"])
    ]
    tr["fwd_u_20"] = [
        fwd_at(paths, int(d), U, int(ts), 20) for d, ts in zip(tr["day"], tr["timestamp"])
    ]

    # Mark01->Mark22 on VEV_5300: gate on vs off
    m = tr[(tr["buyer"] == "Mark 01") & (tr["seller"] == "Mark 22") & (tr["symbol"] == VEV_5300)]
    gsum = []
    for label, mask in [("gate_on", m["sonic_tight"]), ("gate_off", ~m["sonic_tight"])]:
        x = pd.to_numeric(m.loc[mask, "fwd5300_20"], errors="coerce").dropna()
        gsum.append({"slice": label, "n": len(x), "mean_fwd5300_20": float(x.mean()) if len(x) else float("nan")})
    pd.DataFrame(gsum).to_csv(OUT / "r4_phase3_m01_m22_vev5300_fwd20_by_gate.csv", index=False)

    # Day stability: gate-on subset per day
    stab = []
    for d in (1, 2, 3):
        mm = m[m["day"] == d]
        x = pd.to_numeric(mm.loc[mm["sonic_tight"], "fwd5300_20"], errors="coerce").dropna()
        stab.append({"day": d, "n_gate_on": len(x), "mean_fwd": float(x.mean()) if len(x) else float("nan")})
    pd.DataFrame(stab).to_csv(OUT / "r4_phase3_m01_m22_vev5300_gate_on_by_day.csv", index=False)

    # Burst-B (same as phase2): Mark01->Mark22 and >=3 VEV symbols at (day,ts)
    burst_keys: set[tuple[int, int]] = set()
    for (day, ts), g in tr.groupby(["day", "timestamp"]):
        if len(g) < 3:
            continue
        if not ((g["buyer"] == "Mark 01") & (g["seller"] == "Mark 22")).any():
            continue
        vev_syms = {s for s in g["symbol"] if str(s).startswith("VEV_")}
        if len(vev_syms) >= 3:
            burst_keys.add((int(day), int(ts)))
    tr["burst_B"] = [int((int(d), int(ts)) in burst_keys) for d, ts in zip(tr["day"], tr["timestamp"])]

    bsum = []
    for burst in (0, 1):
        for tight in (False, True):
            sub = tr[(tr["burst_B"] == burst) & (tr["sonic_tight"] == tight) & (tr["symbol"] == VEV_5300)]
            x = pd.to_numeric(sub["fwd5300_20"], errors="coerce").dropna()
            bsum.append(
                {
                    "burst_B": burst,
                    "sonic_tight": tight,
                    "n": len(x),
                    "mean_fwd5300_20": float(x.mean()) if len(x) else float("nan"),
                }
            )
    pd.DataFrame(bsum).to_csv(OUT / "r4_phase3_burstB_vev5300_fwd20_gate_cross.csv", index=False)

    # Mark67 buy_agg on U: gate interaction
    bbo = pr_all.set_index(["day", "timestamp", "product"])[["bid_price_1", "ask_price_1"]]

    def is_buy_agg(r) -> bool:
        sym = str(r["symbol"])
        if sym != U:
            return False
        key = (int(r["day"]), int(r["timestamp"]), U)
        if key not in bbo.index:
            return False
        bid1 = float(bbo.loc[key, "bid_price_1"])
        ask1 = float(bbo.loc[key, "ask_price_1"])
        px = float(r["price"])
        return px >= ask1

    tr["u_buy_agg"] = tr.apply(is_buy_agg, axis=1)
    m67 = tr[(tr["buyer"] == "Mark 67") & (tr["u_buy_agg"])]
    u67 = []
    for tight in (False, True):
        x = pd.to_numeric(m67.loc[m67["sonic_tight"] == tight, "fwd_u_20"], errors="coerce").dropna()
        u67.append(
            {
                "sonic_tight": tight,
                "n": len(x),
                "mean_fwd_u_20": float(x.mean()) if len(x) else float("nan"),
            }
        )
    pd.DataFrame(u67).to_csv(OUT / "r4_phase3_mark67_buyagg_fwd_u20_by_gate.csv", index=False)

    print("Phase3 wrote", OUT)


if __name__ == "__main__":
    main()
