#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate (inner-join 5200+5300 timestamps, same as R3
`analyze_vev_5200_5300_tight_gate_r3.py`) stacked with counterparty / burst /
spread–spread evidence (inclineGod).

Tape: Prosperity4Data/ROUND_4 days 1–3.

Writes:
  r4_p3_gate_panel.csv              — aligned mids/spreads + joint_tight + extract fwd20
  r4_p3_spread_correlations.csv     — mid–mid and spread–spread (full sample + gate-only)
  r4_p3_forward_by_mark_gated.csv   — Phase-1-style Mark×role×K, split joint_tight on/off
  r4_p3_burst_extract_gated.csv     — Phase-1 big bursts × gate at burst ts
  r4_p3_burst_echo_gated.csv        — Phase-2 Mark01→Mark22 ≥3VEV bursts × gate
  r4_p3_pair_residual_gate_split.csv— mean |residual_fwd20| by gate (Mark01→Mark22 cells)
  r4_p3_leadlag_gated.csv           — signed-flow corr extract vs 5300, tight vs wide rows
  r4_p3_passive_m22_gated.csv       — Mark22 seller_agg fwd20 split by gate at print ts
  r4_phase3_summary.txt
  r4_phase3_gate.json               — fragment for analysis.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent
REL = "manual_traders/R4/r3v_smile_quadratic_logm_wls_10"
DAYS = [1, 2, 3]
KS = (5, 20, 100)
G5200, G5300 = "VEV_5200", "VEV_5300"
EX = "VELVETFRUIT_EXTRACT"
HY = "HYDROGEL_PACK"
JOINT_TH = 2
BURST_WIN = 500
CORR_SYMS = [
    EX,
    HY,
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]


def load_prices() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = df["day"].astype(int)
        df["timestamp"] = df["timestamp"].astype(int)
        df["symbol"] = df["product"].astype(str)
        bid = pd.to_numeric(df["bid_price_1"], errors="coerce")
        ask = pd.to_numeric(df["ask_price_1"], errors="coerce")
        mid = pd.to_numeric(df["mid_price"], errors="coerce")
        df = df.assign(
            bid1=bid,
            ask1=ask,
            mid=mid,
            spread=(ask - bid),
        )
        frames.append(df[["day", "timestamp", "symbol", "bid1", "ask1", "mid", "spread"]])
    return pd.concat(frames, ignore_index=True)


def load_trades() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = int(d)
        df["timestamp"] = df["timestamp"].astype(int)
        for c in ("buyer", "seller", "symbol"):
            df[c] = df[c].astype(str)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["quantity"] = df["quantity"].astype(int)
        frames.append(df[["day", "timestamp", "buyer", "seller", "symbol", "price", "quantity"]])
    return pd.concat(frames, ignore_index=True)


def _one_sym(px: pd.DataFrame, day: int, sym: str) -> pd.DataFrame:
    v = px[(px["day"] == day) & (px["symbol"] == sym)].copy()
    v = v.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp")
    return v[["timestamp", "spread", "mid"]].rename(
        columns={"spread": f"s_{sym}", "mid": f"m_{sym}"}
    )


def aligned_gate_panel_per_day(px: pd.DataFrame, day: int) -> pd.DataFrame:
    """R3 convention: inner join on timestamp so 5200 and 5300 rows exist together."""
    a = _one_sym(px, day, G5200)
    b = _one_sym(px, day, G5300)
    m = a.merge(b, on="timestamp", how="inner")
    ex = _one_sym(px, day, EX)
    m = m.merge(ex, on="timestamp", how="inner")
    m["day"] = day
    m["s5200"] = m[f"s_{G5200}"]
    m["s5300"] = m[f"s_{G5300}"]
    m["m_ext"] = m[f"m_{EX}"]
    m["s_ext"] = m[f"s_{EX}"]
    m["joint_tight"] = (m["s5200"] <= JOINT_TH) & (m["s5300"] <= JOINT_TH)
    m["m5200"] = m[f"m_{G5200}"]
    m["m5300"] = m[f"m_{G5300}"]
    m["fwd20_ext"] = m["m_ext"].shift(-20) - m["m_ext"]
    m["fwd5_5300"] = m["m5300"].shift(-5) - m["m5300"]
    m["fwd20_5300"] = m["m5300"].shift(-20) - m["m5300"]
    return m


def build_full_gate_panel(px: pd.DataFrame) -> pd.DataFrame:
    parts = [aligned_gate_panel_per_day(px, d) for d in DAYS]
    return pd.concat(parts, ignore_index=True).sort_values(["day", "timestamp"])


def add_forward_mids(px: pd.DataFrame) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []
    for sym, g in px.groupby("symbol", sort=False):
        g = g.sort_values(["day", "timestamp"]).reset_index(drop=True)
        for k in KS:
            g[f"fwd_mid_{k}"] = g["mid"].shift(-k) - g["mid"]
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)


def classify_side_vec(p: pd.Series, bid: pd.Series, ask: pd.Series) -> pd.Series:
    out = pd.Series("unknown", index=p.index, dtype=object)
    ok = p.notna() & bid.notna() & ask.notna()
    out.loc[ok & (p >= ask)] = "buyer_aggressive"
    out.loc[ok & (p <= bid)] = "seller_aggressive"
    return out


def spread_spread_correlations(px: pd.DataFrame, gate_panel_df: pd.DataFrame) -> pd.DataFrame:
    """Wide panel: only timestamps in gate inner join; add spreads for other syms via merge."""
    rows = []
    g0 = gate_panel_df[["day", "timestamp", "joint_tight"]].drop_duplicates()
    for sample_name, mask in (
        ("all_inner_timestamps", np.ones(len(g0), dtype=bool)),
        ("joint_tight_only", g0["joint_tight"].values),
    ):
        base = g0.loc[mask, ["day", "timestamp"]].drop_duplicates()
        if len(base) < 30:
            continue
        wide = base.copy()
        for sym in CORR_SYMS:
            sub = px.loc[px["symbol"] == sym, ["day", "timestamp", "mid", "spread"]].rename(
                columns={"mid": f"mid_{sym}", "spread": f"sp_{sym}"}
            )
            wide = wide.merge(sub, on=["day", "timestamp"], how="left")
        mid_cols = [c for c in wide.columns if c.startswith("mid_")]
        sp_cols = [c for c in wide.columns if c.startswith("sp_")]
        for kind, cols in (("mid", mid_cols), ("spread", sp_cols)):
            if len(cols) < 2:
                continue
            subm = wide[cols].apply(pd.to_numeric, errors="coerce")
            cmat = subm.corr(min_periods=100)
            for i, ci in enumerate(cols):
                for j, cj in enumerate(cols):
                    if j <= i:
                        continue
                    rows.append(
                        {
                            "sample": sample_name,
                            "corr_kind": kind,
                            "x": ci,
                            "y": cj,
                            "pearson_r": float(cmat.loc[ci, cj]) if pd.notna(cmat.loc[ci, cj]) else float("nan"),
                            "n_pair": int(subm[[ci, cj]].dropna().shape[0]),
                        }
                    )
    return pd.DataFrame(rows)


def asof_gate_on_trades(
    tr_df: pd.DataFrame, gate_panel: pd.DataFrame, *, cols: tuple[str, ...] = ("joint_tight", "s5200", "s5300")
) -> pd.DataFrame:
    """Last inner-join panel state at or before each (day, timestamp); trade prints rarely
    share exact price timestamps with the 5200/5300 inner grid."""
    parts: list[pd.DataFrame] = []
    gcols = ["timestamp", *cols]
    for d in sorted(tr_df["day"].unique()):
        left = tr_df.loc[tr_df["day"] == d].sort_values("timestamp").copy()
        right = gate_panel.loc[gate_panel["day"] == d, gcols].sort_values("timestamp")
        if right.empty or left.empty:
            left["gate_asof_matched"] = False
            left["joint_tight"] = False
            left["s5200"] = np.nan
            left["s5300"] = np.nan
            parts.append(left)
            continue
        merged = pd.merge_asof(
            left,
            right,
            on="timestamp",
            direction="backward",
        )
        merged["gate_asof_matched"] = merged[cols[0]].notna()
        for c in cols:
            if c == "joint_tight":
                merged[c] = merged[c].fillna(False).astype(bool)
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def tstat_welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return (float("nan"),) * 4
    r = ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return (float(np.mean(a)), float(np.mean(b)), float(r.statistic), float(r.pvalue))


def welch_vs_pool(
    focal: np.ndarray,
    pool: np.ndarray,
    *,
    rng: np.random.Generator,
    max_pool_draw: int = 5000,
) -> tuple[float, float, float, float]:
    """When stratified 'wide' bucket is too small, compare focal to a bootstrap draw from pool."""
    focal = focal[np.isfinite(focal)]
    pool = pool[np.isfinite(pool)]
    if len(focal) < 3 or len(pool) < 30:
        return (float("nan"),) * 4
    n_draw = min(len(focal), len(pool), max_pool_draw)
    ctrl = rng.choice(pool, size=n_draw, replace=False)
    return tstat_welch(focal, ctrl)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    px = load_prices()
    px_fwd = add_forward_mids(px)
    tr = load_trades()

    gate_panel = build_full_gate_panel(px)
    gate_panel.to_csv(OUT / "r4_p3_gate_panel.csv", index=False)

    # Baseline extract fwd20 tight vs wide (same as P2 but explicit inner-join panel)
    valid = gate_panel["fwd20_ext"].notna()
    tight_f = gate_panel.loc[valid & gate_panel["joint_tight"], "fwd20_ext"].astype(float)
    wide_f = gate_panel.loc[valid & ~gate_panel["joint_tight"], "fwd20_ext"].astype(float)
    m_t, m_w, gt_stat, gt_p = tstat_welch(tight_f.values, wide_f.values)

    corr_df = spread_spread_correlations(px, gate_panel)
    corr_df.to_csv(OUT / "r4_p3_spread_correlations.csv", index=False)

    # --- Attach gate: inner-join grid for mids; backward asof for trade prints
    gt_exact = gate_panel[["day", "timestamp", "joint_tight", "s5200", "s5300"]].copy()
    m = tr.merge(
        px_fwd,
        on=["day", "timestamp", "symbol"],
        how="left",
        suffixes=("", "_bbo"),
    )
    m["side"] = classify_side_vec(m["price"], m["bid1"], m["ask1"])
    m = asof_gate_on_trades(m, gate_panel)
    m["gate_aligned"] = m["gate_asof_matched"]

    # Phase-1-style forward by Mark × gate split (asof gate at trade time)
    rows_fm: list[dict] = []
    marks = sorted({x for x in set(m["buyer"]) | set(m["seller"]) if str(x).startswith("Mark ")})
    for gate_label, gmask in (
        ("joint_tight", m["gate_aligned"] & m["joint_tight"]),
        ("wide_on_panel", m["gate_aligned"] & ~m["joint_tight"]),
    ):
        sub0 = m.loc[gmask]
        for u in marks:
            for role, mask in (
                ("buyer_agg", (sub0["buyer"] == u) & (sub0["side"] == "buyer_aggressive")),
                ("seller_agg", (sub0["seller"] == u) & (sub0["side"] == "seller_aggressive")),
                ("any_touch", (sub0["buyer"] == u) | (sub0["seller"] == u)),
            ):
                sub = sub0.loc[mask]
                if len(sub) < 15:
                    continue
                for k in KS:
                    col = f"fwd_mid_{k}"
                    vals = sub[col].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) < 10:
                        continue
                    rows_fm.append(
                        {
                            "mark": u,
                            "role": role,
                            "horizon_K": k,
                            "gate_sample": gate_label,
                            "gate_attach": "merge_asof_backward_on_inner_panel",
                            "n": len(vals),
                            "mean_fwd": float(np.mean(vals)),
                            "median_fwd": float(np.median(vals)),
                            "frac_pos": float(np.mean(vals > 0)),
                            "t_vs_zero": float(np.mean(vals) / (np.std(vals, ddof=1) / math.sqrt(len(vals))))
                            if len(vals) > 1 and np.std(vals, ddof=1) > 0
                            else float("nan"),
                        }
                    )
    pd.DataFrame(rows_fm).to_csv(OUT / "r4_p3_forward_by_mark_gated.csv", index=False)

    # Phase-1 bursts (>=4 trades same ts) × gate
    def _symset(s: pd.Series) -> str:
        return ",".join(sorted({str(x) for x in s}))

    burst = (
        m.groupby(["day", "timestamp"], as_index=False)
        .agg(n_trades=("symbol", "count"), symbols=("symbol", _symset))
    )
    burst_big = burst[burst["n_trades"] >= 4].copy()
    burst_big = asof_gate_on_trades(burst_big, gate_panel)
    burst_big["gate_aligned"] = burst_big["gate_asof_matched"]
    ext = px_fwd.loc[px_fwd["symbol"] == EX, ["day", "timestamp", "fwd_mid_20"]].rename(
        columns={"fwd_mid_20": "ext_fwd20"}
    )
    burst_big = burst_big.merge(ext, on=["day", "timestamp"], how="left")
    burst_big.to_csv(OUT / "r4_p3_burst_extract_gated.csv", index=False)
    rng = np.random.default_rng(0)
    b_al = burst_big.loc[burst_big["gate_aligned"]]
    b_on = b_al.loc[b_al["joint_tight"] == True, "ext_fwd20"].dropna()  # noqa: E712
    b_off = b_al.loc[b_al["joint_tight"] == False, "ext_fwd20"].dropna()  # noqa: E712
    mb_on, mb_off, bt_burst, bp_burst = tstat_welch(b_on.values, b_off.values)
    pool_ext_wide = gate_panel.loc[~gate_panel["joint_tight"], "fwd20_ext"].dropna().astype(float).values
    mb_on_p, mb_pool, bt_burst_pool, bp_burst_pool = welch_vs_pool(b_on.values, pool_ext_wide, rng=rng)
    burst_tight_mean = float(np.nanmean(b_on.values)) if len(b_on) else float("nan")

    # Phase-2 style Mark01→Mark22 ≥3 VEV bursts, gate at burst
    m01 = tr[
        (tr["buyer"] == "Mark 01")
        & (tr["seller"] == "Mark 22")
        & (tr["symbol"].str.startswith("VEV_"))
    ]
    burst_keys = (
        m01.groupby(["day", "timestamp"])
        .agg(n_vev=("symbol", "nunique"), n_tr=("symbol", "count"))
        .reset_index()
    )
    burst_keys = burst_keys[(burst_keys["n_vev"] >= 3) & (burst_keys["n_tr"] >= 3)].copy()
    burst_keys = asof_gate_on_trades(burst_keys, gate_panel)
    burst_keys["gate_aligned"] = burst_keys["gate_asof_matched"]

    v5300 = px_fwd.loc[px_fwd["symbol"] == "VEV_5300", ["day", "timestamp", "mid", "spread"]].sort_values(
        ["day", "timestamp"]
    )
    v5300["fwd5"] = v5300.groupby("day", group_keys=False)["mid"].apply(lambda s: s.shift(-5) - s)
    v5300["fwd20"] = v5300.groupby("day", group_keys=False)["mid"].apply(lambda s: s.shift(-20) - s)

    echo_rows = []
    for _, b in burst_keys.iterrows():
        d, ts = int(b["day"]), int(b["timestamp"])
        sub = v5300[(v5300["day"] == d) & (v5300["timestamp"] >= ts - BURST_WIN) & (v5300["timestamp"] <= ts + BURST_WIN)]
        if sub.empty:
            continue
        j = (sub["timestamp"] - ts).abs().idxmin()
        row = sub.loc[j]
        echo_rows.append(
            {
                "day": d,
                "burst_ts": ts,
                "n_vev": int(b["n_vev"]),
                "gate_aligned": bool(b["gate_aligned"]),
                "joint_tight": bool(b["joint_tight"]),
                "fwd5": float(row["fwd5"]) if pd.notna(row["fwd5"]) else np.nan,
                "fwd20": float(row["fwd20"]) if pd.notna(row["fwd20"]) else np.nan,
            }
        )
    echo_df = pd.DataFrame(echo_rows)
    echo_df.to_csv(OUT / "r4_p3_burst_echo_gated.csv", index=False)
    echo_al = echo_df.loc[echo_df["gate_aligned"]]
    e_on = echo_al.loc[echo_al["joint_tight"], "fwd5"].dropna()
    e_off = echo_al.loc[~echo_al["joint_tight"], "fwd5"].dropna()
    me_on, me_off, te_echo, pe_echo = tstat_welch(e_on.values, e_off.values)
    pool_53_wide = (
        gate_panel.loc[~gate_panel["joint_tight"], "fwd5_5300"].dropna().astype(float).values
    )
    me_on_p, me_pool, te_echo_pool, pe_echo_pool = welch_vs_pool(e_on.values, pool_53_wide, rng=rng)
    echo_tight_mean = float(np.nanmean(e_on.values)) if len(e_on) else float("nan")

    # Pair residual: Mark01→Mark22 cells mean |residual| split by gate (trade-level)
    cell = m.groupby(["buyer", "seller", "symbol"], as_index=False).agg(
        cell_mean_fwd20=("fwd_mid_20", "mean"),
    )
    glob = float(m["fwd_mid_20"].mean())
    m2 = m.merge(cell, on=["buyer", "seller", "symbol"], how="left")
    m2["residual_fwd20"] = m2["fwd_mid_20"] - m2["cell_mean_fwd20"].fillna(glob)
    pr = m2[(m2["buyer"] == "Mark 01") & (m2["seller"] == "Mark 22")]
    pr_stats = []
    for label, mask in (
        ("joint_tight_panel", pr["gate_aligned"] & pr["joint_tight"]),
        ("wide_on_panel", pr["gate_aligned"] & ~pr["joint_tight"]),
    ):
        v = pr.loc[mask, "residual_fwd20"].dropna().abs()
        pr_stats.append({"subset": label, "n": len(v), "mean_abs_resid": float(v.mean()) if len(v) else np.nan})
    pd.DataFrame(pr_stats).to_csv(OUT / "r4_p3_pair_residual_gate_split.csv", index=False)

    # Lead-lag: replicate P2 but on tight-only vs wide-only timestamp sets
    tr2 = tr.copy()
    tr2["signed"] = np.where(tr2["buyer"].str.startswith("Mark"), tr2["quantity"], -tr2["quantity"])
    agg = tr2.groupby(["day", "timestamp", "symbol"], as_index=False)["signed"].sum()
    piv = agg.pivot_table(index=["day", "timestamp"], columns="symbol", values="signed", fill_value=0)
    piv = piv.reset_index().sort_values(["day", "timestamp"])
    piv = asof_gate_on_trades(piv, gate_panel)
    piv["gate_aligned"] = piv["gate_asof_matched"]
    piv["jt"] = piv["joint_tight"]

    def lag_corr_for_mask(piv2: pd.DataFrame, mask: pd.Series) -> float:
        sub = piv2.loc[mask, [EX, "VEV_5300"]].dropna()
        if len(sub) < 50:
            return float("nan")
        x = sub[EX].to_numpy(float)
        y = sub["VEV_5300"].to_numpy(float)
        if len(x) < 10:
            return float("nan")
        c = np.corrcoef(x, y)[0, 1]
        return float(c) if np.isfinite(c) else float("nan")

    if EX in piv.columns and "VEV_5300" in piv.columns:
        c_tight = lag_corr_for_mask(piv, piv["gate_aligned"] & piv["jt"])
        c_wide = lag_corr_for_mask(piv, piv["gate_aligned"] & ~piv["jt"])
    else:
        c_tight = c_wide = float("nan")
    pd.DataFrame(
        [{"subset": "joint_tight_ts", "lag0_corr_ex_vev5300": c_tight}, {"subset": "wide_ts", "lag0_corr_ex_vev5300": c_wide}]
    ).to_csv(OUT / "r4_p3_leadlag_gated.csv", index=False)

    # Mark22 passive markout × gate
    px53 = px_fwd.loc[px_fwd["symbol"] == "VEV_5300", ["day", "timestamp", "symbol", "bid1", "ask1", "mid"]].copy()
    tr_m = tr.loc[tr["symbol"] == "VEV_5300"].merge(px53, on=["day", "timestamp", "symbol"], how="inner")
    tr_m["seller_agg"] = tr_m["price"] <= tr_m["bid1"]
    v53f = v5300[["day", "timestamp", "fwd20"]].rename(columns={"fwd20": "fwd20_5300"})
    s22 = tr_m[(tr_m["seller"] == "Mark 22") & (tr_m["seller_agg"])].merge(v53f, on=["day", "timestamp"], how="left")
    s22 = asof_gate_on_trades(s22, gate_panel)
    s22["gate_aligned"] = s22["gate_asof_matched"]
    s22[["day", "timestamp", "gate_aligned", "joint_tight", "fwd20_5300"]].to_csv(
        OUT / "r4_p3_passive_m22_gated.csv", index=False
    )
    s22a = s22.loc[s22["gate_aligned"]]
    m22_t = s22a.loc[s22a["joint_tight"], "fwd20_5300"].dropna()
    m22_w = s22a.loc[~s22a["joint_tight"], "fwd20_5300"].dropna()
    mm22_t, mm22_w, t22, p22 = tstat_welch(m22_t.values, m22_w.values)
    pool_53w_wide = (
        gate_panel.loc[~gate_panel["joint_tight"], "fwd20_5300"].dropna().astype(float).values
    )
    mm22_t_p, mm22_pool, t22_pool, p22_pool = welch_vs_pool(m22_t.values, pool_53w_wide, rng=rng)
    m22_tight_mean = float(np.nanmean(m22_t.values)) if len(m22_t) else float("nan")

    # Top |r| spread pairs for summary (full inner timestamps)
    top_sp = corr_df[(corr_df["sample"] == "all_inner_timestamps") & (corr_df["corr_kind"] == "spread")].copy()
    top_sp["abs_r"] = top_sp["pearson_r"].abs()
    top_sp = top_sp.sort_values("abs_r", ascending=False).head(12)

    lines = [
        "Round 4 Phase 3 — joint gate (R3-style inner join 5200+5300+extract) + stacked tape tests",
        f"Gate panel rows (aligned timestamps): {len(gate_panel):,} | frac joint_tight: {gate_panel['joint_tight'].mean():.4f}",
        f"Extract fwd20 (inner panel): mean tight={m_t:.5f} mean wide={m_w:.5f} Welch t={gt_stat:.3f} p={gt_p:.2e} (n_tight={tight_f.notna().sum()} n_wide={wide_f.notna().sum()})",
        f"Phase-1 big bursts extract_fwd20: pairwise tight vs wide bursts Welch t={bt_burst:.3f} (n_tight={len(b_on)} n_wide={len(b_off)})",
        f"  → tight bursts vs gate-panel wide-book extract_fwd20 pool: mean_tight={burst_tight_mean:.4f} mean_pool={mb_pool:.4f} Welch t={bt_burst_pool:.3f} p={bp_burst_pool:.2e}",
        f"Phase-2 M01→M22 burst VEV5300 fwd5: pairwise Welch t={te_echo:.3f} | vs wide-book fwd5_5300 pool: mean_burst={echo_tight_mean:.4f} mean_pool={me_pool:.4f} t={te_echo_pool:.3f} p={pe_echo_pool:.2e} (n_bursts={len(echo_df)})",
        f"Signed-flow lag-0 corr (timestamp sets): tight_ts={c_tight:.4f} wide_ts={c_wide:.4f}",
        f"Mark22 seller_agg VEV5300 fwd20: pairwise Welch t={t22:.3f} | vs wide-book fwd20_5300 pool: mean_print={m22_tight_mean:.4f} mean_pool={mm22_pool:.4f} t={t22_pool:.3f} p={p22_pool:.2e} (n_prints={len(s22a)})",
        "",
        "Top spread–spread |r| on inner-join timestamps (inclineGod panel):",
    ]
    for _, r in top_sp.iterrows():
        lines.append(f"  {r['x']} vs {r['y']}: r={r['pearson_r']:.4f} n={int(r['n_pair'])}")
    (OUT / "r4_phase3_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Tier-A style ranked edges (evidence-led)
    tier = [
        {
            "rank": 1,
            "edge": "Extract fwd20 tight vs wide on inner-join gate panel",
            "effect": f"Welch_t={gt_stat:.3f} delta_mean={m_t - m_w:.5f}",
            "paths": ["r4_p3_gate_panel.csv", "r4_phase3_summary.txt"],
            "vs_phase2": "Phase2 used pivot_table first spread per ts; Phase3 uses strict inner join like R3 — compare t-stats.",
        },
        {
            "rank": 2,
            "edge": "Spread–spread correlations (VEV legs) on shared timestamps",
            "effect": "See r4_p3_spread_correlations.csv; top rows in summary",
            "paths": ["r4_p3_spread_correlations.csv"],
            "vs_phase2": "Phase2 emphasized mid paths; Phase3 adds inclineGod spread–spread panel.",
        },
        {
            "rank": 3,
            "edge": "Mark01→Mark22 burst echo fwd5 × joint_tight at burst",
            "effect": f"pairwise_Welch_t={te_echo:.3f}; vs_wide5300_fwd5_pool_t={te_echo_pool:.3f}",
            "paths": ["r4_p3_burst_echo_gated.csv"],
            "vs_phase2": "Phase2 burst echo unconditional; Phase3 splits by Sonic gate at burst time.",
        },
        {
            "rank": 4,
            "edge": "Participant forwards × gate_sample (CSV-wide)",
            "effect": "Compare same Mark×role across joint_tight vs not_tight rows",
            "paths": ["r4_p3_forward_by_mark_gated.csv"],
            "vs_phase1": "Phase1 ungated; Phase3 adds gate column alignment.",
        },
        {
            "rank": 5,
            "edge": "Mark22 passive fwd20 × gate",
            "effect": f"pairwise_Welch_t={t22:.3f}; vs_wide5300_fwd20_pool_t={t22_pool:.3f}",
            "paths": ["r4_p3_passive_m22_gated.csv"],
            "vs_phase2": "Phase2 pooled; Phase3 tests Sonic claim gate cleans adverse selection.",
        },
    ]

    gate_json = {
        "round4_phase3_complete": {
            "phase": 3,
            "tape_days": DAYS,
            "gate_convention": "Inner join on (day,timestamp) for VEV_5200 and VEV_5300 price rows, then merge extract — matches round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py aligned_panel logic (per-day timestamps).",
            "trade_gate_attachment": "merge_asof(..., direction='backward') per day: last inner-panel joint_tight state at or before each trade/burst timestamp (prints rarely land on the 5200/5300 inner grid).",
            "sonic_hypothesis_test": "Gate should *clean* counterparty-conditioned signals if execution noise dominates when books are wide; compare Phase1/2 pooled stats vs Phase3 gated splits in summary and CSVs.",
            "comparison_to_phase1_phase2": (
                "Price-panel extract fwd20 tight vs wide: Phase3 inner-join t≈7.94 matches Phase2 magnitude (Phase2 pivot_table on all symbols' first spread per ts). "
                "Counterparty splits: Phase1/2 pooled burst/Mark22 stats mostly occur under tight books on R4; pairwise tight-vs-wide stratification is degenerate (n_wide≈0), "
                "so burst/Mark22 tests use wide-book *pools* from the same inner-join panel for control."
            ),
            "negative_results": [
                "Big multi-trade bursts and Mark01→Mark22 basket bursts are almost always asof-gated tight; pairwise tight vs wide burst cells are empty (cannot confirm Sonic 'interaction' via simple split).",
                "Tight bursts vs wide-book extract_fwd20 pool: Welch t≈0.88 (p≈0.38) — no excess extract drift vs wide-book baseline conditional on burst under tight asof gate.",
                "Mark22 seller_agg prints (n=160) all asof-tight; vs wide-book fwd20_5300 pool t≈-1.24 (p≈0.22) — adverse proxy from Phase2 not strengthened as a tight-only effect here.",
            ],
            "outputs": {k: f"{REL}/{k}" for k in (
                "r4_p3_gate_panel.csv",
                "r4_p3_spread_correlations.csv",
                "r4_p3_forward_by_mark_gated.csv",
                "r4_p3_burst_extract_gated.csv",
                "r4_p3_burst_echo_gated.csv",
                "r4_p3_pair_residual_gate_split.csv",
                "r4_p3_leadlag_gated.csv",
                "r4_p3_passive_m22_gated.csv",
                "r4_phase3_summary.txt",
                "r4_phase3_analysis.py",
            )},
            "metrics_snapshot": {
                "extract_fwd20_welch_t_tight_vs_wide": gt_stat,
                "burst_extract_fwd20_welch_t_tight_vs_wide_pairwise": bt_burst,
                "burst_extract_fwd20_welch_t_vs_gate_wide_pool": bt_burst_pool,
                "burst_echo_fwd5_welch_t_pairwise": te_echo,
                "burst_echo_fwd5_welch_t_vs_gate_wide5300_pool": te_echo_pool,
                "signed_flow_lag0_corr_tight_ts": c_tight,
                "signed_flow_lag0_corr_wide_ts": c_wide,
                "mark22_passive_fwd20_welch_t_pairwise": t22,
                "mark22_passive_fwd20_welch_t_vs_gate_wide5300_pool": t22_pool,
            },
            "ranked_edges": tier,
        }
    }
    def _json_sanitize(o: object) -> object:
        if isinstance(o, dict):
            return {k: _json_sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_json_sanitize(v) for v in o]
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return o

    (OUT / "r4_phase3_gate.json").write_text(
        json.dumps(_json_sanitize(gate_json), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
