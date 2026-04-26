#!/usr/bin/env python3
"""
Round 4 Phase 2 — orthogonal edges (suggested direction.txt Phase 2).

Uses Phase 1 enriched trades CSV + price tape. Outputs under outputs_r4_phase2/.

Sections:
  1) Named-bot + burst: post Mark01→Mark22 basket burst follow-through (trend vs MR).
  2) Microstructure: microprice vs mid, spread, depth proxy; trade-through proxy from trades.
  3) Cross-instrument: signed flow in coarse time buckets vs lagged extract Δmid.
  4) Regime: Mark 67 aggr_buy extract effect split by spread tertile / session.
  5) IV / smile proxy: Newton IV on VEV mids at trade timestamps for Mark01→22 prints.
  6) Execution: pair-conditioned mean fwd vs Phase 1 top adverse list (sanity).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
P1 = Path(__file__).resolve().parent / "outputs_r4_phase1" / "r4_p1_trades_enriched.csv"
OUT = Path(__file__).resolve().parent / "outputs_r4_phase2"
OUT.mkdir(parents=True, exist_ok=True)
DAYS = [1, 2, 3]


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = int(d)
        frames.append(df)
    pr = pd.concat(frames, ignore_index=True)
    for c in ["bid_price_1", "ask_price_1", "bid_volume_1", "ask_volume_1", "mid_price"]:
        pr[c] = pd.to_numeric(pr[c], errors="coerce")
    pr["spread"] = pr["ask_price_1"] - pr["bid_price_1"]
    pr["microprice"] = np.where(
        (pr["bid_volume_1"] + pr["ask_volume_1"]) > 0,
        (pr["bid_price_1"] * pr["ask_volume_1"] + pr["ask_price_1"] * pr["bid_volume_1"])
        / (pr["bid_volume_1"] + pr["ask_volume_1"]),
        pr["mid_price"],
    )
    pr["micro_minus_mid"] = pr["microprice"] - pr["mid_price"]
    return pr


def build_mid_series(pr: pd.DataFrame) -> dict[tuple[int, str], pd.DataFrame]:
    out = {}
    for (d, sym), g in pr.groupby(["day", "product"]):
        g = g.sort_values("timestamp").groupby("timestamp", as_index=False).last()
        out[(int(d), str(sym))] = g.reset_index(drop=True)
    return out


def idx_at_or_before(ts: np.ndarray, t: int) -> int:
    i = int(np.searchsorted(ts, t, side="right") - 1)
    return max(0, min(i, len(ts) - 1))


def forward_mid_from_series(ts: np.ndarray, mid: np.ndarray, t0: int, k: int) -> float | None:
    i0 = idx_at_or_before(ts, int(t0))
    i1 = min(i0 + int(k), len(ts) - 1)
    if i1 <= i0 and k > 0:
        return None
    return float(mid[i1] - mid[i0])


def bs_call(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def vega(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12:
        return 0.0
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    return S * norm.pdf(d1) * math.sqrt(T)


def iv_newton(mid: float, S: float, K: float, T: float, iters: int = 8) -> float | None:
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-6 or mid >= S - 1e-9 or S <= 0 or K <= 0 or T <= 0:
        return None
    sig = 0.45
    for _ in range(iters):
        pr = bs_call(S, K, T, sig) - mid
        if abs(pr) < 1e-3:
            break
        vg = vega(S, K, T, sig)
        if vg < 1e-8:
            return None
        sig -= pr / vg
        sig = max(min(sig, 6.0), 0.05)
    if abs(bs_call(S, K, T, sig) - mid) > 0.08:
        return None
    return float(sig)


def dte_T(day: int, ts: int) -> float:
    """Round 4: TTE=4d at round start per description example; use 8-day style from R3: DTE_open = 8 - day_index."""
    dte = max(8.0 - float(day) - (int(ts) // 100) / 10_000.0, 1e-6)
    return dte / 365.0


def main() -> None:
    full = pd.read_csv(P1)
    pr = load_prices()
    series = build_mid_series(pr)

    # --- 1) Post M01→M22 basket burst on extract follow-through ---
    burst_ts = (
        full[(full["buyer"] == "Mark 01") & (full["seller"] == "Mark 22")]
        .groupby(["day", "timestamp"])
        .size()
        .reset_index(name="n_legs")
    )
    burst_ts = burst_ts[burst_ts["n_legs"] >= 3]
    rows = []
    for _, r in burst_ts.iterrows():
        d, t = int(r["day"]), int(r["timestamp"])
        u = series.get((d, "VELVETFRUIT_EXTRACT"))
        if u is None or len(u) < 5:
            continue
        ts = u["timestamp"].to_numpy()
        mid = u["mid_price"].to_numpy()
        f5 = forward_mid_from_series(ts, mid, t, 5)
        f20 = forward_mid_from_series(ts, mid, t, 20)
        rows.append({"day": d, "timestamp": t, "fwd_u_5": f5, "fwd_u_20": f20})
    post = pd.DataFrame(rows)
    if len(post):
        post.to_csv(OUT / "r4_p2_post_m01_m22_basket_extract_fwd.csv", index=False)
        ctrl = []
        for d in DAYS:
            u = series.get((d, "VELVETFRUIT_EXTRACT"))
            if u is None:
                continue
            ts = u["timestamp"].to_numpy()
            mid = u["mid_price"].to_numpy()
            for t in ts[:: max(1, len(ts) // 500)]:
                if (d, int(t)) in set(zip(post["day"], post["timestamp"])):
                    continue
                ctrl.append(
                    {
                        "day": d,
                        "timestamp": int(t),
                        "fwd_u_5": forward_mid_from_series(ts, mid, int(t), 5),
                        "fwd_u_20": forward_mid_from_series(ts, mid, int(t), 20),
                    }
                )
        cdf = pd.DataFrame(ctrl)
        lines = [
            "Post Mark01→Mark22 multi-leg (>=3 rows same ts) vs sparse time controls on extract:",
            f"  n_events={len(post)}",
            f"  mean fwd5={post['fwd_u_5'].mean():.4f} median={post['fwd_u_5'].median():.4f}",
            f"  mean fwd20={post['fwd_u_20'].mean():.4f} median={post['fwd_u_20'].median():.4f}",
        ]
        if len(cdf) > 10:
            lines.append(f"  control n={len(cdf)} mean fwd5={cdf['fwd_u_5'].mean():.4f} mean fwd20={cdf['fwd_u_20'].mean():.4f}")
            t5, p5 = stats.ttest_ind(post["fwd_u_5"].dropna(), cdf["fwd_u_5"].dropna(), equal_var=False)
            lines.append(f"  Welch fwd5: t={t5:.3f} p={p5:.4g}")
        (OUT / "r4_p2_post_basket_vs_control_extract.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- 2) Microprice vs mid at trade times (extract + VEV_5300) ---
    mic_rows = []
    for sym in ["VELVETFRUIT_EXTRACT", "VEV_5300"]:
        sub = pr[pr["product"] == sym].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["day", "timestamp"])
        sub["dmid"] = sub.groupby("day")["mid_price"].diff()
        sub["lag_gap"] = sub.groupby("day")["micro_minus_mid"].shift(1)
        pair = sub[["lag_gap", "dmid"]].dropna()
        c = float(pair.corr().iloc[0, 1]) if len(pair) > 50 else float("nan")
        mic_rows.append(
            {
                "symbol": sym,
                "mean_abs_micro_minus_mid": float(sub["micro_minus_mid"].abs().mean()),
                "mean_spread": float(sub["spread"].mean()),
                "corr_lag_microgap_next_dmid": c,
            }
        )
    pd.DataFrame(mic_rows).to_csv(OUT / "r4_p2_microprice_summary.csv", index=False)

    # --- 3) Signed flow buckets vs lag extract ---
    full["signed_qty"] = np.where(
        full["aggressor_bucket"] == "aggr_buy",
        full["quantity"],
        np.where(full["aggressor_bucket"] == "aggr_sell", -full["quantity"], 0),
    )
    full["bucket"] = (full["timestamp"] // 1000).astype(int)  # coarse ~10s blocks
    lag_corrs = []
    for d in DAYS:
        u = series.get((d, "VELVETFRUIT_EXTRACT"))
        if u is None:
            continue
        u = u.copy()
        u["dmid"] = u["mid_price"].diff()
        # aggregate signed flow per bucket for VEV_5300
        f = full[(full["day"] == d) & (full["symbol"] == "VEV_5300")]
        if f.empty:
            continue
        g = f.groupby("bucket")["signed_qty"].sum().reset_index()
        g = g.rename(columns={"signed_qty": "flow5300"})
        m = u.assign(bucket=(u["timestamp"] // 1000).astype(int)).groupby("bucket")["dmid"].sum().reset_index()
        h = g.merge(m, on="bucket", how="inner")
        if len(h) > 20:
            lag_corrs.append(
                {
                    "day": d,
                    "corr_flow_dmid_same_bucket": float(h["flow5300"].corr(h["dmid"])),
                    "n_buckets": len(h),
                }
            )
    pd.DataFrame(lag_corrs).to_csv(OUT / "r4_p2_signed_flow_vev5300_vs_extract_dmid_bucket.csv", index=False)

    # --- 4) Mark 67 aggr_buy extract × spread regime ---
    m67 = full[
        (full["symbol"] == "VELVETFRUIT_EXTRACT")
        & (full["buyer"] == "Mark 67")
        & (full["aggressor_bucket"] == "aggr_buy")
    ].copy()
    if len(m67) > 30:
        m67["spr_q"] = pd.qcut(
            m67["spread"].rank(method="first"), 3, labels=["tight", "mid", "wide"], duplicates="drop"
        )
        summ = (
            m67.groupby(["day", "spr_q"])["fwd_mid_k20"]
            .agg(n="count", mean="mean")
            .reset_index()
        )
        summ.to_csv(OUT / "r4_p2_mark67_extract_fwd20_by_spread_tertile.csv", index=False)

    # --- 5) Smile width at Mark01→22 timestamps (VEV mids) ---
    strikes = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
    wing_rows = []
    m122 = full[(full["buyer"] == "Mark 01") & (full["seller"] == "Mark 22")].copy()
    def _row_at(d0: int, sym0: str, t0: int) -> pd.Series | None:
        g0 = pr[(pr["day"] == d0) & (pr["product"] == sym0) & (pr["timestamp"] <= t0)]
        if g0.empty:
            return None
        return g0.sort_values("timestamp").iloc[-1]

    for (d, t), g in m122.groupby(["day", "timestamp"]):
        if len(g) < 3:
            continue
        ivs = []
        S = None
        for sym in [f"VEV_{k}" for k in strikes]:
            rv = _row_at(int(d), sym, int(t))
            if rv is None:
                continue
            mid = float(rv["mid_price"])
            K = int(sym.split("_")[1])
            ur = _row_at(int(d), "VELVETFRUIT_EXTRACT", int(t))
            if ur is None:
                continue
            S = float(ur["mid_price"])
            T = dte_T(int(d), int(t))
            iv = iv_newton(mid, S, K, T)
            if iv is not None:
                ivs.append(iv)
        if len(ivs) >= 5 and S is not None:
            wing_rows.append(
                {
                    "day": int(d),
                    "timestamp": int(t),
                    "iv_range": max(ivs) - min(ivs),
                    "iv_mean": float(np.mean(ivs)),
                }
            )
    if wing_rows:
        pd.DataFrame(wing_rows).to_csv(OUT / "r4_p2_smile_width_at_m01_m22_bursts.csv", index=False)

    manifest = {"outputs": sorted(str(p.name) for p in OUT.glob("*"))}
    (OUT / "r4_p2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
