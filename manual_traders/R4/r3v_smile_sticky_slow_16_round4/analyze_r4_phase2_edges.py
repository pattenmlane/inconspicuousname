"""
Round 4 Phase 2 — orthogonal edges (named-bot bursts, microprice, lead-lag, IV vs Marks, day splits).

Prereq: round4_phase1_complete in analysis.json (Phase 1 script already run).

Outputs under analysis_outputs/:
  r4_phase2_burst_typeB_markout.json
  r4_phase2_microprice_spread_vol.json
  r4_phase2_leadlag_signed_flow.json
  r4_phase2_iv_residual_vs_marks.json
  r4_phase2_day_split_oos.json

Run: python3 manual_traders/R4/r3v_smile_sticky_slow_16_round4/analyze_r4_phase2_edges.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"
VEVS = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
KS = (5, 20)


def days() -> list[int]:
    return sorted(int(p.stem.split("_")[-1]) for p in DATA.glob("prices_round_4_day_*.csv"))


def load_px() -> pd.DataFrame:
    frames = []
    for d in days():
        df = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        df["day"] = d
        bv1 = pd.to_numeric(df.get("bid_volume_1"), errors="coerce").fillna(0)
        av1 = pd.to_numeric(df.get("ask_volume_1"), errors="coerce").fillna(0)
        bp = pd.to_numeric(df.get("bid_price_1"), errors="coerce")
        ap = pd.to_numeric(df.get("ask_price_1"), errors="coerce")
        mp = pd.to_numeric(df.get("mid_price"), errors="coerce")
        df["bb"], df["ba"], df["mid"] = bp, ap, mp
        df["spread"] = (ap - bp).astype(float)
        # microprice at L1
        den = (bv1 + av1).replace(0, np.nan)
        df["micro"] = (bp * av1 + ap * bv1) / den
        df["bv1"], df["av1"] = bv1, av1
        frames.append(df[["day", "timestamp", "product", "mid", "bb", "ba", "spread", "micro", "bv1", "av1"]])
    return pd.concat(frames, ignore_index=True)


def load_tr() -> pd.DataFrame:
    frames = []
    for d in days():
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_idx(px: pd.DataFrame):
    idx = {}
    for (d, sym), g in px.groupby(["day", "product"]):
        g = g.sort_values("timestamp")
        idx[(int(d), str(sym))] = (
            g["timestamp"].astype(int).to_numpy(),
            g["mid"].astype(float).to_numpy(),
            g["bb"].astype(float).to_numpy(),
            g["ba"].astype(float).to_numpy(),
        )
    return idx


def mid_at(idx, d, sym, ts):
    key = (int(d), str(sym))
    if key not in idx:
        return None, None, None
    tss, mids, bb, ba = idx[key]
    p = int(np.searchsorted(tss, ts, side="left"))
    if p >= len(tss) or tss[p] != ts:
        return None, None, None
    return float(mids[p]), float(bb[p]) if math.isfinite(bb[p]) else None, float(ba[p]) if math.isfinite(ba[p]) else None


def mid_fwd(idx, d, sym, ts, k):
    m0, _, _ = mid_at(idx, d, sym, ts)
    if m0 is None:
        return None
    key = (int(d), str(sym))
    tss, mids, _, _ = idx[key]
    p = int(np.searchsorted(tss, ts, side="left"))
    j = p + k
    if j >= len(tss):
        return None
    return float(mids[j]) - m0


def bs_call(S, K, T, sig):
    if T <= 0 or sig <= 1e-12:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def iv_from_mid(m, S, K, T):
    if T <= 0 or S <= 0 or K <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if m <= intrinsic + 1e-6 or m >= S - 1e-6:
        return None

    def f(sig):
        return bs_call(S, K, T, sig) - m

    try:
        if f(1e-5) > 0 or f(12.0) < 0:
            return None
        return float(brentq(f, 1e-5, 12.0, xtol=1e-6, rtol=1e-6))
    except ValueError:
        return None


def dte_eff(day: int, ts: int) -> float:
    # round4: TTE=4d at round open; historical day index maps per description — use 5 - day for open DTE then intraday
    dte_open = 5 - int(day)
    prog = (int(ts) // 100) / 10_000.0
    return max(float(dte_open) - prog, 1e-6)


def main():
    px = load_px()
    tr = load_tr()
    idx = build_idx(px)
    bbo = px.drop_duplicates(subset=["day", "timestamp", "product"], keep="first").set_index(["day", "timestamp", "product"])

    # --- 1) Burst type B: Mark01->Mark22 multi-VEV at same (day,ts)
    burst_b = set()
    for (d, ts), g in tr.groupby(["day", "timestamp"]):
        if len(g) < 2:
            continue
        if not ((g["buyer"] == "Mark 01").all() and (g["seller"] == "Mark 22").all()):
            continue
        syms = set(g["symbol"].astype(str))
        if len(syms & set(VEVS)) >= 2:
            burst_b.add((int(d), int(ts)))

    burst_fwd = {k: [] for k in KS}
    ctrl = {k: [] for k in KS}
    rng = np.random.default_rng(1)
    all_ts = {d: tr.loc[tr["day"] == d, "timestamp"].astype(int).unique() for d in days()}
    for (d, ts) in burst_b:
        for k in KS:
            v = mid_fwd(idx, d, U, ts, k)
            if v is not None:
                burst_fwd[k].append(v)
        for k in KS:
            pool = [t for t in all_ts[d] if (d, int(t)) not in burst_b]
            if len(pool) < 5:
                continue
            for _ in range(2):
                ts2 = int(rng.choice(pool))
                v = mid_fwd(idx, d, U, ts2, k)
                if v is not None:
                    ctrl[k].append(v)

    oos = {}
    for d in days():
        burst_d = {(dd, t) for (dd, t) in burst_b if dd == d}
        fd = {k: [] for k in KS}
        for (dd, ts) in burst_d:
            for k in KS:
                v = mid_fwd(idx, dd, U, ts, k)
                if v is not None:
                    fd[k].append(v)
        oos[str(d)] = {str(k): (float(np.mean(v)), len(v)) if v else (None, 0) for k, v in fd.items()}

    (OUT / "r4_phase2_burst_typeB_markout.json").write_text(
        json.dumps(
            {
                "n_burst_timestamps": len(burst_b),
                "extract_fwd_mean": {str(k): float(np.mean(v)) if v else None for k, v in burst_fwd.items()},
                "control_mean": {str(k): float(np.mean(v)) if v else None for k, v in ctrl.items()},
                "per_day_burst_extract_fwd20_mean": {k: oos[k].get("20") for k in oos},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- 2) Microprice: micro - mid vs next |d_mid| (vol proxy) at extract
    u = px[px["product"] == U].sort_values(["day", "timestamp"])
    u["d_mid"] = u.groupby("day")["mid"].diff().abs()
    u["dsp"] = u.groupby("day")["spread"].diff().abs()
    u["micro_gap"] = (u["micro"] - u["mid"]).abs()
    # correlation micro_gap vs future abs sum of d_mid over next 5 rows
    u["fwd_vol5"] = u.groupby("day")["d_mid"].transform(lambda s: s.shift(-1).rolling(5, min_periods=1).sum().shift(-4))
    u = u.replace([np.inf, -np.inf], np.nan)
    corr = float(u[["micro_gap", "fwd_vol5"]].dropna().corr().iloc[0, 1]) if len(u) > 50 else None
    (OUT / "r4_phase2_microprice_spread_vol.json").write_text(
        json.dumps({"corr_abs_micro_minus_mid_vs_fwd5sum_abs_dmid": corr, "n": int(u[["micro_gap", "fwd_vol5"]].dropna().shape[0])}, indent=2),
        encoding="utf-8",
    )

    # --- 3) Signed flow per (day, ts) on extract: sum qty * sign (aggressor)
    flow_u = []
    for _, r in tr[tr["symbol"] == U].iterrows():
        d, ts = int(r["day"]), int(r["timestamp"])
        m, bb, ba = mid_at(idx, d, U, ts)
        if m is None or bb is None or ba is None:
            continue
        pr = float(r["price"])
        q = int(r["quantity"])
        if pr >= ba:
            flow_u.append({"day": d, "timestamp": ts, "f": q})
        elif pr <= bb:
            flow_u.append({"day": d, "timestamp": ts, "f": -q})
    fu = pd.DataFrame(flow_u)
    if len(fu) > 20:
        fu = fu.groupby(["day", "timestamp"], as_index=False)["f"].sum()
        um = px[px["product"] == U][["day", "timestamp", "mid"]]
        m = fu.merge(um, on=["day", "timestamp"], how="inner").sort_values(["day", "timestamp"])
        m["d_mid"] = m.groupby("day")["mid"].diff()
        m = m.dropna(subset=["d_mid"])
        m = m[np.isfinite(m["f"]) & np.isfinite(m["d_mid"])]
        max_lag = 10
        cors = []
        for lag in range(0, max_lag + 1):
            if lag == 0:
                x = m["f"].to_numpy()
                y = m["d_mid"].to_numpy()
            else:
                x = m["f"].to_numpy()[:-lag]
                y = m["d_mid"].to_numpy()[lag:]
            n = min(len(x), len(y))
            if n < 200:
                continue
            xv, yv = x[:n], y[:n]
            if float(np.std(xv)) < 1e-9 or float(np.std(yv)) < 1e-9:
                cors.append({"lag_ticks": lag, "corr_flow_dmid": None})
                continue
            c = float(np.corrcoef(xv, yv)[0, 1])
            cors.append({"lag_ticks": lag, "corr_flow_dmid": c if math.isfinite(c) else None})
        (OUT / "r4_phase2_leadlag_signed_flow.json").write_text(
            json.dumps({"extract_signed_flow_vs_future_dmid": cors}, indent=2, allow_nan=False).replace("NaN", "null"),
            encoding="utf-8",
        )
    else:
        (OUT / "r4_phase2_leadlag_signed_flow.json").write_text("{}", encoding="utf-8")

    # --- 5) IV residual 5200: IV(mid) - rolling mean IV; mean when Mark01|22 trade vs not
    k520 = "VEV_5200"
    rows_iv = []
    for _, r in px[(px["product"] == k520)].iterrows():
        d, ts = int(r["day"]), int(r["timestamp"])
        Su, _, _ = mid_at(idx, d, U, ts)
        m = float(r["mid"])
        if Su is None:
            continue
        T = dte_eff(d, ts) / 365.0
        iv = iv_from_mid(m, Su, 5200.0, T)
        if iv is None:
            continue
        rows_iv.append({"day": d, "timestamp": ts, "iv": iv})
    ivdf = pd.DataFrame(rows_iv)
    if len(ivdf) > 500:
        ivdf = ivdf.sort_values(["day", "timestamp"])
        ivdf["iv_ma50"] = ivdf.groupby("day")["iv"].transform(lambda s: s.rolling(50, min_periods=20).mean())
        ivdf["res"] = ivdf["iv"] - ivdf["iv_ma50"]
        tr_m = tr[(tr["buyer"] == "Mark 01") & (tr["seller"] == "Mark 22") & (tr["symbol"].astype(str).str.startswith("VEV"))]
        ev_ts = set(zip(tr_m["day"].astype(int), tr_m["timestamp"].astype(int)))
        ivdf["m01_22_vev"] = ivdf.apply(lambda x: 1 if (int(x["day"]), int(x["timestamp"])) in ev_ts else 0, axis=1)
        a = ivdf.loc[ivdf["m01_22_vev"] == 1, "res"].dropna()
        b = ivdf.loc[ivdf["m01_22_vev"] == 0, "res"].dropna()
        tt = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit") if len(a) > 10 and len(b) > 10 else None
        (OUT / "r4_phase2_iv_residual_vs_marks.json").write_text(
            json.dumps(
                {
                    "mean_res_when_m01_m22_vev_print": float(a.mean()) if len(a) else None,
                    "mean_res_otherwise": float(b.mean()) if len(b) else None,
                    "n_a": int(len(a)),
                    "n_b": int(len(b)),
                    "welch_t": float(tt.statistic) if tt else None,
                    "p": float(tt.pvalue) if tt else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        (OUT / "r4_phase2_iv_residual_vs_marks.json").write_text("{}", encoding="utf-8")

    # day split for burst K=20
    (OUT / "r4_phase2_day_split_oos.json").write_text(json.dumps({"burst_extract_fwd20_by_day": oos}, indent=2), encoding="utf-8")

    # --- 3b) Hydro signed flow vs next extract d_mid (cross-instrument)
    fh = []
    for _, r in tr[tr["symbol"] == H].iterrows():
        d, ts = int(r["day"]), int(r["timestamp"])
        m, bb, ba = mid_at(idx, d, H, ts)
        if m is None or bb is None or ba is None:
            continue
        pr = float(r["price"])
        q = int(r["quantity"])
        if pr >= ba:
            fh.append({"day": d, "timestamp": ts, "fh": q})
        elif pr <= bb:
            fh.append({"day": d, "timestamp": ts, "fh": -q})
    hf = pd.DataFrame(fh)
    if len(hf) > 20:
        hf = hf.groupby(["day", "timestamp"], as_index=False)["fh"].sum()
        um = px[px["product"] == U][["day", "timestamp", "mid"]]
        mh = hf.merge(um, on=["day", "timestamp"], how="inner").sort_values(["day", "timestamp"])
        mh["d_mid_u"] = mh.groupby("day")["mid"].diff()
        mh = mh.dropna(subset=["d_mid_u"])
        c0 = float(np.corrcoef(mh["fh"].to_numpy(), mh["d_mid_u"].to_numpy())[0, 1]) if len(mh) > 200 else None
        c1 = (
            float(np.corrcoef(mh["fh"].to_numpy()[:-1], mh["d_mid_u"].to_numpy()[1:])[0, 1]) if len(mh) > 201 else None
        )
        (OUT / "r4_phase2_cross_hydro_extract.json").write_text(
            json.dumps({"corr_hydro_signed_flow_extract_dmid_same_ts": c0, "corr_hydro_flow_vs_extract_dmid_lead1": c1, "n": int(len(mh))}, indent=2),
            encoding="utf-8",
        )

    # --- 7) Cumulative Mark 55 signed flow on extract vs extract mid (inventory proxy)
    f55 = []
    for _, r in tr[(tr["symbol"] == U) & ((tr["buyer"] == "Mark 55") | (tr["seller"] == "Mark 55"))].iterrows():
        d, ts = int(r["day"]), int(r["timestamp"])
        m, bb, ba = mid_at(idx, d, U, ts)
        if m is None or bb is None or ba is None:
            continue
        pr, q = float(r["price"]), int(r["quantity"])
        sgn = 0
        if str(r["buyer"]) == "Mark 55" and pr >= ba:
            sgn = q
        elif str(r["seller"]) == "Mark 55" and pr <= bb:
            sgn = -q
        if sgn != 0:
            f55.append({"day": d, "timestamp": ts, "f55": sgn})
    c55 = pd.DataFrame(f55)
    if len(c55) > 10:
        c55 = c55.groupby(["day", "timestamp"], as_index=False)["f55"].sum().sort_values(["day", "timestamp"])
        um = px[px["product"] == U][["day", "timestamp", "mid"]]
        c55 = c55.merge(um, on=["day", "timestamp"], how="inner")
        c55["cum55"] = c55.groupby("day")["f55"].cumsum()
        c55["mid"] = c55["mid"].astype(float)
        corrs = []
        for d0, g in c55.groupby("day"):
            if len(g) < 10:
                continue
            if float(g["cum55"].std()) < 1e-6 or float(g["mid"].std()) < 1e-6:
                continue
            corrs.append(float(g["cum55"].corr(g["mid"])))
        r_c = float(np.nanmean(corrs)) if corrs else None
        (OUT / "r4_phase2_inventory_proxy_mark55.json").write_text(
            json.dumps({"mean_per_day_corr_cum55_mark_flow_extract_mid": r_c, "n_days_used": len(corrs), "n_rows": int(len(c55))}, indent=2),
            encoding="utf-8",
        )

    print("Wrote", OUT)


if __name__ == "__main__":
    main()
