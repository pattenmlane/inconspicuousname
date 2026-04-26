#!/usr/bin/env python3
"""
Round 4 Phase 2 — orthogonal edges (named-bot + burst windows, microstructure, lead–lag,
regime splits, adverse markouts). Reads ROUND_4 days 1–3.

Outputs: manual_traders/R4/r3v_volume_weighted_residual_05/analysis_outputs/phase2/
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)

W_BURST = 0  # same timestamp window only (tape is discrete timestamps)
VEVS = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
K_SHORT = 20


def load_prices() -> pd.DataFrame:
    frames = []
    for d in (1, 2, 3):
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = df["day"].astype(int)
        df["timestamp"] = df["timestamp"].astype(int)
        df["product"] = df["product"].astype(str)
        for c, col in [
            ("mid", "mid_price"),
            ("bid1", "bid_price_1"),
            ("ask1", "ask_price_1"),
            ("bv1", "bid_volume_1"),
            ("av1", "ask_volume_1"),
        ]:
            df[c] = pd.to_numeric(df[col], errors="coerce")
        df["spr"] = df["ask1"] - df["bid1"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_paths(pr: pd.DataFrame) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (day, sym), g in pr.groupby(["day", "product"]):
        g = g.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
        out[(int(day), str(sym))] = (
            g["timestamp"].to_numpy(np.int64),
            g["mid"].to_numpy(np.float64),
            g["spr"].to_numpy(np.float64),
        )
    return out


def fwd_mid(paths: dict, day: int, sym: str, t: int, k: int) -> float:
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


def load_trades() -> pd.DataFrame:
    frames = []
    for d in (1, 2, 3):
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        t = pd.read_csv(p, sep=";")
        t["day"] = d
        frames.append(t)
    tr = pd.concat(frames, ignore_index=True)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = tr["quantity"].astype(int)
    return tr


def microprice_row(bid: float, ask: float, bv: float, av: float) -> float:
    w = bv + av
    if not np.isfinite(w) or w <= 0 or not np.isfinite(bid) or not np.isfinite(ask):
        return float("nan")
    return float((bid * av + ask * bv) / w)


def main() -> None:
    pr = load_prices()
    paths = build_paths(pr)
    bbo = pr.set_index(["day", "timestamp", "product"])[
        ["bid1", "ask1", "bv1", "av1", "spr", "mid"]
    ]

    tr = load_trades()
    tr["buyer"] = tr["buyer"].astype(str)
    tr["seller"] = tr["seller"].astype(str)
    tr["symbol"] = tr["symbol"].astype(str)

    def agg_row(r: pd.Series) -> str:
        key = (int(r["day"]), int(r["timestamp"]), str(r["symbol"]))
        if key not in bbo.index:
            return "unk"
        bid1, ask1 = float(bbo.loc[key, "bid1"]), float(bbo.loc[key, "ask1"])
        px = float(r["price"])
        if px >= ask1:
            return "buy_agg"
        if px <= bid1:
            return "sell_agg"
        return "unk"

    tr["aggressor"] = tr.apply(agg_row, axis=1)

    # --- Burst signature B: Mark01->Mark22 and >=3 distinct VEV symbols at (day,ts) ---
    burst_keys: set[tuple[int, int]] = set()
    for (day, ts), g in tr.groupby(["day", "timestamp"]):
        if len(g) < 3:
            continue
        if not ((g["buyer"] == "Mark 01") & (g["seller"] == "Mark 22")).any():
            continue
        syms = set(g["symbol"])
        vev_syms = {s for s in syms if s.startswith("VEV_")}
        if len(vev_syms) >= 3:
            burst_keys.add((int(day), int(ts)))

    tr["near_m01_m22_burst"] = tr.apply(
        lambda r: int((int(r["day"]), int(r["timestamp"])) in burst_keys), axis=1
    )

    # Forward VEV_5300 mid K=5,20 for each trade row
    for k in (5, 20):
        tr[f"fwd_vev5300_{k}"] = [
            fwd_mid(paths, int(d), "VEV_5300", int(t), k)
            for d, t in zip(tr["day"].to_numpy(), tr["timestamp"].to_numpy())
        ]

    burst_rows = tr[tr["near_m01_m22_burst"] == 1]
    ctrl_rows = tr[tr["near_m01_m22_burst"] == 0]
    rows = []
    for k in (5, 20):
        c = f"fwd_vev5300_{k}"
        for label, g in [("burst_B", burst_rows), ("non_burst", ctrl_rows)]:
            x = pd.to_numeric(g[c], errors="coerce").dropna()
            rows.append(
                {
                    "slice": label,
                    "K": k,
                    "n": len(x),
                    "mean": float(x.mean()) if len(x) else float("nan"),
                    "frac_pos": float((x > 0).mean()) if len(x) else float("nan"),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "r4_phase2_burstB_vev5300_fwd.csv", index=False)

    # Mean-revert vs trend: sign of fwd at K=20 vs sign of fwd at K=5 (same row)
    tr["mr_score"] = np.sign(tr["fwd_vev5300_5"]) * np.sign(tr["fwd_vev5300_20"])
    mr = tr.groupby(["near_m01_m22_burst", "mr_score"]).size().reset_index(name="n")
    mr.to_csv(OUT / "r4_phase2_burstB_meanrevert_vs_trend.csv", index=False)

    # --- Microstructure: U microprice deviation vs future abs mid change ---
    u = pr[pr["product"] == "VELVETFRUIT_EXTRACT"].copy()
    u["micro"] = u.apply(
        lambda r: microprice_row(float(r["bid1"]), float(r["ask1"]), float(r["bv1"]), float(r["av1"])),
        axis=1,
    )
    u["micro_dev"] = u["micro"] - u["mid"]
    u["fwd_abs_mid_20"] = u.apply(
        lambda r: abs(fwd_mid(paths, int(r["day"]), "VELVETFRUIT_EXTRACT", int(r["timestamp"]), 20)),
        axis=1,
    )
    u["fwd_mid_20"] = u.apply(
        lambda r: fwd_mid(paths, int(r["day"]), "VELVETFRUIT_EXTRACT", int(r["timestamp"]), 20),
        axis=1,
    )
    u2 = u.dropna(subset=["micro_dev", "fwd_abs_mid_20"])
    corr = (
        u2["micro_dev"].corr(u2["fwd_abs_mid_20"]),
        u2["spr"].corr(u2["fwd_abs_mid_20"]),
        u2["spr"].corr(np.sign(u2["fwd_mid_20"])),
    )
    (OUT / "r4_phase2_u_microprice_corr.txt").write_text(
        f"corr(micro_dev, abs_fwd_mid_20)={corr[0]:.4f}\n"
        f"corr(U_spread, abs_fwd_mid_20)={corr[1]:.4f}\n"
        f"corr(U_spread, sign(fwd_mid_20))={corr[2]:.4f}\n"
        f"n_rows={len(u2)}\n"
    )

    # Spread compression: negative delta spr over 20 steps vs fwd abs mid
    u["spr_fwd20"] = u.groupby("day")["spr"].shift(-20)
    u["dspr_20"] = u["spr_fwd20"] - u["spr"]
    u3 = u.dropna(subset=["dspr_20", "fwd_abs_mid_20"])
    (OUT / "r4_phase2_u_spread_compression_corr.txt").write_text(
        f"corr(dspr_20, abs_fwd_mid_20)={u3['dspr_20'].corr(u3['fwd_abs_mid_20']):.4f} n={len(u3)}\n"
    )

    # --- Cross-instrument: per-timestamp U aggressive signed flow vs VEV_5300 fwd 20 ---
    tr["signed_qty"] = np.where(
        (tr["symbol"] == "VELVETFRUIT_EXTRACT") & (tr["aggressor"] == "buy_agg"),
        tr["quantity"],
        np.where(
            (tr["symbol"] == "VELVETFRUIT_EXTRACT") & (tr["aggressor"] == "sell_agg"),
            -tr["quantity"],
            0,
        ),
    )
    uf = (
        tr[tr["symbol"] == "VELVETFRUIT_EXTRACT"]
        .groupby(["day", "timestamp"])["signed_qty"]
        .sum()
        .reset_index(name="u_signed_flow")
    )
    lag0, lag1 = [], []
    for day in (1, 2, 3):
        g = uf[uf["day"] == day].copy()
        g["fwd5300_20"] = [fwd_mid(paths, day, "VEV_5300", int(t), 20) for t in g["timestamp"].to_numpy()]
        z = g.dropna(subset=["u_signed_flow", "fwd5300_20"])
        if len(z) > 50:
            lag0.append(
                {
                    "day": day,
                    "n": len(z),
                    "corr_u_flow_fwd5300_20": float(z["u_signed_flow"].corr(z["fwd5300_20"])),
                }
            )
        z2 = g.assign(flow_lag1=g["u_signed_flow"].shift(1)).dropna()
        if len(z2) > 50:
            lag1.append(
                {
                    "day": day,
                    "n": len(z2),
                    "corr_u_flow_lag1_fwd5300_20": float(z2["flow_lag1"].corr(z2["fwd5300_20"])),
                }
            )
    pd.DataFrame(lag0).to_csv(OUT / "r4_phase2_leadlag_u_flow_fwd20_same_ts.csv", index=False)
    pd.DataFrame(lag1).to_csv(OUT / "r4_phase2_leadlag_u_flow_lag1row_fwd20.csv", index=False)

    # --- Regime: tight U spread (<=6) vs wide for Mark67 buy_agg fwd U ---
    m67 = tr[
        (tr["symbol"] == "VELVETFRUIT_EXTRACT")
        & (tr["buyer"] == "Mark 67")
        & (tr["aggressor"] == "buy_agg")
    ].copy()
    spr_u = []
    for _, r in m67.iterrows():
        key = (int(r["day"]), int(r["timestamp"]), "VELVETFRUIT_EXTRACT")
        spr_u.append(float(bbo.loc[key, "spr"]) if key in bbo.index else float("nan"))
    m67["u_spr"] = spr_u
    m67["fwd_u_20"] = [
        fwd_mid(paths, int(d), "VELVETFRUIT_EXTRACT", int(t), 20)
        for d, t in zip(m67["day"].to_numpy(), m67["timestamp"].to_numpy())
    ]
    reg = []
    for label, mask in [
        ("tight_u_spr<=6", m67["u_spr"] <= 6),
        ("wide_u_spr>6", m67["u_spr"] > 6),
    ]:
        x = pd.to_numeric(m67.loc[mask, "fwd_u_20"], errors="coerce").dropna()
        reg.append(
            {
                "slice": label,
                "n": len(x),
                "mean_fwd_u_20": float(x.mean()) if len(x) else float("nan"),
            }
        )
    pd.DataFrame(reg).to_csv(OUT / "r4_phase2_mark67_buyagg_regime_fwd_u20.csv", index=False)

    # --- Adverse: mean fwd_mid_5 for prints where Mark22 is seller on VEV_5300 ---
    m22_sell = tr[(tr["symbol"] == "VEV_5300") & (tr["seller"] == "Mark 22")].copy()
    m22_sell["fwd5"] = [
        fwd_mid(paths, int(d), "VEV_5300", int(t), 5)
        for d, t in zip(m22_sell["day"].to_numpy(), m22_sell["timestamp"].to_numpy())
    ]
    x = pd.to_numeric(m22_sell["fwd5"], errors="coerce").dropna()
    (OUT / "r4_phase2_mark22_seller_vev5300_fwd5_summary.txt").write_text(
        f"n={len(x)} mean_fwd5={float(x.mean()):.4f} frac_pos={float((x>0).mean()):.4f}\n"
    )

    # --- Per-pair adverse on VEV_5300 K=5 for top pairs ---
    top_pairs = [("Mark 01", "Mark 22"), ("Mark 14", "Mark 38"), ("Mark 67", "Mark 49")]
    adv = []
    for b, s in top_pairs:
        g = tr[(tr["symbol"] == "VEV_5300") & (tr["buyer"] == b) & (tr["seller"] == s)].copy()
        g["fwd5"] = [
            fwd_mid(paths, int(d), "VEV_5300", int(t), 5)
            for d, t in zip(g["day"].to_numpy(), g["timestamp"].to_numpy())
        ]
        x = pd.to_numeric(g["fwd5"], errors="coerce").dropna()
        if len(x) == 0:
            continue
        adv.append(
            {
                "buyer": b,
                "seller": s,
                "n": len(x),
                "mean_fwd5": float(x.mean()),
                "frac_pos": float((x > 0).mean()),
            }
        )
    pd.DataFrame(adv).to_csv(OUT / "r4_phase2_pair_adverse_vev5300_fwd5.csv", index=False)

    # --- Lightweight IV residual (subsample): 5200/5300/5400 every 400th U row ---
    sqrtT = math.sqrt(4.0 / 365.0)  # coarse TTE ~4d open; intraday ignored for sketch

    def ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call(S: float, K: float, T: float, sig: float) -> float:
        if T <= 0 or sig <= 1e-12:
            return max(S - K, 0.0)
        v = sig * math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
        d2 = d1 - v
        return S * ncdf(d1) - K * ncdf(d2)

    def iv(mid: float, S: float, K: float, T: float) -> float | None:
        intrinsic = max(S - K, 0.0)
        if mid <= intrinsic + 1e-9 or mid >= S - 1e-9:
            return None
        lo, hi = 1e-4, 5.0
        for _ in range(40):
            m = 0.5 * (lo + hi)
            if bs_call(S, K, T, m) > mid:
                hi = m
            else:
                lo = m
        return 0.5 * (lo + hi)

    iv_rows = []
    u_sub = u.iloc[::400].head(2000)
    for r in u_sub.itertuples(index=False):
        S = float(r.mid)
        if not np.isfinite(S):
            continue
        ts = int(r.timestamp)
        day = int(r.day)
        rec: dict = {"day": day, "timestamp": ts, "S": S}
        for sym, K in [("VEV_5200", 5200.0), ("VEV_5300", 5300.0), ("VEV_5400", 5400.0)]:
            kk = (day, ts, sym)
            if kk not in bbo.index:
                continue
            mid = 0.5 * (float(bbo.loc[kk, "bid1"]) + float(bbo.loc[kk, "ask1"]))
            si = iv(mid, S, K, sqrtT)
            rec[sym + "_iv"] = si
        if len([k for k in rec if k.endswith("_iv") and rec[k] is not None]) >= 3:
            iv_rows.append(rec)
    iv_df = pd.DataFrame(iv_rows)
    if len(iv_df) > 10:
        # quadratic smile residual in log-moneyness
        out_iv = []
        for _, row in iv_df.iterrows():
            S = float(row["S"])
            xs, ys = [], []
            for sym, K in [("VEV_5200", 5200.0), ("VEV_5300", 5300.0), ("VEV_5400", 5400.0)]:
                ivv = row.get(sym + "_iv")
                if ivv is None or not np.isfinite(ivv):
                    continue
                m = math.log(K / S) / sqrtT
                xs.append(m)
                ys.append(float(ivv))
            if len(xs) < 3:
                continue
            X = np.c_[np.array(xs) ** 2, xs, np.ones(len(xs))]
            coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
            for (sym, _K), m, y in zip(
                [("VEV_5200", 5200.0), ("VEV_5300", 5300.0), ("VEV_5400", 5400.0)],
                xs,
                ys,
            ):
                pred = float(coef[0] * m * m + coef[1] * m + coef[2])
                out_iv.append(
                    {
                        "day": int(row["day"]),
                        "timestamp": int(row["timestamp"]),
                        "symbol": sym,
                        "resid_iv": float(y - pred),
                    }
                )
        ivr = pd.DataFrame(out_iv)
        # attach: any Mark01 trade in ±1 price timestamp? use same ts only
        m01_ts = set(
            zip(
                tr.loc[tr["buyer"] == "Mark 01", "day"],
                tr.loc[tr["buyer"] == "Mark 01", "timestamp"],
            )
        )
        ivr["m01_print_same_ts"] = ivr.apply(
            lambda r: int((int(r["day"]), int(r["timestamp"])) in m01_ts), axis=1
        )
        ivr.groupby("m01_print_same_ts")["resid_iv"].agg(["mean", "count"]).reset_index().to_csv(
            OUT / "r4_phase2_iv_residual_vs_m01_same_ts.csv", index=False
        )

    print("Phase2 wrote", OUT)


if __name__ == "__main__":
    main()
