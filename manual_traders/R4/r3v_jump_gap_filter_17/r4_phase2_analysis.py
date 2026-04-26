#!/usr/bin/env python3
"""
Round 4 Phase 2 — microstructure, lead–lag, burst-conditioned 01→22 stats,
joint 5200+5300 gate splits, rough IV residual vs Mark prints, signed-flow panel.

Run from repo root:
  python3 manual_traders/R4/r3v_jump_gap_filter_17/r4_phase2_analysis.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "manual_traders/R4/r3v_jump_gap_filter_17"))
from r4_t_years import t_years_effective  # noqa: E402

DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
TICK = 100
MAX_TS = 999900
EX = "VELVETFRUIT_EXTRACT"
HY = "HYDROGEL_PACK"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
BURST_WINDOW_TICKS = 5
R = 0.0


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sig * sig) * T) / v
    d2 = d1 - v
    return S * _ncdf(d1) - K * math.exp(-R * T) * _ncdf(d2)


def iv_solve(price: float, S: float, K: float, T: float) -> float | None:
    if price <= max(S - K, 0.0) + 1e-6 or T <= 0 or S <= 0 or K <= 0:
        return None
    lo, hi = 1e-4, 5.0
    if bs_call(S, K, T, hi) < price:
        return None

    def f(s: float) -> float:
        return bs_call(S, K, T, s) - price

    for _ in range(55):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def microprice_row(r: pd.Series) -> float:
    bb, ba = r["bid_price_1"], r["ask_price_1"]
    if pd.isna(bb) or pd.isna(ba):
        return float("nan")
    bv = float(r["bid_volume_1"]) if pd.notna(r["bid_volume_1"]) else 0.0
    av = float(abs(r["ask_volume_1"])) if pd.notna(r["ask_volume_1"]) else 0.0
    tot = bv + av
    if tot <= 0:
        return 0.5 * (float(bb) + float(ba))
    return (float(bb) * av + float(ba) * bv) / tot


def tob_spread(r: pd.Series) -> float:
    if pd.isna(r["ask_price_1"]) or pd.isna(r["bid_price_1"]):
        return float("nan")
    return float(r["ask_price_1"]) - float(r["bid_price_1"])


def joint_tight(pr_day: pd.DataFrame, ts: int) -> bool:
    def sp(sym: str) -> float | None:
        sub = pr_day[(pr_day["product"] == sym) & (pr_day["timestamp"] == ts)]
        if len(sub) != 1:
            return None
        return tob_spread(sub.iloc[0])

    s52, s53 = sp("VEV_5200"), sp("VEV_5300")
    if s52 is None or s53 is None:
        return False
    return s52 <= 2.0 and s53 <= 2.0


def main() -> None:
    pr = load_prices()
    pr["mid"] = pr["mid_price"].astype(float)
    pr["spread"] = pr.apply(tob_spread, axis=1)
    pr["micro"] = pr.apply(microprice_row, axis=1)
    pr["micro_skew"] = pr["micro"] - pr["mid"]
    d1 = pr["bid_volume_1"].fillna(0).astype(float)
    a1 = pr["ask_volume_1"].fillna(0).astype(float).abs()
    pr["depth1"] = d1 + a1

    keys = list(zip(pr["day"].astype(int), pr["product"].astype(str), pr["timestamp"].astype(int)))
    mid_ix = dict(zip(keys, pr["mid"].astype(float)))

    def fwd_mid(d: int, sym: str, ts: int, k: int) -> float | None:
        t2 = min(ts + k * TICK, MAX_TS)
        a, b = mid_ix.get((d, sym, ts)), mid_ix.get((d, sym, t2))
        if a is None or b is None:
            return None
        return float(b - a)

    # --- Extract panel: microstructure vs forward |Δmid| ---
    rows = []
    for day in DAYS:
        pd_ = pr[pr["day"] == day]
        ex = pd_[pd_["product"] == EX].sort_values("timestamp").reset_index(drop=True)
        mids, ts, spr = ex["mid"].to_numpy(), ex["timestamp"].to_numpy(), ex["spread"].to_numpy()
        msk = ex["micro_skew"].to_numpy()
        dep = ex["depth1"].to_numpy()
        for i in range(len(ex) - 21):
            t0 = int(ts[i])
            d5 = abs(float(mids[i + 5]) - float(mids[i]))
            d20 = abs(float(mids[i + 20]) - float(mids[i]))
            rows.append(
                {
                    "day": day,
                    "timestamp": t0,
                    "spread_ex": float(spr[i]),
                    "micro_skew_ex": float(msk[i]) if np.isfinite(msk[i]) else np.nan,
                    "depth_ex": float(dep[i]),
                    "abs_fwd5": d5,
                    "abs_fwd20": d20,
                    "joint_tight": joint_tight(pd_, t0),
                }
            )
    panel = pd.DataFrame(rows)
    panel["spread_q"] = panel.groupby("day")["spread_ex"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
    )
    panel.groupby(["day", "spread_q"])["abs_fwd5"].mean().reset_index().to_csv(
        OUT / "extract_spread_quartile_vs_fwd_vol.csv", index=False
    )
    panel.groupby(["day", "joint_tight"])["abs_fwd5"].agg(["mean", "count"]).reset_index().to_csv(
        OUT / "joint_gate_vs_extract_short_vol.csv", index=False
    )
    ms = panel.dropna(subset=["micro_skew_ex"]).copy()
    if len(ms) > 50:
        ms["ms_bin"] = ms.groupby("day")["micro_skew_ex"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "hi"], duplicates="drop")
        )
        ms.groupby(["day", "ms_bin"])["abs_fwd5"].mean().reset_index().to_csv(
            OUT / "microprice_skew_vs_fwd_vol_extract.csv", index=False
        )

    # Depth quartile vs vol
    panel["depth_q"] = panel.groupby("day")["depth_ex"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["d1", "d2", "d3", "d4"], duplicates="drop")
    )
    panel.groupby(["day", "depth_q"])["abs_fwd5"].mean().reset_index().to_csv(
        OUT / "extract_depth_quartile_vs_fwd_vol.csv", index=False
    )

    # --- Lead–lag: product Δmid cross-moment vs lagged other ---
    lag_rows = []
    for day in DAYS:
        pd_ = pr[pr["day"] == day]
        ex = pd_[pd_["product"] == EX][["timestamp", "mid"]].sort_values("timestamp")
        tsx = ex["timestamp"].to_numpy()
        dex = ex["mid"].diff().fillna(0.0).to_numpy()
        for vsym in [HY, "VEV_5300", "VEV_5200"]:
            v = pd_[pd_["product"] == vsym][["timestamp", "mid"]].sort_values("timestamp")
            vm = dict(zip(v["timestamp"].astype(int), v["mid"].diff().fillna(0.0).to_numpy()))
            for lag in range(0, 11):
                xs = []
                for i in range(lag, len(tsx)):
                    tlag = int(tsx[i - lag])
                    if tlag in vm:
                        xs.append(float(dex[i]) * float(vm[tlag]))
                if len(xs) > 100:
                    lag_rows.append(
                        {
                            "day": day,
                            "other": vsym,
                            "lag_ticks": lag,
                            "mean_cross": float(np.mean(xs)),
                            "n": len(xs),
                        }
                    )
    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(OUT / "leadlag_signed_extract_x_other.csv", index=False)
    if len(lag_df) > 0:
        idx = lag_df.groupby(["day", "other"])["mean_cross"].apply(lambda s: s.abs().idxmax())
        lag_df.loc[idx.values].to_csv(OUT / "leadlag_best_lag_by_day.csv", index=False)

    # --- Burst ± window around Mark 01→22 timestamps; VEV_5300 fwd5 vs fwd20 ---
    tr = load_trades()
    b01 = tr[(tr["buyer"] == "Mark 01") & (tr["seller"] == "Mark 22")]
    burst_01_22_ts = set(zip(b01["day"].astype(int), b01["timestamp"].astype(int)))

    def near_burst(d: int, ts: int) -> bool:
        for dt in range(-BURST_WINDOW_TICKS, BURST_WINDOW_TICKS + 1):
            t2 = ts + dt * TICK
            if 0 <= t2 <= MAX_TS and (d, t2) in burst_01_22_ts:
                return True
        return False

    mr_rows = []
    for day in DAYS:
        pd_ = pr[pr["day"] == day]
        for _, r in pd_[pd_["product"] == "VEV_5300"].iterrows():
            ts = int(r["timestamp"])
            if not near_burst(day, ts):
                continue
            f5 = fwd_mid(day, "VEV_5300", ts, 5)
            f20 = fwd_mid(day, "VEV_5300", ts, 20)
            if f5 is None or f20 is None:
                continue
            mr_rows.append({"day": day, "timestamp": ts, "fwd5": f5, "fwd20": f20})
    mr_df = pd.DataFrame(mr_rows)
    if len(mr_df) > 0:
        mr_df["mr"] = (mr_df["fwd5"] * mr_df["fwd20"] < 0).astype(int)
        mr_df.groupby("day").agg(mean_fwd5=("fwd5", "mean"), mean_fwd20=("fwd20", "mean"), frac_mr=("mr", "mean"), n=("fwd5", "count")).to_csv(
            OUT / "burst01_22_near_vev5300_mr_summary.csv"
        )

    # --- Phase 1 extract aggr_buy × seller split by Sonic joint gate ---
    tr["price"] = tr["price"].astype(float)
    bb = pr.rename(columns={"product": "symbol"})[
        ["day", "timestamp", "symbol", "ask_price_1", "bid_price_1", "mid"]
    ]
    te = tr.merge(bb, on=["day", "timestamp", "symbol"], how="left")
    te["aggr_buy"] = te["price"] >= te["ask_price_1"]
    sub_ex = te[(te["symbol"] == EX) & te["aggr_buy"] & te["seller"].isin(["Mark 22", "Mark 49"])].copy()
    jt = []
    for d, t in zip(sub_ex["day"].astype(int), sub_ex["timestamp"].astype(int)):
        jt.append(joint_tight(pr[pr["day"] == d], t))
    sub_ex["joint_tight"] = jt
    ev2 = []
    for _, r in sub_ex.iterrows():
        d, ts = int(r["day"]), int(r["timestamp"])
        t5 = min(ts + 5 * TICK, MAX_TS)
        m0, m1 = mid_ix.get((d, EX, ts)), mid_ix.get((d, EX, t5))
        if m0 is None or m1 is None:
            continue
        ev2.append({"day": d, "joint_tight": r["joint_tight"], "seller": str(r["seller"]), "fwd5": m1 - m0})
    if ev2:
        pd.DataFrame(ev2).groupby(["day", "joint_tight", "seller"])["fwd5"].agg(["mean", "count"]).reset_index().to_csv(
            OUT / "phase1_extract_signal_by_joint_gate.csv", index=False
        )

    # --- IV residual at Mark 01→22 VEV prints (rough ATM vs strike IV) ---
    iv_rows = []
    for _, r in b01[b01["symbol"].str.startswith("VEV_")].iterrows():
        day, ts = int(r["day"]), int(r["timestamp"])
        sym = str(r["symbol"])
        try:
            K = int(sym.split("_")[1])
        except ValueError:
            continue
        S = mid_ix.get((day, EX, ts))
        mp = mid_ix.get((day, sym, ts))
        if S is None or mp is None or S <= 0 or mp <= 0:
            continue
        T = float(t_years_effective(day, ts))
        iv_st = iv_solve(float(mp), float(S), float(K), T)
        if iv_st is None:
            continue
        atm_k = min(STRIKES, key=lambda kk: abs(float(kk) - float(S)))
        atm_sym = f"VEV_{atm_k}"
        atm_mid = mid_ix.get((day, atm_sym, ts))
        if atm_mid is None:
            continue
        iv_atm = iv_solve(float(atm_mid), float(S), float(atm_k), T)
        if iv_atm is None:
            continue
        iv_rows.append(
            {
                "day": day,
                "symbol": sym,
                "iv_strike": iv_st,
                "iv_atm": iv_atm,
                "iv_resid": iv_st - iv_atm,
                "joint_tight": joint_tight(pr[pr["day"] == day], ts),
            }
        )
    if iv_rows:
        ivdf = pd.DataFrame(iv_rows)
        ivdf.groupby(["day", "symbol"])["iv_resid"].agg(["mean", "count"]).reset_index().to_csv(
            OUT / "iv_residual_mark01_mark22_by_symbol.csv", index=False
        )

    # --- Signed flow imbalance rolling vs extract fwd (inventory-style) ---
    imb_rows = []
    win = 20
    for day in DAYS:
        pd_ = pr[pr["day"] == day]
        trd = tr[tr["day"] == day]
        ex = pd_[pd_["product"] == EX].sort_values("timestamp").reset_index(drop=True)
        ts_list = ex["timestamp"].astype(int).tolist()
        for i, ts in enumerate(ts_list):
            if i + win >= len(ts_list):
                break
            sub = trd[(trd["timestamp"] >= ts) & (trd["timestamp"] < ts_list[i + win]) & (trd["symbol"] == EX)]
            buy_vol = sub.loc[sub["buyer"] == "Mark 55", "quantity"].sum()
            sell_vol = sub.loc[sub["seller"] == "Mark 55", "quantity"].sum()
            imb = float(buy_vol - sell_vol) if len(sub) else 0.0
            t2 = min(ts + 5 * TICK, MAX_TS)
            m0, m1 = mid_ix.get((day, EX, ts)), mid_ix.get((day, EX, t2))
            if m0 is None or m1 is None:
                continue
            imb_rows.append({"day": day, "imb_mark55": imb, "fwd5_ex": float(m1 - m0)})
    imb_df = pd.DataFrame(imb_rows)
    if len(imb_df) > 20:
        imb_df["imb_q"] = imb_df.groupby("day")["imb_mark55"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "hi"], duplicates="drop")
        )
        imb_df.groupby(["day", "imb_q"])["fwd5_ex"].mean().reset_index().to_csv(
            OUT / "mark55_flow_imbalance_vs_extract_fwd5.csv", index=False
        )

    # --- Execution: summarize phase1 adverse file if present ---
    adv_path = Path(__file__).resolve().parent / "outputs" / "phase1" / "adverse_proxy_k20_by_pair.csv"
    if adv_path.is_file():
        adv = pd.read_csv(adv_path)
        adv.nsmallest(15, "mean_fwd_same_k20").to_csv(OUT / "worst_markouts_k20_tail.csv", index=False)

    # Text summary
    lines = ["Round 4 Phase 2 — key tables written to outputs/phase2/"]
    if len(panel):
        lines.append("Extract |Δmid| over 5 ticks by joint_tight:")
        lines.append(panel.groupby(["day", "joint_tight"])["abs_fwd5"].mean().to_string())
    if len(lag_df):
        lines.append("\nBest |mean_cross| lead–lag rows:")
        lines.append(lag_df.reindex(lag_df.groupby(["day", "other"])["mean_cross"].apply(lambda s: s.abs().idxmax())).to_string())
    (OUT / "phase2_automated_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
