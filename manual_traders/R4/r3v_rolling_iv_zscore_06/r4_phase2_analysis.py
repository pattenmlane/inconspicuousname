#!/usr/bin/env python3
"""
Round 4 Phase 2 — orthogonal edges (suggested direction.txt Phase 2 block).

Prereq: Phase 1 outputs exist (same merge conventions as r4_phase1_counterparty.py).

Writes:
  r4_p2_burst_01_22_vev5300_fwd.csv       — burst-conditioned forward V5300 mid
  r4_p2_microprice_vs_mid.txt             — microprice - mid stats by spread bucket
  r4_p2_signed_flow_lags.csv              — cross-instrument lagged corr (day-pooled)
  r4_p2_iv_residual_marks.csv             — IV residual vs Mark01 print density (coarse)
  r4_v23_signals.json                     — sparse long V5300 touch signals for trader_v23
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
T_YEAR = 4.0 / 365.0  # round4 example TTE=4d round4


def N_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, t: float, sig: float) -> float:
    if t <= 0 or sig <= 0:
        return max(S - K, 0.0)
    sr = sig * math.sqrt(t)
    if sr < 1e-12:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sig * sig * t) / sr
    d2 = d1 - sr
    return S * N_cdf(d1) - K * N_cdf(d2)


def implied_vol(S: float, K: float, t: float, px: float) -> float | None:
    intr = max(S - K, 0.0)
    if px <= intr + 1e-9:
        return None
    lo, hi = 1e-4, 4.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        th = bs_call(S, K, t, mid)
        if th > px:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def load_px() -> pd.DataFrame:
    fs = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            fs.append(df)
    return pd.concat(fs, ignore_index=True)


def load_tr() -> pd.DataFrame:
    fs = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            fs.append(df)
    return pd.concat(fs, ignore_index=True)


def merge_trades_tr(tr: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    px2 = px.rename(columns={"product": "symbol"})
    fixed = []
    for d in DAYS:
        for sym in tr["symbol"].unique():
            tsub = tr[(tr["day"] == d) & (tr["symbol"] == sym)].sort_values("timestamp")
            psub = px2[(px2["day"] == d) & (px2["symbol"] == sym)].sort_values("timestamp")
            if tsub.empty or psub.empty:
                continue
            mm = pd.merge_asof(
                tsub,
                psub[["timestamp", "bid_price_1", "ask_price_1", "mid_price"]],
                on="timestamp",
                direction="backward",
            )
            mm["day"] = d
            mm["symbol"] = sym
            fixed.append(mm)
    return pd.concat(fixed, ignore_index=True)


def forward_delta(ts: np.ndarray, mid: np.ndarray, t0: int, k: int) -> float:
    i = int(np.searchsorted(ts, t0, side="left"))
    if i >= len(ts):
        return float("nan")
    if ts[i] != t0:
        i = int(np.searchsorted(ts, t0, side="right") - 1)
    i = max(0, min(i, len(ts) - 1))
    j = min(i + k, len(mid) - 1)
    if j <= i:
        return float("nan")
    return float(mid[j] - mid[i])


def series_pair(px: pd.DataFrame, day: int, sym: str) -> tuple[np.ndarray, np.ndarray]:
    g = px[(px["day"] == day) & (px["product"] == sym)].sort_values("timestamp")
    return g["timestamp"].to_numpy(dtype=np.int64), g["mid_price"].astype(float).to_numpy()


def main() -> None:
    px = load_px()
    tr = load_tr()
    tr["qty"] = tr["quantity"].astype(int)
    mtr = merge_trades_tr(tr, px)

    # --- 1) Burst Mark01->Mark22 on VEV*, forward V5300 ---
    burst_rows = []
    for (d, ts), g in tr.groupby(["day", "timestamp"]):
        if len(g) < 3:
            continue
        if not ((g["buyer"] == "Mark 01") & (g["seller"] == "Mark 22")).any():
            continue
        if not g["symbol"].str.startswith("VEV_").all():
            continue
        ts530, mid530 = series_pair(px, int(d), "VEV_5300")
        burst_rows.append(
            {
                "day": int(d),
                "timestamp": int(ts),
                "n_prints": len(g),
                "vev5300_fwd20": forward_delta(ts530, mid530, int(ts), 20),
            }
        )
    bdf = pd.DataFrame(burst_rows)
    bdf.to_csv(OUT / "r4_p2_burst_01_22_vev5300_fwd.csv", index=False)

    # --- 2) Microprice - mid (extract) by spread bucket ---
    g = px[(px["product"] == "VELVETFRUIT_EXTRACT")].copy()
    rows_mp = []
    for _, r in g.iterrows():
        bp1, ap1 = float(r["bid_price_1"]), float(r["ask_price_1"])
        bv1, av1 = float(r["bid_volume_1"]), abs(float(r["ask_volume_1"]))
        mid = float(r["mid_price"])
        if bv1 + av1 <= 0:
            continue
        micro = (bp1 * av1 + ap1 * bv1) / (bv1 + av1)
        sp = ap1 - bp1
        bucket = "<=2" if sp <= 2 else ("<=6" if sp <= 6 else ">6")
        rows_mp.append({"day": int(r["day"]), "spread_bucket": bucket, "micro_minus_mid": micro - mid})
    mpdf = pd.DataFrame(rows_mp)
    summ = mpdf.groupby("spread_bucket")["micro_minus_mid"].agg(["mean", "std", "count"])
    with open(OUT / "r4_p2_microprice_vs_mid.txt", "w") as f:
        f.write(str(summ))

    # --- 3) Signed flow lag corr: aggregate per timestamp signed qty by symbol, align extract ---
    # signed: buyer side +qty for buy from tape... trade row: if price>=ask aggressive buy +qty else if price<=bid sell -qty else 0
    def sign_row(row) -> int:
        b, a, p = float(row["bid_price_1"]), float(row["ask_price_1"]), float(row["price"])
        q = int(row["qty"])
        if p >= a - 1e-9:
            return q
        if p <= b + 1e-9:
            return -q
        return 0

    mtr["signed"] = mtr.apply(sign_row, axis=1)
    agg = mtr.groupby(["day", "timestamp", "symbol"])["signed"].sum().reset_index()
    piv = agg.pivot_table(index=["day", "timestamp"], columns="symbol", values="signed", fill_value=0)
    piv = piv.reset_index().sort_values(["day", "timestamp"])
    ext = piv["VELVETFRUIT_EXTRACT"] if "VELVETFRUIT_EXTRACT" in piv.columns else pd.Series(0, index=piv.index)
    v53 = piv["VEV_5300"] if "VEV_5300" in piv.columns else pd.Series(0, index=piv.index)
    lags = []
    for L in [0, 1, 2, 3, 5, 10]:
        x = v53.shift(L).fillna(0).to_numpy(dtype=float)
        y = ext.fillna(0).to_numpy(dtype=float)
        if len(x) > 10 and np.std(x) > 0 and np.std(y) > 0:
            c = float(np.corrcoef(x, y)[0, 1])
        else:
            c = float("nan")
        lags.append({"lag_ticks": L, "corr_signed_v53_to_ext": c})
    pd.DataFrame(lags).to_csv(OUT / "r4_p2_signed_flow_lags.csv", index=False)

    # --- 4) IV residual vs Mark01 print rate (per timestamp grid extract) ---
    # For each (day, ts) with full strip, compute median IV across strikes minus IV5300
    px2 = px.rename(columns={"product": "symbol"})
    iv_res = []
    mark01_count = defaultdict(int)
    for _, r in tr[(tr["buyer"] == "Mark 01") | (tr["seller"] == "Mark 01")].iterrows():
        mark01_count[(int(r["day"]), int(r["timestamp"]))] += 1
    for d in DAYS:
        ts_list = sorted(px2[(px2["day"] == d)]["timestamp"].unique())
        for ts in ts_list[::50]:  # subsample for speed
            g2 = px2[(px2["day"] == d) & (px2["timestamp"] == ts)]
            row_u = g2[g2["symbol"] == "VELVETFRUIT_EXTRACT"]
            if row_u.empty:
                continue
            S = float(row_u.iloc[0]["mid_price"])
            ivs = []
            for K in STRIKES:
                sym = f"VEV_{K}"
                rv = g2[g2["symbol"] == sym]
                if rv.empty:
                    continue
                c = float(rv.iloc[0]["mid_price"])
                iv = implied_vol(S, K, T_YEAR, c)
                if iv:
                    ivs.append(iv)
            if len(ivs) < 4:
                continue
            iv530 = implied_vol(S, 5300, T_YEAR, float(g2[g2["symbol"] == "VEV_5300"]["mid_price"].iloc[0]))
            if iv530 is None:
                continue
            med = float(np.median(ivs))
            m1 = mark01_count.get((d, ts), 0)
            iv_res.append({"day": d, "timestamp": ts, "iv5300": iv530, "median_iv": med, "residual": iv530 - med, "mark01_events": m1})
    pd.DataFrame(iv_res).to_csv(OUT / "r4_p2_iv_residual_marks.csv", index=False)

    # --- Triggers for trader_v23: Mark 67 aggressive buy on extract (Phase 1 edge) ---
    W = 50_000  # ~500 ticks @ step 100
    ev = mtr[(mtr["symbol"] == "VELVETFRUIT_EXTRACT") & (mtr["buyer"] == "Mark 67")]
    ev = ev[ev.apply(lambda r: float(r["price"]) >= float(r["ask_price_1"]) - 1e-9, axis=1)]
    # Merged timeline = ResultMerger: each next day offset = last_ts_prev + 100
    day_list = sorted(DAYS)
    max_ts = {d: int(px[px["day"] == d]["timestamp"].max()) for d in day_list}
    cum = {}
    off = 0
    for d in day_list:
        cum[d] = off
        off += max_ts[d] + 100
    triggers = sorted([int(cum[int(r["day"])] + int(r["timestamp"])) for _, r in ev.iterrows()])
    with open(OUT / "r4_v23_signals.json", "w") as f:
        json.dump(
            {
                "mark67_extract_buy_aggr_abs_ts": triggers,
                "day_cum_offset": {str(k): v for k, v in cum.items()},
                "day_max_ts": {str(k): v for k, v in max_ts.items()},
                "window_ts": W,
                "rule": "Phase1_Mark67_aggressive_buy_EXTRACT_merged_timeline",
            },
            f,
        )

    print("phase2 wrote", OUT)


if __name__ == "__main__":
    main()
