#!/usr/bin/env python3
"""
Round 4 Phase 2 — orthogonal edges (see round4work/ping_followup_phases.md).

Builds on Phase 1 horizons (K tape steps) and price grid. Outputs under
analysis_outputs/r4_phase2_*.csv plus r4_phase2_run_log.txt.

Sections: (1) burst-conditioned + Mark01→22 proximity, (2) microprice/spread
compression, (3) signed-flow cross-lag, (4) regime×pair interactions,
(5) two-strike IV skew vs Mark prints, (6) adverse refinement, (7) rolling
Mark22 sell pressure vs extract drift.
"""
from __future__ import annotations

import bisect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
KS = (5, 20, 100)
NEAR_BURST_W = 500  # ±5 price ticks (100-unit grid)
_N = NormalDist()


def load_px() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            if "day" not in df.columns:
                df["day"] = d
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_tr() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            frames.append(df)
    tr = pd.concat(frames, ignore_index=True)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)
    return tr


def prep_ts_mid(px: pd.DataFrame) -> tuple[dict[int, np.ndarray], dict[tuple[int, int, str], float], dict[tuple[int, int, str], dict]]:
    """Per (day, ts, product): mid, spread, L1 volumes, microprice proxy."""
    px["mid"] = pd.to_numeric(px["mid_price"], errors="coerce")
    px["bb"] = pd.to_numeric(px["bid_price_1"], errors="coerce")
    px["ba"] = pd.to_numeric(px["ask_price_1"], errors="coerce")
    px["bv"] = pd.to_numeric(px["bid_volume_1"], errors="coerce").fillna(0)
    px["av"] = pd.to_numeric(px["ask_volume_1"], errors="coerce").fillna(0)
    px["spread"] = px["ba"] - px["bb"]
    # Stoikov-style microprice (L1 only)
    den = px["bv"] + px["av"]
    px["micro"] = np.where(den > 0, (px["bb"] * px["av"] + px["ba"] * px["bv"]) / den, px["mid"])

    ts_sorted: dict[int, np.ndarray] = {}
    mid_lk: dict[tuple[int, int, str], float] = {}
    row_lk: dict[tuple[int, int, str], dict] = {}
    for d in DAYS:
        sub = px[px["day"] == d]
        ts_sorted[d] = np.sort(sub["timestamp"].unique())
        for _, r in sub.iterrows():
            k = (int(r["day"]), int(r["timestamp"]), str(r["product"]))
            mid_lk[k] = float(r["mid"])
            row_lk[k] = {
                "spread": float(r["spread"]),
                "micro": float(r["micro"]),
                "mid": float(r["mid"]),
                "bb": float(r["bb"]),
                "ba": float(r["ba"]),
            }
    return ts_sorted, mid_lk, row_lk


def fwd_ts(tsu: np.ndarray, t: int, k: int) -> int | None:
    i = bisect.bisect_left(tsu, t)
    if i >= len(tsu) or tsu[i] != t:
        return None
    j = i + k
    if j >= len(tsu):
        return None
    return int(tsu[j])


def cdf(x: float) -> float:
    return _N.cdf(x)


def bs_call(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    st = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / st
    d2 = d1 - st
    return S * cdf(d1) - K * cdf(d2)


def iv(mid: float, S: float, K: float, T: float) -> float | None:
    if mid <= 0 or S <= 0 or K <= 0 or T <= 0 or mid < max(S - K, 0) - 1e-9:
        return None
    lo, hi = 1e-4, 3.0
    for _ in range(50):
        sig = 0.5 * (lo + hi)
        if bs_call(S, K, T, sig) > mid:
            hi = sig
        else:
            lo = sig
    return 0.5 * (lo + hi)


def tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan")
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    if s < 1e-12:
        return float("nan")
    return m / (s / math.sqrt(len(x)))


def main() -> int:
    log: list[str] = []

    def ln(s: str) -> None:
        print(s)
        log.append(s)

    px = load_px()
    tr = load_tr()
    ts_sorted, mid_lk, row_lk = prep_ts_mid(px)

    # --- Burst keys: >=3 rows; Mark01→Mark22 "basket" burst: has row with both ---
    g = tr.groupby(["day", "timestamp"])
    burst_n = g.size().reset_index(name="n")
    burst_n = burst_n[burst_n["n"] >= 3]
    burst_keys = set(zip(burst_n["day"], burst_n["timestamp"]))

    def burst_has_01_22(d: int, ts: int) -> bool:
        sub = tr[(tr["day"] == d) & (tr["timestamp"] == ts)]
        return bool(
            ((sub["buyer"] == "Mark 01") & (sub["seller"] == "Mark 22")).any()
        )

    burst_01_22 = {(d, ts) for d, ts in burst_keys if burst_has_01_22(int(d), int(ts))}

    def near_set(bset: set[tuple[int, int]], w: int) -> set[tuple[int, int]]:
        out = set()
        for d, ts in bset:
            for dt in range(-w, w + 1, 100):
                out.add((d, ts + dt))
        return out

    near_any = near_set(burst_keys, NEAR_BURST_W)
    near_01_22 = near_set(burst_01_22, NEAR_BURST_W)

    # Build markout frame (reuse Phase 1 logic condensed)
    rows = []
    for _, r in tr.iterrows():
        d, ts, sym = int(r["day"]), int(r["timestamp"]), str(r["symbol"])
        tsu = ts_sorted.get(d)
        if tsu is None:
            continue
        m0 = mid_lk.get((d, ts, sym))
        if m0 is None or math.isnan(m0):
            continue
        bb = row_lk.get((d, ts, sym), {}).get("bb")
        ba = row_lk.get((d, ts, sym), {}).get("ba")
        pr = float(r["price"])
        ag = "unk"
        if bb is not None and ba is not None and not (math.isnan(bb) or math.isnan(ba)):
            if pr >= ba:
                ag = "buy_agg"
            elif pr <= bb:
                ag = "sell_agg"
            else:
                ag = "mid"
        sp = row_lk.get((d, ts, sym), {}).get("spread", float("nan"))
        ext0 = mid_lk.get((d, ts, "VELVETFRUIT_EXTRACT"))
        for K in KS:
            fts = fwd_ts(tsu, ts, K)
            if fts is None:
                continue
            m1 = mid_lk.get((d, fts, sym))
            ext1 = mid_lk.get((d, fts, "VELVETFRUIT_EXTRACT"))
            d_mid = (m1 - m0) if m1 is not None else float("nan")
            d_ext = (ext1 - ext0) if ext0 is not None and ext1 is not None else float("nan")
            spr0 = row_lk.get((d, ts, sym), {}).get("spread", float("nan"))
            spr1 = row_lk.get((d, fts, sym), {}).get("spread", float("nan"))
            d_spread = (spr1 - spr0) if spr0 == spr0 and spr1 == spr1 else float("nan")
            mic0 = row_lk.get((d, ts, sym), {}).get("micro", float("nan"))
            mic1 = row_lk.get((d, fts, sym), {}).get("micro", float("nan"))
            d_micro = (mic1 - mic0) if mic0 == mic0 and mic1 == mic1 else float("nan")
            rows.append(
                {
                    "day": d,
                    "timestamp": ts,
                    "symbol": sym,
                    "buyer": str(r["buyer"]),
                    "seller": str(r["seller"]),
                    "qty": int(r["quantity"]),
                    "aggressor": ag,
                    "K": K,
                    "d_mid": d_mid,
                    "d_ext": d_ext,
                    "d_spread": d_spread,
                    "d_micro": d_micro,
                    "burst": (d, ts) in burst_keys,
                    "burst_01_22": (d, ts) in burst_01_22,
                    "near_any_burst": (d, ts) in near_any,
                    "near_01_22_burst": (d, ts) in near_01_22,
                }
            )
    df = pd.DataFrame(rows)
    ln(f"Phase2 markout rows: {len(df):,}")

    # (1) Pair × burst proximity × K
    pburst = (
        df.groupby(["buyer", "seller", "symbol", "K", "near_01_22_burst"])["d_mid"]
        .agg(n="count", mean="mean", t=tstat)
        .reset_index()
    )
    pburst = pburst[pburst["n"] >= 20]
    pburst.to_csv(OUT / "r4_phase2_pair_near_01_22_burst.csv", index=False)
    ln(f"Wrote r4_phase2_pair_near_01_22_burst.csv ({len(pburst)} rows)")

    # Mean-revert vs trend on VEV_5300 after Mark01→22 burst at same ts
    sub530 = df[(df["symbol"] == "VEV_5300") & (df["burst_01_22"])]
    mr = sub530.groupby("K")["d_mid"].agg(n="count", mean="mean", t=tstat).reset_index()
    mr.to_csv(OUT / "r4_phase2_vev5300_same_ts_burst_01_22_followthrough.csv", index=False)
    ln(str(mr))

    # LOSO: Mark 67 extract K=5 buy_agg tight — reuse phase1 definition via merge tertiles
    tert: dict[int, tuple[float, float]] = {}
    for d in DAYS:
        s = px[(px["day"] == d) & (px["product"] == "VELVETFRUIT_EXTRACT")]["spread"].dropna()
        if len(s) > 10:
            tert[d] = (float(s.quantile(0.33)), float(s.quantile(0.66)))

    def ext_bucket(d: int, ts: int) -> str:
        sp = row_lk.get((d, ts, "VELVETFRUIT_EXTRACT"), {}).get("spread", float("nan"))
        if sp != sp:
            return "unk"
        lo, hi = tert.get(d, (0, 999))
        if sp <= lo:
            return "tight"
        if sp >= hi:
            return "wide"
        return "mid"

    loso_eval = []
    for _, r in tr.iterrows():
        d, ts = int(r["day"]), int(r["timestamp"])
        if str(r["buyer"]) != "Mark 67" or str(r["symbol"]) != "VELVETFRUIT_EXTRACT":
            continue
        if ext_bucket(d, ts) != "tight":
            continue
        pr = float(r["price"])
        bb = row_lk.get((d, ts, "VELVETFRUIT_EXTRACT"), {}).get("bb")
        ba = row_lk.get((d, ts, "VELVETFRUIT_EXTRACT"), {}).get("ba")
        if bb is None or ba is None or pr < ba:
            continue
        tsu = ts_sorted[d]
        fts = fwd_ts(tsu, ts, 5)
        if fts is None:
            continue
        m0 = mid_lk.get((d, ts, "VELVETFRUIT_EXTRACT"))
        m1 = mid_lk.get((d, fts, "VELVETFRUIT_EXTRACT"))
        if m0 is None or m1 is None:
            continue
        loso_eval.append({"day": d, "d_mid": float(m1 - m0)})
    if loso_eval:
        pd.DataFrame(loso_eval).groupby("day")["d_mid"].agg(["count", "mean", "std"]).reset_index().to_csv(
            OUT / "r4_phase2_mark67_extract_buyagg_k5_tight_by_day.csv", index=False
        )
        ln("Wrote r4_phase2_mark67_extract_buyagg_k5_tight_by_day.csv")

    # (2) Spread compression → next-K absolute mid change (VEV_5200)
    comp_rows = []
    for d in DAYS:
        tsu = ts_sorted[d]
        for i in range(len(tsu) - 6):
            t0, t1 = int(tsu[i]), int(tsu[i + 5])
            r0 = row_lk.get((d, t0, "VEV_5200"))
            r1 = row_lk.get((d, t1, "VEV_5200"))
            if not r0 or not r1:
                continue
            d_sp = r1["spread"] - r0["spread"]
            m0, m5 = r0["mid"], r1["mid"]
            comp_rows.append({"day": d, "d_spread_5": d_sp, "abs_dm5": abs(m5 - m0)})
    comp_df = pd.DataFrame(comp_rows)
    comp_sum_rows = []
    for d, g in comp_df.groupby("day"):
        comp_sum_rows.append(
            {
                "day": d,
                "corr_dspread_absdm": float(g["d_spread_5"].corr(g["abs_dm5"])),
                "n": len(g),
            }
        )
    comp_sum = pd.DataFrame(comp_sum_rows)
    comp_sum.to_csv(OUT / "r4_phase2_spread_compression_vs_vol_vev5200.csv", index=False)
    ln(f"Wrote r4_phase2_spread_compression_vs_vol_vev5200.csv\n{comp_sum}")

    # (3) Signed flow: Mark 01 buy qty on VEV_5300 per timestamp → lag corr extract returns
    flow = (
        tr[(tr["buyer"] == "Mark 01") & (tr["symbol"] == "VEV_5300")]
        .groupby(["day", "timestamp"])["quantity"]
        .sum()
        .reset_index(name="q_buy")
    )
    ext_ret = []
    for d in DAYS:
        tsu = ts_sorted[d]
        for i in range(len(tsu) - 1):
            t0, t1 = int(tsu[i]), int(tsu[i + 1])
            m0 = mid_lk.get((d, t0, "VELVETFRUIT_EXTRACT"))
            m1 = mid_lk.get((d, t1, "VELVETFRUIT_EXTRACT"))
            if m0 is None or m1 is None:
                continue
            ext_ret.append({"day": d, "timestamp": t0, "ext_ret": float(m1 - m0)})
    er = pd.DataFrame(ext_ret)
    merged = er.merge(flow, on=["day", "timestamp"], how="left").fillna({"q_buy": 0})
    lag_corrs = []
    for lag in (0, 1, 2, 3, 5):
        m = merged.copy()
        m["q_lag"] = m.groupby("day")["q_buy"].shift(lag)
        for d in DAYS:
            sub = m[m["day"] == d]
            if len(sub) > 50:
                lag_corrs.append(
                    {
                        "day": d,
                        "lag_ticks": lag,
                        "corr": float(sub["q_lag"].corr(sub["ext_ret"])),
                        "n": len(sub),
                    }
                )
    pd.DataFrame(lag_corrs).to_csv(OUT / "r4_phase2_mark01_vev5300_flow_lag_vs_extract_ret.csv", index=False)
    ln("Wrote r4_phase2_mark01_vev5300_flow_lag_vs_extract_ret.csv")

    # (5) IV skew sample: join Mark01→22 prints on VEV_5200 with skew at ts
    TTE = 4 / 365.0
    skew_rows = []
    for _, r in tr[(tr["buyer"] == "Mark 01") & (tr["seller"] == "Mark 22") & (tr["symbol"] == "VEV_5200")].iterrows():
        d, ts = int(r["day"]), int(r["timestamp"])
        S = mid_lk.get((d, ts, "VELVETFRUIT_EXTRACT"))
        m52 = mid_lk.get((d, ts, "VEV_5200"))
        m53 = mid_lk.get((d, ts, "VEV_5300"))
        if S is None or m52 is None or m53 is None:
            continue
        iv52, iv53 = iv(m52, S, 5200, TTE), iv(m53, S, 5300, TTE)
        if iv52 and iv53:
            skew_rows.append({"day": d, "timestamp": ts, "skew": iv53 - iv52})
    if skew_rows:
        sk = pd.DataFrame(skew_rows)
        sk.to_csv(OUT / "r4_phase2_iv_skew_at_mark01_22_vev5200_prints.csv", index=False)
        ln(f"Wrote r4_phase2_iv_skew_at_mark01_22_vev5200_prints.csv n={len(sk)}")

    # (7) Rolling Mark22 sell qty (30 ticks window) vs forward extract K=20
    win = 3  # 3*100 = 300 time units = same as NEAR window half
    roll_rows = []
    for d in DAYS:
        tsu = ts_sorted[d]
        sells = (
            tr[(tr["day"] == d) & (tr["seller"] == "Mark 22")]
            .groupby("timestamp")["quantity"]
            .sum()
        )
        for i in range(len(tsu) - 25):
            t0 = int(tsu[i])
            acc = 0
            for j in range(max(0, i - win), i + 1):
                acc += int(sells.get(int(tsu[j]), 0))
            fts = fwd_ts(tsu, t0, 20)
            if fts is None:
                continue
            e0 = mid_lk.get((d, t0, "VELVETFRUIT_EXTRACT"))
            e1 = mid_lk.get((d, fts, "VELVETFRUIT_EXTRACT"))
            if e0 is None or e1 is None:
                continue
            roll_rows.append({"day": d, "m22_sell_qty_roll": acc, "d_ext_20": float(e1 - e0)})
    if roll_rows:
        rr = pd.DataFrame(roll_rows)
        _rrs = []
        for d, g in rr.groupby("day"):
            _rrs.append({"day": d, "corr": float(g["m22_sell_qty_roll"].corr(g["d_ext_20"])), "n": len(g)})
        pd.DataFrame(_rrs).to_csv(OUT / "r4_phase2_mark22_roll_sell_vs_extract_fwd20_by_day.csv", index=False)
        ln("Wrote r4_phase2_mark22_roll_sell_vs_extract_fwd20_by_day.csv")

    (OUT / "r4_phase2_run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'r4_phase2_run_log.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
