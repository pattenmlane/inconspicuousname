"""
Round-3: Pearson correlation between Δlog(extract mid) and Δ(IV residual)
for VEV_5400 and VEV_5500. Quadratic IV(m_t) fit from all strikes each step.
T: csv day 0,1,2 -> 8,7,6 d at open; intraday DTE wind-down per plot_iv_smile_round3.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "wing_resid_vs_extract_v7.json"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOU = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"
STEP = 20


def intraday_progress(ts: int) -> float:
    return (int(ts) // 100) / 10_000.0


def t_years(day: int, ts: int) -> float:
    dte = max(8.0 - float(day) - intraday_progress(ts), 1e-6)
    return dte / 365.0


def bs(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def implied_vol(m: float, S: float, K: float, T: float) -> float:
    lo, hi = 1e-4, 12.0
    intr = max(S - K, 0.0)
    if m <= intr + 1e-6 or m >= S - 1e-6 or S <= 0 or T <= 0:
        return float("nan")

    def f(sig: float) -> float:
        return bs(S, K, T, sig) - m

    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    return float(brentq(f, lo, hi))


def fit_quad(row: pd.Series, S: float, T: float) -> tuple[float, float, float] | None:
    srt = math.sqrt(T)
    xs, ys = [], []
    for v in VOU:
        if v not in row.index or pd.isna(row[v]):
            continue
        m = float(row[v])
        K = int(v.split("_")[1])
        ivv = implied_vol(m, S, K, T)
        if not np.isfinite(ivv):
            continue
        m_t = math.log(K / S) / srt
        xs.append(m_t)
        ys.append(ivv)
    if len(xs) < 6:
        return None
    coeff = np.polyfit(np.asarray(xs), np.asarray(ys), 2)
    return float(coeff[0]), float(coeff[1]), float(coeff[2])


def one_resid(
    a: float, b: float, c2: float, S: float, T: float, m: float, K: int
) -> float | None:
    srt = math.sqrt(T)
    ivv = implied_vol(m, S, K, T)
    if not np.isfinite(ivv):
        return None
    m_t = math.log(K / S) / srt
    hat = a * m_t * m_t + b * m_t + c2
    return float(ivv - hat)


def series_for_day(df_pivot: pd.DataFrame, day: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = df_pivot.sort_index()
    sseq, r54, r55 = [], [], []
    for ts in p.index[::STEP]:
        row = p.loc[ts]
        S = float(row[U])
        T = t_years(day, int(ts))
        if S <= 0 or T <= 0:
            continue
        cfit = fit_quad(row, S, T)
        if cfit is None:
            continue
        a, b, c2 = cfit
        if "VEV_5400" not in row.index or "VEV_5500" not in row.index:
            continue
        if pd.isna(row["VEV_5400"]) or pd.isna(row["VEV_5500"]):
            continue
        m54, m55 = float(row["VEV_5400"]), float(row["VEV_5500"])
        x54 = one_resid(a, b, c2, S, T, m54, 5400)
        x55 = one_resid(a, b, c2, S, T, m55, 5500)
        if x54 is None or x55 is None:
            continue
        sseq.append(S)
        r54.append(x54)
        r55.append(x55)
    return np.asarray(sseq, float), np.asarray(r54, float), np.asarray(r55, float)


def corr_diffs(
    sseq: np.ndarray, r1: np.ndarray, r2: np.ndarray
) -> tuple[int, tuple[float, float] | None, tuple[float, float] | None]:
    if len(sseq) < 3:
        return 0, None, None
    dlog = np.diff(np.log(sseq))
    d1, d2 = np.diff(r1), np.diff(r2)
    n = min(len(dlog), len(d1), len(d2))
    if n < 5:
        return n, None, None
    p1 = stats.pearsonr(dlog[:n], d1[:n])
    p2 = stats.pearsonr(dlog[:n], d2[:n])
    return n, (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))


def main() -> None:
    by_day: dict = {}
    dlog_p: list[float] = []
    d54_p: list[float] = []
    d55_p: list[float] = []

    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        p = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        cols = [U] + [c for c in VOU if c in p.columns]
        p = p[cols]
        sseq, r54, r55 = series_for_day(p, day)
        n, p54, p55 = corr_diffs(sseq, r54, r55)
        by_day[str(day)] = {
            "len_series": int(len(sseq)),
            "n_diffs": n,
            "pearson_dlogS_vs_dresid_VEV_5400": p54,
            "pearson_dlogS_vs_dresid_VEV_5500": p55,
        }
        if len(sseq) > 1:
            dlog = np.diff(np.log(sseq))
            d1, d2 = np.diff(r54), np.diff(r55)
            nn = min(len(dlog), len(d1), len(d2))
            dlog_p.extend(dlog[:nn].tolist())
            d54_p.extend(d1[:nn].tolist())
            d55_p.extend(d2[:nn].tolist())

    n2 = min(len(dlog_p), len(d54_p), len(d55_p))
    pool54 = pool55 = None
    if n2 > 50:
        a = np.asarray(dlog_p[:n2])
        b1 = np.asarray(d54_p[:n2])
        b2 = np.asarray(d55_p[:n2])
        s54 = stats.pearsonr(a, b1)
        s55 = stats.pearsonr(a, b2)
        pool54 = (float(s54[0]), float(s54[1]))
        pool55 = (float(s55[0]), float(s55[1]))

    payload = {
        "subsample": STEP,
        "method": "Correlate first differences of log(S) and IV residuals; residuals from same-step quadratic in m_t=log(K/S)/sqrt(T)",
        "by_csv_day": by_day,
        "pooled": {"n": n2, "pearson_5400": pool54, "pearson_5500": pool55},
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
