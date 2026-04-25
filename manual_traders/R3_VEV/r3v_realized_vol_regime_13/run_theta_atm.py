"""
ATM call theta / carry ratio vs IV−RV (Round 3 tapes, subsample).
Theta: dC/dT in $/year from BS; carry_ratio = max(0, -theta)/S.
Timing: T_years = DTE_eff/365 with DTE_eff = 8-csv_day - (ts//100)/10000.
IV from BS inversion of VEV_5200 wall mid, RV from extract log-return std.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
U = "VELVETFRUIT_EXTRACT"
FOCAL = "VEV_5200"
K0 = 5200


def t_years(day: int, ts: int) -> float:
    dte = max(float(8 - int(day)) - (int(ts) // 100) / 10_000.0, 1e-6)
    return dte / 365.0


def bs(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def iv_mkt(mkt: float, S: float, K: float, T: float) -> float:
    intr = max(S - K, 0.0)
    if mkt <= intr + 1e-9 or mkt >= S - 1e-9 or S <= 0 or T <= 0:
        return float("nan")

    def f(s: float) -> float:
        return bs(S, K, T, s) - mkt

    try:
        if f(1e-5) > 0 or f(15.0) < 0:
            return float("nan")
        return float(brentq(f, 1e-5, 15.0))
    except ValueError:
        return float("nan")


def theta_call(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    v = sig * sqrtT
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return -S * pdf * sig / (2.0 * sqrtT)  # r=0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        p = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        for ts in sorted(p.index)[::200]:
            if U not in p.columns or FOCAL not in p.columns:
                continue
            S = float(p.loc[ts, U])
            m = float(p.loc[ts, FOCAL])
            if S <= 0:
                continue
            T = t_years(day, int(ts))
            iv = iv_mkt(m, S, float(K0), T)
            if not np.isfinite(iv):
                continue
            th = theta_call(S, float(K0), T, iv)
            carry = max(0.0, -th) / S
            idx = list(sorted(p.index))
            i0 = idx.index(ts)
            rets: list[float] = []
            for a, b in zip(idx[max(0, i0 - 30) : i0 + 1], idx[max(0, i0 - 30) + 1 : i0 + 2]):
                sa = float(p.loc[a, U])
                sb = float(p.loc[b, U])
                if sa > 0 and sb > 0:
                    rets.append(math.log(sb / sa))
            rv = float(np.std(rets, ddof=1) * math.sqrt(252.0 * 10_000.0)) if len(rets) >= 3 else float("nan")
            rows.append(
                {
                    "day": day,
                    "timestamp": int(ts),
                    "iv_focal": float(iv),
                    "theta_per_year_focal": float(th),
                    "carry_neg_theta_over_S": float(carry),
                    "rv_extract_ann": float(rv) if np.isfinite(rv) else None,
                    "iv_minus_rv": float(iv - rv) if np.isfinite(rv) else None,
                }
            )
    tbl = pd.DataFrame(rows)
    p_csv = OUT / "theta_atm5200_vs_iv_minus_rv.csv"
    tbl.to_csv(p_csv, index=False)
    z = tbl.dropna(subset=["iv_minus_rv", "carry_neg_theta_over_S"])
    c = float(z["iv_minus_rv"].corr(z["carry_neg_theta_over_S"])) if len(z) > 3 else None
    summ = {
        "focal": FOCAL,
        "K": K0,
        "n_rows": int(len(tbl)),
        "corr_iv_minus_rv_vs_carry": c,
        "method": "r=0 BS call; theta = -S*phi(d1)*sig/(2*sqrt(T)); carry=max(0,-theta)/S. IV from mid vs S,K,T. RV: rolling 30-tick (timestamp steps) log vol of extract, annualized with sqrt(252*10000).",
    }
    (OUT / "theta_atm5200_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(summ)


if __name__ == "__main__":
    main()
