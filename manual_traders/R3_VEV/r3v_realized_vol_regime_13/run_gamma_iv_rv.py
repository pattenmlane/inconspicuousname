"""
ATM BS gamma (at IV from mid) vs IV−RV of extract (Round 3 tapes, subsample 200).
IV from BS inversion of nearest-strike voucher mid; RV from extract log-return std.
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
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOU = [f"VEV_{k}" for k in STRIKES]


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


def gamma(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12 or S <= 0 or K <= 0:
        return 0.0
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return pdf / (S * v)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        p = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        idx = sorted(p.index)
        for ts in idx[::200]:
            if U not in p.columns:
                continue
            S = float(p.loc[ts, U])
            if S <= 0:
                continue
            T = t_years(day, int(ts))
            k0 = min(STRIKES, key=lambda k: abs(float(k) - S))
            vname = f"VEV_{k0}"
            if vname not in p.columns:
                continue
            mid = float(p.loc[ts, vname])
            iv0 = iv_mkt(mid, S, float(k0), T)
            if not np.isfinite(iv0):
                continue
            gam = gamma(S, float(k0), T, iv0)
            i0 = idx.index(ts)
            seg = idx[max(0, i0 - 30) : i0 + 1]
            rets = []
            for a, b in zip(seg[:-1], seg[1:]):
                sa = float(p.loc[a, U])
                sb = float(p.loc[b, U])
                if sa > 0 and sb > 0:
                    rets.append(math.log(sb / sa))
            rv = (
                float(np.std(rets, ddof=1) * math.sqrt(252.0 * 10_000.0))
                if len(rets) >= 5
                else float("nan")
            )
            rows.append(
                {
                    "day": day,
                    "timestamp": int(ts),
                    "gamma_atm": float(gam),
                    "iv_atm": float(iv0),
                    "rv_extract": float(rv) if np.isfinite(rv) else None,
                    "iv_minus_rv": float(iv0 - rv) if np.isfinite(rv) else None,
                }
            )
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "gamma_atm_vs_iv_minus_rv.csv", index=False)
    z = tbl.dropna(subset=["iv_minus_rv", "gamma_atm"])
    c = float(z["iv_minus_rv"].corr(z["gamma_atm"])) if len(z) > 5 else None
    summ = {
        "n_rows": int(len(tbl)),
        "corr_iv_minus_rv_vs_atm_gamma": c,
        "method": "Gamma from BS with IV=BS-1(mid); nearest strike; same T and RV as prior analyses.",
    }
    (OUT / "gamma_atm_vs_iv_minus_rv_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(summ)


if __name__ == "__main__":
    main()
