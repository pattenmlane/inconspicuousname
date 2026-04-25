#!/usr/bin/env python3
"""Regime diagnostics for family-13 pivot: calm/neutral/stressed from RV-IV z-score.

Outputs:
- occupancy by regime
- transition matrix
- approximate tape PnL proxy by regime via one-step mean-reversion sign on near-ATM voucher
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "rv_iv_regime_stats.json"
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(s: float, k: float, t: float, sig: float) -> float:
    if t <= 1e-12 or sig <= 1e-12:
        return max(s - k, 0.0)
    v = sig * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sig * sig * t) / v
    d2 = d1 - v
    return s * norm_cdf(d1) - k * norm_cdf(d2)


def iv_bisect(px: float, s: float, k: float, t: float) -> float | None:
    if px <= max(s - k, 0.0) + 1e-6 or px >= s - 1e-6:
        return None
    lo, hi = 1e-4, 12.0
    if bs_call(s, k, t, lo) - px > 0 or bs_call(s, k, t, hi) - px < 0:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if bs_call(s, k, t, mid) >= px:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def dte_eff(day: int, ts: int) -> float:
    return max(8.0 - float(day) - (int(ts) // 100) / 10_000.0, 1e-6)


def regime_from_z(z: float, calm_thr: float = -0.8, stress_thr: float = 0.8) -> str:
    if z <= calm_thr:
        return "calm"
    if z >= stress_thr:
        return "stressed"
    return "neutral"


def main() -> None:
    rows: list[dict] = []
    for day in (0, 1, 2):
        df = pd.read_csv(
            REPO / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{day}.csv",
            sep=";",
        )
        pvt = df.pivot_table(
            index="timestamp", columns="product", values="mid_price", aggfunc="first"
        )
        ts_sorted = sorted(pvt.index.to_list())
        s_hist: list[float] = []
        for ts in ts_sorted:
            if "VELVETFRUIT_EXTRACT" not in pvt.columns:
                continue
            s = float(pvt.at[ts, "VELVETFRUIT_EXTRACT"])
            s_hist.append(s)
            if len(s_hist) < 35:
                continue
            # Realized vol proxy from log returns over recent window.
            rets = np.diff(np.log(np.array(s_hist[-35:], dtype=float)))
            rv_ann = float(np.std(rets) * math.sqrt(10_000.0))
            t = dte_eff(day, int(ts)) / 365.0

            # Near-ATM IV proxy: median IV of 3 closest strikes by |K-S|.
            ivs: list[tuple[float, float]] = []
            for k in STRIKES:
                sym = f"VEV_{k}"
                if sym not in pvt.columns or pd.isna(pvt.at[ts, sym]):
                    continue
                iv = iv_bisect(float(pvt.at[ts, sym]), s, float(k), t)
                if iv is None or iv != iv:
                    continue
                ivs.append((abs(float(k) - s), iv))
            if len(ivs) < 2:
                continue
            ivs.sort(key=lambda x: x[0])
            iv_atm = float(np.median([v for _, v in ivs[:3]]))
            rv_iv = rv_ann - iv_atm
            rows.append(
                {
                    "day": day,
                    "timestamp": int(ts),
                    "S": s,
                    "rv_ann": rv_ann,
                    "iv_atm": iv_atm,
                    "rv_iv": rv_iv,
                }
            )

    if not rows:
        raise RuntimeError("no rows produced for regime analysis")
    reg = pd.DataFrame(rows).sort_values(["day", "timestamp"]).reset_index(drop=True)
    reg["z"] = (reg["rv_iv"] - reg["rv_iv"].rolling(600, min_periods=120).mean()) / reg[
        "rv_iv"
    ].rolling(600, min_periods=120).std()
    reg = reg.replace([np.inf, -np.inf], np.nan).dropna(subset=["z"])
    reg["regime"] = reg["z"].apply(regime_from_z)

    # Occupancy and transitions.
    occ = reg["regime"].value_counts(normalize=True).to_dict()
    trans_counts = {
        r: {c: 0 for c in ("calm", "neutral", "stressed")}
        for r in ("calm", "neutral", "stressed")
    }
    prev = None
    for r in reg["regime"].to_list():
        if prev is not None:
            trans_counts[prev][r] += 1
        prev = r
    trans_prob: dict[str, dict[str, float]] = {}
    for r, d in trans_counts.items():
        s = sum(d.values())
        trans_prob[r] = {k: (v / s if s else 0.0) for k, v in d.items()}

    # Simple PnL proxy by regime:
    # sign(mean-reversion on nearest-strike option) * next-step option return.
    pnl_by_regime = {"calm": 0.0, "neutral": 0.0, "stressed": 0.0}
    cnt_by_regime = {"calm": 0, "neutral": 0, "stressed": 0}
    by_key = {(int(r.day), int(r.timestamp)): r for r in reg.itertuples()}
    for r in reg.itertuples():
        day, ts = int(r.day), int(r.timestamp)
        nxt = (day, ts + 100)
        if nxt not in by_key:
            continue
        # nearest listed strike to S.
        k = min(STRIKES, key=lambda kk: abs(float(kk) - float(r.S)))
        sym = f"VEV_{k}"
        # recover mids from source for current+next
        # (cheap reload not needed; use rv/iv proxy direction only)
        # proxy alpha: if rv_iv z high -> short option, low -> long option
        side = -1.0 if r.z > 0 else 1.0
        ret = float(by_key[nxt].iv_atm - r.iv_atm)
        pnl_by_regime[r.regime] += side * ret
        cnt_by_regime[r.regime] += 1

    out = {
        "method": "Regime from zscore(rv_ann - iv_atm). rv_ann from rolling log-return std on extract, annualized by sqrt(10000). iv_atm from median IV of 3 strikes closest to spot. Calm z<=-0.8, stressed z>=0.8, else neutral.",
        "n_rows": int(len(reg)),
        "occupancy": occ,
        "transition_prob": trans_prob,
        "pnl_proxy_by_regime": {
            k: {
                "sum": float(v),
                "n": int(cnt_by_regime[k]),
                "avg": float(v / cnt_by_regime[k]) if cnt_by_regime[k] else 0.0,
            }
            for k, v in pnl_by_regime.items()
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
