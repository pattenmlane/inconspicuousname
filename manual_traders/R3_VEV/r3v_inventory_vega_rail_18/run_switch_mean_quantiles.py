#!/usr/bin/env python3
"""
Empirical distribution of Frankfurt `switch_mean` (EMA of |theo_diff - EMA(theo_diff)|) on
ROUND_3 tapes, same windows as traders (20 / 100). TTE as plot_iv_smile_round3: DTE 8,7,6
at csv day 0,1,2; intraday DTE_eff = dte_open - (ts//100)/10000; T = DTE/365, r=0.
Smile: global quadratic coeffs in traders. Uses mid-based wall_mid; subsample per row for speed.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "switch_mean_quantiles.json"
COEFFS = [0.14215151147708086, -0.0016298611395181932, 0.23576325646627055]
W1, W2 = 20, 100
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]


def dte_effective(day: int, ts: int) -> float:
    return max(float(8 - day) - (ts // 100) / 10_000.0, 1e-6)


def t_years(day: int, ts: int) -> float:
    return dte_effective(day, ts) / 365.0


def iv_smile(S: float, K: float, T: float) -> float:
    m = math.log(K / S) / math.sqrt(T)
    return float(np.polyval(np.array(COEFFS), m))


def bs_call(S: float, K: float, T: float, s: float) -> float:
    d1 = (math.log(S / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
    d2 = d1 - s * math.sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def ema(old: float, w: int, x: float) -> float:
    a = 2.0 / (w + 1.0)
    return a * x + (1.0 - a) * old


def main() -> None:
    switches: list[float] = []
    for day in (0, 1, 2):
        path = DATA / f"prices_round_3_day_{day}.csv"
        by_row: dict[tuple[int, int], dict] = {}
        with path.open() as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["timestamp"]) % 500 != 0:
                    continue
                t = int(row["timestamp"])
                p = row["product"]
                by_row[(t, day)] = by_row.get((t, day), {}) | {p: row}
        ema_t: dict[str, float] = {}
        ema_s: dict[str, float] = {}
        for (ts, d) in sorted(by_row):
            m = by_row[(ts, d)]
            u = m.get("VELVETFRUIT_EXTRACT")
            if not u or not u.get("bid_price_1") or not u.get("ask_price_1"):
                continue
            S = 0.5 * (int(u["bid_price_1"]) + int(u["ask_price_1"]))
            T = t_years(d, ts)
            for v in VOUCHERS:
                rr = m.get(v)
                if not rr or not rr.get("bid_price_1") or not rr.get("ask_price_1"):
                    continue
                bb, ba = int(rr["bid_price_1"]), int(rr["ask_price_1"])
                wm = 0.5 * (bb + ba)
                K = int(v.split("_")[1])
                sig = iv_smile(S, K, T)
                if not math.isfinite(sig) or sig <= 0:
                    continue
                theo = bs_call(S, K, T, sig)
                diff = wm - theo
                k1 = f"{v}_td"
                k2 = f"{v}_sw"
                m_new = ema(ema_t.get(k1, 0.0), W1, diff)
                ema_t[k1] = m_new
                s_new = ema(ema_s.get(k2, 0.0), W2, abs(diff - m_new))
                ema_s[k2] = s_new
                switches.append(s_new)

    arr = np.array(switches, dtype=float) if switches else np.array([], dtype=float)
    qs = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]
    out = {
        "n_samples": int(arr.size),
        "quantiles": {str(q): float(np.quantile(arr, q)) for q in qs} if arr.size else {},
        "frac_ge": {
            "0.55": float(np.mean(arr >= 0.55)) if arr.size else 0.0,
            "0.65": float(np.mean(arr >= 0.65)) if arr.size else 0.0,
            "0.7": float(np.mean(arr >= 0.7)) if arr.size else 0.0,
        },
        "method": "subsample every 500 timestamps; EMAs reset per (day) block; same TTE and smile as traders",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
