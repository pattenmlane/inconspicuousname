#!/usr/bin/env python3
"""Extended extract->voucher propagation: up vs down beta, IV change proxy, spread change on shocks."""
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
OUT = REPO / "manual_traders/R3_VEV/r3v_inventory_vega_rail_18/analysis_outputs/extract_propagation_extended.json"
COEFF = [0.14215151147708086, -0.0016298611395181932, 0.23576325646627055]
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEVS = [f"VEV_{k}" for k in STRIKES]


def t_years(day: int, ts: int) -> float:
    return max(float(8 - day) - (ts // 100) / 10_000.0, 1e-6) / 365.0


def iv_poly(S: float, K: float, T: float) -> float:
    m = math.log(K / S) / math.sqrt(T)
    return float(np.polyval(np.array(COEFF), m))


def bs_vega(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * math.sqrt(T))
    return float(S * float(norm.pdf(d1)) * math.sqrt(T))


def ols_beta(du: np.ndarray, y: np.ndarray) -> float | None:
    if du.size == 0 or y.size == 0:
        return None
    den = float(np.dot(du, du))
    if den < 1e-12:
        return None
    return float(np.dot(du, y) / den)


def main() -> None:
    series: dict[str, list[tuple[float, float, float | None, float, int]]] = {
        v: [] for v in VEVS
    }
    du_for_q: list[float] = []

    for day in (0, 1, 2):
        by_ts: dict[int, dict] = defaultdict(dict)
        with (DATA / f"prices_round_3_day_{day}.csv").open() as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                by_ts[int(row["timestamp"])][row["product"]] = row

        prev_u: float | None = None
        prev_mid: dict[str, float] = {}
        prev_spr: dict[str, int] = {}

        for ts in sorted(by_ts):
            d = by_ts[ts]
            u = d.get("VELVETFRUIT_EXTRACT")
            if not u or not u.get("bid_price_1") or not u.get("ask_price_1"):
                continue
            S = 0.5 * (int(u["bid_price_1"]) + int(u["ask_price_1"]))
            T = t_years(day, ts)
            du = 0.0 if prev_u is None else float(S - prev_u)
            prev_u = float(S)
            if du != 0.0:
                du_for_q.append(abs(du))

            for v in VEVS:
                K = int(v.split("_")[1])
                rr = d.get(v)
                if not rr or not rr.get("bid_price_1") or not rr.get("ask_price_1"):
                    continue
                mid = 0.5 * (int(rr["bid_price_1"]) + int(rr["ask_price_1"]))
                spr = int(rr["ask_price_1"]) - int(rr["bid_price_1"])
                pm = prev_mid.get(v)
                ps = prev_spr.get(v)
                dmid = None if pm is None else float(mid - pm)
                d_iv = None
                if pm is not None and prev_u is not None and T > 0:
                    sig0 = iv_poly(float(S - du), K, T)
                    sig1 = iv_poly(S, K, T)
                    if math.isfinite(sig0) and math.isfinite(sig1):
                        d_iv = float(sig1 - sig0)
                vg = bs_vega(S, K, T, iv_poly(S, K, T)) if T > 0 else 0.0
                dspr = None if ps is None else int(spr - ps)
                if dmid is not None:
                    series[v].append((du, dmid, d_iv, vg, dspr if dspr is not None else 0))
                prev_mid[v] = mid
                prev_spr[v] = spr

    q90 = float(np.quantile(du_for_q, 0.9)) if du_for_q else 0.0

    contract: dict = {}
    for v in VEVS:
        arr = series[v]
        if not arr:
            continue
        du = np.array([a[0] for a in arr], dtype=float)
        dm = np.array([a[1] for a in arr], dtype=float)
        div = [a[2] for a in arr if a[2] is not None]
        du2 = np.array([a[0] for a in arr if a[2] is not None], dtype=float)
        diva = np.array([float(x) for x in div], dtype=float)
        up = du > 0
        dn = du < 0
        d_spr = np.array([a[4] for a in arr], dtype=float)

        shock = np.abs(du) >= q90
        contract[v] = {
            "n": len(arr),
            "beta_all": ols_beta(du, dm),
            "beta_up": ols_beta(du[up], dm[up]) if up.any() else None,
            "beta_down": ols_beta(du[dn], dm[dn]) if dn.any() else None,
            "d_iv_regress": ols_beta(du2, diva) if diva.size else None,
            "mean_d_spr_shock": float(np.mean(d_spr[shock])) if shock.any() else None,
            "mean_d_spr_non": float(np.mean(d_spr[~shock])) if (~shock).any() else None,
        }

    out = {
        "method": "Per tick: du_extract, dmid per voucher, d(IV) from same smile poly, BS vega at tick; TTE from round3; shock = |du| >= 90p over non-zero |du|.",
        "q90_abs_du": q90,
        "contract": contract,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
