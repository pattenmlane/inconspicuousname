#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs"

VEV_SYMS = [
    "VEV_4000","VEV_4500","VEV_5000","VEV_5100","VEV_5200",
    "VEV_5300","VEV_5400","VEV_5500","VEV_6000","VEV_6500",
]
STRIKES = {s: int(s.split("_")[1]) for s in VEV_SYMS}
EXTRACT = "VELVETFRUIT_EXTRACT"
R = 0.0
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    sv = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / sv
    d2 = d1 - sv
    return S * norm_cdf(d1) - K * math.exp(-R * T) * norm_cdf(d2)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sv = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / sv
    return S * math.sqrt(T) * norm_pdf(d1)


def implied_vol(S: float, K: float, T: float, price: float) -> float | None:
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if price < intrinsic - 1e-9:
        return None
    lo, hi = 1e-6, 4.5
    if bs_call(S, K, T, hi) < price:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def wall_mid(bid_p: float, bid_v: float, ask_p: float, ask_v: float) -> float:
    tot = bid_v + ask_v
    if tot <= 0:
        return 0.5 * (bid_p + ask_p)
    return (bid_p * ask_v + ask_p * bid_v) / tot


def t_from_day(day_idx: int) -> float:
    return (8 - day_idx) / 365.25


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def to_float(v: str) -> float | None:
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def analyze_day(day_idx: int) -> list[dict[str, float | int | str]]:
    rows = read_rows(DATA / f"prices_round_3_day_{day_idx}.csv")
    by_ts: dict[int, dict[str, dict[str, str]]] = {}
    for r in rows:
        ts = int(r["timestamp"])
        by_ts.setdefault(ts, {})[r["product"]] = r

    T = t_from_day(day_idx)
    out: list[dict[str, float | int | str]] = []

    for ts, prod in by_ts.items():
        ex = prod.get(EXTRACT)
        if ex is None:
            continue
        ex_bb = to_float(ex.get("bid_price_1", ""))
        ex_bv = to_float(ex.get("bid_volume_1", ""))
        ex_aa = to_float(ex.get("ask_price_1", ""))
        ex_av = to_float(ex.get("ask_volume_1", ""))
        if None in (ex_bb, ex_bv, ex_aa, ex_av):
            continue
        S = wall_mid(float(ex_bb), float(ex_bv), float(ex_aa), float(ex_av))

        xs, ys = [], []
        snap = {}
        for sym in VEV_SYMS:
            r = prod.get(sym)
            if r is None:
                continue
            bb = to_float(r.get("bid_price_1", ""))
            bv = to_float(r.get("bid_volume_1", ""))
            aa = to_float(r.get("ask_price_1", ""))
            av = to_float(r.get("ask_volume_1", ""))
            if None in (bb, bv, aa, av):
                continue
            mid = wall_mid(float(bb), float(bv), float(aa), float(av))
            iv = implied_vol(S, STRIKES[sym], T, mid)
            if iv is None:
                continue
            xs.append(math.log(STRIKES[sym]))
            ys.append(iv)
            snap[sym] = (float(bb), float(aa), mid, iv)

        if len(xs) < 5:
            continue

        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)
        o = np.argsort(x)
        x, y = x[o], y[o]
        if (y.max() - y.min()) > 0.35 and len(y) >= 6:
            med = float(np.median(y))
            drop = int(np.argmax(np.abs(y - med)))
            x = np.delete(x, drop)
            y = np.delete(y, drop)
        if len(x) < 4:
            continue

        cs = CubicSpline(x, y, bc_type="natural")
        for sym, (bb, aa, mid, iv) in snap.items():
            K = STRIKES[sym]
            iv_fit = float(cs(math.log(K)))
            fair = bs_call(S, K, T, iv_fit)
            vega = bs_vega(S, K, T, max(iv_fit, 1e-5))
            out.append({
                "day": day_idx,
                "timestamp": ts,
                "symbol": sym,
                "strike": K,
                "S": S,
                "mid": mid,
                "bid": bb,
                "ask": aa,
                "spread": aa - bb,
                "iv": iv,
                "iv_fit": iv_fit,
                "iv_err": iv - iv_fit,
                "vega": vega,
                "vega_iv_err": vega * (iv - iv_fit),
                "buy_edge": fair - aa,
                "sell_edge": bb - fair,
                "fair": fair,
            })
    return out


def summarize(all_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str | int]]:
    out = []
    for sym in VEV_SYMS:
        rs = [r for r in all_rows if r["symbol"] == sym]
        if not rs:
            continue
        buy_pos = [r["buy_edge"] for r in rs if r["buy_edge"] > 0]
        sell_pos = [r["sell_edge"] for r in rs if r["sell_edge"] > 0]
        vega_abs = [abs(r["vega_iv_err"]) for r in rs]
        out.append({
            "symbol": sym,
            "n": len(rs),
            "mean_spread": statistics.mean(r["spread"] for r in rs),
            "buy_hit_rate": len(buy_pos) / len(rs),
            "sell_hit_rate": len(sell_pos) / len(rs),
            "mean_buy_edge_pos": statistics.mean(buy_pos) if buy_pos else 0.0,
            "mean_sell_edge_pos": statistics.mean(sell_pos) if sell_pos else 0.0,
            "mean_abs_vega_iv_err": statistics.mean(vega_abs) if vega_abs else 0.0,
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in (0, 1, 2):
        rows.extend(analyze_day(d))

    detailed = OUT / "strike_edge_greeks_detailed.csv"
    with detailed.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    srows = summarize(rows)
    summary = OUT / "strike_edge_greeks_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)

    print(f"wrote {detailed}")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
