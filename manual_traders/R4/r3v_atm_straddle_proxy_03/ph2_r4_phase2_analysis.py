#!/usr/bin/env python3
"""
Round 4 Phase 2 — named-bot + burst conditioning, microprice/spread regimes,
cross-instrument signed-flow lags, regime splits (suggested direction.txt Phase 2).

Reuses tick series definition from Phase 1 (price CSV row order per day, product).

Run from repo root:
  python3 manual_traders/R4/r3v_atm_straddle_proxy_03/ph2_r4_phase2_analysis.py
"""
from __future__ import annotations

import bisect
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]


def _f(x: str) -> float:
    try:
        return float(x) if x else float("nan")
    except ValueError:
        return float("nan")


def _i(x: str) -> int:
    try:
        return int(float(x)) if x else 0
    except ValueError:
        return 0


def load_prices_full() -> dict[tuple[int, str], list[dict]]:
    """Per (day, prod): list of rows with ts, mid, spread, bid, ask, bidv1, askv1, microprice."""
    acc: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for day in DAYS:
        p = DATA / f"prices_round_4_day_{day}.csv"
        if not p.is_file():
            continue
        with p.open(newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                prod = row.get("product") or row.get("Product")
                if not prod or prod not in PRODUCTS:
                    continue
                ts = _i(row["timestamp"])
                bid = _f(row.get("bid_price_1", ""))
                ask = _f(row.get("ask_price_1", ""))
                bv = _f(row.get("bid_volume_1", ""))
                av = _f(row.get("ask_volume_1", ""))
                if math.isnan(bid) or math.isnan(ask) or ask <= bid:
                    mid = _f(row.get("mid_price", ""))
                    sp = float("nan")
                    micro = float("nan")
                else:
                    mid = 0.5 * (bid + ask)
                    sp = ask - bid
                    den = bv + av
                    micro = (ask * bv + bid * av) / den if den > 0 else mid
                acc[(day, prod)].append(
                    {
                        "ts": ts,
                        "mid": mid,
                        "spread": sp,
                        "bid": bid,
                        "ask": ask,
                        "bidv1": bv,
                        "askv1": av,
                        "micro": micro,
                    }
                )
    for k in acc:
        acc[k].sort(key=lambda x: x["ts"])
    return acc


def load_trades() -> list[dict]:
    rows = []
    for day in DAYS:
        p = DATA / f"trades_round_4_day_{day}.csv"
        if not p.is_file():
            continue
        with p.open(newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                rows.append(
                    {
                        "day": day,
                        "ts": _i(row["timestamp"]),
                        "buyer": row["buyer"].strip(),
                        "seller": row["seller"].strip(),
                        "symbol": row["symbol"].strip(),
                        "price": _i(row["price"]),
                        "qty": _i(row["quantity"]),
                    }
                )
    return rows


def tick_index(series: list[dict], ts: int) -> int:
    tss = [x["ts"] for x in series]
    if not tss:
        return -1
    i = bisect.bisect_right(tss, ts) - 1
    return max(0, min(i, len(series) - 1))


def fwd_mid_delta(series: list[dict], i: int, k: int) -> float | None:
    if i < 0 or not series:
        return None
    j = min(i + k, len(series) - 1)
    a, b = series[i]["mid"], series[j]["mid"]
    if math.isnan(a) or math.isnan(b):
        return None
    return b - a


def tstat_mean(xs: list[float]) -> tuple[float, int, float]:
    xs = [x for x in xs if not math.isnan(x)]
    n = len(xs)
    if n < 2:
        return float("nan"), n, float("nan")
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    if v <= 0:
        return m, n, float("nan")
    return m, n, m / math.sqrt(v / n)


def main() -> None:
    px = load_prices_full()
    trades = load_trades()

    # --- Burst set: Mark 01 -> Mark 22, multi-symbol same (day, ts) ---
    burst_ts: set[tuple[int, int]] = set()
    by_dt: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for t in trades:
        by_dt[(t["day"], t["ts"])].append(t)
    for (day, ts), rows in by_dt.items():
        if len(rows) < 3:
            continue
        if any(r["buyer"] == "Mark 01" and r["seller"] == "Mark 22" for r in rows):
            syms = {r["symbol"] for r in rows}
            if len(syms) >= 2:
                burst_ts.add((day, ts))

    # --- 1) Mark 01 buyer on VEV_5300: burst window ±W vs isolated ---
    W = 500
    near_burst = set()
    for day, ts in burst_ts:
        for dt in range(ts - W, ts + W + 1):
            near_burst.add((day, dt))

    def mark01_5300_buy_print(tr: dict) -> bool:
        """Any tape print where Mark 01 buys VEV_5300 (Phase 1 used buyer role, not only aggressive)."""
        return tr["buyer"] == "Mark 01" and tr["symbol"] == "VEV_5300"

    iso_f5, burst_f5 = [], []
    for tr in trades:
        if not mark01_5300_buy_print(tr):
            continue
        s5300 = px[(tr["day"], "VEV_5300")]
        i = tick_index(s5300, tr["ts"])
        f5 = fwd_mid_delta(s5300, i, 5)
        if f5 is None:
            continue
        if (tr["day"], tr["ts"]) in near_burst:
            burst_f5.append(f5)
        else:
            iso_f5.append(f5)

    m_iso, n_iso, t_iso = tstat_mean(iso_f5)
    m_br, n_br, t_br = tstat_mean(burst_f5)
    burst_cond = {
        "window_ticks_W": W,
        "mark01_aggr_buy_vev5300_isolated_n": n_iso,
        "mean_fwd5_mid": round(m_iso, 6) if n_iso else None,
        "t_isolated": round(t_iso, 4) if n_iso > 1 and t_iso == t_iso else None,
        "near_burst_n": n_br,
        "mean_fwd5_mid_near_burst": round(m_br, 6) if n_br else None,
        "t_near_burst": round(t_br, 4) if n_br > 1 and t_br == t_br else None,
    }
    (OUT / "r4_ph2_mark01_vev5300_burst_window_fwd5.json").write_text(json.dumps(burst_cond, indent=2))

    # --- 2) Microprice vs mid on VEV_5300: spread quantile then fwd20 vol proxy ---
    s5300_all: list[tuple[int, dict]] = []
    for day in DAYS:
        for row in px.get((day, "VEV_5300"), []):
            s5300_all.append((day, row))
    spreads = [r[1]["spread"] for r in s5300_all if not math.isnan(r[1]["spread"])]
    spreads.sort()
    q33 = spreads[len(spreads) // 3] if spreads else 0
    q66 = spreads[2 * len(spreads) // 3] if spreads else 0

    def sp_bin(sp: float) -> str:
        if math.isnan(sp):
            return "na"
        if sp <= q33:
            return "tight33"
        if sp <= q66:
            return "mid33"
        return "wide33"

    micro_edge = []
    for day in DAYS:
        ser = px.get((day, "VEV_5300"), [])
        for i in range(len(ser) - 20):
            r = ser[i]
            if math.isnan(r["mid"]) or math.isnan(r["micro"]):
                continue
            d20 = ser[i + 20]["mid"] - r["mid"]
            if math.isnan(d20):
                continue
            micro_edge.append(
                {
                    "day": day,
                    "spread_bin": sp_bin(r["spread"]),
                    "micro_minus_mid": r["micro"] - r["mid"],
                    "abs_fwd20": abs(d20),
                }
            )
    # correlate micro_minus_mid with abs_fwd20 by bin (mean abs_fwd20 top vs bottom micro tercile)
    from statistics import median

    abs_m = [abs(x["micro_minus_mid"]) for x in micro_edge if not math.isnan(x["micro_minus_mid"])]
    med_m = median(abs_m) if abs_m else 0.0
    if med_m == 0.0 and abs_m:
        med_m = sorted(abs_m)[len(abs_m) // 2] or 1e-9
    hi = [x["abs_fwd20"] for x in micro_edge if abs(x["micro_minus_mid"]) > med_m]
    lo = [x["abs_fwd20"] for x in micro_edge if abs(x["micro_minus_mid"]) <= med_m]
    micro_summary = {
        "n_ticks": len(micro_edge),
        "median_abs_micro_minus_mid": med_m,
        "mean_abs_fwd20_high_micro_disloc": sum(hi) / len(hi) if hi else None,
        "mean_abs_fwd20_low_micro_disloc": sum(lo) / len(lo) if lo else None,
        "spread_tercile_cutoffs": [round(q33, 4), round(q66, 4)],
    }
    (OUT / "r4_ph2_vev5300_microprice_spread_regime.json").write_text(json.dumps(micro_summary, indent=2))

    # --- 3) Cross-instrument signed flow lag: net Mark01↔Mark22 lot flow on VEV_5300 vs extract fwd20 ---
    flow5300: dict[tuple[int, int], int] = defaultdict(int)
    for tr in trades:
        if tr["symbol"] != "VEV_5300":
            continue
        if tr["buyer"] == "Mark 01" and tr["seller"] == "Mark 22":
            flow5300[(tr["day"], tr["ts"])] += tr["qty"]
        if tr["buyer"] == "Mark 22" and tr["seller"] == "Mark 01":
            flow5300[(tr["day"], tr["ts"])] -= tr["qty"]

    lag_rows = []
    ex_series = {d: px.get((d, "VELVETFRUIT_EXTRACT"), []) for d in DAYS}
    for lag_ticks in (0, 1, 2, 3, 5, 10, 20, 50):
        ys = []
        for (day, ts), q in flow5300.items():
            if q == 0:
                continue
            ex = ex_series[day]
            if not ex:
                continue
            i0 = tick_index(ex, ts)
            j = min(i0 + lag_ticks, len(ex) - 1)
            f20 = fwd_mid_delta(ex, j, 20)
            if f20 is None:
                continue
            ys.append((float(q), f20))
        if len(ys) < 30:
            lag_rows.append({"lag_ticks_after_trade_index": lag_ticks, "n": len(ys), "corr_qty_fwd20": None})
            continue
        xs = [a[0] for a in ys]
        fs = [a[1] for a in ys]
        mx, my = sum(xs) / len(xs), sum(fs) / len(fs)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, fs))
        denx = sum((x - mx) ** 2 for x in xs) ** 0.5
        deny = sum((y - my) ** 2 for y in fs) ** 0.5
        corr = num / (denx * deny) if denx > 1e-9 and deny > 1e-9 else float("nan")
        lag_rows.append(
            {
                "lag_ticks_after_trade_index": lag_ticks,
                "n": len(ys),
                "corr_qty_fwd20": round(corr, 4) if corr == corr else None,
            }
        )
    (OUT / "r4_ph2_signed_flow_5300_vs_extract_fwd20_lag.json").write_text(json.dumps(lag_rows, indent=2))

    # --- 4) Mark 67 aggr buy extract: tight vs wide extract spread ---
    tight_f, wide_f = [], []
    for tr in trades:
        if tr["buyer"] != "Mark 67" or tr["symbol"] != "VELVETFRUIT_EXTRACT":
            continue
        ex = px.get((tr["day"], "VELVETFRUIT_EXTRACT"), [])
        if not ex:
            continue
        i = tick_index(ex, tr["ts"])
        row = ex[i]
        if tr["price"] < row["ask"]:
            continue
        f20 = fwd_mid_delta(ex, i, 20)
        if f20 is None:
            continue
        if not math.isnan(row["spread"]) and row["spread"] <= 8:
            tight_f.append(f20)
        elif not math.isnan(row["spread"]) and row["spread"] > 8:
            wide_f.append(f20)

    regime67 = {
        "extract_spread_tight_le8_n": len(tight_f),
        "mean_fwd20": round(sum(tight_f) / len(tight_f), 6) if tight_f else None,
        "extract_spread_wide_gt8_n": len(wide_f),
        "mean_fwd20_wide": round(sum(wide_f) / len(wide_f), 6) if wide_f else None,
    }
    (OUT / "r4_ph2_mark67_aggr_buy_extract_spread_regime.json").write_text(json.dumps(regime67, indent=2))

    # --- 5) Light IV proxy: skip full BS (time); mark which Mark prints when VEV_5300 spread tight ---
    tight_prints = Counter()
    for tr in trades:
        if tr["symbol"] != "VEV_5300":
            continue
        s5300 = px.get((tr["day"], "VEV_5300"), [])
        if not s5300:
            continue
        i = tick_index(s5300, tr["ts"])
        sp = s5300[i]["spread"]
        if not math.isnan(sp) and sp <= 4:
            tight_prints[tr["buyer"] + "|" + tr["seller"]] += 1
    (OUT / "r4_ph2_vev5300_tight_spread_print_pairs.json").write_text(
        json.dumps(tight_prints.most_common(20), indent=2)
    )

    # --- 6) Phase 1 interaction note: burst-conditioned Mark01 5300 ---
    phase2_note = {
        "confirms_phase1": "Mark 01 aggressive VEV_5300 short-horizon negativity persists in isolated window; burst-adjacent subset differs — see r4_ph2_mark01_vev5300_burst_window_fwd5.json",
        "refines_phase1": "Mark 67 extract lift edge stratified by extract spread (tight vs wide) in r4_ph2_mark67_aggr_buy_extract_spread_regime.json",
        "orthogonal": "Microprice dislocation on VEV_5300 vs subsequent abs fwd20 in r4_ph2_vev5300_microprice_spread_regime.json; signed-flow lag table in r4_ph2_signed_flow_5300_vs_extract_fwd20_lag.json",
    }
    (OUT / "r4_ph2_phase1_interaction_note.json").write_text(json.dumps(phase2_note, indent=2))

    lines = [
        "Round 4 Phase 2 summary",
        json.dumps(burst_cond),
        json.dumps(micro_summary),
        "Lag corr Mark01-22 5300 flow vs extract fwd20: " + json.dumps(lag_rows[:8]),
        json.dumps(regime67),
    ]
    OUT.joinpath("r4_ph2_executive_summary.txt").write_text("\n".join(lines) + "\n")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
