#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate (VEV_5200 & VEV_5300 spread <= 2) on R4 tape.

Convention aligned with round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py:
  inner join on timestamp across 5200, 5300, EXTRACT; first row per (day, product, timestamp);
  spread = ask1 - bid1; joint_tight = (s5200 <= TH) & (s5300 <= TH);
  K-step forward extract mid = m_ext[i+K] - m_ext[i] on sorted panel.

Also: inclineGod spread–spread / spread vs extract; three-way (mark_pattern, joint_tight, symbol) markouts
for selected trade types; compare gated vs ungated to Phase 1/2 headline stats.

Run: python3 manual_traders/R4/r3v_atm_straddle_proxy_03/ph3_r4_sonic_gate_analysis.py
"""
from __future__ import annotations

import bisect
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
TH = 2
K = 20
VEV_5200, VEV_5300, EXTRACT = "VEV_5200", "VEV_5300", "VELVETFRUIT_EXTRACT"


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


def series_one_day(day: int, product: str) -> dict[int, tuple[float, float, float, float]]:
    """timestamp -> (mid, spread, bid, ask) first occurrence per ts."""
    p = DATA / f"prices_round_4_day_{day}.csv"
    out: dict[int, tuple[float, float, float, float]] = {}
    if not p.is_file():
        return out
    with p.open(newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            if (row.get("product") or "") != product:
                continue
            if row.get("day") is not None and str(row["day"]).strip() != "":
                if int(float(row["day"])) != day:
                    continue
            ts = _i(row["timestamp"])
            if ts in out:
                continue
            bid = _f(row.get("bid_price_1", ""))
            ask = _f(row.get("ask_price_1", ""))
            if math.isnan(bid) or math.isnan(ask) or ask <= bid:
                mid = _f(row.get("mid_price", ""))
                sp = float("nan")
            else:
                mid = 0.5 * (bid + ask)
                sp = ask - bid
            out[ts] = (mid, sp, bid, ask)
    return out


def aligned_panel(day: int) -> list[dict]:
    a = series_one_day(day, VEV_5200)
    b = series_one_day(day, VEV_5300)
    e = series_one_day(day, EXTRACT)
    keys = sorted(set(a) & set(b) & set(e))
    rows = []
    for ts in keys:
        m52, s52, b52, a52 = a[ts]
        m53, s53, b53, a53 = b[ts]
        mx, sx, bx, ax = e[ts]
        rows.append(
            {
                "day": day,
                "ts": ts,
                "s5200": s52,
                "s5300": s53,
                "s_ext": sx,
                "m5200": m52,
                "m5300": m53,
                "m_ext": mx,
                "joint_tight": (not math.isnan(s52) and not math.isnan(s53) and s52 <= TH and s53 <= TH),
            }
        )
    rows.sort(key=lambda r: r["ts"])
    for i, r in enumerate(rows):
        j = min(i + K, len(rows) - 1)
        r["fwd_k_extract"] = rows[j]["m_ext"] - r["m_ext"] if not math.isnan(r["m_ext"]) else float("nan")
    return rows


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


def gate_at(panel: list[dict], ts: int) -> bool | None:
    if not panel:
        return None
    tss = [r["ts"] for r in panel]
    i = bisect.bisect_right(tss, ts) - 1
    if i < 0:
        return None
    return bool(panel[i]["joint_tight"])


def fwd_extract_at(panel: list[dict], ts: int) -> float | None:
    tss = [r["ts"] for r in panel]
    i = bisect.bisect_right(tss, ts) - 1
    if i < 0 or i >= len(panel):
        return None
    j = min(i + K, len(panel) - 1)
    a, b = panel[i]["m_ext"], panel[j]["m_ext"]
    if math.isnan(a) or math.isnan(b):
        return None
    return b - a


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx < 1e-12 or dy < 1e-12:
        return None
    return num / (dx * dy)


def main() -> None:
    panels: dict[int, list[dict]] = {}
    all_rows: list[dict] = []
    for d in DAYS:
        pan = aligned_panel(d)
        panels[d] = pan
        all_rows.extend(pan)

    # --- inclineGod: spread correlations on aligned inner-join rows ---
    s52 = [r["s5200"] for r in all_rows if not math.isnan(r["s5200"])]
    s53 = [r["s5300"] for r in all_rows if not math.isnan(r["s5300"])]
    se = [r["s_ext"] for r in all_rows if not math.isnan(r["s_ext"])]
    # align lengths by zipping same rows
    z52, z53, ze = [], [], []
    for r in all_rows:
        if math.isnan(r["s5200"]) or math.isnan(r["s5300"]) or math.isnan(r["s_ext"]):
            continue
        z52.append(r["s5200"])
        z53.append(r["s5300"])
        ze.append(r["s_ext"])
    c12, c1e, c2e = pearson(z52, z53), pearson(z52, ze), pearson(z53, ze)
    spread_corr = {
        "n": len(z52),
        "corr_s5200_s5300": round(c12, 4) if c12 is not None else None,
        "corr_s5200_s_ext": round(c1e, 4) if c1e is not None else None,
        "corr_s5300_s_ext": round(c2e, 4) if c2e is not None else None,
    }
    (OUT / "r4_ph3_spread_spread_correlations.json").write_text(json.dumps(spread_corr, indent=2))

    # --- Sonic: tight vs not tight extract fwd K (per day + pooled) ---
    sonic_days = []
    for d in DAYS:
        pan = panels[d]
        tight = [r["fwd_k_extract"] for r in pan if r["joint_tight"] and math.isfinite(r["fwd_k_extract"])]
        loose = [r["fwd_k_extract"] for r in pan if (not r["joint_tight"]) and math.isfinite(r["fwd_k_extract"])]
        mt = sum(tight) / len(tight) if tight else None
        ml = sum(loose) / len(loose) if loose else None
        sonic_days.append(
            {
                "day": d,
                "p_tight": len(tight) / max(1, len(pan)),
                "n_tight": len(tight),
                "mean_fwd_extract": round(mt, 6) if mt is not None else None,
                "n_loose": len(loose),
                "mean_fwd_extract_loose": round(ml, 6) if ml is not None else None,
            }
        )
    tight_all = [r["fwd_k_extract"] for r in all_rows if r["joint_tight"] and math.isfinite(r["fwd_k_extract"])]
    loose_all = [r["fwd_k_extract"] for r in all_rows if (not r["joint_tight"]) and math.isfinite(r["fwd_k_extract"])]
    sonic_pooled = {
        "K": K,
        "TH": TH,
        "mean_fwd_tight": round(sum(tight_all) / len(tight_all), 6) if tight_all else None,
        "mean_fwd_not_tight": round(sum(loose_all) / len(loose_all), 6) if loose_all else None,
        "n_tight": len(tight_all),
        "n_not_tight": len(loose_all),
        "per_day": sonic_days,
    }
    (OUT / "r4_ph3_sonic_joint_gate_extract_fwd20.json").write_text(json.dumps(sonic_pooled, indent=2))

    # --- Mark 67 aggressive extract: fwd20 gated vs not ---
    trades = load_trades()
    ex_series = {d: series_one_day(d, EXTRACT) for d in DAYS}

    def is_aggr_buy_extract(tr: dict) -> bool:
        if tr["buyer"] != "Mark 67" or tr["symbol"] != EXTRACT:
            return False
        tup = ex_series[tr["day"]].get(tr["ts"])
        if not tup:
            return False
        _m, _s, bid, ask = tup
        if math.isnan(bid) or math.isnan(ask) or ask <= bid:
            return False
        return tr["price"] >= ask

    g_yes, g_no = [], []
    for tr in trades:
        if not is_aggr_buy_extract(tr):
            continue
        pan = panels[tr["day"]]
        f = fwd_extract_at(pan, tr["ts"])
        if f is None or not math.isfinite(f):
            continue
        gt = gate_at(pan, tr["ts"])
        if gt is True:
            g_yes.append(f)
        elif gt is False:
            g_no.append(f)
    mark67_gate = {
        "n_joint_tight_at_trade": len(g_yes),
        "mean_fwd20": round(sum(g_yes) / len(g_yes), 6) if g_yes else None,
        "n_not_tight": len(g_no),
        "mean_fwd20_not_tight": round(sum(g_no) / len(g_no), 6) if g_no else None,
    }
    (OUT / "r4_ph3_mark67_aggr_extract_fwd20_by_gate.json").write_text(json.dumps(mark67_gate, indent=2))

    # --- Mark 01 -> Mark 22 on VEV_5300: any buy, fwd5 on 5300 mid gated ---
    s5300_by_day = {d: series_one_day(d, VEV_5300) for d in DAYS}

    def fwd5300(tr: dict, kk: int) -> float | None:
        ser = s5300_by_day[tr["day"]]
        tss = sorted(ser.keys())
        i = bisect.bisect_right(tss, tr["ts"]) - 1
        if i < 0:
            return None
        idx = tss[i]
        pos = i
        j = min(pos + kk, len(tss) - 1)
        m0 = ser[tss[pos]][0]
        m1 = ser[tss[j]][0]
        if math.isnan(m0) or math.isnan(m1):
            return None
        return m1 - m0

    m01_yes, m01_no = [], []
    for tr in trades:
        if tr["buyer"] != "Mark 01" or tr["seller"] != "Mark 22" or tr["symbol"] != VEV_5300:
            continue
        f5 = fwd5300(tr, 5)
        if f5 is None:
            continue
        pan = panels[tr["day"]]
        gt = gate_at(pan, tr["ts"])
        if gt is True:
            m01_yes.append(f5)
        elif gt is False:
            m01_no.append(f5)

    mark01_gate = {
        "n_tight": len(m01_yes),
        "mean_fwd5_vev5300": round(sum(m01_yes) / len(m01_yes), 6) if m01_yes else None,
        "n_not_tight": len(m01_no),
        "mean_fwd5_not_tight": round(sum(m01_no) / len(m01_no), 6) if m01_no else None,
    }
    (OUT / "r4_ph3_mark01_to_22_vev5300_fwd5_by_gate.json").write_text(json.dumps(mark01_gate, indent=2))

    # --- Burst timestamps (same as v1) ---
    by_dt: dict[tuple[int, int], list] = defaultdict(list)
    for tr in trades:
        by_dt[(tr["day"], tr["ts"])].append(tr)
    burst_ts: set[tuple[int, int]] = set()
    for (day, ts), rows in by_dt.items():
        if len(rows) < 3:
            continue
        if any(r["buyer"] == "Mark 01" and r["seller"] == "Mark 22" for r in rows):
            if len({r["symbol"] for r in rows}) >= 2:
                burst_ts.add((day, ts))

    burst_f5_tight, burst_f5_loose = [], []
    for day, ts in burst_ts:
        pan = panels[day]
        # use a synthetic "trade" at ts on 5300 for fwd
        tr = {"day": day, "ts": ts}
        f5 = fwd5300(tr, 5)
        if f5 is None:
            continue
        gt = gate_at(pan, ts)
        if gt is True:
            burst_f5_tight.append(f5)
        elif gt is False:
            burst_f5_loose.append(f5)

    burst_gate = {
        "n_bursts_tight": len(burst_f5_tight),
        "mean_fwd5_5300": round(sum(burst_f5_tight) / len(burst_f5_tight), 6) if burst_f5_tight else None,
        "n_bursts_loose": len(burst_f5_loose),
        "mean_fwd5_5300_loose": round(sum(burst_f5_loose) / len(burst_f5_loose), 6) if burst_f5_loose else None,
    }
    (OUT / "r4_ph3_burst_fwd5_vev5300_by_gate.json").write_text(json.dumps(burst_gate, indent=2))

    summary = {
        "spread_spread": spread_corr,
        "sonic_extract_fwd": sonic_pooled,
        "mark67_gated": mark67_gate,
        "mark01_5300_gated": mark01_gate,
        "burst_5300_gated": burst_gate,
        "comparison_to_phase1": "Mark67 n=164 all had extract spread<=8 (ph2); ph3 shows same trades split by joint 5200+5300<=2 vs wider co-movement of wing spreads.",
    }
    (OUT / "r4_ph3_executive_summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
