#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate (VEV_5200 & VEV_5300 L1 spread <= 2) + counterparty.

Matches round3work/vouchers_final_strategy convention: spread = ask_price_1 - bid_price_1,
inner-join 5200, 5300, VELVETFRUIT_EXTRACT on timestamp per day (Round 4 days 1-3).

Outputs:
- r4_phase3_gate_summary.json: P(tight), extract fwd20 tight vs loose Welch-style stats,
  Phase-1 style cells (Mark67 buyer_agg extract) split by gate at trade ts,
  Mark01->Mark22 on VEV_5300 fwd20 split by gate,
  burst (>=4 trades at ts) extract fwd20 split by gate at ts.
- r4_phase3_spread_spread.json: Pearson corr (spread pairs) full sample vs tight-only vs loose-only.

Run: python3 manual_traders/R4/r4_counterparty_phase1/analyze_r4_phase3_sonic_gate_r4.py
"""
from __future__ import annotations

import csv
import json
import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent
DAYS = (1, 2, 3)
TH = 2.0
K = 20
VE5200, VE5300, EX = "VEV_5200", "VEV_5300", "VELVETFRUIT_EXTRACT"
PRODUCTS = [
    EX,
    "HYDROGEL_PACK",
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]


@dataclass
class BookSnap:
    bid: float
    ask: float
    mid: float


def load_price_index() -> dict[tuple[int, str], tuple[list[int], list[float], list[float], list[float]]]:
    idx: dict[tuple[int, str], tuple[list[int], list[float], list[float], list[float]]] = {}
    for day in DAYS:
        path = DATA / f"prices_round_4_day_{day}.csv"
        if not path.is_file():
            continue
        by_sym: dict[str, dict[int, BookSnap]] = defaultdict(dict)
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["day"]) != day:
                    continue
                sym = row["product"]
                if sym not in PRODUCTS:
                    continue
                ts = int(row["timestamp"])
                bid = _float(row["bid_price_1"])
                ask = _float(row["ask_price_1"])
                mid = _float(row["mid_price"])
                if math.isnan(mid):
                    continue
                if math.isnan(bid) or math.isnan(ask):
                    bid, ask = mid, mid
                by_sym[sym][ts] = BookSnap(bid=bid, ask=ask, mid=mid)
        for sym, mp in by_sym.items():
            tss = sorted(mp)
            mids = [mp[t].mid for t in tss]
            bids = [mp[t].bid for t in tss]
            asks = [mp[t].ask for t in tss]
            idx[(day, sym)] = (tss, mids, bids, asks)
    return idx


def _float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_aligned(day: int) -> dict[int, dict]:
    """timestamp -> {s5200,s5300,s_ext,m_ext,...}"""
    by_ts: dict[int, dict[str, float]] = defaultdict(dict)
    path = DATA / f"prices_round_4_day_{day}.csv"
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            if int(row["day"]) != day:
                continue
            sym = row["product"]
            if sym not in (VE5200, VE5300, EX):
                continue
            ts = int(row["timestamp"])
            bid = _float(row["bid_price_1"])
            ask = _float(row["ask_price_1"])
            mid = _float(row["mid_price"])
            if math.isnan(bid) or math.isnan(ask):
                sp = 0.0
            else:
                sp = ask - bid
            by_ts[ts][sym + "_spr"] = sp
            by_ts[ts][sym + "_mid"] = mid
    rows = {}
    for ts, d in by_ts.items():
        if not all(k in d for k in (VE5200 + "_spr", VE5300 + "_spr", EX + "_mid")):
            continue
        s52 = d[VE5200 + "_spr"]
        s53 = d[VE5300 + "_spr"]
        tight = (s52 <= TH) and (s53 <= TH)
        rows[ts] = {
            "s5200": s52,
            "s5300": s53,
            "s_ext": d.get(EX + "_spr", float("nan")),
            "m_ext": d[EX + "_mid"],
            "m5200": d.get(VE5200 + "_mid", float("nan")),
            "m5300": d.get(VE5300 + "_mid", float("nan")),
            "tight": tight,
        }
    return rows


def sorted_ts_mids(rows: dict[int, dict], key_mid: str) -> tuple[list[int], list[float]]:
    tss = sorted(rows)
    mids = [rows[t][key_mid] for t in tss]
    return tss, mids


def fwd_delta(tss: list[int], mids: list[float], ts: int, k: int) -> float | None:
    i = bisect_right(tss, ts) - 1
    if i < 0:
        i = 0
    j = i + k
    if j >= len(mids):
        return None
    return mids[j] - mids[i]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = min(len(xs), len(ys))
    if n < 10:
        return None
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    vx = sum((x - mx) ** 2 for x in xs[:n])
    vy = sum((y - my) ** 2 for y in ys[:n])
    if vx <= 1e-18 or vy <= 1e-18:
        return None
    c = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return c / math.sqrt(vx * vy)


def welch_t(a: list[float], b: list[float]) -> tuple[float | None, float | None]:
    a = [x for x in a if x is not None and math.isfinite(x)]
    b = [x for x in b if x is not None and math.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return None, None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se <= 1e-18:
        return None, None
    t = (ma - mb) / se
    return t, ma - mb


def main() -> None:
    all_rows: list[tuple[int, int, dict]] = []
    gate_at: dict[tuple[int, int], bool] = {}
    for day in DAYS:
        rows = load_aligned(day)
        for ts, row in rows.items():
            all_rows.append((day, ts, row))
            gate_at[(day, ts)] = row["tight"]

    # extract forward K on per-day merged ts list
    fwd_ext_tight: list[float] = []
    fwd_ext_loose: list[float] = []
    for day in DAYS:
        rows = {ts: r for d, ts, r in all_rows if d == day}
        if not rows:
            continue
        tss = sorted(rows)
        mids = [rows[t]["m_ext"] for t in tss]
        for ts in tss:
            fk = fwd_delta(tss, mids, ts, K)
            if fk is None:
                continue
            if rows[ts]["tight"]:
                fwd_ext_tight.append(fk)
            else:
                fwd_ext_loose.append(fk)

    t_stat, mean_diff = welch_t(fwd_ext_tight, fwd_ext_loose)
    n_tight = len(fwd_ext_tight)
    n_loose = len(fwd_ext_loose)
    p_tight = n_tight / (n_tight + n_loose) if (n_tight + n_loose) else 0
    mean_t = sum(fwd_ext_tight) / n_tight if n_tight else None
    mean_l = sum(fwd_ext_loose) / n_loose if n_loose else None

    # per-trade Phase1 splits (reuse price index from day files)
    pidx = load_price_index()

    def markout(sym: str, day: int, ts: int, k: int) -> float | None:
        key = (day, sym)
        if key not in pidx:
            return None
        tss, mids, bids, asks = pidx[key]
        return fwd_delta(tss, mids, ts, k)

    def aggressor_price(day: int, sym: str, ts: int, price: float) -> str | None:
        key = (day, sym)
        if key not in pidx:
            return None
        tss, mids, bids, asks = pidx[key]
        i = bisect_right(tss, ts) - 1
        if i < 0:
            i = 0
        bid, ask, mid = bids[i], asks[i], mids[i]
        if price >= ask - 1e-9:
            return "buyer_agg"
        if price <= bid + 1e-9:
            return "seller_agg"
        if price > mid:
            return "buyer_agg"
        if price < mid:
            return "seller_agg"
        return None

    m67_t_t: list[float] = []
    m67_t_f: list[float] = []
    m01_m22_5300_t: list[float] = []
    m01_m22_5300_f: list[float] = []

    burst_ext_fwd_t: list[float] = []
    burst_ext_fwd_f: list[float] = []

    burst_count_by: dict[tuple[int, int], int] = defaultdict(int)
    for day in DAYS:
        tp = DATA / f"trades_round_4_day_{day}.csv"
        with open(tp, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                burst_count_by[(day, int(row["timestamp"]))] += 1

    for day in DAYS:
        tp = DATA / f"trades_round_4_day_{day}.csv"
        with open(tp, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                ts = int(row["timestamp"])
                sym = row["symbol"]
                buyer, seller = str(row["buyer"]).strip(), str(row["seller"]).strip()
                price = float(row["price"])
                side = aggressor_price(day, sym, ts, price)
                g = gate_at.get((day, ts), False)
                fk = markout(sym, day, ts, K)
                if fk is None:
                    continue
                if buyer == "Mark 67" and sym == EX and side == "buyer_agg":
                    (m67_t_t if g else m67_t_f).append(fk)
                if buyer == "Mark 01" and seller == "Mark 22" and sym == VE5300:
                    (m01_m22_5300_t if g else m01_m22_5300_f).append(fk)

        for (d, ts), bn in burst_count_by.items():
            if d != day or bn < 4:
                continue
            fk = markout(EX, day, ts, K)
            if fk is None:
                continue
            g = gate_at.get((day, ts), False)
            (burst_ext_fwd_t if g else burst_ext_fwd_f).append(fk)

    # spread-spread correlations (subsample every 5th ts for speed)
    xs52, xs53, xse = [], [], []
    xs52_t, xs53_t, xse_t = [], [], []
    xs52_l, xs53_l, xse_l = [], [], []
    for day in DAYS:
        rows = load_aligned(day)
        for j, ts in enumerate(sorted(rows)):
            if j % 5 != 0:
                continue
            r = rows[ts]
            xs52.append(r["s5200"])
            xs53.append(r["s5300"])
            se = r["s_ext"]
            if math.isfinite(se):
                xse.append(se)
            else:
                xse.append(0.0)
            if r["tight"]:
                xs52_t.append(r["s5200"])
                xs53_t.append(r["s5300"])
                xse_t.append(se if math.isfinite(se) else 0.0)
            else:
                xs52_l.append(r["s5200"])
                xs53_l.append(r["s5300"])
                xse_l.append(se if math.isfinite(se) else 0.0)

    spread_json = {
        "corr_s5200_s5300_all": pearson(xs52, xs53),
        "corr_s5200_s5300_tight_only": pearson(xs52_t, xs53_t),
        "corr_s5200_s5300_loose_only": pearson(xs52_l, xs53_l),
        "corr_s5200_s_ext_all": pearson(xs52, xse),
        "corr_s5300_s_ext_all": pearson(xs53, xse),
        "n_sample_all": len(xs52),
        "n_sample_tight": len(xs52_t),
        "n_sample_loose": len(xs52_l),
    }

    out = {
        "sonic_TH": TH,
        "forward_K_price_rows": K,
        "P_joint_tight": round(p_tight, 6),
        "n_tight_rows": n_tight,
        "n_loose_rows": n_loose,
        "extract_fwd20_mean_tight": round(mean_t, 6) if mean_t is not None else None,
        "extract_fwd20_mean_loose": round(mean_l, 6) if mean_l is not None else None,
        "extract_fwd20_welch_t_tight_minus_loose": round(t_stat, 4) if t_stat is not None else None,
        "extract_fwd20_mean_diff_tight_minus_loose": round(mean_diff, 6) if mean_diff is not None else None,
        "mark67_buyer_agg_extract_fwd20": {
            "n_tight": len(m67_t_t),
            "mean_tight": round(sum(m67_t_t) / len(m67_t_t), 6) if m67_t_t else None,
            "n_loose": len(m67_t_f),
            "mean_loose": round(sum(m67_t_f) / len(m67_t_f), 6) if m67_t_f else None,
        },
        "m01_m22_VEV5300_fwd20": {
            "n_tight": len(m01_m22_5300_t),
            "mean_tight": round(sum(m01_m22_5300_t) / len(m01_m22_5300_t), 6)
            if m01_m22_5300_t
            else None,
            "n_loose": len(m01_m22_5300_f),
            "mean_loose": round(sum(m01_m22_5300_f) / len(m01_m22_5300_f), 6)
            if m01_m22_5300_f
            else None,
        },
        "burst_ge4_extract_fwd20": {
            "n_tight": len(burst_ext_fwd_t),
            "mean_tight": round(sum(burst_ext_fwd_t) / len(burst_ext_fwd_t), 6)
            if burst_ext_fwd_t
            else None,
            "n_loose": len(burst_ext_fwd_f),
            "mean_loose": round(sum(burst_ext_fwd_f) / len(burst_ext_fwd_f), 6)
            if burst_ext_fwd_f
            else None,
        },
        "spread_spread_pearson": spread_json,
    }

    (OUT / "r4_phase3_gate_summary.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (OUT / "r4_phase3_spread_spread.json").write_text(
        json.dumps(spread_json, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("wrote r4_phase3_gate_summary.json and r4_phase3_spread_spread.json")


if __name__ == "__main__":
    main()
