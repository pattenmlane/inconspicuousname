#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned markouts (suggested direction.txt).

- Prices: Prosperity4Data/ROUND_4/prices_round_4_day_{1,2,3}.csv
- Trades: Prosperity4Data/ROUND_4/trades_round_4_day_{1,2,3}.csv
- Horizons K in {5, 20, 100}: forward steps in the **tape's timestamp ordering**
  (same-symbol rows sorted by timestamp; K-th next row mid - current mid).
- Aggressor: trade price vs L1 bid/ask at (day, timestamp, symbol); else vs mid.
- Outputs JSON summaries under this folder for analysis.json gate.

Run: python3 manual_traders/R4/r4_counterparty_phase1/analyze_r4_phase1_counterparty_edges.py
"""
from __future__ import annotations

import csv
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent

DAYS = (1, 2, 3)
KS = (5, 20, 100)
PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]


@dataclass
class BookSnap:
    bid: float
    ask: float
    mid: float


def _float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_price_index() -> dict[tuple[int, str], tuple[list[int], list[float], list[float], list[float]]]:
    """Per (day, symbol): sorted timestamps, mids, bids, asks (L1)."""
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


def forward_mid_delta(
    tss: list[int],
    mids: list[float],
    ts: int,
    k: int,
) -> float | None:
    if not tss:
        return None
    i = bisect_right(tss, ts) - 1
    if i < 0:
        i = 0
    j = i + k
    if j >= len(mids):
        return None
    return mids[j] - mids[i]


def aggressor_side(price: float, bid: float, ask: float, mid: float) -> str | None:
    if price >= ask - 1e-9:
        return "buyer_agg"
    if price <= bid + 1e-9:
        return "seller_agg"
    if price > mid:
        return "buyer_agg"
    if price < mid:
        return "seller_agg"
    return None


def t_stat_mean(xs: list[float]) -> tuple[float, int]:
    n = len(xs)
    if n < 2:
        return float("nan"), n
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    if v <= 1e-18:
        return float("nan"), n
    return m / (math.sqrt(v / n)), n


def main() -> None:
    pidx = load_price_index()
    # burst counts per (day, ts)
    burst_mult: dict[tuple[int, int], int] = Counter()
    trades: list[dict] = []
    for day in DAYS:
        tp = DATA / f"trades_round_4_day_{day}.csv"
        if not tp.is_file():
            continue
        with open(tp, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            rows = list(r)
        by_ts: dict[int, int] = Counter()
        for row in rows:
            by_ts[int(row["timestamp"])] += 1
        for ts, c in by_ts.items():
            burst_mult[(day, ts)] = c
        for row in rows:
            trades.append(
                {
                    "day": day,
                    "ts": int(row["timestamp"]),
                    "buyer": str(row["buyer"]).strip(),
                    "seller": str(row["seller"]).strip(),
                    "sym": str(row["symbol"]).strip(),
                    "price": float(row["price"]),
                    "qty": int(float(row["quantity"])),
                }
            )

    # --- per-trade markouts ---
    marks: list[dict] = []
    for tr in trades:
        day, sym, ts = tr["day"], tr["sym"], tr["ts"]
        key = (day, sym)
        if key not in pidx:
            continue
        tss, mids, bids, asks = pidx[key]
        i = bisect_right(tss, ts) - 1
        if i < 0:
            i = 0
        bid, ask, mid = bids[i], asks[i], mids[i]
        side = aggressor_side(tr["price"], bid, ask, mid)
        spr = ask - bid
        burst = burst_mult.get((day, ts), 1)
        hour_bin = (ts // 10000) % 12  # coarse session bucket (tape units)
        row = {
            **tr,
            "side": side,
            "spread": spr,
            "burst_n": burst,
            "hour_bin": hour_bin,
        }
        for K in KS:
            d = forward_mid_delta(tss, mids, ts, K)
            row[f"fwd_{K}"] = d
        # cross: extract forward same K
        ek = (day, "VELVETFRUIT_EXTRACT")
        if ek in pidx:
            etss, emids, _, _ = pidx[ek]
            for K in KS:
                row[f"ex_fwd_{K}"] = forward_mid_delta(etss, emids, ts, K)
        else:
            for K in KS:
                row[f"ex_fwd_{K}"] = None
        marks.append(row)

    # spread quantile per symbol (global over marks)
    spr_by_sym: dict[str, list[float]] = defaultdict(list)
    for m in marks:
        if not math.isnan(m["spread"]):
            spr_by_sym[m["sym"]].append(m["spread"])
    cut_q: dict[str, tuple[float, float, float]] = {}
    for sym, arr in spr_by_sym.items():
        sa = sorted(arr)
        n = len(sa)
        if n == 0:
            continue
        cut_q[sym] = (
            sa[int(0.25 * (n - 1))],
            sa[int(0.5 * (n - 1))],
            sa[int(0.75 * (n - 1))],
        )

    def spr_q(m: dict) -> str:
        sym = m["sym"]
        if sym not in cut_q:
            return "q_na"
        q1, q2, q3 = cut_q[sym]
        s = m["spread"]
        if math.isnan(s):
            return "q_na"
        if s <= q1:
            return "tight"
        if s <= q2:
            return "midlo"
        if s <= q3:
            return "midhi"
        return "wide"

    for m in marks:
        m["spr_q"] = spr_q(m)

    # --- (1) Participant predictability: focus U x side x sym x K, pool days ---
    cells: dict[tuple, list[float]] = defaultdict(list)
    for m in marks:
        if m["side"] is None:
            continue
        U = m["buyer"] if m["side"] == "buyer_agg" else m["seller"]
        for K in KS:
            fk = m.get(f"fwd_{K}")
            if fk is None or math.isnan(fk):
                continue
            cells[(U, m["side"], m["sym"], K)].append(fk)

    part_summary = []
    for (U, side, sym, K), xs in sorted(cells.items(), key=lambda kv: -len(kv[1]))[
        :200
    ]:
        n = len(xs)
        if n < 30:
            continue
        mu = sum(xs) / n
        pos = sum(1 for x in xs if x > 0) / n
        ts, _ = t_stat_mean(xs)
        part_summary.append(
            {
                "U": U,
                "side": side,
                "symbol": sym,
                "K": K,
                "n": n,
                "mean_fwd_mid": round(mu, 6),
                "frac_pos": round(pos, 4),
                "t_stat": round(ts, 3) if not math.isnan(ts) else None,
            }
        )
    part_summary.sort(key=lambda x: (-abs(x["mean_fwd_mid"]) * math.sqrt(x["n"]), -x["n"]))

    # stratified examples (K=20 same symbol)
    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    strat = []
    for label, pred in (
        ("M01_to_M22_VEV5300_burst_ge4", lambda m: m["buyer"] == "Mark 01" and m["seller"] == "Mark 22" and m["sym"] == "VEV_5300" and m["burst_n"] >= 4),
        ("M01_to_M22_VEV5300_singleton", lambda m: m["buyer"] == "Mark 01" and m["seller"] == "Mark 22" and m["sym"] == "VEV_5300" and m["burst_n"] == 1),
        ("all_trades_fwd20", lambda m: True),
    ):
        xs = [m["fwd_20"] for m in marks if m.get("fwd_20") is not None and pred(m)]
        m = _mean(xs)
        strat.append(
            {
                "bucket": label,
                "n": len(xs),
                "mean_fwd20": None if math.isnan(m) else round(m, 6),
            }
        )

    # --- (2) Baseline cell mean residual: (buyer, seller, symbol, spr_q) for K=20 ---
    cell20: dict[tuple, list[float]] = defaultdict(list)
    for m in marks:
        fk = m.get("fwd_20")
        if fk is None:
            continue
        cell20[(m["buyer"], m["seller"], m["sym"], m["spr_q"])].append(fk)
    baseline_mean = {k: sum(v) / len(v) for k, v in cell20.items() if len(v) >= 5}
    resids = []
    for m in marks:
        fk = m.get("fwd_20")
        if fk is None:
            continue
        key = (m["buyer"], m["seller"], m["sym"], m["spr_q"])
        bmean = baseline_mean.get(key)
        if bmean is None:
            continue
        resids.append(fk - bmean)
    resid_mean = sum(resids) / len(resids) if resids else float("nan")
    resid_abs_mean = sum(abs(x) for x in resids) / len(resids) if resids else float("nan")

    # --- (3) Graph buyer->seller ---
    pair_c = Counter()
    pair_notional = Counter()
    for m in marks:
        pair_c[(m["buyer"], m["seller"])] += 1
        pair_notional[(m["buyer"], m["seller"])] += abs(m["price"]) * m["qty"]
    top_pairs = pair_c.most_common(25)

    # 2-hop counts A->B->C on consecutive trades same day (cheap proxy)
    by_day: dict[int, list[dict]] = defaultdict(list)
    for m in marks:
        by_day[m["day"]].append(m)
    for d in by_day:
        by_day[d].sort(key=lambda x: (x["ts"], x["sym"]))
    hop2 = Counter()
    for d, seq in by_day.items():
        for i in range(len(seq) - 2):
            a, b, c = seq[i], seq[i + 1], seq[i + 2]
            hop2[(a["buyer"], a["seller"], b["buyer"], b["seller"])] += 1
    top_hop2 = hop2.most_common(15)

    # --- (4) Burst event study: first row per (day,ts) burst>=4, forward extract ---
    burst_keys = {(d, t) for (d, t), n in burst_mult.items() if n >= 4}
    ex_fwd_burst = []
    ex_fwd_ctrl = []
    for m in marks:
        fk = m.get("ex_fwd_20")
        if fk is None:
            continue
        if (m["day"], m["ts"]) in burst_keys:
            ex_fwd_burst.append(fk)
        elif m["burst_n"] == 1:
            ex_fwd_ctrl.append(fk)
    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    mb, mc = mean(ex_fwd_burst), mean(ex_fwd_ctrl)
    burst_study = {
        "n_burst_rows": len(ex_fwd_burst),
        "n_control_singleton": len(ex_fwd_ctrl),
        "mean_ex_fwd20_burst": None if math.isnan(mb) else round(mb, 6),
        "mean_ex_fwd20_singleton": None if math.isnan(mc) else round(mc, 6),
    }

    # --- (5) Adverse: for each buyer when buyer_agg, mean fwd same sym K=20 ---
    adv_buy = defaultdict(list)
    for m in marks:
        if m["side"] != "buyer_agg":
            continue
        fk = m.get("fwd_20")
        if fk is None:
            continue
        adv_buy[m["buyer"]].append(fk)
    adverse = sorted(
        [
            {"buyer_when_agg": u, "n": len(vs), "mean_fwd20": round(sum(vs) / len(vs), 6)}
            for u, vs in adv_buy.items()
            if len(vs) >= 20
        ],
        key=lambda x: x["mean_fwd20"],
    )

    out_main = {
        "n_trades_parsed": len(trades),
        "n_marks_with_fwd": len(marks),
        "participant_cells_top": part_summary[:40],
        "stratified_baseline_note": strat,
        "baseline_fwd20_residual_mean": round(resid_mean, 8),
        "baseline_fwd20_residual_abs_mean": round(resid_abs_mean, 8),
        "top_buyer_seller_pairs": [
            {"buyer": a, "seller": b, "count": n} for (a, b), n in top_pairs
        ],
        "top_2hop_trade_chain_patterns": [
            {"pattern": k, "count": n} for k, n in top_hop2
        ],
        "burst_event_study_extract_fwd20": burst_study,
        "buyer_agg_adverse_fwd20_by_buyer": adverse[:15],
    }
    (OUT / "r4_phase1_markout_summary.json").write_text(
        json.dumps(out_main, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    # CSV sample for audit
    csv_path = OUT / "r4_phase1_trade_markouts_sample.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "day",
                "ts",
                "buyer",
                "seller",
                "sym",
                "price",
                "side",
                "spread",
                "spr_q",
                "burst_n",
                "fwd_5",
                "fwd_20",
                "fwd_100",
                "ex_fwd_20",
            ],
        )
        w.writeheader()
        for m in marks[:5000]:
            w.writerow(
                {
                    "day": m["day"],
                    "ts": m["ts"],
                    "buyer": m["buyer"],
                    "seller": m["seller"],
                    "sym": m["sym"],
                    "price": m["price"],
                    "side": m["side"],
                    "spread": m["spread"],
                    "spr_q": m["spr_q"],
                    "burst_n": m["burst_n"],
                    "fwd_5": m.get("fwd_5"),
                    "fwd_20": m.get("fwd_20"),
                    "fwd_100": m.get("fwd_100"),
                    "ex_fwd_20": m.get("ex_fwd_20"),
                }
            )

    print("wrote", OUT / "r4_phase1_markout_summary.json")
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
