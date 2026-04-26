#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned markouts (suggested direction.txt).

Tick definition: per (day, symbol), rows from prices CSV ordered by (timestamp, file order).
Forward K ticks: mid at index min(i+K, n-1) minus mid at index i (trade-aligned index).

Aggressive side: at trade (day, ts, sym, price), BBO from price row with same day, product, timestamp
(if missing, skip side split for that row).

Horizons K in {5, 20, 100}. Cross-asset: same forward for VELVETFRUIT_EXTRACT and HYDROGEL_PACK
using extract/hydro tick series aligned by timestamp index (nearest <= trade ts).

Outputs under manual_traders/R4/r3v_atm_straddle_proxy_03/analysis_outputs/
Run from repo root:
  python3 manual_traders/R4/r3v_atm_straddle_proxy_03/ph1_r4_counterparty_analysis.py
"""
from __future__ import annotations

import bisect
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
KS = (5, 20, 100)
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


def load_prices() -> dict[tuple[int, str], list[tuple[int, float, float, float, float]]]:
    """(day, product) -> list of (ts, mid, spread, bid1, ask1) in file order."""
    acc: dict[tuple[int, str], list[tuple[int, float, float, float, float]]] = defaultdict(list)
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
                if math.isnan(bid) or math.isnan(ask) or ask <= bid:
                    mid = _f(row.get("mid_price", ""))
                    sp = float("nan")
                else:
                    mid = 0.5 * (bid + ask)
                    sp = ask - bid
                acc[(day, prod)].append((ts, mid, sp, bid, ask))
    for k in list(acc.keys()):
        acc[k].sort(key=lambda t: (t[0],))
    return acc


def load_trades() -> list[dict]:
    rows: list[dict] = []
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
    rows.sort(key=lambda x: (x["day"], x["ts"], x["symbol"]))
    return rows


def tick_index_for_ts(series: list[tuple[int, float, float, float, float]], ts: int) -> int:
    tss = [s[0] for s in series]
    if not tss:
        return -1
    i = bisect.bisect_right(tss, ts) - 1
    return max(0, min(i, len(series) - 1))


def forward_delta(series: list[tuple[int, float, float, float, float]], i: int, k: int) -> float | None:
    if i < 0 or not series:
        return None
    j = min(i + k, len(series) - 1)
    mi, mj = series[i][1], series[j][1]
    if math.isnan(mi) or math.isnan(mj):
        return None
    return mj - mi


def aggressive_side(
    series: list[tuple[int, float, float, float, float]], ts: int, price: int
) -> str | None:
    i = tick_index_for_ts(series, ts)
    if i < 0:
        return None
    bid, ask = series[i][3], series[i][4]
    if math.isnan(bid) or math.isnan(ask) or ask <= bid:
        return None
    if price >= ask:
        return "aggr_buy"
    if price <= bid:
        return "aggr_sell"
    return "mid_inside"


def spread_bin(sp: float) -> str:
    if math.isnan(sp) or sp < 0:
        return "unknown"
    if sp <= 4:
        return "tight"
    if sp <= 12:
        return "mid"
    return "wide"


def hour_bucket(ts: int) -> str:
    # timestamps are simulation ticks; bucket by coarse ranges
    if ts < 250_000:
        return "early"
    if ts < 500_000:
        return "midday"
    return "late"


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
    prices = load_prices()
    trades = load_trades()

    # Pre-index price series per day for extract/hydro for cross-asset
    def series(day: int, sym: str):
        return prices.get((day, sym), [])

    # --- Per-trade markouts ---
    mark_rows: list[dict] = []
    for tr in trades:
        day, ts, sym = tr["day"], tr["ts"], tr["symbol"]
        if sym not in PRODUCTS:
            continue
        s_sym = series(day, sym)
        if not s_sym:
            continue
        i_sym = tick_index_for_ts(s_sym, ts)
        sp = s_sym[i_sym][2]
        side = aggressive_side(s_sym, ts, tr["price"])
        hb = hour_bucket(ts)
        sb = spread_bin(sp)

        row: dict = {
            **tr,
            "spread_bin": sb,
            "hour_bin": hb,
            "aggressive": side,
        }
        for K in KS:
            d_sym = forward_delta(s_sym, i_sym, K)
            row[f"fwd_{K}_sym"] = d_sym
            for cross in ("VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"):
                s2 = series(day, cross)
                if not s2:
                    row[f"fwd_{K}_{cross}"] = None
                    continue
                i2 = tick_index_for_ts(s2, ts)
                row[f"fwd_{K}_{cross}"] = forward_delta(s2, i2, K)
        mark_rows.append(row)

    # Write markout sample JSONL (trim size: summary stats primary)
    sample_path = OUT / "r4_ph1_markout_per_trade_sample.json"
    sample_path.write_text(json.dumps(mark_rows[:500], indent=2))

    # --- 1) Participant-level: aggregate by (name, role as buyer vs seller, aggressive subset) ---
    names = sorted({r["buyer"] for r in mark_rows} | {r["seller"] for r in mark_rows})

    def collect_for_party(U: str, role: str) -> list[dict]:
        out = []
        for r in mark_rows:
            if role == "buyer" and r["buyer"] != U:
                continue
            if role == "seller" and r["seller"] != U:
                continue
            out.append(r)
        return out

    ph1_predict = []
    for U in names:
        for role in ("buyer", "seller"):
            sub = collect_for_party(U, role)
            if len(sub) < 30:
                continue
            for sym in sorted({r["symbol"] for r in sub}):
                s2 = [r for r in sub if r["symbol"] == sym]
                if len(s2) < 20:
                    continue
                for K in KS:
                    key = f"fwd_{K}_sym"
                    vals = [r[key] for r in s2 if r.get(key) is not None]
                    m, n, t = tstat_mean([float(v) for v in vals])
                    frac_pos = sum(1 for v in vals if v > 0) / max(1, len(vals))
                    ph1_predict.append(
                        {
                            "party": U,
                            "role": role,
                            "symbol": sym,
                            "K": K,
                            "n": n,
                            "mean_fwd_mid": round(m, 6) if not math.isnan(m) else None,
                            "t_stat": round(t, 4) if not math.isnan(t) else None,
                            "frac_pos": round(frac_pos, 4),
                        }
                    )

    (OUT / "r4_ph1_participant_predictivity.json").write_text(json.dumps(ph1_predict, indent=2))

    # Stratify: buyer Mark 01, symbol VEV_5300, spread bins
    strat = []
    for r in mark_rows:
        if r["buyer"] != "Mark 01" or r["symbol"] != "VEV_5300":
            continue
        for K in KS:
            v = r.get(f"fwd_{K}_VELVETFRUIT_EXTRACT")
            if v is None:
                continue
            strat.append(
                {
                    "K": K,
                    "spread_bin": r["spread_bin"],
                    "hour_bin": r["hour_bin"],
                    "fwd_ex": v,
                }
            )
    (OUT / "r4_ph1_mark01_vev5300_stratify_extract.json").write_text(json.dumps(strat[:8000], indent=2))

    # --- 2) Baseline: cell mean by (buyer, seller, symbol) ---
    cell_key = lambda r: (r["buyer"], r["seller"], r["symbol"])
    cells: dict[tuple, list[float]] = defaultdict(list)
    for r in mark_rows:
        v = r.get("fwd_20_sym")
        if v is None:
            continue
        cells[cell_key(r)].append(float(v))

    cell_mean = {k: sum(v) / len(v) for k, v in cells.items() if len(v) >= 5}
    residuals = []
    for r in mark_rows:
        v = r.get("fwd_20_sym")
        if v is None:
            continue
        cm = cell_mean.get(cell_key(r))
        if cm is None:
            continue
        residuals.append(
            {
                "buyer": r["buyer"],
                "seller": r["seller"],
                "symbol": r["symbol"],
                "resid": float(v) - cm,
                "day": r["day"],
            }
        )
    cell_rows = [{"key": list(k), "n": len(cells[k]), "mean": cell_mean[k]} for k in cell_mean]
    cell_rows.sort(key=lambda r: -r["n"])
    (OUT / "r4_ph1_baseline_cell_means_fwd20.json").write_text(json.dumps(cell_rows[:80], indent=2))
    (OUT / "r4_ph1_residuals_fwd20.json").write_text(json.dumps(residuals[:12000], indent=2))

    # --- 3) Graph buyer -> seller ---
    pair_cnt: Counter[tuple[str, str]] = Counter()
    pair_notional: Counter[tuple[str, str]] = Counter()
    for r in trades:
        pair_cnt[(r["buyer"], r["seller"])] += 1
        pair_notional[(r["buyer"], r["seller"])] += abs(r["price"] * r["qty"])

    top_pairs = [(list(k), v, pair_notional[k]) for k, v in pair_cnt.most_common(25)]
    (OUT / "r4_ph1_graph_top_pairs.json").write_text(json.dumps(top_pairs, indent=2))

    # 2-hop: count A->B->C patterns on consecutive trades same day (coarse)
    trades_sorted = sorted(trades, key=lambda x: (x["day"], x["ts"], x["symbol"]))
    hop2 = Counter()
    for i in range(len(trades_sorted) - 2):
        a = (trades_sorted[i]["buyer"], trades_sorted[i]["seller"])
        b = (trades_sorted[i + 1]["buyer"], trades_sorted[i + 1]["seller"])
        c = (trades_sorted[i + 2]["buyer"], trades_sorted[i + 2]["seller"])
        hop2[(a, b, c)] += 1
    top_hop = hop2.most_common(15)
    (OUT / "r4_ph1_graph_2hop_trade_order.json").write_text(json.dumps([{"chain": list(k[0]) + list(k[1]) + list(k[2]), "n": v} for k, v in top_hop], indent=2))

    # --- 4) Bursts: same (day, ts) multi-row ---
    burst_groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in trades:
        burst_groups[(r["day"], r["ts"])].append(r)
    bursts = {k: v for k, v in burst_groups.items() if len(v) >= 3}
    burst_events = []
    for (day, ts), rows in sorted(bursts.items(), key=lambda x: -len(x[1]))[:200]:
        buyers = [x["buyer"] for x in rows]
        orch = Counter(buyers).most_common(1)[0][0]
        syms = sorted({x["symbol"] for x in rows})
        # forward extract at K=20 from first symbol in burst (use extract series)
        ex = series(day, "VELVETFRUIT_EXTRACT")
        i0 = tick_index_for_ts(ex, ts)
        f20 = forward_delta(ex, i0, 20) if ex else None
        burst_events.append(
            {
                "day": day,
                "ts": ts,
                "n_prints": len(rows),
                "orchestrator_buyer_mode": orch,
                "symbols": syms[:12],
                "fwd20_extract": f20,
            }
        )
    (OUT / "r4_ph1_burst_events.json").write_text(json.dumps(burst_events, indent=2))
    bvals = [e["fwd20_extract"] for e in burst_events if e["fwd20_extract"] is not None]
    ctrl = []
    rng = list({(r["day"], r["ts"]) for r in trades})
    import random

    random.seed(42)
    for _ in range(min(500, max(1, len(rng)))):
        day, ts = random.choice(rng)
        if (day, ts) in bursts:
            continue
        ex = series(day, "VELVETFRUIT_EXTRACT")
        if not ex:
            continue
        i0 = tick_index_for_ts(ex, ts)
        f20 = forward_delta(ex, i0, 20)
        if f20 is not None:
            ctrl.append(f20)
    m_b, n_b, _ = tstat_mean([float(x) for x in bvals])
    m_c, n_c, _ = tstat_mean([float(x) for x in ctrl])
    (OUT / "r4_ph1_burst_vs_control_extract_fwd20.txt").write_text(
        f"burst_n={len(bvals)} mean_fwd20_extract={m_b}\ncontrol_n={n_c} mean_fwd20_extract={m_c}\n"
    )

    # --- 5) Adverse selection proxy: when U is buyer, fwd20 sym (seller receives inventory) ---
    adv = []
    for U in names:
        buys = [r for r in mark_rows if r["buyer"] == U and r.get("aggressive") == "aggr_buy"]
        sells = [r for r in mark_rows if r["seller"] == U and r.get("aggressive") == "aggr_sell"]
        for label, arr in (("as_buyer_aggr", buys), ("as_seller_aggr", sells)):
            vals = [r["fwd_20_sym"] for r in arr if r.get("fwd_20_sym") is not None]
            m, n, t = tstat_mean([float(v) for v in vals])
            if n >= 15:
                adv.append({"party": U, "leg": label, "n": n, "mean_fwd20_sym": m, "t": t})
    (OUT / "r4_ph1_adverse_aggressor_fwd20.json").write_text(json.dumps(adv, indent=2))

    # Summary text
    lines = [
        "Round 4 Phase 1 — summary (see JSON artifacts in same folder)",
        f"Trade rows analyzed (with price mid path): {len(mark_rows)}",
        f"Distinct participants: {len(names)}",
        "",
        "Top buyer->seller pairs (count, notional):",
    ]
    for item in top_pairs[:8]:
        lines.append(f"  {item[0][0]} -> {item[0][1]}: n={item[1]} notional~{item[2]}")
    lines.append("")
    lines.append("Burst vs control extract fwd20 (see r4_ph1_burst_vs_control_extract_fwd20.txt)")
    lines.append(f"Markout party table rows: {len(ph1_predict)}")
    OUT.joinpath("r4_ph1_executive_summary.txt").write_text("\n".join(lines) + "\n")

    # Per-day stability: Mark 67 aggressive buy fwd20 on traded symbol
    stab = defaultdict(list)
    for r in mark_rows:
        if r["buyer"] != "Mark 67" or r.get("aggressive") != "aggr_buy":
            continue
        v = r.get("fwd_20_sym")
        if v is None:
            continue
        stab[r["day"]].append(float(v))
    stab_out = {str(d): {"n": len(vs), "mean": sum(vs) / len(vs)} for d, vs in sorted(stab.items())}
    (OUT / "r4_ph1_mark67_aggr_buy_fwd20_by_day.json").write_text(json.dumps(stab_out, indent=2))

    print("Wrote", OUT)


if __name__ == "__main__":
    main()
