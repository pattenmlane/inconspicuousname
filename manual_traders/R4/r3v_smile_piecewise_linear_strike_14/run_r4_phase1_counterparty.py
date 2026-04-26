#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-aware tape analysis (suggested direction.txt).

Horizon K: K-th *subsequent* price snapshot for the same (day, symbol) after trade
timestamp (strictly greater timestamps), K in {5, 20, 100}. Forward move = mid[t+K] - mid[t].

Aggressor: at trade timestamp, if trade price >= ask -> buyer aggressive; if price <= bid ->
seller aggressive; else ambiguous (both).

Outputs under this folder (JSON/CSV summaries for analysis.json gate).
"""
from __future__ import annotations

import bisect
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path("Prosperity4Data/ROUND_4")
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = (1, 2, 3)
K_HORIZONS = (5, 20, 100)
CROSS = ("VELVETFRUIT_EXTRACT", "HYDROGEL_PACK")
ALL_VEV = [
    f"VEV_{k}"
    for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
]


@dataclass
class Snap:
    ts: int
    mid: float
    bid: int
    ask: int
    spread: int


def load_prices(day: int) -> dict[str, list[Snap]]:
    path = DATA / f"prices_round_4_day_{day}.csv"
    by_sym: dict[str, list[Snap]] = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if int(r["day"]) != day:
                continue
            sym = r["product"]
            ts = int(r["timestamp"])
            try:
                bb = int(float(r["bid_price_1"]))
                ba = int(float(r["ask_price_1"]))
            except (KeyError, ValueError):
                continue
            mid = float(r["mid_price"])
            by_sym[sym].append(Snap(ts, mid, bb, ba, ba - bb))
    for sym in by_sym:
        by_sym[sym].sort(key=lambda s: s.ts)
        # dedupe same ts keep last
        dedup: dict[int, Snap] = {}
        for s in by_sym[sym]:
            dedup[s.ts] = s
        by_sym[sym] = [dedup[t] for t in sorted(dedup)]
    return dict(by_sym)


def snap_at(series: list[Snap], ts: int) -> Snap | None:
    """Latest snapshot with snap.ts <= ts (trade references concurrent book)."""
    tss = [s.ts for s in series]
    i = bisect.bisect_right(tss, ts) - 1
    if i < 0:
        return None
    return series[i]


def forward_mid(series: list[Snap], ts: int, k: int) -> float | None:
    """Mid at k-th strictly future snapshot (same symbol)."""
    tss = [s.ts for s in series]
    i = bisect.bisect_right(tss, ts)
    j = i + k - 1
    if j >= len(series):
        return None
    return float(series[j].mid)


def load_trades(day: int) -> list[dict]:
    path = DATA / f"trades_round_4_day_{day}.csv"
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows.append(
                {
                    "day": day,
                    "ts": int(r["timestamp"]),
                    "buyer": (r.get("buyer") or "").strip(),
                    "seller": (r.get("seller") or "").strip(),
                    "sym": r["symbol"],
                    "price": float(r["price"]),
                    "qty": float(r["quantity"]),
                }
            )
    rows.sort(key=lambda x: (x["ts"], x["sym"]))
    return rows


def hour_bucket(ts: int) -> int:
    return (ts // 10000) % 24


def main() -> None:
    price_by_day_sym: dict[int, dict[str, list[Snap]]] = {}
    for d in DAYS:
        price_by_day_sym[d] = load_prices(d)

    # --- Bursts: (day, ts) -> list of trades
    trades_by_dt: dict[tuple[int, int], list[dict]] = defaultdict(list)
    all_trades: list[dict] = []
    for d in DAYS:
        for tr in load_trades(d):
            trades_by_dt[(d, tr["ts"])].append(tr)
            all_trades.append(tr)

    burst_keys = {k for k, v in trades_by_dt.items() if len(v) >= 3}
    burst_set = set(burst_keys)

    # --- Graph buyer -> seller
    edge_count: Counter[tuple[str, str]] = Counter()
    edge_notional: defaultdict[tuple[str, str], float] = defaultdict(float)
    for tr in all_trades:
        b, s = tr["buyer"], tr["seller"]
        if not b or not s:
            continue
        edge_count[(b, s)] += 1
        edge_notional[(b, s)] += abs(tr["price"] * tr["qty"])

    graph_lines = ["Top directed pairs (count, notional):", ""]
    for (b, s), c in edge_count.most_common(25):
        graph_lines.append(f"  {b} -> {s}: n={c}, notional={edge_notional[(b,s)]:.1f}")
    (OUT / "r4_graph_top_pairs.txt").write_text("\n".join(graph_lines))

    burst_summary = {
        "n_burst_timestamps": len(burst_set),
        "example": list(sorted(burst_set))[:15],
    }
    (OUT / "r4_burst_summary.json").write_text(json.dumps(burst_summary, indent=2))

    # --- Per-participant markouts (aggregate across days for stability)
    # key: (U, role, sym, spread_bin, hour_bin, burst) -> list of fwd deltas for each K
    # spread_bin: 0 tight <=2, 1 mid 3-5, 2 wide >=6 for traded symbol
    cells: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    cross_cells: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    def spread_bin(sp: int) -> int:
        if sp <= 2:
            return 0
        if sp <= 5:
            return 1
        return 2

    skipped = 0
    for tr in all_trades:
        d, ts, sym = tr["day"], tr["ts"], tr["sym"]
        ser = price_by_day_sym.get(d, {}).get(sym)
        if not ser:
            skipped += 1
            continue
        sn = snap_at(ser, ts)
        if sn is None:
            skipped += 1
            continue
        px = int(round(tr["price"]))
        role = "ambig"
        if px >= sn.ask:
            role = "buyer_agg"
        elif px <= sn.bid:
            role = "seller_agg"
        spb = spread_bin(sn.spread)
        hb = hour_bucket(ts)
        bur = 1 if (d, ts) in burst_set else 0
        for party in (tr["buyer"], tr["seller"]):
            if not party:
                continue
            pr = "buy" if party == tr["buyer"] else "sell"
            # participant touched trade
            for K in K_HORIZONS:
                fm = forward_mid(ser, ts, K)
                if fm is None:
                    continue
                delta = fm - sn.mid
                key = (party, pr, role, sym, spb, hb, bur)
                cells[key][K].append(delta)
            for csym in CROSS:
                if csym == sym:
                    continue
                cser = price_by_day_sym[d].get(csym)
                if not cser:
                    continue
                csn = snap_at(cser, ts)
                if csn is None:
                    continue
                for K in K_HORIZONS:
                    fm = forward_mid(cser, ts, K)
                    if fm is None:
                        continue
                    delta = fm - csn.mid
                    ckey = (party, pr, role, sym, spb, hb, bur, csym)
                    cross_cells[ckey][K].append(delta)

    def summarize(xs: list[float]) -> dict:
        if len(xs) < 8:
            return {"n": len(xs)}
        m = statistics.mean(xs)
        med = statistics.median(xs)
        pos = sum(1 for x in xs if x > 0) / len(xs)
        # Welch-style normal approx t
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        se = sd / math.sqrt(len(xs)) if len(xs) > 1 else 1.0
        tstat = m / se if se > 1e-12 else 0.0
        return {
            "n": len(xs),
            "mean": m,
            "median": med,
            "frac_pos": pos,
            "t_approx": tstat,
        }

    # Aggregate simpler: (party, sym, role) for K=20 only for screening
    screen: dict[tuple, list[float]] = defaultdict(list)
    for tr in all_trades:
        d, ts, sym = tr["day"], tr["ts"], tr["sym"]
        ser = price_by_day_sym.get(d, {}).get(sym)
        if not ser:
            continue
        sn = snap_at(ser, ts)
        if sn is None:
            continue
        px = int(round(tr["price"]))
        role = "ambig"
        if px >= sn.ask:
            role = "buyer_agg"
        elif px <= sn.bid:
            role = "seller_agg"
        K = 20
        fm = forward_mid(ser, ts, K)
        if fm is None:
            continue
        delta = fm - sn.mid
        for party in (tr["buyer"], tr["seller"]):
            if party:
                screen[(party, sym, role)].append(delta)

    screen_rows = []
    for (party, sym, role), xs in screen.items():
        if len(xs) < 15:
            continue
        s = summarize(xs)
        s["party"] = party
        s["symbol"] = sym
        s["role"] = role
        screen_rows.append(s)
    screen_rows.sort(key=lambda r: -abs(r.get("t_approx", 0)))

    (OUT / "r4_participant_screen_k20.json").write_text(
        json.dumps(screen_rows[:80], indent=2)
    )

    # --- Baseline: mean fwd by (buyer, seller, symbol) for K=20, buyer_agg only
    base_cells: dict[tuple, list[float]] = defaultdict(list)
    for tr in all_trades:
        d, ts, sym = tr["day"], tr["ts"], tr["sym"]
        ser = price_by_day_sym.get(d, {}).get(sym)
        if not ser:
            continue
        sn = snap_at(ser, ts)
        if sn is None:
            continue
        if int(round(tr["price"])) < sn.ask:
            continue  # buyer not aggressive
        fm = forward_mid(ser, ts, 20)
        if fm is None:
            continue
        base_cells[(tr["buyer"], tr["seller"], sym)].append(fm - sn.mid)

    baseline_means = {
        str(k): {"n": len(v), "mean": statistics.mean(v)}
        for k, v in base_cells.items()
        if len(v) >= 5
    }
    (OUT / "r4_baseline_buyer_agg_fwd20.json").write_text(
        json.dumps(dict(sorted(baseline_means.items(), key=lambda kv: -kv[1]["n"])[:40]), indent=2)
    )

    residuals = []
    for tr in all_trades:
        d, ts, sym = tr["day"], tr["ts"], tr["sym"]
        ser = price_by_day_sym.get(d, {}).get(sym)
        if not ser:
            continue
        sn = snap_at(ser, ts)
        if sn is None:
            continue
        if int(round(tr["price"])) < sn.ask:
            continue
        fm = forward_mid(ser, ts, 20)
        if fm is None:
            continue
        key = (tr["buyer"], tr["seller"], sym)
        if key not in base_cells or len(base_cells[key]) < 10:
            continue
        exp = statistics.mean(base_cells[key])
        residuals.append((fm - sn.mid) - exp)

    res_summary = {
        "n_residuals": len(residuals),
        "mean": statistics.mean(residuals) if residuals else None,
        "stdev": statistics.stdev(residuals) if len(residuals) > 1 else None,
    }
    (OUT / "r4_residuals_buyer_agg_fwd20.json").write_text(json.dumps(res_summary, indent=2))

    # --- Burst vs control: mean extract fwd20 after burst timestamp
    ext_series = {d: price_by_day_sym[d]["VELVETFRUIT_EXTRACT"] for d in DAYS}
    burst_fwd = []
    for d, ts in sorted(burst_set):
        ser = ext_series[d]
        sn = snap_at(ser, ts)
        if sn is None:
            continue
        fm = forward_mid(ser, ts, 20)
        if fm is not None:
            burst_fwd.append(fm - sn.mid)
    # control: random non-burst timestamps with trade
    control_fwd = []
    import random

    rng = random.Random(42)
    candidates = [(tr["day"], tr["ts"]) for tr in all_trades if (tr["day"], tr["ts"]) not in burst_set]
    sample = rng.sample(candidates, min(200, len(candidates))) if candidates else []
    for d, ts in sample:
        ser = ext_series[d]
        sn = snap_at(ser, ts)
        if sn is None:
            continue
        fm = forward_mid(ser, ts, 20)
        if fm is not None:
            control_fwd.append(fm - sn.mid)

    burst_study = {
        "burst_n": len(burst_fwd),
        "burst_mean_fwd20_extract": statistics.mean(burst_fwd) if burst_fwd else None,
        "control_n": len(control_fwd),
        "control_mean_fwd20_extract": statistics.mean(control_fwd) if control_fwd else None,
    }
    (OUT / "r4_burst_vs_control_extract_fwd20.json").write_text(
        json.dumps(burst_study, indent=2)
    )

    by_day_burst: dict[int, list[float]] = defaultdict(list)
    for d, ts in sorted(burst_set):
        ser = ext_series[d]
        sn = snap_at(ser, ts)
        if sn is None:
            continue
        fm = forward_mid(ser, ts, 20)
        if fm is not None:
            by_day_burst[d].append(fm - sn.mid)
    by_day_ctrl: dict[int, list[float]] = defaultdict(list)
    for d, ts in sample:
        if (d, ts) in burst_set:
            continue
        ser = ext_series[d]
        sn = snap_at(ser, ts)
        if sn is None:
            continue
        fm = forward_mid(ser, ts, 20)
        if fm is not None:
            by_day_ctrl[d].append(fm - sn.mid)
    day_table: dict = {}
    for d in DAYS:
        b = by_day_burst.get(d, [])
        c = by_day_ctrl.get(d, [])
        day_table[str(d)] = {
            "burst_n": len(b),
            "burst_mean": statistics.mean(b) if len(b) >= 3 else None,
            "ctrl_n": len(c),
            "ctrl_mean": statistics.mean(c) if len(c) >= 3 else None,
        }
    (OUT / "r4_burst_vs_control_by_day.json").write_text(json.dumps(day_table, indent=2))

    # --- Full cell table sample for Phase 1 doc (Mark 01, 22, VEV_5300, K=20)
    focal = []
    for tr in all_trades:
        if tr["sym"] != "VEV_5300":
            continue
        if tr["buyer"] != "Mark 01" or tr["seller"] != "Mark 22":
            continue
        d, ts = tr["day"], tr["ts"]
        ser = price_by_day_sym[d]["VEV_5300"]
        sn = snap_at(ser, ts)
        if sn is None:
            continue
        for K in K_HORIZONS:
            fm = forward_mid(ser, ts, K)
            if fm is None:
                continue
            focal.append(fm - sn.mid)
    focal_s = (
        summarize(focal)
        if len(focal) >= 8
        else {"n": len(focal), "note": "small n"}
    )
    focal_s["label"] = "VEV_5300 Mark01->Mark22 all aggressor contexts fwd mid delta"
    (OUT / "r4_focal_pair_vev5300.json").write_text(json.dumps(focal_s, indent=2))

    meta = {
        "days": list(DAYS),
        "horizons_K": list(K_HORIZONS),
        "definition": "K = K-th later price row for same (day, symbol), timestamps strictly after trade ts",
        "trades_total": len(all_trades),
        "skipped_no_snap": skipped,
        "outputs": sorted(p.name for p in OUT.glob("r4_*")),
    }
    (OUT / "r4_phase1_run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
