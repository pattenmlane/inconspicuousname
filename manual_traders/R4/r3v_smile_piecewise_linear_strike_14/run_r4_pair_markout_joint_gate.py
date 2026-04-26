#!/usr/bin/env python3
"""
Round 4 — Extract forward markouts by (buyer -> seller) stratified by Sonic joint-tight
at trade time (5200 & 5300 L1 spread <=2 from concurrent price snapshots).

Uses Phase-1 conventions: snap_at(ts) = last snapshot with ts<=trade ts; forward_mid =
K-th strictly future snapshot mid - mid at snap.

Outputs: analysis_outputs/r4_pair_extract_fwd_joint_gate.json
"""
from __future__ import annotations

import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("Prosperity4Data/ROUND_4")

DAYS = (1, 2, 3)
K_LIST = (5, 20, 100)
EXTRACT = "VELVETFRUIT_EXTRACT"
S5200 = "VEV_5200"
S5300 = "VEV_5300"
TH = 2


class Snap:
    __slots__ = ("ts", "mid", "bid", "ask", "spread")

    def __init__(self, ts: int, mid: float, bid: int, ask: int, spread: int) -> None:
        self.ts = ts
        self.mid = mid
        self.bid = bid
        self.ask = ask
        self.spread = spread


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
        dedup: dict[int, Snap] = {}
        for s in by_sym[sym]:
            dedup[s.ts] = s
        by_sym[sym] = [dedup[t] for t in sorted(dedup)]
    return dict(by_sym)


def snap_at(series: list[Snap], ts: int) -> Snap | None:
    tss = [s.ts for s in series]
    i = bisect.bisect_right(tss, ts) - 1
    if i < 0:
        return None
    return series[i]


def forward_mid(series: list[Snap], ts: int, k: int) -> float | None:
    tss = [s.ts for s in series]
    i = bisect.bisect_right(tss, ts)
    j = i + k - 1
    if j >= len(series):
        return None
    return float(series[j].mid)


def joint_tight(day_sym: dict[str, list[Snap]], ts: int) -> bool | None:
    a = snap_at(day_sym.get(S5200, []), ts)
    b = snap_at(day_sym.get(S5300, []), ts)
    if a is None or b is None:
        return None
    return a.spread <= TH and b.spread <= TH


def main() -> None:
    by_day: dict[int, dict[str, list[Snap]]] = {d: load_prices(d) for d in DAYS}
    ext = {d: by_day[d][EXTRACT] for d in DAYS}

    # (pair, tight, K) -> list of deltas
    cells: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    per_day: dict[int, dict[tuple[str, int, int], list[float]]] = {
        d: defaultdict(list) for d in DAYS
    }
    n_skip = 0

    for d in DAYS:
        ser_e = ext[d]
        if not ser_e:
            continue
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["symbol"] != EXTRACT:
                    continue
                ts = int(r["timestamp"])
                buyer = (r.get("buyer") or "").strip()
                seller = (r.get("seller") or "").strip()
                if not buyer or not seller:
                    n_skip += 1
                    continue
                sn = snap_at(ser_e, ts)
                if sn is None:
                    n_skip += 1
                    continue
                jt = joint_tight(by_day[d], ts)
                if jt is None:
                    n_skip += 1
                    continue
                tight = 1 if jt else 0
                pair = f"{buyer}->{seller}"
                for K in K_LIST:
                    fm = forward_mid(ser_e, ts, K)
                    if fm is None:
                        continue
                    delta = fm - sn.mid
                    key = (pair, tight, K)
                    cells[key].append(delta)
                    per_day[d][key].append(delta)

    def summarize(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
        return {"n": len(xs), "mean": m, "std": v**0.5}

    # Pooled: top pairs by n in tight K=20
    tight_counts: dict[str, int] = defaultdict(int)
    for (pair, tight, K), xs in cells.items():
        if tight == 1 and K == 20:
            tight_counts[pair] += len(xs)

    top_pairs = sorted(tight_counts.keys(), key=lambda p: -tight_counts[p])[:15]

    out: dict = {
        "method": "snap_at concurrent book; joint_tight = s5200<=2 and s5300<=2 at same ts; fwd = mid[t+K]-mid[t] extract",
        "n_skipped_no_book_or_gate": n_skip,
        "top_pairs_tight_k20": [{"pair": p, "n": tight_counts[p]} for p in top_pairs],
        "pooled_pair_gate": {},
        "per_day_pair_gate": {},
    }

    for p in top_pairs:
        out["pooled_pair_gate"][p] = {}
        for tight in (0, 1):
            out["pooled_pair_gate"][p][f"tight_{tight}"] = {
                str(K): summarize(cells.get((p, tight, K), [])) for K in K_LIST
            }

    out["per_day_pair_gate"] = {}
    for d in DAYS:
        out["per_day_pair_gate"][str(d)] = {}
        for p in top_pairs:
            out["per_day_pair_gate"][str(d)][p] = {}
            for tight in (0, 1):
                pd_cells = per_day[d]
                out["per_day_pair_gate"][str(d)][p][f"tight_{tight}"] = {
                    str(K): summarize(pd_cells.get((p, tight, K), [])) for K in K_LIST
                }

    # Welch-style summary for top pair Mark14->55 tight vs loose K=20 if enough n
    def welch_simple(a: list[float], b: list[float]) -> dict:
        if len(a) < 2 or len(b) < 2:
            return {"na": len(a), "nb": len(b)}
        import statistics

        ma, mb = statistics.mean(a), statistics.mean(b)
        va = statistics.pvariance(a) if len(a) > 1 else 0.0
        vb = statistics.pvariance(b) if len(b) > 1 else 0.0
        se = (va / len(a) + vb / len(b)) ** 0.5
        if se < 1e-12:
            return {"na": len(a), "nb": len(b), "mean_a": ma, "mean_b": mb}
        t = (ma - mb) / se
        return {"na": len(a), "nb": len(b), "mean_a": ma, "mean_b": mb, "t_approx": t}

    focal = "Mark 14->Mark 55"
    if focal in tight_counts:
        ta = cells.get((focal, 1, 20), [])
        la = cells.get((focal, 0, 20), [])
        out["focal_Mark14_to_55_k20_tight_vs_loose"] = welch_simple(ta, la)

    pth = OUT / "r4_pair_extract_fwd_joint_gate.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
