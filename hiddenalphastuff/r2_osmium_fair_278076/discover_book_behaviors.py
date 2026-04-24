#!/usr/bin/env python3
"""
Discover distinct quoting / liquidity behaviors for ASH_COATED_OSMIUM from a
fair-probe session (true FV + book), without assuming tomato-style "bot1/2/3".

Follows Calibration Analysis Philosophy:
  - Primary structure: price placement relative to true FV.
  - Then condition volume, side, timing on that (and joint cells).
  - Use simple tests before claiming non-uniformity; report n in every cell.

Reads:
  prices_round_2_day_1.csv
  osmium_true_fv.csv

Writes:
  osmium_behavior_discovery.txt   (human-readable report)
  osmium_level_observations.csv   (optional long-form for plotting / follow-up)
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent


def chi2_uniform(counts: dict[int, int], support: tuple[int, ...]) -> tuple[float, int]:
    """Chi^2 vs discrete uniform over `support` (categories with 0 observed included)."""
    n = sum(counts.get(k, 0) for k in support)
    if n == 0 or len(support) < 2:
        return float("nan"), 0
    e = n / len(support)
    chi2 = sum((counts.get(k, 0) - e) ** 2 / e for k in support)
    df = len(support) - 1
    return chi2, df


@dataclass
class LevelObs:
    timestamp: int
    true_fv: float
    side: str  # "bid" | "ask"
    depth_slot: int  # 1..3 = L1..L3 in activitiesLog
    price: int
    volume: int
    offset_fv: float  # price - true_fv
    offset_rf: int  # price - round(true_fv), integer grid vs rounded anchor
    offset_int: int  # round(price - true_fv): primary placement vs continuous FV


def cross_label(o: LevelObs) -> str:
    if o.side == "bid":
        if o.price > o.true_fv:
            return "bid_cross"
        if o.price < o.true_fv:
            return "bid_passive"
        return "bid_at"
    if o.price < o.true_fv:
        return "ask_cross"
    if o.price > o.true_fv:
        return "ask_passive"
    return "ask_at"


def load_fv_by_ts(fvcsv: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with fvcsv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            out[int(row["timestamp"])] = float(row["true_fv"])
    return out


def iter_level_obs(rows: list[dict[str, str]], fv_by_ts: dict[int, float]) -> list[LevelObs]:
    obs: list[LevelObs] = []
    for row in rows:
        ts = int(row["timestamp"])
        fv = fv_by_ts[ts]
        rf = round(fv)
        for side in ("bid", "ask"):
            for i in range(1, 4):
                p = row[f"{side}_price_{i}"].strip()
                v = row[f"{side}_volume_{i}"].strip()
                if not p or not v:
                    continue
                price = int(p)
                vol = int(v)
                off_f = price - fv
                obs.append(
                    LevelObs(
                        timestamp=ts,
                        true_fv=fv,
                        side=side,
                        depth_slot=i,
                        price=price,
                        volume=vol,
                        offset_fv=off_f,
                        offset_rf=price - rf,
                        offset_int=int(round(off_f)),
                    )
                )
    return obs


def resolve_data_paths(data_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Return (prices_csv, fv_csv, out_report, out_levels)."""
    data_dir = data_dir.resolve()
    prices_files = sorted(data_dir.glob("prices_round_*_day_*.csv"))
    if len(prices_files) != 1:
        raise SystemExit(
            f"Expected exactly one prices_round_*_day_*.csv in {data_dir}, got: {prices_files}"
        )
    fv = data_dir / "osmium_true_fv.csv"
    if not fv.is_file():
        raise SystemExit(f"Missing {fv}")
    return (
        prices_files[0],
        fv,
        data_dir / "osmium_behavior_discovery.txt",
        data_dir / "osmium_level_observations.csv",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="FV-anchored book behavior discovery for osmium fair logs.")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=HERE,
        help="Directory containing one prices_round_*_day_*.csv and osmium_true_fv.csv",
    )
    args = ap.parse_args()
    prices_path, fv_path, out_report, out_levels = resolve_data_paths(args.data_dir)

    fv_by_ts = load_fv_by_ts(fv_path)
    with prices_path.open(encoding="utf-8") as f:
        price_rows = list(csv.DictReader(f, delimiter=";"))

    obs = iter_level_obs(price_rows, fv_by_ts)
    n_ts = len({o.timestamp for o in obs})
    n_levels = len(obs)

    lines: list[str] = []
    W = lines.append

    W("=" * 72)
    W("ASH_COATED_OSMIUM — behavior discovery (FV-anchored, no preset bot count)")
    W("=" * 72)
    W(f"Timesteps (unique timestamps in prices file): {len(fv_by_ts)}")
    W(f"Parsed book level observations (side × depth slots × time): {n_levels}")
    W("")

    # --- Primary: placement relative to FV (integer grid vs round(FV)) ---
    W("-" * 72)
    W("PRIMARY A: round(price − true_fv)  (offset_int) — vs continuous FV")
    W("PRIMARY B: price − round(true_fv)   (offset_rf) — integer book vs integer anchor")
    W("Use A when FV is fractional so one physical layer does not split across .5 jumps.")
    W("Marginals are maps only; use conditionals below for structure.")
    W("")

    cnt_ib = Counter()
    cnt_ia = Counter()
    for o in obs:
        if o.side == "bid":
            cnt_ib[o.offset_int] += 1
        else:
            cnt_ia[o.offset_int] += 1

    def top_lines_int(title: str, c: Counter, k: int = 20) -> None:
        W(title)
        for off, n in c.most_common(k):
            W(f"  offset_int={off:+4d}  n={n}")
        W("")

    top_lines_int("BID (round(price - FV)):", cnt_ib)
    top_lines_int("ASK (round(price - FV)):", cnt_ia)
    W("")

    cnt_b = Counter()
    cnt_a = Counter()
    for o in obs:
        if o.side == "bid":
            cnt_b[o.offset_rf] += 1
        else:
            cnt_a[o.offset_rf] += 1

    W("PRIMARY B detail: price − round(FV)  (offset_rf), by side")
    W("")

    def top_lines(title: str, c: Counter, k: int = 25) -> None:
        W(title)
        for off, n in c.most_common(k):
            W(f"  offset_rf={off:+4d}  n={n}")
        W("")

    top_lines("BID level counts (how many (t, depth) slots sit at each offset_rf):", cnt_b)
    top_lines("ASK level counts:", cnt_a)

    # Persistence: at how many timestamps does *some* bid exist at offset_rf k?
    bid_ts_by_off: defaultdict[int, set[int]] = defaultdict(set)
    ask_ts_by_off: defaultdict[int, set[int]] = defaultdict(set)
    bid_ts_int: defaultdict[int, set[int]] = defaultdict(set)
    ask_ts_int: defaultdict[int, set[int]] = defaultdict(set)
    for o in obs:
        if o.side == "bid":
            bid_ts_by_off[o.offset_rf].add(o.timestamp)
            bid_ts_int[o.offset_int].add(o.timestamp)
        else:
            ask_ts_by_off[o.offset_rf].add(o.timestamp)
            ask_ts_int[o.offset_int].add(o.timestamp)

    n_ticks = len(fv_by_ts)
    W("-" * 72)
    W("PERSISTENCE: fraction of timesteps where ≥1 bid/ask level exists at offset_rf")
    W("(High persistence + tight offset → stable background layer candidate.)")
    W("")

    def persist_report(title: str, m: defaultdict[int, set[int]], key: str) -> None:
        W(title)
        offs = sorted(m.keys(), key=lambda k: -len(m[k]))
        for off in offs[:20]:
            frac = len(m[off]) / n_ticks
            W(f"  {key}={off:+4d}  timesteps={len(m[off]):4d}  ({frac*100:5.1f}% of {n_ticks})")
        W("")

    persist_report("BID (timesteps with level at offset_rf = price − round(FV)):", bid_ts_by_off, "offset_rf")
    persist_report("ASK:", ask_ts_by_off, "offset_rf")
    W("PERSISTENCE vs offset_int = round(price − FV):")
    persist_report("BID:", bid_ts_int, "offset_int")
    persist_report("ASK:", ask_ts_int, "offset_int")

    # --- CONDITION volume | (side, offset_rf) ---
    W("-" * 72)
    W("CONDITIONAL: volume given (side, offset_rf)")
    W("Uniform volume on a support only where category counts justify a test.")
    W("")

    vol_cell_rf: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    vol_cell_int: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    for o in obs:
        vol_cell_rf[(o.side, o.offset_rf)].append(o.volume)
        vol_cell_int[(o.side, o.offset_int)].append(o.volume)

    MIN_N = 30  # guardrail for "don't trust the histogram"
    for label, cell in [("offset_rf (price - round(FV))", vol_cell_rf), ("offset_int (round(price-FV))", vol_cell_int)]:
        W(f"CONDITIONAL: volume | (side, {label})")
        for side in ("bid", "ask"):
            W(f"  Side = {side}")
            offs = sorted(set(off for s, off in cell if s == side))
            cells = [(off, cell[(side, off)]) for off in offs]
            cells.sort(key=lambda x: -len(x[1]))
            for off, vols in cells:
                n = len(vols)
                if n < MIN_N:
                    W(f"    off={off:+4d}  n={n:4d}  (skip chi²; n < {MIN_N})")
                    continue
                lo, hi = min(vols), max(vols)
                support = tuple(range(lo, hi + 1))
                cdict = Counter(vols)
                chi2, df = chi2_uniform(cdict, support)
                W(
                    f"    off={off:+4d}  n={n:4d}  vol {lo}..{hi}  mean={statistics.mean(vols):.2f}  "
                    f"chi2_uniform={chi2:.2f} df={df}"
                )
        W("")

    # --- CONDITION volume | crossing (price vs true FV), within side ---
    W("-" * 72)
    W("CONDITIONAL: volume | (side, aggressive vs passive vs at_fv)")
    W("Aggressive bid: price > FV; passive bid: price < FV; at: price == FV (rare float).")
    W("Aggressive ask: price < FV; passive ask: price > FV.")
    W("(Reveals whether a marginal volume law is two processes — e.g. large aggressive vs small passive.)")
    W("")

    vol_cross: defaultdict[str, list[int]] = defaultdict(list)
    for o in obs:
        vol_cross[cross_label(o)].append(o.volume)

    for lab in sorted(vol_cross.keys()):
        vs = vol_cross[lab]
        n = len(vs)
        W(f"  {lab:<12} n={n:5d}  mean_vol={statistics.mean(vs):.2f}  min..max={min(vs)}..{max(vs)}")

    W("")
    W("Compare crossing vs passive within same side (rough z on mean ranks skipped;")
    W("use nonparametric or formal test offline if n large).")
    # Simple: bid_cross vs bid_passive mean volume
    bc = vol_cross.get("bid_cross", [])
    bp = vol_cross.get("bid_passive", [])
    ac = vol_cross.get("ask_cross", [])
    ap = vol_cross.get("ask_passive", [])
    if len(bc) > 10 and len(bp) > 10:
        W(
            f"  bid_cross mean={statistics.mean(bc):.2f} (n={len(bc)})  "
            f"bid_passive mean={statistics.mean(bp):.2f} (n={len(bp)})"
        )
    if len(ac) > 10 and len(ap) > 10:
        W(
            f"  ask_cross mean={statistics.mean(ac):.2f} (n={len(ac)})  "
            f"ask_passive mean={statistics.mean(ap):.2f} (n={len(ap)})"
        )
    W("")

    # --- Joint: volume | (side, offset_rf bucket coarse) already above ---
    # --- Timing: gap between appearances of same (side, offset_rf) ---
    W("-" * 72)
    W("TIMING (coarse): inter-arrival of timesteps with ANY bid at offset_rf = best bid touch")
    W("(proxy for inner quote refresh if you pick dominant inner offset from persistence table).")
    W("")

    dom_b = cnt_ib.most_common(1)[0][0] if cnt_ib else None
    dom_a = cnt_ia.most_common(1)[0][0] if cnt_ia else None
    if dom_b is not None:
        ts_list = sorted(bid_ts_int[dom_b])
        gaps = [b - a for a, b in zip(ts_list, ts_list[1:])]
        if gaps:
            W(
                f"Dominant bid offset_int={dom_b}: inter-timestep gaps "
                f"(100 = 1 step): min={min(gaps)} median={statistics.median(gaps)} max={max(gaps)}"
            )
    if dom_a is not None:
        ts_list = sorted(ask_ts_int[dom_a])
        gaps = [b - a for a, b in zip(ts_list, ts_list[1:])]
        if gaps:
            W(
                f"Dominant ask offset_int={dom_a}: inter-timestep gaps: "
                f"min={min(gaps)} median={statistics.median(gaps)} max={max(gaps)}"
            )
    W("")

    # --- Calibration philosophy: explicit checklist (marginals vs conditionals) ---
    W("=" * 72)
    W("PHILOSOPHY CHECKLIST: conditionals (never trust pooled marginals alone)")
    W("Reference: hiddenalphastuff/ANALYSIS_PHILOSOPHY.md")
    W("")

    MIN_CELL = 35  # slightly lower than chi² block; still report with caveat if 20–34

    def pstdev_safe(xs: list[int]) -> float:
        return statistics.pstdev(xs) if len(xs) > 1 else 0.0

    def gap_summary(ts_sorted: list[int]) -> str:
        if len(ts_sorted) < 2:
            return "n_ts<2"
        gaps = [ts_sorted[i + 1] - ts_sorted[i] for i in range(len(ts_sorted) - 1)]
        gaps_s = sorted(gaps)
        p90_i = min(len(gaps_s) - 1, int(0.9 * (len(gaps_s) - 1)))
        p90 = gaps_s[p90_i]
        return (
            f"n_gap={len(gaps)}  min={min(gaps)}  med={statistics.median(gaps):.0f}  "
            f"p90={p90}  max={max(gaps)}"
        )

    # volume | side (POOLED — marginal that can hide two processes)
    W("-" * 72)
    W("volume | side   (POOLED over all prices / depths — compare to blocks below)")
    W("")
    for side in ("bid", "ask"):
        vs = [o.volume for o in obs if o.side == side]
        if vs:
            W(
                f"  {side}: n={len(vs):5d}  mean_vol={statistics.mean(vs):.2f}  "
                f"median={statistics.median(vs):.1f}  min..max={min(vs)}..{max(vs)}"
            )
    W("  If this looks 'simple' but volume|(side,offset) is bimodal, the marginal lied.")
    W("")

    # volume | (side, absolute price)
    vol_sp: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    ts_sp: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    for o in obs:
        k = (o.side, o.price)
        vol_sp[k].append(o.volume)
        ts_sp[k].append(o.timestamp)

    W("-" * 72)
    W("volume | (side, absolute price)   — top cells by count (discrete price ticks)")
    W("")
    ranked = sorted(vol_sp.items(), key=lambda kv: -len(kv[1]))
    shown = 0
    for (side, price), vols in ranked:
        n = len(vols)
        if n < MIN_CELL:
            break
        W(
            f"  {side} price={price:6d}  n={n:5d}  mean_vol={statistics.mean(vols):.2f}  "
            f"std={pstdev_safe(vols):.2f}  range={min(vols)}..{max(vols)}"
        )
        shown += 1
        if shown >= 18:
            break
    W("")

    W("-" * 72)
    W("volume | price   (POOLED across sides — FV moves, so absolute price mixes regimes)")
    W("(Prefer volume|(side,price) or placement-offset conditionals for structural inference.)")
    W("")
    vol_price: defaultdict[int, list[int]] = defaultdict(list)
    ts_price: defaultdict[int, list[int]] = defaultdict(list)
    for o in obs:
        vol_price[o.price].append(o.volume)
        ts_price[o.price].append(o.timestamp)
    for price, vols in sorted(vol_price.items(), key=lambda kv: -len(kv[1]))[:12]:
        if len(vols) < MIN_CELL:
            break
        W(
            f"  price={price:6d}  n={len(vols):5d}  mean_vol={statistics.mean(vols):.2f}  "
            f"std={pstdev_safe(vols):.2f}  range={min(vols)}..{max(vols)}"
        )
    W("")

    W("-" * 72)
    W("timing | price   (pooled sides — inter-arrival at each absolute price tick)")
    W("")
    for price, vols in sorted(vol_price.items(), key=lambda kv: -len(kv[1]))[:12]:
        if len(vols) < MIN_CELL:
            break
        tsu = sorted(set(ts_price[price]))
        W(f"  price={price:6d}  {gap_summary(tsu)}")
    W("")

    W("-" * 72)
    W("price | side   (distribution of quoted price conditional on side)")
    W("")
    for side in ("bid", "ask"):
        ps = [o.price for o in obs if o.side == side]
        if len(ps) < 2:
            continue
        W(
            f"  {side}: n={len(ps)}  mean_price={statistics.mean(ps):.2f}  "
            f"std={pstdev_safe(ps):.2f}  min={min(ps)}  max={max(ps)}"
        )
    W("")

    W("-" * 72)
    W("timing | (side, absolute price)   inter-arrival of timestamps with that quote")
    W("")
    for (side, price), vols in ranked[:25]:
        n = len(vols)
        if n < MIN_CELL:
            continue
        tsu = sorted(set(ts_sp[(side, price)]))
        W(f"  {side} price={price:6d}  {gap_summary(tsu)}")

    W("")
    W("-" * 72)
    W("timing | (side, offset_int)   same, for structural offsets (inner + wall)")
    W("")
    timing_pairs = [
        ("bid", -11),
        ("bid", -10),
        ("bid", -8),
        ("ask", 8),
        ("ask", 10),
        ("ask", 11),
    ]
    for side, off in timing_pairs:
        tsu = sorted({o.timestamp for o in obs if o.side == side and o.offset_int == off})
        if len(tsu) < 2:
            W(f"  {side} offset_int={off:+3d}  too_few_timesteps")
            continue
        W(f"  {side} offset_int={off:+3d}  {gap_summary(tsu)}")

    W("")

    W("-" * 72)
    W("volume | (side, offset_int, crossing)   — split each placement bucket by FV side")
    W("(If mean volume shifts here, a 'uniform' marginal at that offset was hiding two processes.)")
    W("")
    struct_cross = [("bid", -11), ("bid", -10), ("bid", -8), ("ask", 8), ("ask", 10), ("ask", 11)]
    for side, off in struct_cross:
        sub = [o for o in obs if o.side == side and o.offset_int == off]
        if len(sub) < MIN_CELL:
            W(f"  {side} offset_int={off:+4d}  n={len(sub)}  (skip split; n < {MIN_CELL})")
            continue
        by_c: dict[str, list[int]] = defaultdict(list)
        for o in sub:
            by_c[cross_label(o)].append(o.volume)
        W(f"  {side} offset_int={off:+4d}  n={len(sub)}")
        for cl in sorted(by_c):
            vs = by_c[cl]
            W(
                f"      {cl:<12} n={len(vs):4d}  mean_vol={statistics.mean(vs):.2f}  "
                f"min..max={min(vs)}..{max(vs)}"
            )
    W("")

    W("-" * 72)
    W("volume | (side, depth_slot)   L1 vs L2 vs L3 (same side, all offsets pooled)")
    W("")
    for side in ("bid", "ask"):
        for slot in (1, 2, 3):
            vs = [o.volume for o in obs if o.side == side and o.depth_slot == slot]
            if not vs:
                continue
            W(
                f"  {side} L{slot}: n={len(vs):5d}  mean_vol={statistics.mean(vs):.2f}  "
                f"median={statistics.median(vs):.1f}  range={min(vs)}..{max(vs)}"
            )
    W("")

    W("-" * 72)
    W("volume | (side, offset_int, depth_slot)   — inner/wall only (|offset| in 8,10,11)")
    W("")
    for side in ("bid", "ask"):
        relevant_offs = (-11, -10, -8) if side == "bid" else (8, 10, 11)
        for off in relevant_offs:
            for slot in (1, 2, 3):
                vs = [
                    o.volume
                    for o in obs
                    if o.side == side and o.offset_int == off and o.depth_slot == slot
                ]
                if len(vs) < 15:
                    continue
                W(
                    f"  {side} off={off:+4d} L{slot}  n={len(vs):4d}  "
                    f"mean_vol={statistics.mean(vs):.2f}  range={min(vs)}..{max(vs)}"
                )
        W("")

    W("-" * 72)
    W("PAIRWISE REMINDER")
    W("  Any cell above with low n is *not* 'done' — pool another fair day or widen bins.")
    W("  If volume|(side,off,cross) differs from volume|(side,off), model crossing as explicit state.")
    W("")

    # --- Candidate "behaviors" (names are ours, not IMC's) ---
    W("=" * 72)
    W("CANDIDATE BEHAVIORS (rename after you validate rules)")
    W("Assign names only after a proposed rule hits high match rate on a second day.")
    W("")
    W("Heuristic from THIS pass only:")
    W("  • Layers with ≥~95% timestep persistence at a fixed offset_rf → background makers.")
    W("  • Offsets with moderate persistence + volume law differing by crossing → taker residue / inner.")
    W("  • Rare offsets + short runs → opportunistic third process (cf. tomato 'near-FV').")
    W("")

    persist_sorted = []
    for off in bid_ts_int:
        frac = len(bid_ts_int[off]) / n_ticks
        persist_sorted.append(("bid", "int", off, frac, len(bid_ts_int[off])))
    for off in ask_ts_int:
        frac = len(ask_ts_int[off]) / n_ticks
        persist_sorted.append(("ask", "int", off, frac, len(ask_ts_int[off])))
    persist_sorted.sort(key=lambda x: -x[3])
    W("Top persistence (side, offset_int, frac):")
    for side, _k, off, frac, kk in persist_sorted[:14]:
        W(f"  {side:3s}  offset_int={off:+4d}  persist={frac*100:5.1f}%  timesteps={kk}")
    W("")

    out_report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_report}")

    with out_levels.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(
            [
                "timestamp",
                "true_fv",
                "side",
                "depth_slot",
                "price",
                "volume",
                "offset_fv",
                "offset_rf",
                "offset_int",
                "cross_label",
            ]
        )
        for o in obs:
            w.writerow(
                [
                    o.timestamp,
                    f"{o.true_fv:.10f}",
                    o.side,
                    o.depth_slot,
                    o.price,
                    o.volume,
                    f"{o.offset_fv:.10f}",
                    o.offset_rf,
                    o.offset_int,
                    cross_label(o),
                ]
            )
    print(f"Wrote {out_levels} ({len(obs)} rows)")


if __name__ == "__main__":
    main()
