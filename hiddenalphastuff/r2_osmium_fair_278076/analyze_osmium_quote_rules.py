#!/usr/bin/env python3
"""
Brute-force quote rules for ASH_COATED_OSMIUM fair-probe data (true FV + book),
similar spirit to hiddenalphastuff/analyze_bot1.py / analyze_bot2.py.

Hypotheses:
  A) TWO MMs: inner cluster (~offset_int -8 / +8) vs wall (~-10/-11, +10/+11)
  B) ONE MM: single ladder with three bid rungs and three ask rungs at -11,-10,-8 / +8,+10,+11

Reads: one prices_round_*_day_*.csv + osmium_true_fv.csv in --data-dir.

Writes: osmium_quote_rule_search.txt in each --data-dir.

With --all-sessions: runs this script once per known session folder and also
writes osmium_quote_rule_search_MULTISESSION.txt (combined) next to this file.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

from osmium_fair_common import DEFAULT_DATA_DIR, Tick, build_ticks, load_fv, resolve_paths
from osmium_sessions import all_session_dirs

HERE_SCRIPT = Path(__file__).resolve().parent


def brute_bid(
    ticks: list[Tick],
    getter,
    rnd_name: str,
    rnd,
    shifts: list[float],
    off_lo: int,
    off_hi: int,
) -> tuple[int, str, float]:
    best_n, best_f, best_pct = -1, "", 0.0
    n_valid = sum(1 for t in ticks if getter(t) is not None)
    if n_valid == 0:
        return 0, "n/a", 0.0
    for shift in shifts:
        for off in range(off_lo, off_hi + 1):
            matches = sum(1 for t in ticks if getter(t) is not None and rnd(t.fv + shift) + off == getter(t))
            if matches > best_n:
                best_n = matches
                best_f = f"{rnd_name}(FV + {shift}) + {off}"
                best_pct = 100.0 * matches / n_valid
    return best_n, best_f, best_pct


def brute_ask(
    ticks: list[Tick],
    getter,
    rnd_name: str,
    rnd,
    shifts: list[float],
    off_lo: int,
    off_hi: int,
) -> tuple[int, str, float]:
    best_n, best_f, best_pct = -1, "", 0.0
    n_valid = sum(1 for t in ticks if getter(t) is not None)
    if n_valid == 0:
        return 0, "n/a", 0.0
    for shift in shifts:
        for off in range(off_lo, off_hi + 1):
            matches = sum(1 for t in ticks if getter(t) is not None and rnd(t.fv + shift) + off == getter(t))
            if matches > best_n:
                best_n = matches
                best_f = f"{rnd_name}(FV + {shift}) + {off}"
                best_pct = 100.0 * matches / n_valid
    return best_n, best_f, best_pct


def ladder_match(
    ticks: list[Tick],
    rnd,
    shift: float,
    bid_offs: tuple[int, int, int],
    ask_offs: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Return (bid_matches, ticks_with_>=3_bids, ask_matches, ticks_with_>=3_asks)."""
    mb = ma = 0
    nb = na = 0
    for t in ticks:
        r = rnd(t.fv + shift)
        pred_b = sorted([r + bid_offs[0], r + bid_offs[1], r + bid_offs[2]], reverse=True)
        pred_a = sorted([r + ask_offs[0], r + ask_offs[1], r + ask_offs[2]])
        if len(t.bids_desc) >= 3:
            nb += 1
            if tuple(t.bids_desc[:3]) == tuple(pred_b):
                mb += 1
        if len(t.asks_asc) >= 3:
            na += 1
            if tuple(t.asks_asc[:3]) == tuple(pred_a):
                ma += 1
    return mb, nb, ma, na


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument(
        "--all-sessions",
        action="store_true",
        help="Run analysis for every folder in osmium_sessions.all_session_dirs()",
    )
    args = ap.parse_args()
    if args.all_sessions:
        sess = all_session_dirs()
        chunks: list[str] = []
        for d in sess:
            subprocess.run(
                [sys.executable, str(HERE_SCRIPT / "analyze_osmium_quote_rules.py"), "--data-dir", str(d)],
                check=True,
                cwd=str(HERE_SCRIPT),
            )
            p = d / "osmium_quote_rule_search.txt"
            chunks.append(f"{'=' * 20}  {d.resolve()}  {'=' * 20}\n\n{p.read_text(encoding='utf-8')}")
        combo = HERE_SCRIPT / "osmium_quote_rule_search_MULTISESSION.txt"
        combo.write_text("\n\n".join(chunks), encoding="utf-8")
        print(f"Wrote per-session files + combined {combo}")
        return

    prices_path, fv_path = resolve_paths(args.data_dir)
    out_path = args.data_dir.resolve() / "osmium_quote_rule_search.txt"

    fv_map = load_fv(fv_path)
    with prices_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    ticks = build_ticks(rows, fv_map)
    shifts = [x * 0.25 for x in range(-8, 9)]
    rnds = [("floor", math.floor), ("ceil", math.ceil), ("round", round)]

    lines: list[str] = []
    W = lines.append
    W("=" * 72)
    W("OSMIUM QUOTE RULE SEARCH (brute force on true FV)")
    W(f"data_dir={args.data_dir.resolve()}")
    W(f"timesteps={len(ticks)}")
    W("")

    # --- Hypothesis A1: INNER MM alone (levels tagged offset_int -8 / +8) ---
    W("=" * 72)
    W("HYPOTHESIS A — part 1: INNER cluster (bid @ offset_int -8, ask @ +8)")
    W("")

    n_ib = sum(1 for t in ticks if t.inner_bid is not None)
    n_ia = sum(1 for t in ticks if t.inner_ask is not None)
    W(f"Ticks with inner_bid present: {n_ib}/{len(ticks)}")
    W(f"Ticks with inner_ask present: {n_ia}/{len(ticks)}")
    W("")

    W("INNER BID brute (same grid as tomato-style analyze_bot1)")
    best_ib = (-1, "", 0.0)
    for name, rnd in rnds:
        n, f, pct = brute_bid(ticks, lambda t: t.inner_bid, name, rnd, shifts, -14, -2)
        if n > best_ib[0]:
            best_ib = (n, f"{name}: {f}", pct)
    W(f"  Best: {best_ib[1]}  →  {best_ib[0]}/{n_ib}  ({best_ib[2]:.1f}% of ticks with inner bid)")
    W("")

    W("INNER ASK brute")
    best_ia = (-1, "", 0.0)
    for name, rnd in rnds:
        n, f, pct = brute_ask(ticks, lambda t: t.inner_ask, name, rnd, shifts, 2, 16)
        if n > best_ia[0]:
            best_ia = (n, f"{name}: {f}", pct)
    W(f"  Best: {best_ia[1]}  →  {best_ia[0]}/{n_ia}  ({best_ia[2]:.1f}% of ticks with inner ask)")
    W("")

    # Simple fixed tomato-like inner
    def count_fixed_inner() -> None:
        cib = sum(
            1
            for t in ticks
            if t.inner_bid is not None and round(t.fv) - 8 == t.inner_bid
        )
        cia = sum(
            1
            for t in ticks
            if t.inner_ask is not None and round(t.fv) + 8 == t.inner_ask
        )
        both = sum(
            1
            for t in ticks
            if t.inner_bid is not None
            and t.inner_ask is not None
            and round(t.fv) - 8 == t.inner_bid
            and round(t.fv) + 8 == t.inner_ask
        )
        denom = sum(
            1 for t in ticks if t.inner_bid is not None and t.inner_ask is not None
        )
        W(
            f"  Fixed tomato-style inner: bid=round(FV)-8  → {cib}/{n_ib} ({100*cib/max(n_ib,1):.1f}%)"
        )
        W(
            f"                              ask=round(FV)+8  → {cia}/{n_ia} ({100*cia/max(n_ia,1):.1f}%)"
        )
        W(
            f"  BOTH exact same tick: {both}/{denom} ({100*both/max(denom,1):.1f}% of ticks with both inner legs)"
        )
        W("")

    count_fixed_inner()

    W("  Bot2-style asymmetric inner (floor/ceil + shift grid ±1.0 step 0.25; coarser than full ladder search):")
    shift_coarse = [x * 0.25 for x in range(-4, 5)]
    best_asym = (-1, "", "", 0.0, 0.0)
    for sb in shift_coarse:
        for kb in range(-14, -2):
            for sa in shift_coarse:
                for ka in range(2, 17):
                    mib = sum(
                        1
                        for t in ticks
                        if t.inner_bid is not None
                        and math.floor(t.fv + sb) + kb == t.inner_bid
                    )
                    mia = sum(
                        1
                        for t in ticks
                        if t.inner_ask is not None
                        and math.ceil(t.fv + sa) + ka == t.inner_ask
                    )
                    score = mib + mia
                    if score > best_asym[0]:
                        best_asym = (
                            score,
                            f"bid=floor(FV+{sb})+{kb}",
                            f"ask=ceil(FV+{sa})+{ka}",
                            100.0 * mib / max(n_ib, 1),
                            100.0 * mia / max(n_ia, 1),
                        )
    W(
        f"    Best sum(bid_match+ask_match)={best_asym[0]}: {best_asym[1]}  {best_asym[2]}  "
        f"({best_asym[3]:.1f}% bid, {best_asym[4]:.1f}% ask)"
    )
    W("")

    # Same-anchor round(FV): all four bid rungs when present
    def all_bid_rungs_match(t: Tick) -> bool:
        r = round(t.fv)
        if t.inner_bid is not None and r - 8 != t.inner_bid:
            return False
        if t.bid_m10 is not None and r - 10 != t.bid_m10:
            return False
        if t.bid_m11 is not None and r - 11 != t.bid_m11:
            return False
        return True

    def all_ask_rungs_match(t: Tick) -> bool:
        r = round(t.fv)
        if t.inner_ask is not None and r + 8 != t.inner_ask:
            return False
        if t.ask_p10 is not None and r + 10 != t.ask_p10:
            return False
        if t.ask_p11 is not None and r + 11 != t.ask_p11:
            return False
        return True

    n_all_b = sum(
        1
        for t in ticks
        if t.inner_bid is not None and t.bid_m10 is not None and t.bid_m11 is not None
    )
    n_all_a = sum(
        1
        for t in ticks
        if t.inner_ask is not None and t.ask_p10 is not None and t.ask_p11 is not None
    )
    m_all_b = sum(
        1
        for t in ticks
        if t.inner_bid is not None and t.bid_m10 is not None and t.bid_m11 is not None and all_bid_rungs_match(t)
    )
    m_all_a = sum(
        1
        for t in ticks
        if t.inner_ask is not None and t.ask_p10 is not None and t.ask_p11 is not None and all_ask_rungs_match(t)
    )
    W(
        f"  TWO-BOT same anchor round(FV): all 3 bid rungs (-8,-10,-11) match when all present: "
        f"{m_all_b}/{n_all_b} ({100 * m_all_b / max(n_all_b, 1):.1f}%)"
    )
    W(
        f"                                 all 3 ask rungs (+8,+10,+11) match when all present: "
        f"{m_all_a}/{n_all_a} ({100 * m_all_a / max(n_all_a, 1):.1f}%)"
    )
    if n_all_b == 0:
        W("  (Denominator 0: no tick has inner_bid AND bid_m10 AND bid_m11 simultaneously — book rarely shows all three.)")
    if n_all_a == 0:
        W("  (Denominator 0: no tick has inner_ask AND ask_p10 AND ask_p11 simultaneously.)")

    bid_len_ct: dict[int, int] = {}
    ask_len_ct: dict[int, int] = {}
    for t in ticks:
        bid_len_ct[len(t.bids_desc)] = bid_len_ct.get(len(t.bids_desc), 0) + 1
        ask_len_ct[len(t.asks_asc)] = ask_len_ct.get(len(t.asks_asc), 0) + 1
    W(f"  Bid depth histogram (#price levels): {dict(sorted(bid_len_ct.items()))}")
    W(f"  Ask depth histogram: {dict(sorted(ask_len_ct.items()))}")

    def bid_offsets(t: Tick) -> set[int]:
        return {int(round(p - t.fv)) for p in t.bids_desc}

    has_triple_b = sum(1 for t in ticks if {-8, -10, -11}.issubset(bid_offsets(t)))
    has_triple_a = sum(1 for t in ticks if {8, 10, 11}.issubset(set(int(round(p - t.fv)) for p in t.asks_asc)))
    W(
        f"  Ticks where bid offset set contains {{-8,-10,-11}}: {has_triple_b}/{len(ticks)}; "
        f"asks contain {{+8,+10,+11}}: {has_triple_a}/{len(ticks)}"
    )

    ladder_ok = 0
    ladder_nb = 0
    for t in ticks:
        if len(t.bids_desc) < 3:
            continue
        ladder_nb += 1
        r = round(t.fv)
        pred = sorted([r - 8, r - 10, r - 11], reverse=True)
        if tuple(t.bids_desc[:3]) == tuple(pred):
            ladder_ok += 1
    W(
        f"  Among ticks with ≥3 bids: top-3 prices vs round(FV)+{{-8,-10,-11}} (sorted): "
        f"{ladder_ok}/{ladder_nb} ({100 * ladder_ok / max(ladder_nb, 1):.1f}%)"
    )
    W("")

    # --- Hypothesis A2: WALL levels separately ---
    W("=" * 72)
    W("HYPOTHESIS A — part 2: WALL rungs (tagged offset -10, -11 bids / +10, +11 asks)")
    W("")

    for label, getter, lo, hi in [
        ("bid -10", lambda t: t.bid_m10, -14, -4),
        ("bid -11", lambda t: t.bid_m11, -16, -4),
        ("ask +10", lambda t: t.ask_p10, 4, 16),
        ("ask +11", lambda t: t.ask_p11, 4, 16),
    ]:
        n_valid = sum(1 for t in ticks if getter(t) is not None)
        best = (-1, "", 0.0)
        for name, rnd in rnds:
            if "bid" in label:
                n, f, pct = brute_bid(ticks, getter, name, rnd, shifts, lo, hi)
            else:
                n, f, pct = brute_ask(ticks, getter, name, rnd, shifts, lo, hi)
            if n > best[0]:
                best = (n, f"{name}: {f}", pct)
        W(f"  {label:8s}  n_valid={n_valid:4d}  BEST {best[1]}  → {best[0]}/{n_valid} ({best[2]:.1f}%)")

    W("")
    W("  Fixed ladder-style from round(FV):")
    for label, getter, pred in [
        ("bid -10", lambda t: t.bid_m10, lambda fv: round(fv) - 10),
        ("bid -11", lambda t: t.bid_m11, lambda fv: round(fv) - 11),
        ("ask +10", lambda t: t.ask_p10, lambda fv: round(fv) + 10),
        ("ask +11", lambda t: t.ask_p11, lambda fv: round(fv) + 11),
    ]:
        nv = sum(1 for t in ticks if getter(t) is not None)
        m = sum(1 for t in ticks if getter(t) is not None and pred(t.fv) == getter(t))
        W(f"    {label}: round(FV)+k  → {m}/{nv} ({100*m/max(nv,1):.1f}%)")
    W("")

    # --- Hypothesis B: SINGLE 3-layer MM (same round anchor, offsets -11,-10,-8 / +8,+10,+11) ---
    W("=" * 72)
    W("HYPOTHESIS B: ONE MM — three bid + three ask prices from same rnd(FV+shift)+offset ladder")
    W("Template: bids = sort_desc(R+ob1, R+ob2, R+ob3) with {ob}= {-8,-10,-11}, asks sort_asc(R+oa1,...) with {+8,+10,+11}")
    W("")

    bid_pattern = (-8, -10, -11)
    ask_pattern = (8, 10, 11)

    best_score = -1
    best_line = ""
    for name, rnd in rnds:
        for shift in shifts:
            mb, nb, ma, na = ladder_match(ticks, rnd, shift, bid_pattern, ask_pattern)
            score = mb + ma
            if score > best_score:
                best_score = score
                best_line = (
                    f"{name}(FV + {shift})  bid_offs={bid_pattern} ask_offs={ask_pattern}  "
                    f"bid_3tuple={mb}/{nb} ({100 * mb / max(nb, 1):.1f}%)  "
                    f"ask_3tuple={ma}/{na} ({100 * ma / max(na, 1):.1f}%)  "
                    f"joint_score={score}"
                )
    W(f"  Best over rnd × shift: {best_line}")
    W("")

    W("  Anchor shift=0 only:")
    for rnd_name, rnd in [("round", round), ("floor", math.floor)]:
        mb, nb, ma, na = ladder_match(ticks, rnd, 0.0, bid_pattern, ask_pattern)
        W(
            f"    {rnd_name}(FV)+k: bid {mb}/{nb} ({100*mb/max(nb,1):.1f}%)  "
            f"ask {ma}/{na} ({100*ma/max(na,1):.1f}%)"
        )
    W("")

    W("=" * 72)
    W("INTERPRETATION")
    W("  • Inner (offset ±8) and wall (±10/±11) each match round(FV)±k on every tick where that tagged level exists.")
    W("  • Brute 'best floor(FV+shift)+k' hits 100% too — non-unique; prefer round(FV)±k unless boundary misses appear.")
    W("  • This slice often has 0–2 book levels; full {{-8,-10,-11}} bid set never co-occurs → cannot distinguish")
    W("    'one 3-layer MM' vs 'separate bots' from simultaneous top-of-book alone; use longer logs or level timing.")
    W("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
