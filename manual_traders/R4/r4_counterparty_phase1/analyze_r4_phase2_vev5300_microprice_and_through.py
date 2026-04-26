#!/usr/bin/env python3
"""Phase 2 microstructure: VEV_5300 L1 microprice vs mid, trade-through rate, and
per-(buyer,seller) split on ROUND_4 days 1–3.

- Microprice: (bid*bid_vol + ask*ask_vol) / (bid_vol+ask_vol) at trade timestamp
  (join to prices_round_4_day_* by day,ts,symbol=VEV_5300).
- Microprice offset: (microprice - mid) in tick space (same units as prices).
- Trade-through: buyer-initiated if price >= ask; **through** if price > ask.
  Seller-initiated if price <= bid; through if price < bid.
- Pooled and per-day means of |mp_offset| and through rates. Optional: joint
  tight gate (5200&5300 spr<=2) subsample.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_phase2_vev5300_microprice_through.json"
DAYS = (1, 2, 3)
S5300 = "VEV_5300"
S5200 = "VEV_5200"
TH = 2.0


def load_5300_books() -> dict[tuple[int, int], dict]:
    """(day, ts) -> bid, ask, bvol, avol, mid"""
    by: dict[tuple[int, int], dict] = {}
    for d in DAYS:
        path = DATA / f"prices_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["day"]) != d or row["product"] != S5300:
                    continue
                ts = int(row["timestamp"])
                bid = float(row["bid_price_1"])
                ask = float(row["ask_price_1"])
                bvol = int(float(row["bid_volume_1"] or 0))
                avol = int(float(row["ask_volume_1"] or 0))
                mid = float(row["mid_price"])
                by[(d, ts)] = {
                    "bid": bid,
                    "ask": ask,
                    "bvol": bvol,
                    "avol": avol,
                    "mid": mid,
                }
    return by


def load_5200_spread() -> dict[tuple[int, int], float]:
    sp: dict[tuple[int, int], float] = {}
    for d in DAYS:
        path = DATA / f"prices_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["day"]) != d or row["product"] != S5200:
                    continue
                ts = int(row["timestamp"])
                bid = float(row["bid_price_1"])
                ask = float(row["ask_price_1"])
                sp[(d, ts)] = ask - bid if ask > bid else 0.0
    return sp


def microprice(b: dict) -> float | None:
    """Standard L1 volume-weighted: (bid * bidvol + ask * askvol) / (vol sum)."""
    bv, av = b["bvol"], b["avol"]
    if bv + av <= 0:
        return None
    return (b["bid"] * bv + b["ask"] * av) / (bv + av)


def main() -> None:
    books = load_5300_books()
    sp52 = load_5200_spread()
    per_day: dict[str, list] = {str(d): [] for d in DAYS}
    pooled: list[dict] = []
    pair_buy_through: dict[tuple[str, str], list[int]] = defaultdict(list)  # 0/1
    for d in DAYS:
        tp = DATA / f"trades_round_4_day_{d}.csv"
        with open(tp, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if row["symbol"] != S5300:
                    continue
                ts = int(row["timestamp"])
                key = (d, ts)
                if key not in books:
                    continue
                b = books[key]
                pr = float(row["price"])
                mp = microprice(b)
                if mp is None:
                    continue
                off = pr - b["mid"]  # not mp-mid; report both
                mpo = mp - b["mid"]
                bid, ask = b["bid"], b["ask"]
                buyer = str(row["buyer"]).strip()
                seller = str(row["seller"]).strip()
                thru = 0
                if pr >= ask - 1e-9:
                    thru = 1 if pr > ask + 1e-9 else 0
                elif pr <= bid + 1e-9:
                    thru = 1 if pr < bid - 1e-9 else 0
                spr53 = b["ask"] - b["bid"] if b["ask"] > b["bid"] else 0.0
                tight = spr53 <= TH and (key in sp52 and sp52[key] <= TH)
                rec = {
                    "day": d,
                    "ts": ts,
                    "mp_offset": round(mpo, 6),
                    "abs_mp_offset": abs(mpo),
                    "px_minus_mid": round(off, 6),
                    "joint_tight": tight,
                    "through": thru,
                }
                pooled.append(rec)
                per_day[str(d)].append(rec)
                pair_buy_through[(buyer, seller)].append(thru)

    def mean(xs: list[float]) -> float | None:
        if not xs:
            return None
        return sum(xs) / len(xs)

    def summarize(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0}
        abs_mp = [r["abs_mp_offset"] for r in rows]
        thr = [r["through"] for r in rows]
        tight_r = [r for r in rows if r.get("joint_tight")]
        return {
            "n": len(rows),
            "mean_abs_mp_offset": round(mean(abs_mp), 6) if abs_mp else None,
            "frac_trade_through": round(sum(thr) / len(thr), 6) if thr else None,
            "n_joint_tight": len(tight_r),
            "mean_abs_mp_offset_joint_tight": (
                round(mean([r["abs_mp_offset"] for r in tight_r]), 6) if tight_r else None
            ),
        }

    top_pairs = sorted(
        (
            {
                "buyer": a,
                "seller": b,
                "n": len(vs),
                "frac_through": round(sum(vs) / len(vs), 6) if vs else None,
            }
            for (a, b), vs in pair_buy_through.items()
            if len(vs) >= 20
        ),
        key=lambda x: -x["n"],
    )[:20]

    out = {
        "symbol": S5300,
        "TH_joint_gate": TH,
        "pooled": summarize(pooled),
        "per_day": {d: summarize(rows) for d, rows in per_day.items()},
        "per_buyer_seller_n_ge_20": top_pairs,
        "note": "microprice = L1 (bid*bid_vol + ask*ask_vol)/(bid_vol+ask_vol); through = print strictly beyond best bid or best ask when at touch.",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
