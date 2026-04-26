"""One-off helper: build (day,ts) set where Mark55 aggressive extract sell in (ts-W, ts) precedes tick with Mark67 aggressive buy. Used by trader_v8. Run from repo root."""
from __future__ import annotations

import bisect
import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "r4_m55_lead_suppress_mark67_W2000.json"
DAYS = [1, 2, 3]
W = 2000
EXTRACT = "VELVETFRUIT_EXTRACT"


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


def price_row(day: int, product: str) -> dict[int, tuple[float, float]]:
    """ts -> (bid, ask) first row."""
    p = DATA / f"prices_round_4_day_{day}.csv"
    out: dict[int, tuple[float, float]] = {}
    if not p.is_file():
        return out
    with p.open(newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            if (row.get("product") or "") != product:
                continue
            if row.get("day") and str(row["day"]).strip():
                if int(float(row["day"])) != day:
                    continue
            ts = _i(row["timestamp"])
            if ts in out:
                continue
            bid = _f(row.get("bid_price_1", ""))
            ask = _f(row.get("ask_price_1", ""))
            out[ts] = (bid, ask)
    return out


def main() -> None:
    ex_ba: dict[int, dict[int, tuple[float, float]]] = {}
    sym_ba: dict[tuple[int, str], dict[int, tuple[float, float]]] = {}

    def get_ba(day: int, sym: str, ts: int) -> tuple[float, float] | None:
        if sym == EXTRACT:
            d = ex_ba.setdefault(day, price_row(day, EXTRACT))
        else:
            d = sym_ba.setdefault((day, sym), price_row(day, sym))
        return d.get(ts)

    m55_sell_ts: dict[int, list[int]] = {d: [] for d in DAYS}
    for day in DAYS:
        p = DATA / f"trades_round_4_day_{day}.csv"
        if not p.is_file():
            continue
        exb = price_row(day, EXTRACT)
        ex_ba[day] = exb
        with p.open(newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if row["seller"].strip() != "Mark 55" or row["symbol"].strip() != EXTRACT:
                    continue
                ts = _i(row["timestamp"])
                pr = _i(row["price"])
                ba = exb.get(ts)
                if not ba:
                    continue
                bid, ask = ba
                if math.isnan(bid) or math.isnan(ask) or ask <= bid:
                    continue
                if pr <= bid:
                    m55_sell_ts[day].append(ts)
        m55_sell_ts[day].sort()

    suppress: set[tuple[int, int]] = set()
    m67_ticks = 0
    for day in DAYS:
        p = DATA / f"trades_round_4_day_{day}.csv"
        if not p.is_file():
            continue
        with p.open(newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if row["buyer"].strip() != "Mark 67":
                    continue
                ts = _i(row["timestamp"])
                sym = row["symbol"].strip()
                pr = _i(row["price"])
                ba = get_ba(day, sym, ts)
                if not ba:
                    continue
                bid, ask = ba
                if math.isnan(bid) or math.isnan(ask) or ask <= bid:
                    continue
                if pr < ask:
                    continue
                m67_ticks += 1
                arr = m55_sell_ts[day]
                if not arr:
                    continue
                lo = ts - W
                i = bisect.bisect_right(arr, ts - 1)
                j = bisect.bisect_right(arr, lo)
                if i > j:
                    suppress.add((day, ts))

    meta = {
        "W_ticks": W,
        "mark67_aggressive_buy_ticks": m67_ticks,
        "suppress_tick_count": len(suppress),
        "sample": sorted(suppress)[:20],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(meta, indent=2))
    # Also write a compact list for embedding
    lst_path = Path(__file__).resolve().parent / "analysis_outputs" / "r4_m55_lead_suppress_pairs.json"
    lst_path.write_text(json.dumps(sorted(suppress), indent=2))
    print(meta)


if __name__ == "__main__":
    main()
