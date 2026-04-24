"""
Shared CSV + true FV loading for ASH_COATED_OSMIUM fair-probe analysis.

Used by analyze_osmium_quote_rules.py, validate_osmium_inner.py,
validate_osmium_wall.py, osmium_inner_exact_rule.py.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent


def resolve_paths(data_dir: Path) -> tuple[Path, Path]:
    data_dir = data_dir.resolve()
    ps = sorted(data_dir.glob("prices_round_*_day_*.csv"))
    if len(ps) != 1:
        raise SystemExit(f"Need exactly one prices_round_*_day_*.csv in {data_dir}")
    fv = data_dir / "osmium_true_fv.csv"
    if not fv.is_file():
        raise SystemExit(f"Missing {fv}")
    return ps[0], fv


def load_fv(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            out[int(row["timestamp"])] = float(row["true_fv"])
    return out


def bid_levels(row: dict[str, str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in range(1, 4):
        p, v = row[f"bid_price_{i}"].strip(), row[f"bid_volume_{i}"].strip()
        if p and v:
            out.append((int(p), int(v)))
    return out


def ask_levels(row: dict[str, str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in range(1, 4):
        p, v = row[f"ask_price_{i}"].strip(), row[f"ask_volume_{i}"].strip()
        if p and v:
            out.append((int(p), int(v)))
    return out


@dataclass
class Tick:
    ts: int
    fv: float
    inner_bid: int | None
    inner_ask: int | None
    inner_bid_vol: int | None
    inner_ask_vol: int | None
    bid_m10: int | None
    bid_m11: int | None
    ask_p10: int | None
    ask_p11: int | None
    bid_m10_vol: int | None
    bid_m11_vol: int | None
    ask_p10_vol: int | None
    ask_p11_vol: int | None
    bids_desc: list[int]
    asks_asc: list[int]


def _pick_bid(bids: list[tuple[int, int]], fv: float, target_off: int) -> tuple[int | None, int | None]:
    cand = [(p, v) for p, v in bids if int(round(p - fv)) == target_off]
    if not cand:
        return None, None
    p, v = max(cand, key=lambda x: x[0])
    return p, v


def _pick_ask(asks: list[tuple[int, int]], fv: float, target_off: int) -> tuple[int | None, int | None]:
    cand = [(p, v) for p, v in asks if int(round(p - fv)) == target_off]
    if not cand:
        return None, None
    p, v = min(cand, key=lambda x: x[0])
    return p, v


def build_ticks(rows: list[dict[str, str]], fv_map: dict[int, float]) -> list[Tick]:
    ticks: list[Tick] = []
    for row in rows:
        ts = int(row["timestamp"])
        fv = fv_map[ts]
        bids = bid_levels(row)
        asks = ask_levels(row)
        bp = [p for p, _ in bids]
        ap = [p for p, _ in asks]

        ib, ibv = _pick_bid(bids, fv, -8)
        ia, iav = _pick_ask(asks, fv, 8)
        b10, b10v = _pick_bid(bids, fv, -10)
        b11, b11v = _pick_bid(bids, fv, -11)
        a10, a10v = _pick_ask(asks, fv, 10)
        a11, a11v = _pick_ask(asks, fv, 11)

        ticks.append(
            Tick(
                ts=ts,
                fv=fv,
                inner_bid=ib,
                inner_ask=ia,
                inner_bid_vol=ibv,
                inner_ask_vol=iav,
                bid_m10=b10,
                bid_m11=b11,
                ask_p10=a10,
                ask_p11=a11,
                bid_m10_vol=b10v,
                bid_m11_vol=b11v,
                ask_p10_vol=a10v,
                ask_p11_vol=a11v,
                bids_desc=sorted(bp, reverse=True),
                asks_asc=sorted(ap),
            )
        )
    return ticks


def load_ticks(data_dir: Path) -> tuple[list[Tick], Path]:
    prices_path, fv_path = resolve_paths(data_dir)
    fv_map = load_fv(fv_path)
    with prices_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    return build_ticks(rows, fv_map), prices_path
