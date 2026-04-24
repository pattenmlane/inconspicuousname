"""Extract near-FV book events: |price - true_fv| <= MAX_ABS (tomato Bot 3 style)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from osmium_fair_common import DEFAULT_DATA_DIR, ask_levels, bid_levels, load_fv, resolve_paths

# Inner / wall structural offsets (integer-rounded); exclude from "residual participant" slice
MM_OFFSET_INTS = frozenset({-11, -10, -8, 8, 10, 11})

DEFAULT_MAX_ABS = 4.0


@dataclass
class NearFvEvent:
    timestamp: int
    fv: float
    side: str  # "bid" | "ask"
    price: int
    vol: int
    off_cont: float  # price - fv
    off_int: int  # round(price - fv)
    delta: int  # price - round(fv)
    crossing: bool


def iter_near_fv_events(
    rows: list[dict[str, str]],
    fv_map: dict[int, float],
    *,
    max_abs: float = DEFAULT_MAX_ABS,
    exclude_mm_offsets: bool = False,
) -> Iterator[NearFvEvent]:
    for row in rows:
        ts = int(row["timestamp"])
        fv = fv_map[ts]
        for side, levels in (
            ("bid", bid_levels(row)),
            ("ask", ask_levels(row)),
        ):
            for price, vol in levels:
                off_cont = price - fv
                if abs(off_cont) > max_abs + 1e-12:
                    continue
                off_int = int(round(off_cont))
                if exclude_mm_offsets and off_int in MM_OFFSET_INTS:
                    continue
                delta = price - round(fv)
                if side == "bid":
                    crossing = price > fv
                else:
                    crossing = price < fv
                yield NearFvEvent(
                    timestamp=ts,
                    fv=fv,
                    side=side,
                    price=price,
                    vol=vol,
                    off_cont=off_cont,
                    off_int=off_int,
                    delta=delta,
                    crossing=crossing,
                )


def load_rows_and_fv(data_dir: Path) -> tuple[list[dict[str, str]], dict[int, float], Path]:
    prices_path, fv_path = resolve_paths(data_dir)
    fv_map = load_fv(fv_path)
    with prices_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    return rows, fv_map, prices_path
