"""
Osmium WALL bot simulator — two bid rungs and two ask rungs vs true FV.

Identified by offset_int in {-11,-10} (bids) and {+10,+11} (asks).

Rule (validated per rung when that rung exists on the tape):
    bid_m10 = round(FV) - 10
    bid_m11 = round(FV) - 11
    ask_p10 = round(FV) + 10
    ask_p11 = round(FV) + 11
    vol per rung = randint(20, 30)   # independent draw per displayed slot in sim
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class WallQuotes:
    bid_m10: int
    bid_m11: int
    ask_p10: int
    ask_p11: int


def wall_prices(fv: float) -> WallQuotes:
    r = round(fv)
    return WallQuotes(
        bid_m10=r - 10,
        bid_m11=r - 11,
        ask_p10=r + 10,
        ask_p11=r + 11,
    )


def wall_volumes(rng: random.Random | None = None) -> tuple[int, int, int, int]:
    g = rng or random
    return (
        g.randint(20, 30),
        g.randint(20, 30),
        g.randint(20, 30),
        g.randint(20, 30),
    )
