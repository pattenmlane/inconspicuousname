"""
Osmium INNER bot simulator (ASH_COATED_OSMIUM fair probe).

Identified by book levels with offset_int = round(price - true_fv) in {-8, +8}
for bid / ask respectively.

Rule (validated on fair logs 278076, 248329):
    bid = round(FV) - 8
    ask = round(FV) + 8
    bid_vol = ask_vol = randint(10, 15)   # same on both sides per tick (like tomato bots)

Uses Python's round() (banker's rounding at exact halves; FV is effectively continuous).
"""

from __future__ import annotations

import random


def inner_quote(fv: float) -> tuple[int, int, int]:
    """
    Return (bid_price, ask_price, volume) for the inner / touch layer.

    Volume is one draw reused for both sides for this tick.
    """
    r = round(fv)
    vol = random.randint(10, 15)
    return r - 8, r + 8, vol


def inner_prices(fv: float) -> tuple[int, int]:
    """Deterministic prices only (for validation without RNG)."""
    r = round(fv)
    return r - 8, r + 8
