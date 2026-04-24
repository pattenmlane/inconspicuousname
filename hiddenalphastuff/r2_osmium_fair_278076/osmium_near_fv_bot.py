"""
Osmium NEAR-FV / residual participant simulator (tomato Bot 3 analogue).

Identification: |price - true_fv| <= 4 (continuous), same as tomato.

Empirical osmium (fair logs 278076 + 248329) differs from tomato on the
*price offset*: mass sits on round(FV)+{-3,-2,+1,+2} with almost nothing at
round(FV) (see analyze_osmium_near_fv.py). Volume still splits by crossing
like tomato: crossing → larger (here U(4,10)), passive → smaller U(2,6).

The (side, delta) joint is **not** a product of marginal side × marginal delta
(asks appear with negative deltas often). Use `near_fv_quote_empirical` with
weights from a Counter fit on a reference log for better crossing fidelity.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass


@dataclass
class NearFvParams:
    """Fit from one fair CSV + FV; defaults ≈ pooled two-session averages."""

    p_tick: float = 0.078  # P(at least one event this timestep) ~156/2000
    p_bid: float = 0.453  # P(bid | event) pooled 72/159
    # Price = round(FV) + delta; delta uniform on ring (excludes 0, ±1 sparse in data)
    deltas: tuple[int, ...] = (-3, -2, 1, 2)
    vol_cross_lo: int = 4
    vol_cross_hi: int = 10
    vol_pass_lo: int = 2
    vol_pass_hi: int = 6


def near_fv_quote(
    fv: float,
    rng: random.Random | None = None,
    params: NearFvParams | None = None,
) -> tuple[str, int, int] | None:
    """
    Maybe return one (side, price, vol) quote.

    Returns None if absent this tick.
    """
    g = rng or random
    p = params or NearFvParams()
    if g.random() > p.p_tick:
        return None
    side = "bid" if g.random() < p.p_bid else "ask"
    delta = g.choice(p.deltas)
    price = round(fv) + delta
    crossing = (side == "bid" and price > fv) or (side == "ask" and price < fv)
    if crossing:
        vol = g.randint(p.vol_cross_lo, p.vol_cross_hi)
    else:
        vol = g.randint(p.vol_pass_lo, p.vol_pass_hi)
    return side, price, vol


def fit_params_from_counts(
    *,
    n_timesteps: int,
    n_ts_with_event: int,
    n_bid_events: int,
    n_ask_events: int,
) -> NearFvParams:
    p_tick = n_ts_with_event / max(n_timesteps, 1)
    n_ev = n_bid_events + n_ask_events
    p_bid = n_bid_events / max(n_ev, 1)
    return NearFvParams(p_tick=p_tick, p_bid=p_bid)


def near_fv_quote_empirical(
    fv: float,
    joint: Counter[tuple[str, int]],
    rng: random.Random | None = None,
    *,
    p_tick: float,
    vol_cross_lo: int = 4,
    vol_cross_hi: int = 10,
    vol_pass_lo: int = 2,
    vol_pass_hi: int = 6,
) -> tuple[str, int, int] | None:
    """Sample (side, delta) from empirical joint; then price and vol by crossing."""
    g = rng or random
    if g.random() > p_tick:
        return None
    pairs = list(joint.elements())
    if not pairs:
        return None
    side, delta = g.choice(pairs)
    price = round(fv) + delta
    crossing = (side == "bid" and price > fv) or (side == "ask" and price < fv)
    if crossing:
        vol = g.randint(vol_cross_lo, vol_cross_hi)
    else:
        vol = g.randint(vol_pass_lo, vol_pass_hi)
    return side, price, vol
