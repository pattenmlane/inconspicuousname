"""
v41 — round3work/vouchers_final_strategy/ only (per branch pivot).

- Sonic / STRATEGY.txt: VEV_5200 and VEV_5300 BBO spread both ≤ 2 (ticks) = tight joint surface;
  t-stat / edge is meaningful there; in wide book, execution noise dominates.
- Optional directional read (STRATEGY layer 2): when tight, short-horizon extract mid is more
  favorable; we only lean long VELVETFRUIT_EXTRACT (clip toward a target) when the gate is on.
- When the gate is off: flatten extract (risk-off) — do not run voucher logic.
- No IV smile, no HYDROGEL (PnL objective: extract + VEV; we only touch extract here).

inclineGod: we read spreads per name (5200, 5300) from the order book, not just mids.
"""
from __future__ import annotations

import json
from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
H = "HYDROGEL_PACK"

LIMITS = {
    H: 200,
    U: 200,
    **{v: 300 for v in VOUCHERS},
}

# Must match strategy replication (analyze_vev_5200_5300_tight_gate_r3.py default TH=2)
TIGHT_S5200_S5300_TH = 2
# Long extract target (shares) when joint gate on — sub-limit of 200
EXTRACT_TARGET_LONG = 32
# Per-timestamp clip when moving toward target or flattening
MAX_CLIP = 12


def _mid(depth: OrderDepth) -> tuple[float, int, int] | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    return (bb + ba) / 2.0, bb, ba


def _spread_bbo(depth: OrderDepth) -> int | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return int(min(depth.sell_orders) - max(depth.buy_orders))


def _joint_tight_from_depths(
    d52: OrderDepth | None, d53: OrderDepth | None
) -> bool:
    if d52 is None or d53 is None:
        return False
    s5 = _spread_bbo(d52)
    s3 = _spread_bbo(d53)
    if s5 is None or s3 is None:
        return False
    return s5 <= TIGHT_S5200_S5300_TH and s3 <= TIGHT_S5200_S5300_TH


class Trader:
    def run(self, state: TradingState):
        orders: dict[str, list[Order]] = {p: [] for p in LIMITS}
        d52 = state.order_depths.get(VEV_5200)
        d53 = state.order_depths.get(VEV_5300)
        du = state.order_depths.get(U)

        if du is None or du.buy_orders is None or du.sell_orders is None:
            return orders, 0, json.dumps({})
        if not _mid(du):
            return orders, 0, json.dumps({})

        tight = _joint_tight_from_depths(d52, d53)
        pu = int(state.position.get(U, 0))
        u_lim = LIMITS[U]

        if tight:
            # Long bias toward target when surface is tight (STRATEGY optional layer 2)
            if pu < EXTRACT_TARGET_LONG - 1:
                q = min(EXTRACT_TARGET_LONG - pu, MAX_CLIP, u_lim - pu)
                if q > 0 and du.sell_orders:
                    orders[U].append(Order(U, int(min(du.sell_orders)), int(q)))
            elif pu > EXTRACT_TARGET_LONG + 1:
                q = min(pu - EXTRACT_TARGET_LONG, MAX_CLIP, pu + u_lim)
                if q > 0 and du.buy_orders:
                    orders[U].append(Order(U, int(max(du.buy_orders)), -int(q)))
        else:
            # Wide book: flatten extract (Sonic: do not trust flow / pay noise)
            if pu > 0 and du.buy_orders:
                q = min(pu, MAX_CLIP)
                orders[U].append(Order(U, int(max(du.buy_orders)), -q))
            elif pu < 0 and du.sell_orders:
                q = min(-pu, MAX_CLIP, u_lim + pu)
                if q > 0:
                    orders[U].append(Order(U, int(min(du.sell_orders)), int(q)))

        return orders, 0, json.dumps({})
