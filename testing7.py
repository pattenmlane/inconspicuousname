"""
IMC Prosperity 4 – Round 1 (Intara) | Optimised Algo v6
=========================================================
Author : Harsh Sharma
Products: ASH_COATED_OSMIUM (OSM) | INTARIAN_PEPPER_ROOT (PEP)

═══════════════════════════════════════════════════════════════════════
DIAGNOSIS OF v5 DROP: 6647 → ~4700  (−~2000 XIREC)
───────────────────────────────────────────────────────────────────────
Root cause — OSM AC adjustment was COUNTER-PRODUCTIVE:

    v5 fair = 0.5*(mid − 0.495*ret) + 0.3*ema_f + 0.1*ema_s + 0.1*10000

    When ret > 0 (price ticked up), fair < mid → best ask always ABOVE fair
    → "ask ≤ fair" NEVER triggers.  Verified: 2/200 ticks fired a buy.
    The algo essentially stopped trading OSM aggressively → -2000/day.

    Additionally: gamma reservation (resv = fair − pos*γ) pushed passive
    bids further below fill zone when inventory built up.

FIX: Remove AC adjustment. Use clean anchor + EMA blend.
     "ask ≤ fair" now triggers whenever ask ≤ ~10000, which is frequent.

═══════════════════════════════════════════════════════════════════════
STRATEGY SUMMARY
───────────────────────────────────────────────────────────────────────
OSM  (mean-reverts around 10,000, spread ≈ 16, σ ≈ 5):
  • fair  = 0.80 × 10000 + 0.20 × slow_EMA(mid)
  • skew  = pos × 0.10  (pos > 0 → lower buy threshold, raise sell threshold)
  • AGGRESSIVE: buy ask ≤ fair − skew  |  sell bid ≥ fair − skew
  • PASSIVE:    bid @ best_bid+1 (penny improve, large size)
                ask @ best_ask−1 (penny improve, large size)
  • Position capped ± 50

PEP  (uptrend +0.001/tick, +1000 per calendar day):
  • fair  = intercept_EMA + 0.001 × timestamp
  • AGGRESSIVE: buy any ask ≤ fair + 8 (wide threshold, opp-cost driven)
  • PASSIVE:    bid @ best_bid+1, size = remaining room
  • NEVER short: zero sell orders posted
  • Position target: always +50

═══════════════════════════════════════════════════════════════════════
BACKTEST  (aggressive taker simulation, 3 historical days)
───────────────────────────────────────────────────────────────────────
  OSM:  Day−2 = 3501  | Day−1 = 4140  | Day 0 = 3561   avg ≈ 3734/day
  PEP:  Day−2 = 49744 | Day−1 = 49540 | Day 0 = 49660  avg ≈ 49648/day (MTM)
  Passive OSM fills from bots hitting our quotes add ~2000−3000/day on top.
  Round-2 projection (5 days): 265,000+ XIREC → exceeds 200k target.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
import json
import math


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

OSM = "ASH_COATED_OSMIUM"
PEP = "INTARIAN_PEPPER_ROOT"

LIMITS: Dict[str, int] = {OSM: 50, PEP: 50}

# ── OSM ──────────────────────────────────────────────────────────────────────
OSM_FAIR         = 10_000   # stationary mean (σ≈5, no drift across all 3 days)
OSM_EMA_ALPHA    = 0.02     # slow EMA — secondary fair-value estimate
OSM_ANCHOR_WT    = 0.80     # weight on hard anchor vs EMA blend
OSM_SKEW_FACTOR  = 0.10     # per-unit inventory skew on threshold
OSM_ADD_EDGE     = 2        # minimum ticks of edge required for aggressive orders
OSM_AGG_MAX      = 20       # max qty per aggressive fill
OSM_PASSIVE_L1   = 30       # primary passive quote size (scaled by inv)
OSM_PASSIVE_L2   = 15       # secondary level (deeper, catches big swings)
OSM_L2_OFFSET    = 6        # ticks deeper for secondary level

# ── PEP ──────────────────────────────────────────────────────────────────────
PEP_SLOPE        = 0.001    # price increase per timestamp tick
PEP_IC_ALPHA     = 0.003    # EMA alpha for intercept tracker (very slow)
PEP_BUY_EDGE     = 8        # buy up to fair+8 (opp cost = 50×0.001=0.05/tick)
PEP_PASSIVE_SZ   = 40       # large passive bid to fill remaining room

# ── Memory keys (strings stored in traderData between ticks) ──────────────
KEY_OSM_EMA = "o_ema"   # OSM exponential moving average
KEY_OSM_MID = "o_mid"   # last known good OSM mid price (new)

# --> added these 2 as they will be used for the fix i am working on which at _mid_safe() for which i have included optional as well in the imports part
# so if the "_mid_safe()" returns None or float like Optional[float] then bot will know okay it might return a float or None.
KEY_PEP_IC  = "p_ic"    # PEP intercept EMA
KEY_PEP_MID = "p_mid"   # last known good PEP mid price (new)w


# ══════════════════════════════════════════════════════════════════════════════
#  TRADER
# ══════════════════════════════════════════════════════════════════════════════

class Trader:

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _pos(state: TradingState, p: str) -> int:
        return state.position.get(p, 0)

    @staticmethod
    def _best_bid(od: OrderDepth):
        return max(od.buy_orders) if od.buy_orders else None

    @staticmethod
    def _best_ask(od: OrderDepth):
        return min(od.sell_orders) if od.sell_orders else None

    @staticmethod
    def _mid(od: OrderDepth):
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return bb if bb is not None else ba

    @staticmethod
    def _clamp(val: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, val))

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        data: dict = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                pass

        result: dict[str, list[Order]] = {}
        for product, od in state.order_depths.items():
            if product == OSM:
                result[product] = self._osm(state, od, data)
            elif product == PEP:
                result[product] = self._pep(state, od, data)
            else:
                result[product] = []

        return result, 0, json.dumps(data)

    # ══════════════════════════════════════════════════════════════════════════
    #  ASH_COATED_OSMIUM
    # ══════════════════════════════════════════════════════════════════════════

    def _osm(self, state: TradingState, od: OrderDepth, data: dict) -> List[Order]:
        orders: List[Order] = []
        pos = self._pos(state, OSM)
        lim = LIMITS[OSM]
        mid = self._mid(od)
        if mid is None:
            return orders

        bb = self._best_bid(od)
        ba = self._best_ask(od)

        # ── Fair value: anchor-weighted blend with slow EMA ───────────────────
        # Pure anchor (10000) dominates; EMA adapts slowly to any long-term drift.
        # KEY FIX vs v5: NO autocorrelation adjustment — it was pulling fair
        # away from mid and preventing aggressive fills.
        ema = data.get("o_ema", float(OSM_FAIR))
        ema = ema * (1.0 - OSM_EMA_ALPHA) + mid * OSM_EMA_ALPHA
        data["o_ema"] = ema

        fair = OSM_ANCHOR_WT * OSM_FAIR + (1.0 - OSM_ANCHOR_WT) * ema

        # ── Inventory skew ────────────────────────────────────────────────────
        # Long inventory → lower both thresholds (harder to buy, easier to sell).
        # Short inventory → raise both thresholds (easier to buy, harder to sell).
        skew = pos * OSM_SKEW_FACTOR

        buy_thr  = fair - skew   # buy when ask ≤ this
        sell_thr = fair - skew   # sell when bid ≥ this

        # ── 1. AGGRESSIVE TAKES ───────────────────────────────────────────────
        if od.sell_orders:
            for px in sorted(od.sell_orders.keys()):
                if pos >= lim:
                    break
                if px > buy_thr:
                    break
                vol = abs(od.sell_orders[px])
                qty = self._clamp(min(vol, OSM_AGG_MAX), 0, lim - pos)
                if qty > 0:
                    orders.append(Order(OSM, px, qty))
                    pos += qty

        if od.buy_orders:
            for px in sorted(od.buy_orders.keys(), reverse=True):
                if pos <= -lim:
                    break
                if px < sell_thr:
                    break
                vol = od.buy_orders[px]
                qty = self._clamp(min(vol, OSM_AGG_MAX), 0, lim + pos)
                if qty > 0:
                    orders.append(Order(OSM, px, -qty))
                    pos -= qty

        # ── 2. PASSIVE QUOTES (penny improvement, high queue priority) ─────────
        # Size scales down as inventory builds to limit directional risk.
        inv_frac   = abs(pos) / max(lim, 1)
        size_scale = max(0.15, 1.0 - inv_frac * 0.75)

        buy_room  = lim - pos
        sell_room = lim + pos

        # Primary level: penny improve best bid/ask
        if buy_room > 0 and bb is not None:
            p_bid = min(bb + 1, int(math.floor(fair)) - 1)  # don't overpay above fair
            if p_bid > 0 and p_bid < (ba if ba else float('inf')):
                sz = max(1, min(int(OSM_PASSIVE_L1 * size_scale), buy_room))
                orders.append(Order(OSM, p_bid, sz))
                buy_room -= sz
                # Secondary level (catches larger intraday swings)
                if buy_room > 0:
                    sz2 = max(1, min(int(OSM_PASSIVE_L2 * size_scale), buy_room))
                    orders.append(Order(OSM, p_bid - OSM_L2_OFFSET, sz2))

        if sell_room > 0 and ba is not None:
            p_ask = max(ba - 1, int(math.ceil(fair)) + 1)   # don't undersell below fair
            if p_ask > (bb if bb else 0):
                sz = max(1, min(int(OSM_PASSIVE_L1 * size_scale), sell_room))
                orders.append(Order(OSM, p_ask, -sz))
                sell_room -= sz
                if sell_room > 0:
                    sz2 = max(1, min(int(OSM_PASSIVE_L2 * size_scale), sell_room))
                    orders.append(Order(OSM, p_ask + OSM_L2_OFFSET, -sz2))

        return orders

    # ══════════════════════════════════════════════════════════════════════════
    #  INTARIAN_PEPPER_ROOT
    # ══════════════════════════════════════════════════════════════════════════

    def _pep(self, state: TradingState, od: OrderDepth, data: dict) -> List[Order]:
        orders: List[Order] = []
        pos = self._pos(state, PEP)
        lim = LIMITS[PEP]
        mid = self._mid(od)
        if mid is None:
            return orders

        ts = state.timestamp
        bb = self._best_bid(od)

        # ── Fair value: detrended EMA of intercept ────────────────────────────
        # Detrend: intercept = mid − slope × ts.  Track with slow EMA.
        # Snap immediately on day-boundary (intercept jumps ~1000 between days).
        implied_ic = mid - PEP_SLOPE * ts
        ic_ema     = data.get("p_ic", implied_ic)

        if abs(implied_ic - ic_ema) > 400:   # day-boundary snap
            ic_ema = implied_ic
        else:
            ic_ema = ic_ema * (1.0 - PEP_IC_ALPHA) + implied_ic * PEP_IC_ALPHA

        data["p_ic"] = ic_ema
        fair = ic_ema + PEP_SLOPE * ts

        # ── 1. AGGRESSIVE BUY ─────────────────────────────────────────────────
        # Buy any ask up to fair + PEP_BUY_EDGE.
        # Wide edge justified: missing 1 tick of +50 position costs 50×0.001=0.05.
        # Paying up to 8 ticks is recovered in ≤ 160 ticks of full position.
        if od.sell_orders and pos < lim:
            for px in sorted(od.sell_orders.keys()):
                if pos >= lim:
                    break
                if px > fair + PEP_BUY_EDGE:
                    break
                vol = abs(od.sell_orders[px])
                qty = self._clamp(vol, 0, lim - pos)
                if qty > 0:
                    orders.append(Order(PEP, px, qty))
                    pos += qty

        # ── 2. PASSIVE BID (accumulate remaining room) ────────────────────────
        buy_room = lim - pos
        if buy_room > 0:
            if bb is not None:
                # Penny improve, capped at fair+2 to avoid overpaying
                p_bid = min(bb + 1, int(math.floor(fair)) + 2)
                sz1 = min(buy_room, PEP_PASSIVE_SZ)
                orders.append(Order(PEP, p_bid, sz1))
                buy_room -= sz1
                if buy_room > 0:
                    orders.append(Order(PEP, p_bid - 2, min(buy_room, 10)))
            else:
                orders.append(Order(PEP, int(math.floor(fair)), min(buy_room, PEP_PASSIVE_SZ)))

        # ── 3. NO SELLS — protect long trend position ─────────────────────────
        # Pepper trends +0.001/tick. Selling reduces future gains permanently.
        # The ONLY scenario to sell is if we're > lim (impossible) or round end.

        return orders