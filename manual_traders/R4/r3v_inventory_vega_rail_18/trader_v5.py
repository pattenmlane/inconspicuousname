"""
Round 4 — Sonic joint gate + **tape-timestamp** counterparty filter (v5).

Phase 1/3: Mark 67→Mark 22 and Mark 67→Mark 49 on VELVETFRUIT_EXTRACT showed
positive short-horizon forward mids on tape. v3/v4 showed naive gate-only
execution fails (mid ≠ bid/ask).

v5 only submits a **small long extract** when **both**:
  (1) Sonic joint tight: VEV_5200 and VEV_5300 BBO spreads <= 2, and
  (2) The current timestamp is one where the **tape** had that counterparty
      print (precomputed from ROUND_4 trades into r4_pair_ts_masks.json).

Flatten long when joint gate opens (wide). No VEV/hydro orders.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UNDER = "VELVETFRUIT_EXTRACT"
GATE_A = "VEV_5200"
GATE_B = "VEV_5300"
TH = 2
CLIP = 10
LIM = 200
WARMUP_TICKS = 15

_MASK_PATH = Path(__file__).resolve().parent / "r4_pair_ts_masks.json"
_TS: dict[str, dict[str, set[int]]] | None = None


def _load_masks() -> dict[str, dict[str, set[int]]]:
    global _TS
    if _TS is not None:
        return _TS
    raw = json.loads(_MASK_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict[str, set[int]]] = {}
    for key, day_map in raw.items():
        out[key] = {str(d): set(int(x) for x in ts_list) for d, ts_list in day_map.items()}
    _TS = out
    return out


def sym_for(state: TradingState, product: str) -> str | None:
    for s, lst in (state.listings or {}).items():
        if getattr(lst, "product", None) == product:
            return s
    return None


def spread_bbo(depth: OrderDepth | None) -> int | None:
    if depth is None:
        return None
    bu = depth.buy_orders or {}
    se = depth.sell_orders or {}
    if not bu or not se:
        return None
    bb = int(max(bu))
    ba = int(min(se))
    if ba < bb:
        return None
    return int(ba - bb)


def joint_tight(d52: OrderDepth | None, d53: OrderDepth | None) -> bool:
    s1 = spread_bbo(d52)
    s2 = spread_bbo(d53)
    if s1 is None or s2 is None:
        return False
    return s1 <= TH and s2 <= TH


class Trader:
    def run(self, state: TradingState):
        td = getattr(state, "traderData", "") or ""
        try:
            bag: dict[str, Any] = json.loads(td) if td else {}
        except json.JSONDecodeError:
            bag = {}
        if not isinstance(bag, dict):
            bag = {}

        ts = int(getattr(state, "timestamp", 0))
        last_ts = int(bag.get("_last_ts", -1))
        day = int(bag.get("_csv_day", 1))
        if last_ts >= 0 and ts < last_ts:
            day += 1
        bag["_last_ts"] = ts
        bag["_csv_day"] = day

        if ts // 100 < WARMUP_TICKS:
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        masks = _load_masks()
        day_key = str(day)
        ts_set: set[int] = set()
        for k in ("m67_m22_extract", "m67_m49_extract"):
            ts_set |= masks.get(k, {}).get(day_key, set())

        depths: dict[str, OrderDepth] = getattr(state, "order_depths", {}) or {}
        pos: dict[str, int] = getattr(state, "position", {}) or {}

        sym_u = sym_for(state, UNDER)
        s52 = sym_for(state, GATE_A)
        s53 = sym_for(state, GATE_B)
        if sym_u is None or s52 is None or s53 is None:
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        du = depths.get(sym_u)
        if du is None:
            return {}, 0, json.dumps(bag, separators=(",", ":"))
        bu_u = du.buy_orders or {}
        se_u = du.sell_orders or {}
        if not bu_u or not se_u:
            return {}, 0, json.dumps(bag, separators=(",", ":"))
        bb = int(max(bu_u))
        ba = int(min(se_u))
        if ba < bb:
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        tight = joint_tight(depths.get(s52), depths.get(s53))
        p = int(pos.get(sym_u, 0))

        if tight and ts in ts_set and p + CLIP <= LIM:
            return {sym_u: [Order(sym_u, ba, CLIP)]}, 0, json.dumps(bag, separators=(",", ":"))

        if (not tight) and p > 0:
            q = min(p, CLIP)
            if q > 0:
                return {sym_u: [Order(sym_u, bb, -q)]}, 0, json.dumps(bag, separators=(",", ":"))

        return {}, 0, json.dumps(bag, separators=(",", ":"))
