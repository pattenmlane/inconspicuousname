"""
vouchers_final_strategy (STRATEGY.txt, ORIGINAL_DISCORD_QUOTES.txt) — v24.

Sonic: **hard** joint gate s5200<=2 and s5300<=2 (same bar). No other strikes, no hydrogel.

inclineGod: beyond the box, use **spread–spread** tension: imbalance = |s5200-s5300|.
When both are in the tight box, imbalance is usually 0 (see analysis JSON); we scale
order size down when imbalance is positive (less "symmetric" surface).

Continuous risk: quality in [0,1] from (imb, s_ext), then scale passive lot caps.
Extract remains passive-only (STRATEGY mid vs PnL caveat).
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Listing, Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState

TH = 2
STRIKES = (5200, 5300)
EXTRACT = "VELVETFRUIT_EXTRACT"
MAX_EXTRACT_SPREAD = 8

EXTRACT_MID_W = 40
EXTRACT_DRIFT_EDGE = 2
EXTRACT_LIM = 200
VEV_LIM = 30
# Base max lots; actual = max(1, round(base * quality))
LOTS_VEV_BASE = 22
LOTS_EX_BASE = 8


def _symbol_for_product(state: TradingState, product: str) -> str | None:
    listings: dict[str, Listing] = getattr(state, "listings", {}) or {}
    for sym, lst in listings.items():
        if getattr(lst, "product", None) == product:
            return sym
    return None


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _spread_bbo(depth: OrderDepth) -> tuple[int, int, int, int] | None:
    buys = getattr(depth, "buy_orders", {}) or {}
    sells = getattr(depth, "sell_orders", {}) or {}
    if not buys or not sells:
        return None
    bb, ba = int(max(buys.keys())), int(min(sells.keys()))
    return bb, ba, ba - bb, abs(buys[bb]) + abs(sells[ba])


def _improve_bid(bb: int, ba: int) -> int:
    return bb + 1 if ba > bb + 1 else bb


def _improve_ask(bb: int, ba: int) -> int:
    return ba - 1 if ba > bb + 1 else ba


def _quality(s5200: int, s5300: int, s_ext: int) -> tuple[float, float]:
    """Return (imb, q) with q in (0,1]: symmetric tight books and tight extract -> ~1."""
    imb = float(abs(s5200 - s5300))
    # imbalance: 0 best, 2+ in-box still possible -> downweight
    q_leg = max(0.15, 1.0 - 0.45 * imb)
    q_ext = max(0.2, 1.0 - (float(s_ext) / float(max(MAX_EXTRACT_SPREAD, 1))) * 0.55)
    q = max(0.12, min(1.0, q_leg * q_ext))
    return imb, q


def _scale_lot(base: int, q: float) -> int:
    return max(1, int(round(float(base) * q)))


class Trader:
    def bid(self) -> int:
        return 0

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        depths: dict[str, Any] = getattr(state, "order_depths", {}) or {}

        sym_5200 = _symbol_for_product(state, "VEV_5200")
        sym_5300 = _symbol_for_product(state, "VEV_5300")
        sym_e = _symbol_for_product(state, EXTRACT)
        if not sym_5200 or not sym_5300 or not sym_e:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        for s in (sym_5200, sym_5300, sym_e):
            if s not in depths:
                return {}, 0, json.dumps(store, separators=(",", ":"))

        t5200 = _spread_bbo(depths[sym_5200])
        t5300 = _spread_bbo(depths[sym_5300])
        t_e = _spread_bbo(depths[sym_e])
        if t5200 is None or t5300 is None or t_e is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        s5200, s5300, s_ext = t5200[2], t5300[2], t_e[2]
        store["s5200"] = float(s5200)
        store["s5300"] = float(s5300)
        store["s_ext"] = float(s_ext)

        joint = (s5200 <= TH) and (s5300 <= TH) and (s_ext <= MAX_EXTRACT_SPREAD)
        store["joint_tight"] = bool(joint)
        if not joint:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        imb, qv = _quality(s5200, s5300, s_ext)
        store["s_imb"] = float(imb)
        store["qual"] = float(qv)

        lots_v = _scale_lot(LOTS_VEV_BASE, qv)
        lots_ex = _scale_lot(LOTS_EX_BASE, qv)
        pos = getattr(state, "position", {}) or {}
        out: dict[str, list[Order]] = {}

        for strike in STRIKES:
            p = f"VEV_{strike}"
            sym = _symbol_for_product(state, p)
            if not sym or sym not in depths:
                continue
            t = _spread_bbo(depths[sym])
            if t is None:
                continue
            bb, ba, _spr, _ = t
            pk = int(pos.get(sym, 0))
            buy_cap = min(VEV_LIM - pk, lots_v)
            sell_cap = min(VEV_LIM + pk, lots_v)
            bpx, apx = _improve_bid(bb, ba), _improve_ask(bb, ba)
            olist: list[Order] = []
            if buy_cap > 0:
                olist.append(Order(sym, bpx, buy_cap))
            if sell_cap > 0:
                olist.append(Order(sym, apx, -sell_cap))
            if olist:
                out[sym] = olist

        m_e = 0.5 * (float(t_e[0]) + float(t_e[1]))
        hist = store.get("e_mid_hist")
        if not isinstance(hist, list):
            hist = []
        hist = [float(x) for x in hist if isinstance(x, (int, float))][-EXTRACT_MID_W:]
        hist.append(m_e)
        hist = hist[-EXTRACT_MID_W:]
        store["e_mid_hist"] = hist

        prev = store.get("e_mid_prev")
        d = 0.0
        if isinstance(prev, (int, float)):
            d = m_e - float(prev)
        store["e_mid_prev"] = m_e

        if len(hist) < 15:
            return out, 0, json.dumps(store, separators=(",", ":"))

        mean = sum(hist) / float(len(hist))
        pe = int(pos.get(sym_e, 0))
        bbb, bba = t_e[0], t_e[1]
        bcap = min(EXTRACT_LIM - pe, lots_ex)
        scap = min(EXTRACT_LIM + pe, lots_ex)
        eol: list[Order] = []
        if bcap > 0 and m_e < mean - float(EXTRACT_DRIFT_EDGE) and d > -0.5 and imb <= 1.0:
            eol.append(Order(sym_e, _improve_bid(bbb, bba), bcap))
        if scap > 0 and m_e > mean + float(EXTRACT_DRIFT_EDGE) and d < 0.5 and imb <= 1.0:
            eol.append(Order(sym_e, _improve_ask(bbb, bba), -scap))
        if eol:
            out[sym_e] = (out.get(sym_e) or []) + eol
        return out, 0, json.dumps(store, separators=(",", ":"))
