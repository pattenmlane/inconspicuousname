"""
Round 4 — trader_v11 + extra VEV_5300 bid fade when Mark 22 aggressively sells on 5300
this tick under Sonic joint tight (tape: aggr_sell at bid; Phase-1 negative fwd_mid_5).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState, Trade
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState, Trade

U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
LIMITS = {H: 200, U: 200, **{v: 300 for v in VOUCHERS}}

_SIG_M67 = Path(__file__).resolve().parent / "analysis_outputs" / "signals_mark67_aggr_extract_buy.json"
_TH = 2.0
INV_SKEW_STEP = 25
INV_SKEW_MAX = 8
EXTRACT_INV_STEP = 40
EXTRACT_INV_MAX = 3
M67_AGGR_BUY_Q = 22


def book_walls(depth: OrderDepth) -> tuple[int | None, int | None, int | None, int | None]:
    buys = depth.buy_orders or {}
    sells = depth.sell_orders or {}
    if not buys and not sells:
        return None, None, None, None
    sell_prices = list(sells.keys())
    return (
        min(buys.keys()),
        max(sell_prices) if sell_prices else None,
        max(buys.keys()),
        min(sell_prices) if sell_prices else None,
    )


def micro_mid(depth: OrderDepth) -> float | None:
    _, _, bb, ba = book_walls(depth)
    if bb is None or ba is None:
        return None
    return (float(bb) + float(ba)) / 2.0


def bbo_spread(depth: OrderDepth) -> float | None:
    _, _, bb, ba = book_walls(depth)
    if bb is None or ba is None:
        return None
    return float(ba) - float(bb)


def _load_m67_keys() -> set[str]:
    if not _SIG_M67.is_file():
        return set()
    try:
        d = json.loads(_SIG_M67.read_text(encoding="utf-8"))
        return set(str(x) for x in d.get("keys", []))
    except (json.JSONDecodeError, OSError):
        return set()


_M67_KEYS = _load_m67_keys()


def _m01_m22_5300_live(state: TradingState, symbol: str) -> bool:
    mt = getattr(state, "market_trades", None) or {}
    for t in mt.get(symbol, []) or []:
        if not isinstance(t, Trade):
            continue
        if str(t.buyer) == "Mark 01" and str(t.seller) == "Mark 22":
            return True
    return False


def _m22_aggr_sell_5300_live(state: TradingState, symbol: str, depth: OrderDepth) -> bool:
    """Tape aggressive sell: seller hit bid (price <= best bid), seller Mark 22."""
    _, _, bb, ba = book_walls(depth)
    if bb is None or ba is None or float(ba) - float(bb) <= 0:
        return False
    mt = getattr(state, "market_trades", None) or {}
    for t in mt.get(symbol, []) or []:
        if not isinstance(t, Trade):
            continue
        if str(t.seller) != "Mark 22":
            continue
        try:
            p = float(t.price)
        except (TypeError, ValueError):
            continue
        if p <= float(bb) + 1e-9:
            return True
    return False


def _skew_ticks(pos: int, step: int, max_mag: int) -> int:
    if pos == 0:
        return 0
    mag = min(max_mag, abs(pos) // step)
    if mag == 0:
        return 0
    return int(math.copysign(mag, pos))


class Trader:
    SPREAD_TIGHT_TH = 2.0
    EMA_S = 80

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conv = 0
        raw = getattr(state, "traderData", "") or ""
        try:
            td: dict[str, Any] = json.loads(raw) if str(raw).strip() else {}
        except (json.JSONDecodeError, TypeError):
            td = {}

        csv_day = int(getattr(state, "_prosperity4bt_csv_day", td.get("csv_day", 1)))
        td["csv_day"] = csv_day
        ts = int(getattr(state, "timestamp", 0))
        depths: dict = getattr(state, "order_depths", {}) or {}

        def sym(product: str) -> str | None:
            listings = getattr(state, "listings", {}) or {}
            for s, lst in listings.items():
                if getattr(lst, "product", None) == product:
                    return s
            return product if product in depths else None

        du = sym(U)
        if not du or du not in depths:
            return result, conv, json.dumps(td, separators=(",", ":"))

        d_u = depths[du]
        mid_u = micro_mid(d_u)
        if mid_u is None:
            return result, conv, json.dumps(td, separators=(",", ":"))

        sp = bbo_spread(d_u)
        tight = sp is not None and sp <= float(self.SPREAD_TIGHT_TH)
        td["extract_spread_tight"] = bool(tight)
        if sp is not None:
            td["extract_bbo_spread"] = float(sp)

        s5200 = sym("VEV_5200")
        s5300 = sym("VEV_5300")
        joint = False
        if s5200 and s5300 and s5200 in depths and s5300 in depths:
            a = bbo_spread(depths[s5200])
            b = bbo_spread(depths[s5300])
            if a is not None and b is not None:
                joint = a <= _TH and b <= _TH
        td["sonic_joint_tight"] = bool(joint)

        key = f"{csv_day}:{ts}"
        m67_sig = key in _M67_KEYS
        td["mark67_signal_tick"] = bool(m67_sig)

        alpha = 2.0 / (float(self.EMA_S) + 1.0)
        ema = float(td.get("ema_s", mid_u))
        ema = alpha * float(mid_u) + (1.0 - alpha) * ema
        td["ema_s"] = ema

        if tight:
            half_u, qu = 2, 45
            half_v, qv = 3, 18
            if m67_sig:
                half_u, qu = 1, 60
        else:
            half_u, qu = 5, 12
            half_v, qv = 8, 6

        pos = getattr(state, "position", {}) or {}
        pu = int(pos.get(du, 0))
        lim_u = LIMITS[U]
        bu = int(round(ema - half_u))
        au = int(round(ema + half_u))
        bb, ba = book_walls(d_u)[2], book_walls(d_u)[3]
        if bb is not None and ba is not None:
            bu = min(bu, int(ba) - 1)
            au = max(au, int(bb) + 1)
        ex_sk = _skew_ticks(pu, EXTRACT_INV_STEP, EXTRACT_INV_MAX)
        bu -= ex_sk
        au -= ex_sk
        if bb is not None and ba is not None:
            bu = min(bu, int(ba) - 1)
            au = max(au, int(bb) + 1)
        qu = max(1, min(qu, 200))
        if bu < au:
            if pu < lim_u:
                result.setdefault(du, []).append(Order(du, bu, min(qu, lim_u - pu)))
            if pu > -lim_u:
                result.setdefault(du, []).append(Order(du, au, -min(qu, lim_u + pu)))

        if tight and m67_sig and joint and ba is not None and pu < lim_u - 5:
            q_take = min(M67_AGGR_BUY_Q, lim_u - pu)
            if q_take > 0:
                result.setdefault(du, []).append(Order(du, int(ba), q_take))

        sv530 = sym("VEV_5300")
        fade_m22_5300 = (
            bool(joint)
            and sv530 is not None
            and sv530 in depths
            and _m22_aggr_sell_5300_live(state, sv530, depths[sv530])
        )
        td["fade_5300_m22_aggr_sell"] = fade_m22_5300

        for v in VOUCHERS:
            sv = sym(v)
            if not sv or sv not in depths:
                continue
            d = depths[sv]
            m = micro_mid(d)
            if m is None:
                continue
            bbb, baa = book_walls(d)[2], book_walls(d)[3]
            if bbb is None or baa is None:
                continue
            theo = float(m)
            p = int(pos.get(sv, 0))
            lim = LIMITS[v]
            skew = _skew_ticks(p, INV_SKEW_STEP, INV_SKEW_MAX)
            bid_p = int(round(theo - half_v)) - skew
            ask_p = int(round(theo + half_v)) - skew
            bid_p = min(bid_p, int(baa) - 1)
            ask_p = max(ask_p, int(bbb) + 1)
            if bid_p >= ask_p:
                continue
            q = max(1, min(qv, 300))
            fade_m01 = v == "VEV_5300" and joint and _m01_m22_5300_live(state, sv)
            fade_m22 = v == "VEV_5300" and fade_m22_5300
            fade_bid = fade_m01 or fade_m22
            if p < lim and not fade_bid:
                result.setdefault(sv, []).append(Order(sv, bid_p, min(q, lim - p)))
            if p > -lim:
                result.setdefault(sv, []).append(Order(sv, ask_p, -min(q, lim + p)))

        dh = sym(H)
        if dh and dh in depths:
            d_h = depths[dh]
            mh = micro_mid(d_h)
            if mh is not None:
                ph = int(pos.get(dh, 0))
                lim_h = LIMITS[H]
                hh = 3 if tight else 5
                bh = int(round(float(mh) - hh))
                ah = int(round(float(mh) + hh))
                bbh, bah = book_walls(d_h)[2], book_walls(d_h)[3]
                if bbh is not None and bah is not None:
                    bh = min(bh, int(bah) - 1)
                    ah = max(ah, int(bbh) + 1)
                qh = 8 if tight else 5
                if bh < ah:
                    if ph < lim_h:
                        result.setdefault(dh, []).append(Order(dh, bh, min(qh, lim_h - ph)))
                    if ph > -lim_h:
                        result.setdefault(dh, []).append(Order(dh, ah, -min(qh, lim_h + ph)))

        return result, conv, json.dumps(td, separators=(",", ":"))
