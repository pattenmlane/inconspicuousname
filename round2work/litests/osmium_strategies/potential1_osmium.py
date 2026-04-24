from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth

SYMBOL = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80

OSM_K_SS = 0.1353
OSM_FAIR_STATIC = 10001
OSM_TAKE_WIDTH = 2
OSM_CLEAR_WIDTH = 2
OSM_VOLUME_LIMIT = 30
OSM_MAKE_EDGE = 1
OSM_SKEW_UNIT = 12


def osmium_step(
    depth: OrderDepth,
    position: int,
    trader_data: dict[str, Any] | None = None,
) -> tuple[list[Order], dict[str, Any]]:
    """
    One tick of potential1 osmium. Mutates a **copy** of ``trader_data`` and returns it.
    """
    td = dict(trader_data or {})
    orders = _osmium(depth, int(position), td)
    return orders, td


def _osmium(d: OrderDepth, pos: int, td: dict[str, Any]) -> list[Order]:
    if not d.buy_orders or not d.sell_orders:
        return []
    bb = max(d.buy_orders)
    ba = min(d.sell_orders)
    bv_tob = d.buy_orders[bb]
    av_tob = -d.sell_orders[ba]
    tot = bv_tob + av_tob
    micro = (bb * av_tob + ba * bv_tob) / tot if tot > 0 else (bb + ba) / 2.0

    fair = td.get("_osm_f", micro)
    innov = micro - fair
    err_ema = td.get("_osm_err", abs(innov))
    err_ema += OSM_K_SS * (abs(innov) - err_ema)
    td["_osm_err"] = err_ema
    fair += (OSM_K_SS / (1.0 + err_ema)) * innov
    td["_osm_f"] = fair

    lim = POSITION_LIMIT
    static = OSM_FAIR_STATIC
    cw = OSM_CLEAR_WIDTH
    orders: list[Order] = []
    bv = sv = 0

    skew = round(pos / OSM_SKEW_UNIT)
    ask_limit = max(static, fair) - max(0, OSM_TAKE_WIDTH + skew)
    bid_limit = min(static, fair) + max(0, OSM_TAKE_WIDTH - skew)
    for a in sorted(d.sell_orders):
        if a > ask_limit:
            break
        q = min(-d.sell_orders[a], lim - pos - bv)
        if q > 0:
            orders.append(Order(SYMBOL, a, q))
            bv += q
    for b in sorted(d.buy_orders, reverse=True):
        if b < bid_limit:
            break
        q = min(d.buy_orders[b], lim + pos - sv)
        if q > 0:
            orders.append(Order(SYMBOL, b, -q))
            sv += q

    pos_after = pos + bv - sv
    f_bid = int(round(fair - cw))
    f_ask = int(round(fair + cw))
    long_favorable = fair < static
    short_favorable = fair > static
    if pos_after > 0 and not long_favorable:
        cq = min(pos_after, sum(v for p, v in d.buy_orders.items() if p >= f_ask))
        sent = min(lim + pos - sv, cq)
        if sent > 0:
            orders.append(Order(SYMBOL, f_ask, -sent))
            sv += sent
    elif pos_after < 0 and not short_favorable:
        cq = min(-pos_after, sum(-v for p, v in d.sell_orders.items() if p <= f_bid))
        sent = min(lim - pos - bv, cq)
        if sent > 0:
            orders.append(Order(SYMBOL, f_bid, sent))
            bv += sent

    favorable_inv = (pos > 0 and long_favorable) or (pos < 0 and short_favorable)
    if favorable_inv:
        bid_edge = ask_edge = max(1, OSM_MAKE_EDGE)
    else:
        bid_edge = max(1, OSM_MAKE_EDGE + skew)
        ask_edge = max(1, OSM_MAKE_EDGE - skew)
    baaf = min((p for p in d.sell_orders if p > fair + ask_edge - 1), default=None)
    bbbf = max((p for p in d.buy_orders if p < fair - bid_edge + 1), default=None)
    if baaf is not None and bbbf is not None:
        if baaf <= fair + ask_edge and pos <= OSM_VOLUME_LIMIT:
            baaf = int(round(fair + ask_edge + 1))
        if bbbf >= fair - bid_edge and pos >= -OSM_VOLUME_LIMIT:
            bbbf = int(round(fair - bid_edge - 1))
        buy_q = lim - pos - bv
        if buy_q > 0:
            orders.append(Order(SYMBOL, bbbf + 1, buy_q))
        sell_q = lim + pos - sv
        if sell_q > 0:
            orders.append(Order(SYMBOL, baaf - 1, -sell_q))
    return orders


def trader_data_dumps(td: dict[str, Any]) -> str:
    return json.dumps(td, separators=(",", ":"))
