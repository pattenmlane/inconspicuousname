from datamodel import Order, OrderDepth, TradingState
import json


class Trader:

    LIMITS = {"ASH_COATED_OSMIUM": 80}

    OSM_K_SS = 0.1353
    OSM_FAIR_STATIC = 10001
    OSM_TAKE_WIDTH = 2
    OSM_CLEAR_WIDTH = 2
    OSM_VOLUME_LIMIT = 30
    OSM_MAKE_EDGE = 1
    OSM_SKEW_UNIT = 12

    def bid(self) -> int:
        return 1103

    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}
        result: dict[str, list[Order]] = {}
        for symbol, depth in state.order_depths.items():
            if symbol == "ASH_COATED_OSMIUM":
                result[symbol] = self._osmium(depth, state.position.get(symbol, 0), td)
            else:
                result[symbol] = []
        return result, 0, json.dumps(td)

    def _osmium(self, d, pos, td):
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
        err_ema += self.OSM_K_SS * (abs(innov) - err_ema)
        td["_osm_err"] = err_ema
        fair += (self.OSM_K_SS / (1.0 + err_ema)) * innov
        td["_osm_f"] = fair

        lim = self.LIMITS["ASH_COATED_OSMIUM"]
        static = self.OSM_FAIR_STATIC
        cw = self.OSM_CLEAR_WIDTH
        orders = []
        bv = sv = 0

        skew = round(pos / self.OSM_SKEW_UNIT)
        ask_limit = max(static, fair) - max(0, self.OSM_TAKE_WIDTH + skew)
        bid_limit = min(static, fair) + max(0, self.OSM_TAKE_WIDTH - skew)
        for a in sorted(d.sell_orders):
            if a > ask_limit:
                break
            q = min(-d.sell_orders[a], lim - pos - bv)
            if q > 0:
                orders.append(Order("ASH_COATED_OSMIUM", a, q)); bv += q
        for b in sorted(d.buy_orders, reverse=True):
            if b < bid_limit:
                break
            q = min(d.buy_orders[b], lim + pos - sv)
            if q > 0:
                orders.append(Order("ASH_COATED_OSMIUM", b, -q)); sv += q

        pos_after = pos + bv - sv
        f_bid = int(round(fair - cw))
        f_ask = int(round(fair + cw))
        long_favorable = fair < static
        short_favorable = fair > static
        if pos_after > 0 and not long_favorable:
            cq = min(pos_after, sum(v for p, v in d.buy_orders.items() if p >= f_ask))
            sent = min(lim + pos - sv, cq)
            if sent > 0:
                orders.append(Order("ASH_COATED_OSMIUM", f_ask, -sent)); sv += sent
        elif pos_after < 0 and not short_favorable:
            cq = min(-pos_after, sum(-v for p, v in d.sell_orders.items() if p <= f_bid))
            sent = min(lim - pos - bv, cq)
            if sent > 0:
                orders.append(Order("ASH_COATED_OSMIUM", f_bid, sent)); bv += sent

        favorable_inv = (pos > 0 and long_favorable) or (pos < 0 and short_favorable)
        if favorable_inv:
            bid_edge = ask_edge = max(1, self.OSM_MAKE_EDGE)
        else:
            bid_edge = max(1, self.OSM_MAKE_EDGE + skew)
            ask_edge = max(1, self.OSM_MAKE_EDGE - skew)
        baaf = min((p for p in d.sell_orders if p > fair + ask_edge - 1), default=None)
        bbbf = max((p for p in d.buy_orders if p < fair - bid_edge + 1), default=None)
        if baaf is not None and bbbf is not None:
            if baaf <= fair + ask_edge and pos <= self.OSM_VOLUME_LIMIT:
                baaf = int(round(fair + ask_edge + 1))
            if bbbf >= fair - bid_edge and pos >= -self.OSM_VOLUME_LIMIT:
                bbbf = int(round(fair - bid_edge - 1))
            buy_q = lim - pos - bv
            if buy_q > 0:
                orders.append(Order("ASH_COATED_OSMIUM", bbbf + 1, buy_q))
            sell_q = lim + pos - sv
            if sell_q > 0:
                orders.append(Order("ASH_COATED_OSMIUM", baaf - 1, -sell_q))
        return orders