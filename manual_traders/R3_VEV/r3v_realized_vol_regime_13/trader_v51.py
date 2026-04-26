"""
Round 3 — vouchers_final_strategy: joint 5200+5300 BBO spread gate (Sonic) + optional
long-bias on VELVETFRUIT_EXTRACT when the gate is ON (STRATEGY.txt layer 2: forward
mid more favorable; toy policy — not guaranteed at bid/ask).

Additionally: one-tick extract mid momentum; when joint tight and recent mid tick is up,
size extract buys further (inclineGod: not only levels — short-horizon path).

BS fair for VEVs from pooled smile (trader_v0 math); no hydrogel.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

_spec = importlib.util.spec_from_file_location(
    "_vf_gate_math", Path(__file__).resolve().parent / "trader_v0.py"
)
_m = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_m)

U = _m.U
STRIKES = _m.STRIKES
VOUCHERS = _m.VOUCHERS
LIMITS = {U: 200, **{v: 300 for v in VOUCHERS}}


class Trader:
    TIGHT_SPREAD_TH = 2.0
    EMA_S = 80

    # Tight: ~2x edge in community description -> larger clips vs v50, tighter quotes
    TIGHT_VEV_HALF = 1.6
    TIGHT_EX_HALF = 1.6
    TIGHT_VEV_SIZE = 85
    TIGHT_EX_SIZE = 85
    TIGHT_TAKE_MULT = 0.52
    TIGHT_EX_LONG_ONLY = True
    TIGHT_MOMO_EX_BONUS = 30

    WIDE_VEV_HALF = 10.0
    WIDE_EX_HALF = 6.0
    WIDE_VEV_SIZE = 4
    WIDE_EX_SIZE = 3
    WIDE_TAKE_MULT = 0.32

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conv = 0
        raw = getattr(state, "traderData", "") or ""
        try:
            td: dict[str, Any] = json.loads(raw) if str(raw).strip() else {}
        except (json.JSONDecodeError, TypeError):
            td = {}

        csv_day = int(getattr(state, "_prosperity4bt_csv_day", td.get("csv_day", 0)))
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
        s5200 = sym("VEV_5200")
        s5300 = sym("VEV_5300")
        if not du or du not in depths or not s5200 or not s5300:
            return result, conv, json.dumps(td, separators=(",", ":"))
        if s5200 not in depths or s5300 not in depths:
            return result, conv, json.dumps(td, separators=(",", ":"))

        sp5200 = _m.bbo_spread(depths[s5200])
        sp5300 = _m.bbo_spread(depths[s5300])
        if sp5200 is None or sp5300 is None:
            return result, conv, json.dumps(td, separators=(",", ":"))

        th = float(self.TIGHT_SPREAD_TH)
        joint_tight = sp5200 <= th and sp5300 <= th
        td["joint_tight_5200_5300"] = joint_tight
        td["s5200_bbo_spread"] = float(sp5200)
        td["s5300_bbo_spread"] = float(sp5300)
        td["spread_5200_x_5300"] = float(sp5200) * float(sp5300)

        d_u: OrderDepth = depths[du]
        mid_u = _m.micro_mid(d_u)
        if mid_u is None:
            return result, conv, json.dumps(td, separators=(",", ":"))

        prev_mid = td.get("extract_mid_prev")
        one_tick_du = 0.0
        if isinstance(prev_mid, (int, float)) and float(prev_mid) > 0.0:
            one_tick_du = float(mid_u) - float(prev_mid)
        td["extract_mid_prev"] = float(mid_u)
        td["one_tick_extract_dmid"] = one_tick_du

        alpha = 2.0 / (float(self.EMA_S) + 1.0)
        ema = float(td.get("ema_s", mid_u))
        ema = alpha * float(mid_u) + (1.0 - alpha) * ema
        td["ema_s"] = ema

        S = float(mid_u)
        T = _m.t_years(csv_day, ts)
        K0 = _m.nearest_strike(S)
        v_atm = f"VEV_{K0}"
        sv = sym(v_atm)
        iv_atm = 0.32
        if sv and sv in depths:
            wm = _m.wall_mid(depths[sv])
            if wm is not None:
                iv0 = _m.implied_vol_bisect(float(wm), S, float(K0), T, 0.0)
                if iv0 is not None:
                    iv_atm = iv0
        td["iv_atm_nearest_strike"] = float(iv_atm)

        if joint_tight:
            half_vev = float(self.TIGHT_VEV_HALF)
            half_u = float(self.TIGHT_EX_HALF)
            qv = int(self.TIGHT_VEV_SIZE)
            qu = int(self.TIGHT_EX_SIZE)
            take_m = float(self.TIGHT_TAKE_MULT)
            max_take = 60
            if one_tick_du > 0.0 and bool(self.TIGHT_EX_LONG_ONLY):
                qu = min(200, qu + int(self.TIGHT_MOMO_EX_BONUS))
        else:
            half_vev = float(self.WIDE_VEV_HALF)
            half_u = float(self.WIDE_EX_HALF)
            qv = int(self.WIDE_VEV_SIZE)
            qu = int(self.WIDE_EX_SIZE)
            take_m = float(self.WIDE_TAKE_MULT)
            max_take = 8

        pos: dict = getattr(state, "position", {}) or {}

        for v in VOUCHERS:
            symv = sym(v)
            if symv is None or symv not in depths:
                continue
            K = float(v.split("_")[1])
            sig_m = _m.model_iv(S, K, T)
            theo = _m.bs_call(S, K, T, sig_m, 0.0)
            d: OrderDepth = depths[symv]
            bb, ba = _m.book_walls(d)[2], _m.book_walls(d)[3]
            if bb is None or ba is None:
                continue
            p = int(pos.get(symv, 0))
            lim = LIMITS[v]
            bid_p = int(round(theo - half_vev))
            ask_p = int(round(theo + half_vev))
            bid_p = min(bid_p, int(ba) - 1)
            ask_p = max(ask_p, int(bb) + 1)
            if bid_p >= ask_p:
                continue
            q = max(1, min(qv, 300))
            take_edge = max(1.0, half_vev * take_m)
            sells = d.sell_orders or {}
            buys = d.buy_orders or {}
            if p < lim:
                rem_buy = min(max_take, lim - p)
                for ap in sorted(sells.keys()):
                    if rem_buy <= 0:
                        break
                    av = abs(int(sells[ap]))
                    if float(ap) <= theo - take_edge:
                        tqty = min(rem_buy, av)
                        if tqty > 0:
                            result.setdefault(symv, []).append(Order(symv, int(ap), int(tqty)))
                            rem_buy -= tqty
                            p += tqty
                    else:
                        break
            if p > -lim:
                rem_sell = min(max_take, lim + p)
                for bp in sorted(buys.keys(), reverse=True):
                    if rem_sell <= 0:
                        break
                    bv = abs(int(buys[bp]))
                    if float(bp) >= theo + take_edge:
                        tqty = min(rem_sell, bv)
                        if tqty > 0:
                            result.setdefault(symv, []).append(Order(symv, int(bp), -int(tqty)))
                            rem_sell -= tqty
                            p -= tqty
                    else:
                        break
            if p < lim:
                result.setdefault(symv, []).append(Order(symv, bid_p, min(q, lim - p)))
            if p > -lim:
                result.setdefault(symv, []).append(Order(symv, ask_p, -min(q, lim + p)))

        pu = int(pos.get(du, 0))
        lim_u = LIMITS[U]
        bu = int(round(ema - half_u))
        au = int(round(ema + half_u))
        bbu, bau = _m.book_walls(d_u)[2], _m.book_walls(d_u)[3]
        if bbu is not None and bau is not None:
            bu = min(bu, int(bau) - 1)
            au = max(au, int(bbu) + 1)
        qu = max(1, min(qu, 200))
        ex_long = bool(self.TIGHT_EX_LONG_ONLY) and joint_tight
        if ex_long:
            if bu < au and pu < lim_u:
                result.setdefault(du, []).append(Order(du, bu, min(qu, lim_u - pu)))
        elif bu < au:
            if pu < lim_u:
                result.setdefault(du, []).append(Order(du, bu, min(qu, lim_u - pu)))
            if pu > -lim_u:
                result.setdefault(du, []).append(Order(du, au, -min(qu, lim_u + pu)))

        return result, conv, json.dumps(td, separators=(",", ":"))
