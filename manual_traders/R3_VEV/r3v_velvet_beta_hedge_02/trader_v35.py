"""
r3v_velvet_beta_hedge_02 — iteration 35 (parent: v33).

Shared thesis (round3work/vouchers_final_strategy/STRATEGY.txt): joint tight two-leg gate
— VEV_5200 and VEV_5300 both have top-of-book spread <= TH (BBO: ask1-bid1, tape units) =>
risk-on: scale VEV tilt; one-tick-inside VELVETFRUIT_EXTRACT when gate is on. VEV quotes
still use synthetic inside spread as in v22/v33. v33 shock guard on |ret1| retained. Hydrogel
MM restored for backtest comparability; track VEV+U subtotal in results if excluding hydrogel.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from scipy.stats import norm

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
V5200 = "VEV_5200"
V5300 = "VEV_5300"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
LIMITS = {HYDRO: 200, U: 200, **{v: 300 for v in VOUCHERS}}
_COEFFS = [0.14215151147708086, -0.0016298611395181932, 0.23576325646627055]


def _dte_open(csv_day: int) -> int:
    return 8 - int(csv_day)


def _dte_eff(csv_day: int, local_ts: int) -> float:
    return max(float(_dte_open(csv_day)) - (int(local_ts) // 100) / 10_000.0, 1e-6)


def t_years(csv_day: int, local_ts: int) -> float:
    return _dte_eff(csv_day, local_ts) / 365.0


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(S * float(norm.pdf(d1)) * math.sqrt(T))


def iv_smile(S: float, K: float, T: float) -> float:
    if S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    m_t = math.log(K / S) / math.sqrt(T)
    return float(np.polyval(np.asarray(_COEFFS, dtype=float), m_t))


def vega_only(S: float, K: float, T: float) -> float:
    sig = iv_smile(S, K, T)
    if not math.isfinite(sig) or sig <= 0:
        return 0.0
    return bs_vega(S, K, T, sig, 0.0)


def _book(depth: OrderDepth) -> tuple[int | None, int | None, float | None]:
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys and not sells:
        return None, None, None
    bb = max(buys.keys())
    ba = min(sells.keys())
    bw = min(buys.keys())
    aw = max(sells.keys())
    return bb, ba, (float(bw) + float(aw)) / 2.0


def _bbo_spread(depth: OrderDepth) -> int | None:
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None
    bb = max(buys.keys())
    ba = min(sells.keys())
    if ba <= bb:
        return None
    return int(ba - bb)


def _synth(bb: int | None, ba: int | None, wm: float | None) -> tuple[int, int, float]:
    if wm is not None and bb is not None and ba is not None:
        return bb, ba, wm
    if ba is not None and bb is None:
        return int(ba) - 1, int(ba), float(ba) - 0.5
    if bb is not None and ba is None:
        return int(bb), int(bb) + 1, float(bb) + 0.5
    return 0, 0, 0.0


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _merged_day(ts: int) -> tuple[int, int]:
    return int(ts) // 1_000_000, int(ts) % 1_000_000


class Trader:
    MID_HIST = 40
    WARMUP_STEPS = 11
    TILT_RET_SCALE = 2.5
    TILT_Z_SCALE = 1.35
    TILT_AMP = 4.0
    W_RET = 1.0
    W_Z = 0.52
    VEGA_REF = 33.0
    VEV_SIZE = 18
    # Joint tight gate (STRATEGY.txt TH=2 on BBO spread, same tick units as tape)
    TIGHT_SPREAD_TH = 2
    TIGHT_VEV_TILT_MULT = 1.06
    TIGHT_EXTRACT_MM = True
    U_SIZE = 18
    HYDRO_SIZE = 11
    # spread_w = (ba2-bb2) synthetic for voucher
    SPREAD_TILT_CUT1 = 2
    SPREAD_TILT_CUT2 = 5
    SPREAD_TILT_CUT3 = 8
    SPREAD_TILT_MULT1 = 0.84
    SPREAD_TILT_MULT2 = 0.64
    SPREAD_TILT_MULT3 = 0.52
    SHOCK_ABS_RET1 = 1.0
    SHOCK_TILT_MULT = 0.9

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        hist = store.get("u_mid_hist")
        if not isinstance(hist, list):
            hist = []
        hist = [float(x) for x in hist[-self.MID_HIST :] if isinstance(x, (int, float))]

        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}
        pos: dict[str, int] = getattr(state, "position", None) or {}
        ts = int(getattr(state, "timestamp", 0))
        csv_day, local_ts = _merged_day(ts)
        csv_day = min(max(csv_day, 0), 2)

        if U not in depths:
            store["u_mid_hist"] = hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        bb_u, ba_u, _ = _book(depths[U])
        if bb_u is None or ba_u is None:
            store["u_mid_hist"] = hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        u_mid = 0.5 * (float(bb_u) + float(ba_u))
        hist.append(u_mid)
        hist = hist[-self.MID_HIST :]

        ret1 = (hist[-1] - hist[-2]) if len(hist) >= 2 else 0.0
        ar1 = abs(ret1)
        z20 = 0.0
        if len(hist) >= 5:
            a = np.asarray(hist[-20:], dtype=float)
            z20 = (u_mid - float(a.mean())) / (float(a.std()) + 1e-6)

        def clip(x: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, x))

        tilt = self.W_RET * clip(ret1 / self.TILT_RET_SCALE, -1.0, 1.0) + self.W_Z * clip(
            z20 / self.TILT_Z_SCALE, -1.0, 1.0
        )
        tilt_px = tilt * self.TILT_AMP
        shock = self.SHOCK_TILT_MULT if ar1 > self.SHOCK_ABS_RET1 else 1.0

        T = t_years(csv_day, local_ts)
        if ts // 100 < self.WARMUP_STEPS:
            store["u_mid_hist"] = hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        s52 = _bbo_spread(depths[V5200]) if V5200 in depths else None
        s53 = _bbo_spread(depths[V5300]) if V5300 in depths else None
        joint_tight = (
            s52 is not None
            and s53 is not None
            and s52 <= self.TIGHT_SPREAD_TH
            and s53 <= self.TIGHT_SPREAD_TH
        )
        vev_gate = self.TIGHT_VEV_TILT_MULT if joint_tight else 1.0

        orders: dict[str, list[Order]] = {}
        for v in VOUCHERS:
            if v not in depths:
                continue
            bb, ba, wm = _book(depths[v])
            bb2, ba2, _ = _synth(bb, ba, wm)
            if ba2 <= bb2 + 1:
                continue
            spread_w = int(ba2 - bb2)
            if spread_w <= self.SPREAD_TILT_CUT1:
                sm = 1.0
            elif spread_w <= self.SPREAD_TILT_CUT2:
                sm = self.SPREAD_TILT_MULT1
            elif spread_w <= self.SPREAD_TILT_CUT3:
                sm = self.SPREAD_TILT_MULT2
            else:
                sm = self.SPREAD_TILT_MULT3

            K = float(v.split("_")[1])
            sc = min(max(vega_only(u_mid, K, T) / self.VEGA_REF, 0.12), 2.6)
            sh = int(round(tilt_px * sc * sm * shock * vev_gate))
            base_bid = bb2 + 1
            base_ask = ba2 - 1
            if base_bid >= base_ask:
                continue
            bid_p = max(bb2 + 1, min(base_bid + sh, ba2 - 2))
            ask_p = min(ba2 - 1, max(base_ask + sh, bb2 + 2))
            if bid_p >= ask_p:
                continue
            p0 = int(pos.get(v, 0))
            lim = LIMITS[v]
            qb = min(self.VEV_SIZE, max(0, lim - p0))
            qs = min(self.VEV_SIZE, max(0, lim + p0))
            lo: list[Order] = []
            if qb > 0:
                lo.append(Order(v, bid_p, qb))
            if qs > 0:
                lo.append(Order(v, ask_p, -qs))
            if lo:
                orders[v] = lo

        if self.TIGHT_EXTRACT_MM and joint_tight and U in depths:
            bbu, bau, _ = _book(depths[U])
            if bbu is not None and bau is not None and bau > bbu + 1:
                e_b = int(bbu) + 1
                e_a = int(bau) - 1
                if e_b < e_a:
                    pu = int(pos.get(U, 0))
                    qu = min(self.U_SIZE, max(0, LIMITS[U] - pu))
                    qus = min(self.U_SIZE, max(0, LIMITS[U] + pu))
                    uo: list[Order] = []
                    if qu > 0:
                        uo.append(Order(U, e_b, qu))
                    if qus > 0:
                        uo.append(Order(U, e_a, -qus))
                    if uo:
                        orders[U] = uo

        if HYDRO in depths:
            bbh, bah, wh = _book(depths[HYDRO])
            if bbh is not None and bah is not None and wh is not None:
                spread_h = max(1, bah - bbh)
                ph_b = max(bbh + 1, int(round(wh - spread_h // 5)))
                ph_a = min(bah - 1, int(round(wh + spread_h // 5)))
                if ph_b < ph_a:
                    ph0 = int(pos.get(HYDRO, 0))
                    qh = min(self.HYDRO_SIZE, max(0, LIMITS[HYDRO] - ph0))
                    qhs = min(self.HYDRO_SIZE, max(0, LIMITS[HYDRO] + ph0))
                    ho: list[Order] = []
                    if qh > 0:
                        ho.append(Order(HYDRO, ph_b, qh))
                    if qhs > 0:
                        ho.append(Order(HYDRO, ph_a, -qhs))
                    if ho:
                        orders[HYDRO] = ho

        store["u_mid_hist"] = hist
        return orders, 0, json.dumps(store, separators=(",", ":"))
