"""
Round 3 — realized vol vs ATM implied vol regime: joint VEV quote width.

Thesis (r3v_realized_vol_regime_13): when ATM Black–Scholes IV exceeds a short-horizon
realized vol estimate of VELVETFRUIT_EXTRACT, options look rich vs recent moves;
widen all VEV half-spreads together. When IV < RV, tighten spreads.

Timing: T_years from historical CSV day index (0->8d open, 1->7d, 2->6d per
round3work/round3description.txt) plus intraday DTE wind-down as in
plot_iv_smile_round3 (DTE_eff = calendar_dte_open - (timestamp//100)/10000).

Model IV smile: global quadratic in m_t = log(K/S)/sqrt(T), coeffs from
round3work/voucher_work/overall_work/fitted_smile_coeffs.json (embedded).
"""
from __future__ import annotations

import json
import math
from collections import deque
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
LIMITS = {
    H: 200,
    U: 200,
    **{v: 300 for v in VOUCHERS},
}

# np.polyfit coeffs high-to-low for IV in m_t = log(K/S)/sqrt(T)
_COEFFS = (0.14215151147708086, -0.0016298611395181932, 0.23576325646627055)


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)


def bs_delta_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12 or S <= 0 or K <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    return _cdf(d1)


def bs_theta_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """∂C/∂T (years); r=0 => theta = -S·φ(d1)·σ/(2√T) in $/year."""
    if T <= 0 or sigma <= 1e-12 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    v = sigma * sqrtT
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    return -S * _pdf(d1) * sigma / (2.0 * sqrtT) - r * K * math.exp(-r * T) * _cdf(
        (math.log(S / K) + (r - 0.5 * sigma * sigma) * T) / v
    )


def bs_gamma_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """d²C/dS² (per $ of underlying) at given sigma; r=0 standard BS gamma."""
    if T <= 0 or sigma <= 1e-12 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    v = sigma * sqrtT
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    return _pdf(d1) / (S * sigma * sqrtT)


def implied_vol_bisect(price: float, S: float, K: float, T: float, r: float = 0.0) -> float | None:
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-9 or price >= S - 1e-9 or S <= 0 or K <= 0 or T <= 0:
        return None
    lo, hi = 1e-5, 12.0
    if bs_call(S, K, T, lo, r) > price or bs_call(S, K, T, hi, r) < price:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, mid, r) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def model_iv(S: float, K: float, T: float) -> float:
    if S <= 0 or K <= 0 or T <= 0:
        return 0.25
    m_t = math.log(K / S) / math.sqrt(T)
    a, b, c = _COEFFS
    iv = ((a * m_t) + b) * m_t + c
    return max(iv, 1e-4)


def dte_from_csv_day(day: int) -> int:
    return 8 - int(day)


def intraday_progress(ts: int) -> float:
    return (int(ts) // 100) / 10_000.0


def t_years(csv_day: int, ts: int) -> float:
    dte = max(float(dte_from_csv_day(csv_day)) - intraday_progress(ts), 1e-6)
    return dte / 365.0


def book_walls(depth: OrderDepth) -> tuple[int | None, int | None, int | None, int | None]:
    buys = depth.buy_orders or {}
    sells = depth.sell_orders or {}
    if not buys and not sells:
        return None, None, None, None
    bid_wall = min(buys.keys())
    sell_prices = list(sells.keys())
    ask_wall = max(sell_prices) if sell_prices else None
    best_bid = max(buys.keys())
    best_ask = min(sell_prices) if sell_prices else None
    return bid_wall, ask_wall, best_bid, best_ask


def wall_mid(depth: OrderDepth) -> float | None:
    bw, aw, bb, ba = book_walls(depth)
    if bw is not None and aw is not None:
        return (float(bw) + float(aw)) / 2.0
    if bb is not None and ba is not None:
        return (float(bb) + float(ba)) / 2.0
    return None


def micro_mid(depth: OrderDepth) -> float | None:
    _, _, bb, ba = book_walls(depth)
    if bb is None or ba is None:
        return None
    return (float(bb) + float(ba)) / 2.0


def bbo_spread(depth: OrderDepth) -> float | None:
    """Top-of-book spread ask₁−bid₁ (same units as price tape)."""
    _, _, bb, ba = book_walls(depth)
    if bb is None or ba is None:
        return None
    return float(ba) - float(bb)


def nearest_strike(S: float) -> int:
    return min(STRIKES, key=lambda k: abs(float(k) - S))


class Trader:
    # Regime → joint width (grid anchor for v0)
    BASE_VEV_HALF = 2
    K_WIDEN = 10.0
    K_TIGHTEN = 5.0
    RV_WIN = 40
    EMA_S = 80
    BASE_EX_HALF = 2
    REG_EX_SCALE = 0.35
    BASE_H_HALF = 2
    REG_H_SCALE = 0.2
    ORDER_SIZE_VEV = 12
    ORDER_SIZE_U = 8
    ORDER_SIZE_H = 10
    TAKE_EDGE_MULT = 0.55
    MAX_TAKE_PER_SIDE = 24
    VEV_HALF_MIN = 1.0
    VEV_HALF_MAX = 40.0
    # Optional delta hedge vs extract: fraction of -sum(delta_i * pos_i) to offset per tick (0 = disabled).
    DELTA_HEDGE_STRENGTH = 0.0
    MAX_D_HEDGE_QTY = 0
    # Optional: blend ATM time-decay (theta) into the IV−RV regime scalar (0 = off).
    THETA_REGIME_WEIGHT = 0.0
    THETA_REGIME_NORM = 0.04
    # Optional directional short-vol bias from regime sign; keeps same IV-vs-RV thesis
    # but nudges quote center so asks are lower / bids higher when IV > RV.
    REGIME_CENTER_SHIFT = 0.0
    # Optional: scale joint regime by normalized ATM call gamma (0 = off).
    GAMMA_REGIME_WEIGHT = 0.0
    GAMMA_REGIME_NORM = 0.0008
    # Optional: widen joint VEV half-spread when |Δlog S| exceeds threshold (microstructure shock).
    SHOCK_VEV_HALF_ADD = 0.0
    SHOCK_ABS_LOG_DU = 0.0012
    # Optional: per-strike add to half width when |d log S| is large; use sign of du for
    # different maps (tape: low strikes widen more on up-shocks vs down). If both are set,
    # they take precedence over SHOCK_VEV_HALF_ADD_MAP for the signed component.
    UP_SHOCK_VEV_HALF_ADD_MAP: dict | None = None
    DN_SHOCK_VEV_HALF_ADD_MAP: dict | None = None
    # Set False to benchmark extract+VEV only (still allowed to quote other products if needed).
    TRADE_HYDROGEL = True
    # Optional: per-strike add to VEV half-spread after joint regime + shocks (e.g. from tape
    # cross-section of top-of-book spread vs model vega). Scaled by VEV_HALF_LOCAL_ADD_SCALE.
    VEV_HALF_LOCAL_ADD_MAP: dict | None = None
    VEV_HALF_LOCAL_ADD_SCALE = 1.0
    # Optional: "joint tight book" filter (vouchers_final_strategy): both VEV_5200 and VEV_5300
    # BBO spread <= TIGHT_SPREAD_TH at the same tick — risk-on / size up; else scale down and
    # optionally widen VEV and extract half-spreads. Off by default (backward compatible).
    USE_TIGHT_GATE_5200_5300 = False
    TIGHT_SPREAD_TH = 2.0
    TIGHT_GATE_TIGHT_VEV_SIZE_MULT = 1.0
    TIGHT_GATE_TIGHT_EX_SIZE_MULT = 1.0
    TIGHT_GATE_TIGHT_VEV_HALF_MULT = 1.0
    TIGHT_GATE_TIGHT_EX_HALF_MULT = 1.0
    TIGHT_GATE_LOOSE_VEV_SIZE_MULT = 1.0
    TIGHT_GATE_LOOSE_EX_SIZE_MULT = 1.0
    TIGHT_GATE_LOOSE_VEV_HALF_MULT = 1.0
    TIGHT_GATE_LOOSE_EX_HALF_MULT = 1.0

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

        depths = getattr(state, "order_depths", {}) or {}
        ts = int(getattr(state, "timestamp", 0))

        def sym(p: str) -> str | None:
            listings = getattr(state, "listings", {}) or {}
            for s, lst in listings.items():
                if getattr(lst, "product", None) == p:
                    return s
            return p if p in depths else None

        du = sym(U)
        if du is None or du not in depths:
            return result, conv, json.dumps(td, separators=(",", ":"))

        d_u: OrderDepth = depths[du]
        mid_u = micro_mid(d_u)
        if mid_u is None:
            return result, conv, json.dumps(td, separators=(",", ":"))

        alpha_s = 2.0 / (self.EMA_S + 1.0)
        ema_s = float(td.get("ema_s", mid_u))
        ema_s = alpha_s * mid_u + (1.0 - alpha_s) * ema_s
        td["ema_s"] = ema_s

        prev = td.get("s_prev")
        rets_raw = td.get("rets")
        if isinstance(rets_raw, list):
            rets = deque((float(x) for x in rets_raw if isinstance(x, (int, float))), maxlen=self.RV_WIN * 2)
        else:
            rets = deque(maxlen=self.RV_WIN * 2)
        if isinstance(prev, (int, float)) and float(prev) > 0:
            rets.append(math.log(float(mid_u) / float(prev)))
        td["s_prev"] = float(mid_u)
        td["rets"] = list(rets)

        rv_ann = 0.25
        if len(rets) >= 5:
            xs = list(rets)[-self.RV_WIN :]
            m = sum(xs) / len(xs)
            var = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
            sig = math.sqrt(max(var, 0.0))
            rv_ann = sig * math.sqrt(252.0 * 10_000.0)

        T = t_years(csv_day, ts)
        S = float(mid_u)
        K0 = nearest_strike(S)
        v_atm = f"VEV_{K0}"
        sv = sym(v_atm)
        iv_atm = 0.3
        if sv and sv in depths:
            wm = wall_mid(depths[sv])
            if wm is not None:
                iv0 = implied_vol_bisect(float(wm), S, float(K0), T, 0.0)
                if iv0 is not None:
                    iv_atm = iv0

        regime = iv_atm - rv_ann
        if self.THETA_REGIME_WEIGHT > 0.0 and T > 0 and S > 0:
            th = bs_theta_call(S, float(K0), T, iv_atm, 0.0)
            carry = max(0.0, -th) / max(S, 1.0)
            boost = min(2.0, carry / max(self.THETA_REGIME_NORM, 1e-9))
            regime = regime * (1.0 + self.THETA_REGIME_WEIGHT * boost)
        if self.GAMMA_REGIME_WEIGHT > 0.0 and T > 0 and S > 0:
            gam = bs_gamma_call(S, float(K0), T, iv_atm, 0.0)
            gboost = min(2.0, gam / max(self.GAMMA_REGIME_NORM, 1e-12))
            regime = regime * (1.0 + self.GAMMA_REGIME_WEIGHT * gboost)
        half_vev = float(
            self.BASE_VEV_HALF + self.K_WIDEN * max(0.0, regime) - self.K_TIGHTEN * max(0.0, -regime)
        )
        du_inst = 0.0
        if isinstance(prev, (int, float)) and float(prev) > 0:
            du_inst = math.log(float(mid_u) / float(prev))
        td["last_du_extract"] = du_inst

        use_tg = bool(getattr(self, "USE_TIGHT_GATE_5200_5300", False))
        th_tight = float(getattr(self, "TIGHT_SPREAD_TH", 2.0))
        joint_tight = False
        s5200_s: float | None = None
        s5300_s: float | None = None
        u_sz_m = 1.0
        v_sz_m = 1.0
        if use_tg:
            s5k = sym("VEV_5200")
            s5k2 = sym("VEV_5300")
            if s5k and s5k2 and s5k in depths and s5k2 in depths:
                sp0 = bbo_spread(depths[s5k])
                sp1 = bbo_spread(depths[s5k2])
                if sp0 is not None and sp1 is not None:
                    s5200_s, s5300_s = sp0, sp1
                    joint_tight = sp0 <= th_tight and sp1 <= th_tight
            if joint_tight:
                u_sz_m = float(getattr(self, "TIGHT_GATE_TIGHT_EX_SIZE_MULT", 1.0))
                v_sz_m = float(getattr(self, "TIGHT_GATE_TIGHT_VEV_SIZE_MULT", 1.0))
            else:
                u_sz_m = float(getattr(self, "TIGHT_GATE_LOOSE_EX_SIZE_MULT", 1.0))
                v_sz_m = float(getattr(self, "TIGHT_GATE_LOOSE_VEV_SIZE_MULT", 1.0))
        td["joint_tight_5200_5300"] = bool(joint_tight) if use_tg else False
        if s5200_s is not None:
            td["s5200_bbo_spread"] = float(s5200_s)
        if s5300_s is not None:
            td["s5300_bbo_spread"] = float(s5300_s)

        vtg = 1.0
        if use_tg:
            vtg = float(
                getattr(
                    self,
                    "TIGHT_GATE_TIGHT_VEV_HALF_MULT" if joint_tight else "TIGHT_GATE_LOOSE_VEV_HALF_MULT",
                    1.0,
                )
            )
        half_vev *= vtg
        if float(getattr(self, "SHOCK_VEV_HALF_ADD", 0.0)) > 0.0:
            thr = float(getattr(self, "SHOCK_ABS_LOG_DU", 0.0012))
            if abs(du_inst) >= thr:
                half_vev += float(self.SHOCK_VEV_HALF_ADD)
        half_vev = max(self.VEV_HALF_MIN, min(half_vev, self.VEV_HALF_MAX))

        half_u = max(1.0, self.BASE_EX_HALF + self.REG_EX_SCALE * abs(regime))
        if use_tg:
            half_u *= float(
                getattr(
                    self,
                    "TIGHT_GATE_TIGHT_EX_HALF_MULT" if joint_tight else "TIGHT_GATE_LOOSE_EX_HALF_MULT",
                    1.0,
                )
            )
        half_h = max(1.0, self.BASE_H_HALF + self.REG_H_SCALE * abs(regime))

        td["last_iv_atm"] = iv_atm
        td["last_rv_ann"] = rv_ann
        td["last_regime"] = regime
        td["last_half_vev"] = half_vev

        pos = getattr(state, "position", {}) or {}

        # VEVs: same half width, model theoretical as center
        for v in VOUCHERS:
            symv = sym(v)
            if symv is None or symv not in depths:
                continue
            K = float(v.split("_")[1])
            sig_m = model_iv(S, K, T)
            theo = bs_call(S, K, T, sig_m, 0.0)
            d = depths[symv]
            bb, ba = book_walls(d)[2], book_walls(d)[3]
            if bb is None or ba is None:
                continue
            p = int(pos.get(symv, 0))
            lim = LIMITS[v]
            skew = 0
            if p > 15:
                skew = -1
            elif p < -15:
                skew = 1
            center_shift = self.REGIME_CENTER_SHIFT * regime
            half_vev_local = half_vev
            if float(getattr(self, "SHOCK_VEV_HALF_ADD", 0.0)) > 0.0:
                thr = float(getattr(self, "SHOCK_ABS_LOG_DU", 0.0012))
                if abs(du_inst) >= thr:
                    half_vev_local += float(self.SHOCK_VEV_HALF_ADD)
            thr_s = float(getattr(self, "SHOCK_ABS_LOG_DU", 0.0012))
            up_m = getattr(self, "UP_SHOCK_VEV_HALF_ADD_MAP", None)
            dn_m = getattr(self, "DN_SHOCK_VEV_HALF_ADD_MAP", None)
            if (
                isinstance(up_m, dict)
                and isinstance(dn_m, dict)
                and abs(du_inst) >= thr_s
            ):
                if du_inst > 0.0:
                    half_vev_local += float(up_m.get(v, 0.0))
                elif du_inst < 0.0:
                    half_vev_local += float(dn_m.get(v, 0.0))
            else:
                shock_map = getattr(self, "SHOCK_VEV_HALF_ADD_MAP", None)
                if isinstance(shock_map, dict) and abs(du_inst) >= thr_s:
                    half_vev_local += float(shock_map.get(v, 0.0))
            loc_map = getattr(self, "VEV_HALF_LOCAL_ADD_MAP", None)
            loc_sc = float(getattr(self, "VEV_HALF_LOCAL_ADD_SCALE", 1.0))
            if isinstance(loc_map, dict) and loc_sc != 0.0:
                half_vev_local += loc_sc * float(loc_map.get(v, 0.0))
            half_vev_local = max(self.VEV_HALF_MIN, min(half_vev_local, self.VEV_HALF_MAX))
            bid_p = int(round(theo - half_vev_local + skew + center_shift))
            ask_p = int(round(theo + half_vev_local + skew + center_shift))
            bid_p = min(bid_p, int(ba) - 1)
            ask_p = max(ask_p, int(bb) + 1)
            if bid_p >= ask_p:
                continue
            q_default = self.ORDER_SIZE_VEV
            q_map = getattr(self, "ORDER_SIZE_VEV_MAP", None)
            q = int(q_map.get(v, q_default)) if isinstance(q_map, dict) else int(q_default)
            q = max(1, int(round(float(q) * v_sz_m)))
            # Controlled taking when market is clearly mispriced vs same theo anchor.
            # We still use the same shared regime logic; this only improves execution quality.
            take_edge = max(1.0, half_vev_local * self.TAKE_EDGE_MULT)
            sells = d.sell_orders or {}
            buys = d.buy_orders or {}
            max_take = self.MAX_TAKE_PER_SIDE
            take_map = getattr(self, "MAX_TAKE_PER_SIDE_MAP", None)
            if isinstance(take_map, dict):
                max_take = int(take_map.get(v, max_take))
            max_take = max(1, int(round(float(max_take) * v_sz_m)))
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

        # Optional: reduce net option delta against VELVETFRUIT_EXTRACT (call deltas × voucher positions).
        if (
            self.DELTA_HEDGE_STRENGTH > 0.0
            and self.MAX_D_HEDGE_QTY > 0
            and du in depths
        ):
            net_call_delta = 0.0
            for v in VOUCHERS:
                symv = sym(v)
                if symv is None:
                    continue
                K = float(v.split("_")[1])
                sig_m = model_iv(S, K, T)
                dlt = bs_delta_call(S, K, T, sig_m, 0.0)
                net_call_delta += dlt * float(int(pos.get(symv, 0)))
            h_raw = -self.DELTA_HEDGE_STRENGTH * net_call_delta
            hqty = int(round(h_raw))
            if hqty != 0:
                hqty = max(-self.MAX_D_HEDGE_QTY, min(self.MAX_D_HEDGE_QTY, hqty))
                d_u2: OrderDepth = depths[du]
                bb_h, ba_h = book_walls(d_u2)[2], book_walls(d_u2)[3]
                lim_uh = LIMITS[U]
                pu_h = int(pos.get(du, 0))
                if hqty > 0 and ba_h is not None and pu_h < lim_uh:
                    q = min(hqty, lim_uh - pu_h)
                    if q > 0:
                        result.setdefault(du, []).append(Order(du, int(ba_h), q))
                elif hqty < 0 and bb_h is not None and pu_h > -lim_uh:
                    q = min(-hqty, lim_uh + pu_h)
                    if q > 0:
                        result.setdefault(du, []).append(Order(du, int(bb_h), -q))

        # Underlying extract
        pu = int(pos.get(du, 0))
        lim_u = LIMITS[U]
        bu = int(round(ema_s - half_u))
        au = int(round(ema_s + half_u))
        bb, ba = book_walls(d_u)[2], book_walls(d_u)[3]
        if bb is not None and ba is not None:
            bu = min(bu, int(ba) - 1)
            au = max(au, int(bb) + 1)
        q_u = max(1, int(round(float(self.ORDER_SIZE_U) * u_sz_m)))
        if bu < au:
            if pu < lim_u:
                result.setdefault(du, []).append(Order(du, bu, min(q_u, lim_u - pu)))
            if pu > -lim_u:
                result.setdefault(du, []).append(Order(du, au, -min(q_u, lim_u + pu)))

        # Hydrogel: simple MM around mid scaled by regime
        if getattr(self, "TRADE_HYDROGEL", True):
            dh = sym(H)
            if dh and dh in depths:
                d_h = depths[dh]
                mh = micro_mid(d_h)
                if mh is not None:
                    ph = int(pos.get(dh, 0))
                    lim_h = LIMITS[H]
                    bh = int(round(float(mh) - half_h))
                    ah = int(round(float(mh) + half_h))
                    bbh, bah = book_walls(d_h)[2], book_walls(d_h)[3]
                    if bbh is not None and bah is not None:
                        bh = min(bh, int(bah) - 1)
                        ah = max(ah, int(bbh) + 1)
                    if bh < ah:
                        if ph < lim_h:
                            result.setdefault(dh, []).append(Order(dh, bh, min(self.ORDER_SIZE_H, lim_h - ph)))
                        if ph > -lim_h:
                            result.setdefault(dh, []).append(Order(dh, ah, -min(self.ORDER_SIZE_H, lim_h + ph)))

        return result, conv, json.dumps(td, separators=(",", ":"))
