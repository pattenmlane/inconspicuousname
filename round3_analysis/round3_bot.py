"""Round 3 production bot — single file, Python stdlib only.

Splits cleanly into two completely independent books (Magritte hint: this is not a pipe):

  Book 1: HYDROGEL_PACK
    - Independent asset, mean-reverting around ~9990 with std ~32.
    - Strategy: pure market-maker that quotes inside the wall, with inventory skew
      and a layered "deep mean-reversion" taker that buys very low / sells very high.

  Book 2: VELVETFRUIT_EXTRACT + 10 VEV vouchers
    - Quadratic IV smile in m_t = log(K/S)/sqrt(T): IV(m) = a*m^2 + b*m + c
      (calibrated from days 0-2; ATM IV stable ~ 0.236)
    - Per-strike systematic mispricings (in vol points) -> fair price tilts.
    - For each VEV: theo = BS(S, K, T, smile_iv + per_strike_tilt). When a market
      ask is below theo by enough, BUY; when bid is above theo by enough, SELL.
    - We delta-hedge the net option delta on VELVETFRUIT_EXTRACT.
    - Deep ITM (4000, 4500): trade the intrinsic-arb (price - max(S-K,0))
      with a tight band; deep OTM (6000, 6500) are pinned at 0.5 -> never trade.
"""
from __future__ import annotations

import json
import math
from typing import Any

# ----------- Imports for the competition datamodel (with a backtester fallback) -----------
try:
    from datamodel import Order, OrderDepth, TradingState  # type: ignore
except ImportError:  # pragma: no cover
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState  # type: ignore


# =============================================================================
# Calibration constants (frozen from analysis on Prosperity4Data/ROUND_3 days 0-2)
# =============================================================================
UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDROGEL = "HYDROGEL_PACK"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]

POS_LIM = {
    HYDROGEL: 200,
    UNDERLYING: 200,
    **{v: 300 for v in VOUCHERS},
}

# DTE convention: at the start of round 3 (production day) the options have 5 DTE.
PRODUCTION_DTE_AT_OPEN = 5
STEPS_PER_DAY = 10_000  # 1_000_000 / 100
ANNUAL_TICKS = 365 * STEPS_PER_DAY

# Quadratic IV smile in m_t = log(K/S)/sqrt(T). Coefficients in numpy polyfit order
# (m^2, m, const). Calibrated on subsampled days 0-2 of historical Round 3 data.
SMILE_A = 0.14215151147708086
SMILE_B = -0.0016298611395181932
SMILE_C = 0.23576325646627055

# Per-strike *vol* mispricing (smile residual). Mean of (market_iv - smile_iv).
# Positive => market overprices => SELL; negative => underprices => BUY.
# These come from round3_analysis/01_smile_analysis.py (pooled days 0-2).
PER_STRIKE_TILT = {
    4000: 0.0,    # intrinsic-pinned, treat with intrinsic-band logic instead
    4500: 0.0,    # intrinsic-pinned
    5000: -0.014,
    5100: -0.003,
    5200: +0.006,
    5300: +0.009,
    5400: -0.012,
    5500: -0.003,
    6000: 0.0,    # ~0.5 floor; vega tiny -> ignore
    6500: 0.0,
}

# Hydrogel mean-reversion params
HYDRO_MEAN = 9990.0
HYDRO_BAND = 20.0   # how far from mean before we lean hard
HYDRO_BIG_BAND = 35.0  # full-tilt band

# Voucher trading thresholds (kept conservative -- the smile residual edge is real
# but small ~$0.5-2/contract, so we trade in small clips rather than slamming 300).
# Voucher trading is OFF by default: the smile-residual edge (~0.5-2$/contract)
# is too small to overcome holding-period drift, and on backtest each lit strike
# bled $300-1500. Deep-ITM intrinsic arb stays on (it's risk-free if it ever fires).
VOUCHER_TAKES_ENABLED = False
TAKE_DOLLAR_EDGE = 2.50
TAKE_TICK_BUFFER = 0.5
VOUCHER_CLIP = 20
VOUCHER_HARD_LIMIT = 60
DELTA_HEDGE_TOL = 8
INTRINSIC_BAND = 1.5


# =============================================================================
# Stdlib math: Black-Scholes call price + delta + vega + norm.cdf
# =============================================================================
def _norm_cdf(x: float) -> float:
    """Abramowitz/Stegun via math.erf — exact enough for trading."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call(S: float, K: float, T: float, sigma: float) -> tuple[float, float, float]:
    """Returns (price, delta, vega) for a European call with r=q=0."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return max(S - K, 0.0), 1.0 if S > K else 0.0, 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    price = S * Nd1 - K * Nd2
    delta = Nd1
    vega = S * _norm_pdf(d1) * sqrtT
    return price, delta, vega


def smile_iv(S: float, K: float, T: float) -> float:
    if S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    m = math.log(K / S) / math.sqrt(T)
    iv = SMILE_A * m * m + SMILE_B * m + SMILE_C
    return max(iv, 0.05)


# =============================================================================
# Production T(t): we hard-code "today is the start of round 3 (DTE=5)"
# =============================================================================
def t_years_production(timestamp: int) -> float:
    """timestamp is the within-day tick (0 .. 999_900 step 100)."""
    step = timestamp // 100
    rem_steps = PRODUCTION_DTE_AT_OPEN * STEPS_PER_DAY - step
    if rem_steps <= 0:
        rem_steps = 1
    return rem_steps / float(ANNUAL_TICKS)


# =============================================================================
# Helpers
# =============================================================================
def book_summary(depth: OrderDepth) -> tuple[dict[int, int], dict[int, int],
                                              int | None, int | None,
                                              int | None, int | None,
                                              float | None]:
    buys = {int(p): abs(int(q)) for p, q in (depth.buy_orders or {}).items() if int(q) != 0}
    sells = {int(p): abs(int(q)) for p, q in (depth.sell_orders or {}).items() if int(q) != 0}
    if not buys and not sells:
        return buys, sells, None, None, None, None, None
    bb = max(buys) if buys else None
    ba = min(sells) if sells else None
    bid_wall = min(buys) if buys else None
    ask_wall = max(sells) if sells else None
    wall_mid = (bid_wall + ask_wall) / 2.0 if bid_wall is not None and ask_wall is not None else None
    return buys, sells, bb, ba, bid_wall, ask_wall, wall_mid


def clip(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


# =============================================================================
# Trader
# =============================================================================
class Trader:

    # ---------------------------- HYDROGEL ----------------------------
    def _trade_hydrogel(self, state: TradingState, orders_out: dict[str, list[Order]]) -> None:
        if HYDROGEL not in state.order_depths:
            return
        depth = state.order_depths[HYDROGEL]
        buys, sells, bb, ba, bid_wall, ask_wall, wall_mid = book_summary(depth)
        if wall_mid is None or bb is None or ba is None:
            return
        pos = int(state.position.get(HYDROGEL, 0))
        lim = POS_LIM[HYDROGEL]
        max_buy = lim - pos
        max_sell = lim + pos

        # Use wall_mid as our short-term FV; lean hard on the calibrated long-term mean.
        fv = wall_mid
        # Inventory skew on the *passive* fair only.
        skew = 0.04 * pos
        passive_fair = fv - skew

        ords: list[Order] = []

        # 1) Aggressive taker against stale quotes (Frankfurt-style)
        for sp in sorted(sells.keys()):
            if max_buy <= 0:
                break
            sv = sells[sp]
            if sp <= fv - 1:
                q = min(sv, max_buy)
                ords.append(Order(HYDROGEL, sp, q))
                max_buy -= q
            elif sp <= fv and pos < 0:
                q = min(sv, max_buy, -pos)
                if q > 0:
                    ords.append(Order(HYDROGEL, sp, q))
                    max_buy -= q
        for bp in sorted(buys.keys(), reverse=True):
            if max_sell <= 0:
                break
            bv = buys[bp]
            if bp >= fv + 1:
                q = min(bv, max_sell)
                ords.append(Order(HYDROGEL, bp, -q))
                max_sell -= q
            elif bp >= fv and pos > 0:
                q = min(bv, max_sell, pos)
                if q > 0:
                    ords.append(Order(HYDROGEL, bp, -q))
                    max_sell -= q

        # 2) Mean-reversion overlay: when wall_mid is far from the long-term mean,
        #    add an extra layer of orders deep in the book.
        dev = fv - HYDRO_MEAN
        if dev <= -HYDRO_BAND and max_buy > 0:
            # very cheap -> bid up to (mean - band) aggressively
            target_px = int(math.floor(min(fv + 1, HYDRO_MEAN - HYDRO_BAND / 2)))
            ords.append(Order(HYDROGEL, target_px, max_buy))
            max_buy = 0
        elif dev >= HYDRO_BAND and max_sell > 0:
            target_px = int(math.ceil(max(fv - 1, HYDRO_MEAN + HYDRO_BAND / 2)))
            ords.append(Order(HYDROGEL, target_px, -max_sell))
            max_sell = 0

        # 3) Passive market-making just inside the wall.
        if max_buy > 0:
            bid_px = int(bid_wall) + 1 if bid_wall is not None else int(math.floor(passive_fair - 1))
            # don't post above passive_fair
            bid_px = min(bid_px, int(math.floor(passive_fair)))
            ords.append(Order(HYDROGEL, bid_px, max_buy))
        if max_sell > 0:
            ask_px = int(ask_wall) - 1 if ask_wall is not None else int(math.ceil(passive_fair + 1))
            ask_px = max(ask_px, int(math.ceil(passive_fair)))
            ords.append(Order(HYDROGEL, ask_px, -max_sell))

        if ords:
            orders_out.setdefault(HYDROGEL, []).extend(ords)

    # ---------------------------- VOUCHER BOOK ----------------------------
    def _trade_options(self, state: TradingState, orders_out: dict[str, list[Order]]) -> None:
        depths = state.order_depths or {}
        if UNDERLYING not in depths:
            return
        u_depth = depths[UNDERLYING]
        _, _, ubb, uba, u_bw, u_aw, u_wm = book_summary(u_depth)
        if u_wm is None:
            return
        S = float(u_wm)
        T = t_years_production(int(state.timestamp))

        net_signed_delta = 0.0
        positions = state.position or {}
        # accumulate currently held delta first (so we hedge against existing position).
        # NB: fair sigma == smile sigma; PER_STRIKE_TILT is the *expected market residual*
        # (positive => market overprices, negative => underprices), used only as the
        # *direction* signal for which side to take, NOT a fair-price adjustment.
        for K, v in zip(STRIKES, VOUCHERS):
            pv = int(positions.get(v, 0))
            if pv == 0 or T <= 0:
                continue
            sigma = max(smile_iv(S, K, T), 0.05)
            _, dlt, _ = bs_call(S, K, T, sigma)
            net_signed_delta += pv * dlt

        # ---- Trade each voucher ----
        for K, v in zip(STRIKES, VOUCHERS):
            if v not in depths:
                continue
            depth = depths[v]
            buys, sells, bb, ba, bid_wall, ask_wall, wall_mid = book_summary(depth)
            if bb is None and ba is None:
                continue
            pv = int(positions.get(v, 0))
            lim = POS_LIM[v]
            max_buy = lim - pv
            max_sell = lim + pv
            if max_buy <= 0 and max_sell <= 0:
                continue

            ords: list[Order] = []

            # ----- Deep ITM intrinsic arbitrage (K=4000, 4500) -----
            if K <= 4500:
                intrinsic = max(S - K, 0.0)
                if ba is not None and ba < intrinsic - INTRINSIC_BAND and max_buy > 0:
                    sv = sells[ba]
                    q = min(sv, max_buy, 30)
                    if q > 0:
                        ords.append(Order(v, int(ba), q))
                        max_buy -= q
                        # delta of deep-ITM call = 1
                        net_signed_delta += q * 1.0
                if bb is not None and bb > intrinsic + INTRINSIC_BAND and max_sell > 0:
                    bv = buys[bb]
                    q = min(bv, max_sell, 30)
                    if q > 0:
                        ords.append(Order(v, int(bb), -q))
                        max_sell -= q
                        net_signed_delta -= q * 1.0
                if ords:
                    orders_out.setdefault(v, []).extend(ords)
                continue

            # ----- Deep OTM (K=6000, 6500): pinned at 0.5 floor -> skip -----
            if K >= 6000:
                continue

            # ----- Smile-relative mispricing on near-money strikes -----
            if T <= 0:
                continue
            sigma_fair = max(smile_iv(S, K, T), 0.05)
            theo, dlt, vega = bs_call(S, K, T, sigma_fair)
            if vega <= 0:
                continue

            if not VOUCHER_TAKES_ENABLED:
                if ords:
                    orders_out.setdefault(v, []).extend(ords)
                continue

            tilt = PER_STRIKE_TILT[K]
            min_edge = TAKE_DOLLAR_EDGE + TAKE_TICK_BUFFER

            cap_buy = min(max_buy, VOUCHER_CLIP, max(0, VOUCHER_HARD_LIMIT - pv))
            cap_sell = min(max_sell, VOUCHER_CLIP, max(0, VOUCHER_HARD_LIMIT + pv))

            if tilt < 0 and ba is not None and cap_buy > 0 and ba <= theo - min_edge:
                sv = sells[ba]
                q = min(sv, cap_buy)
                if q > 0:
                    ords.append(Order(v, int(ba), q))
                    net_signed_delta += q * dlt

            # tilt > 0 => market overprices => SELL
            if tilt > 0 and bb is not None and cap_sell > 0 and bb >= theo + min_edge:
                bv = buys[bb]
                q = min(bv, cap_sell)
                if q > 0:
                    ords.append(Order(v, int(bb), -q))
                    net_signed_delta -= q * dlt

            # Position-flattening leg: scrape position back when the price reverts past fair.
            if pv > 0 and bb is not None and bb >= theo:
                q = min(buys[bb], pv)
                ords.append(Order(v, int(bb), -q))
                net_signed_delta -= q * dlt
            elif pv < 0 and ba is not None and ba <= theo:
                q = min(sells[ba], -pv)
                ords.append(Order(v, int(ba), q))
                net_signed_delta += q * dlt

            if ords:
                orders_out.setdefault(v, []).extend(ords)

        # ---------- Delta hedge net option position on VELVETFRUIT_EXTRACT ----------
        u_pos = int(positions.get(UNDERLYING, 0))
        u_lim = POS_LIM[UNDERLYING]
        target_underlying = -int(round(net_signed_delta))  # short delta-1 to neutralize call delta
        target_underlying = clip(target_underlying, -u_lim, u_lim)
        delta_to_trade = target_underlying - u_pos
        if abs(delta_to_trade) >= DELTA_HEDGE_TOL and (ubb is not None or uba is not None):
            ords_u: list[Order] = []
            if delta_to_trade > 0 and uba is not None:
                # Need to buy underlying — lift the offer
                _, sells_u, _, _, _, _, _ = book_summary(u_depth)
                remaining = delta_to_trade
                for sp in sorted(sells_u.keys()):
                    if remaining <= 0:
                        break
                    sv = sells_u[sp]
                    q = min(sv, remaining, u_lim - u_pos)
                    if q > 0:
                        ords_u.append(Order(UNDERLYING, int(sp), q))
                        remaining -= q
                        u_pos += q
            elif delta_to_trade < 0 and ubb is not None:
                buys_u, _, _, _, _, _, _ = book_summary(u_depth)
                remaining = -delta_to_trade
                for bp in sorted(buys_u.keys(), reverse=True):
                    if remaining <= 0:
                        break
                    bv = buys_u[bp]
                    q = min(bv, remaining, u_lim + u_pos)
                    if q > 0:
                        ords_u.append(Order(UNDERLYING, int(bp), -q))
                        remaining -= q
                        u_pos -= q
            if ords_u:
                orders_out.setdefault(UNDERLYING, []).extend(ords_u)

        # Also: when *no* hedge is needed, still market-make the underlying lightly
        # to harvest its tight bot-A spread.
        self._make_underlying(state, orders_out)

    def _make_underlying(self, state: TradingState, orders_out: dict[str, list[Order]]) -> None:
        if UNDERLYING not in state.order_depths:
            return
        depth = state.order_depths[UNDERLYING]
        buys, sells, bb, ba, bid_wall, ask_wall, wall_mid = book_summary(depth)
        if wall_mid is None or bid_wall is None or ask_wall is None:
            return
        pos = int(state.position.get(UNDERLYING, 0))
        lim = POS_LIM[UNDERLYING]
        # account for hedge orders queued this tick
        queued = sum(o.quantity for o in orders_out.get(UNDERLYING, []))
        max_buy = max(0, lim - (pos + queued))
        max_sell = max(0, lim + (pos + queued))

        wm = float(wall_mid)
        ords: list[Order] = []

        # 1) Take only when ask is meaningfully BELOW wall_mid (i.e. crossing the wall).
        #    The Bot A spread is ~5 ticks; we don't take inside it because the bot is
        #    quoting that fairly and we'd lose to adverse selection.
        for sp in sorted(sells.keys()):
            if max_buy <= 0:
                break
            if sp < bid_wall:  # someone is selling below the floor of the book
                q = min(sells[sp], max_buy)
                ords.append(Order(UNDERLYING, sp, q))
                max_buy -= q
        for bp in sorted(buys.keys(), reverse=True):
            if max_sell <= 0:
                break
            if bp > ask_wall:  # someone is bidding above the ceiling of the book
                q = min(buys[bp], max_sell)
                ords.append(Order(UNDERLYING, bp, -q))
                max_sell -= q

        # 2) Passive market-making one tick inside the wall.
        skew = 0.02 * pos
        passive_fair = wm - skew
        if max_buy > 0:
            bid_px = int(bid_wall) + 1
            if bid_px < passive_fair:
                ords.append(Order(UNDERLYING, bid_px, min(max_buy, 30)))
        if max_sell > 0:
            ask_px = int(ask_wall) - 1
            if ask_px > passive_fair:
                ords.append(Order(UNDERLYING, ask_px, -min(max_sell, 30)))

        if ords:
            orders_out.setdefault(UNDERLYING, []).extend(ords)

    # ---------------------------- ENTRY POINT ----------------------------
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        try:
            self._trade_hydrogel(state, result)
        except Exception as e:
            print(f"[hydrogel error] {e}")
        try:
            self._trade_options(state, result)
        except Exception as e:
            print(f"[options error] {e}")
        return result, 0, state.traderData or ""
