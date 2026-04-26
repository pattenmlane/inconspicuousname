"""
Round-3: **vouchers_final_strategy** only (no smile/jump/neighbor-legacy from prior iters).

Source: `round3work/vouchers_final_strategy/STRATEGY.txt`, `ORIGINAL_DISCORD_QUOTES.txt` (Sonic: gate on
**both** 5200 and 5300 **spread ≤ 2**; inclineGod: spreads per contract, not just mids), plus
`outputs/r3_tight_spread_summary.txt` (forward extract mid / tight vs not).

**Logic**
- **Tight** = (ask1−bid1) for VEV_5200 and VEV_5300 both ≤ `TIGHT_TOB` (2).
- **Tight regime:** "risk on" per STRATEGY — larger clips, default MM edges, **+1 tick** long bias on
  extract around EMA of wall mid (optional directional read of tight-gate; mid≠PnL, caveat in STRATEGY).
- **Wide regime:** wider take/make on VEVs, half MM/take size, small extract clip, no long bias
  (Sonic: t-stat decays in wide books).
- **Fair value:** Black–Scholes with **r=0**, **T = t_years_effective** (round3description; see
  `round3work/round3description.txt` + `plot_iv_smile_round3`). One **ATM** IV (strike nearest
  **extract** S from wall-mid) from voucher mid, applied **flat** across strikes = simple IV
  *surface* proxy (Greeks/IV thread without cubic spline).
- Trades: **VELVETFRUIT_EXTRACT** + 10 VEVs only. **No HYDROGEL_PACK**.

**traderData key:** `vfs20` (vouchers final strategy 20).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "round3work" / "plotting" / "original_method" / "combined_analysis"))
from plot_iv_smile_round3 import t_years_effective  # noqa: E402

R = 0.0
_SQRT2PI = math.sqrt(2.0 * math.pi)
_TD_KEY = "vfs20"

EXTRACT = "VELVETFRUIT_EXTRACT"
VEV_SYMS = [
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]
STRIKES = [int(s.split("_")[1]) for s in VEV_SYMS]
S5200 = "VEV_5200"
S5300 = "VEV_5300"
TIGHT_TOB = 2
EM_EX = 0.12

LIMITS: dict[str, int] = {"VELVETFRUIT_EXTRACT": 200, **{s: 300 for s in VEV_SYMS}}


def _tape_day() -> int:
    e = os.environ.get("PROSPERITY4_BACKTEST_DAY")
    if e is not None and e.lstrip("-").isdigit():
        return int(e)
    return 0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * _norm_cdf(d1) - K * math.exp(-R * T) * _norm_cdf(d2)


def bs_delta_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / v
    return float(_norm_cdf(d1))


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / v
    return S * math.sqrt(T) * _norm_pdf(d1)


def implied_vol(
    S: float, K: float, T: float, price: float, initial: float | None = None
) -> float | None:
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    if price < max(S - K, 0.0) - 1e-9:
        return None
    if bs_call(S, K, T, 4.5) < price - 1e-9:
        return None
    sigma = 0.28 if initial is None else max(1e-4, min(float(initial), 4.5))
    for _ in range(8):
        th = bs_call(S, K, T, sigma) - price
        if abs(th) < 1e-7:
            return sigma
        vg = bs_vega(S, K, T, sigma)
        if vg < 1e-14:
            break
        sigma -= th / vg
        sigma = max(1e-6, min(sigma, 4.5))
    lo, hi = 1e-5, 4.5
    if bs_call(S, K, T, lo) > price or bs_call(S, K, T, hi) < price:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def wall_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    bv, av = depth.buy_orders[bb], -depth.sell_orders[ba]
    tot = bv + av
    if tot <= 0:
        return 0.5 * (float(bb) + float(ba))
    return (float(bb) * av + float(ba) * bv) / tot


def tob_spread(depth: OrderDepth) -> int | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return int(min(depth.sell_orders)) - int(max(depth.buy_orders))


def joint_tight_books(d52: OrderDepth | None, d53: OrderDepth | None) -> bool:
    if d52 is None or d53 is None:
        return False
    if not d52.buy_orders or not d52.sell_orders or not d53.buy_orders or not d53.sell_orders:
        return False
    s52, s53 = tob_spread(d52), tob_spread(d53)
    if s52 is None or s53 is None:
        return False
    return s52 <= TIGHT_TOB and s53 <= TIGHT_TOB


def nearest_strike_strike(S: float) -> int:
    return int(min(STRIKES, key=lambda k: abs(float(k) - S)))


class Trader:
    def run(self, state: TradingState):
        bu: dict[str, Any] = {}
        if state.traderData:
            try:
                o = json.loads(state.traderData)
                if isinstance(o, dict) and _TD_KEY in o and isinstance(o[_TD_KEY], dict):
                    bu = o[_TD_KEY]
            except (json.JSONDecodeError, TypeError, KeyError):
                bu = {}

        out: dict[str, list[Order]] = {EXTRACT: []}
        for s in VEV_SYMS:
            out[s] = []

        day = _tape_day()
        ts = int(getattr(state, "timestamp", 0))
        T = float(t_years_effective(int(day), int(ts)))

        exd = state.order_depths.get(EXTRACT)
        if exd is None or not exd.buy_orders or not exd.sell_orders:
            return out, 0, json.dumps({_TD_KEY: bu}, separators=(",", ":"))

        S = wall_mid(exd)
        if S is None or S <= 0 or T <= 0:
            return out, 0, json.dumps({_TD_KEY: bu}, separators=(",", ":"))

        s_opt = float(S)
        f_ex = bu.get("f_ex")
        if f_ex is None:
            f_ex = s_opt
        else:
            f_ex = float(f_ex) + EM_EX * (s_opt - float(f_ex))
        bu["f_ex"] = f_ex

        d52 = state.order_depths.get(S5200)
        d53 = state.order_depths.get(S5300)
        tight = joint_tight_books(d52, d53)

        ivp = bu.get("ivp")
        if not isinstance(ivp, dict):
            ivp = {}

        mids: dict[str, float] = {}
        for sym, K in zip(VEV_SYMS, STRIKES):
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                continue
            wm = wall_mid(d)
            if wm is None:
                continue
            mids[sym] = float(wm)

        if len(mids) < 3:
            self._append_extract(
                out, exd, int(state.position.get(EXTRACT, 0) or 0), float(f_ex), tight, 0.0
            )
            return out, 0, json.dumps({_TD_KEY: bu}, separators=(",", ":"))

        atm_k = nearest_strike_strike(s_opt)
        atm_sym = f"VEV_{atm_k}"
        atm_mid = mids.get(atm_sym)
        p0 = ivp.get(atm_sym)
        init0 = float(p0) if isinstance(p0, (int, float)) else None
        sigma_atm: float | None = None
        if atm_mid is not None:
            sigma_atm = implied_vol(s_opt, float(atm_k), T, float(atm_mid), initial=init0)
        if sigma_atm is None:
            for sym, K in zip(VEV_SYMS, STRIKES):
                m = mids.get(sym)
                if m is None:
                    continue
                p1 = ivp.get(sym)
                iv = implied_vol(s_opt, float(K), T, float(m), initial=float(p1) if isinstance(p1, (int, float)) else None)
                if iv is not None:
                    sigma_atm = iv
                    break
        if sigma_atm is None:
            self._append_extract(
                out, exd, int(state.position.get(EXTRACT, 0) or 0), float(f_ex), tight, 0.0
            )
            return out, 0, json.dumps({_TD_KEY: bu}, separators=(",", ":"))

        for sym, K in zip(VEV_SYMS, STRIKES):
            if sym in mids and sym in state.order_depths:
                p2 = ivp.get(sym)
                iv0 = implied_vol(
                    s_opt, float(K), T, float(mids[sym]),
                    initial=float(p2) if isinstance(p2, (int, float)) else None,
                )
                if iv0 is not None:
                    ivp[sym] = float(iv0)
        ivp[atm_sym] = float(sigma_atm) if sigma_atm is not None else ivp.get(atm_sym, 0.28)
        bu["ivp"] = ivp

        sigf = float(sigma_atm)
        net_delta = 0.0
        for sym, K in zip(VEV_SYMS, STRIKES):
            pos = int(state.position.get(sym, 0) or 0)
            if pos == 0:
                continue
            net_delta += pos * bs_delta_call(s_opt, float(K), T, sigf)

        self._append_extract(
            out, exd, int(state.position.get(EXTRACT, 0) or 0), float(f_ex), tight, float(net_delta)
        )

        if tight:
            te, me = 2.0, 1.0
            mms, tks = 20, 24
        else:
            te, me = 4.0, 2.0
            mms, tks = 10, 12

        for sym, K in zip(VEV_SYMS, STRIKES):
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders or sym not in mids:
                continue
            fair = bs_call(s_opt, float(K), T, sigf)
            pos = int(state.position.get(sym, 0) or 0)
            lim = LIMITS[sym]
            out[sym].extend(
                self._vev_orders(sym, d, pos, lim, float(fair), te, me, mms, tks)
            )

        return out, 0, json.dumps({_TD_KEY: bu}, separators=(",", ":"))

    def _append_extract(
        self,
        out: dict,
        depth: OrderDepth,
        pos: int,
        fair: float,
        tight: bool,
        net_delta: float,
    ) -> None:
        skew = int(round(max(-3.0, min(3.0, 0.02 * net_delta))))
        bias = 1 if tight else 0
        fi = int(round(fair)) + skew + bias
        edge = 2
        limu = LIMITS[EXTRACT]
        mq = 22 if tight else 8
        if not depth.buy_orders or not depth.sell_orders:
            return
        bb, ba = max(depth.buy_orders), min(depth.sell_orders)
        bid_p = min(int(bb) + 1, fi - edge)
        if bid_p >= 1 and bid_p < int(ba) and pos < limu:
            out[EXTRACT].append(Order(EXTRACT, bid_p, min(mq, limu - pos)))
        ask_p = max(int(ba) - 1, fi + edge)
        if ask_p > int(bb) and pos > -limu:
            out[EXTRACT].append(Order(EXTRACT, ask_p, -min(mq, limu + pos)))

    def _vev_orders(
        self,
        sym: str,
        depth: OrderDepth,
        pos: int,
        lim: int,
        fair: float,
        take_edge: float,
        make_edge: float,
        mm_size: int,
        take_size: int,
    ) -> list[Order]:
        olist: list[Order] = []
        bb, ba = max(depth.buy_orders), min(depth.sell_orders)
        best_bid, best_ask = int(bb), int(ba)
        if best_ask <= fair - take_edge + 1e-9 and pos < lim:
            q = min(take_size, lim - pos)
            if q > 0:
                olist.append(Order(sym, best_ask, q))
        if best_bid >= fair + take_edge - 1e-9 and pos > -lim:
            q = min(take_size, lim + pos)
            if q > 0:
                olist.append(Order(sym, best_bid, -q))
        bid_p = min(best_bid + 1, int(math.floor(fair - make_edge)))
        bid_p = max(0, bid_p)
        if bid_p < best_ask and pos < lim:
            q = min(mm_size, lim - pos)
            if q > 0:
                olist.append(Order(sym, bid_p, q))
        ask_p = max(best_ask - 1, int(math.ceil(fair + make_edge)))
        if ask_p > best_bid and pos > -lim:
            q = min(mm_size, lim + pos)
            if q > 0:
                olist.append(Order(sym, ask_p, -q))
        return olist
