"""
Round 3 — velvet beta hedge (r3v_velvet_beta_hedge_02), iteration 0.

Concept: tilt VEV quotes using short-horizon VELVETFRUIT_EXTRACT return and z-score so
the quoted surface tracks underlying moves (beta-style).

TTE (round3work/round3description.txt + intraday winding per combined_analysis):
  Historical CSV day d in {0,1,2} -> DTE at open = 8 - d.
  DTE_eff = DTE_open - (local_timestamp // 100) / 10000; T = DTE_eff / 365, r = 0.

Backtester merges days: csv_day = timestamp // 1_000_000; local_ts = timestamp % 1_000_000.
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
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
LIMITS = {
    HYDRO: 200,
    U: 200,
    **{v: 300 for v in VOUCHERS},
}

# Global smile (same coeffs as round3work/voucher_work/5200_work/calibration.json)
_COEFFS = [0.14215151147708086, -0.0016298611395181932, 0.23576325646627055]

# --- TTE ---
def _dte_open(csv_day: int) -> int:
    return 8 - int(csv_day)


def _dte_eff(csv_day: int, local_ts: int) -> float:
    return max(float(_dte_open(csv_day)) - (int(local_ts) // 100) / 10_000.0, 1e-6)


def t_years(csv_day: int, local_ts: int) -> float:
    return _dte_eff(csv_day, local_ts) / 365.0


def _cdf(x: float) -> float:
    return float(norm.cdf(x))


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> tuple[float, float]:
    if T <= 0 or sigma <= 1e-12:
        return max(S - K, 0.0), 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    price = S * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)
    delta = _cdf(d1)
    return float(price), float(delta)


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


def theo_delta_vega(S: float, K: float, T: float) -> tuple[float, float, float]:
    sig = iv_smile(S, K, T)
    if not math.isfinite(sig) or sig <= 0:
        return float("nan"), float("nan"), float("nan")
    th, de = bs_call(S, K, T, sig, 0.0)
    ve = bs_vega(S, K, T, sig, 0.0)
    return th, de, ve


def _book(depth: OrderDepth) -> tuple[int | None, int | None, float | None]:
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys and not sells:
        return None, None, None
    bb = max(buys.keys())
    ba = min(sells.keys())
    # wall mid = outer bid/ask walls (Frankfurt convention)
    bw = min(buys.keys())
    aw = max(sells.keys())
    wm = (float(bw) + float(aw)) / 2.0
    return bb, ba, wm


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
    """Return (csv_day_index, local_timestamp within day)."""
    d = int(ts) // 1_000_000
    loc = int(ts) % 1_000_000
    return d, loc


class Trader:
    """Tilt VEV quotes with extract short-horizon return + z; light MM on hydrogel."""

    MID_HIST = 40
    WARMUP_STEPS = 15
    TILT_RET_SCALE = 2.0
    TILT_Z_SCALE = 1.5
    TILT_AMP = 4.0
    W_RET = 1.0
    W_Z = 0.55
    VEV_SIZE = 22
    HYDRO_SIZE = 15
    EXTRACT_SIZE = 6

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
        if csv_day not in (0, 1, 2):
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

        ret1 = 0.0
        if len(hist) >= 2:
            ret1 = hist[-1] - hist[-2]

        z20 = 0.0
        if len(hist) >= 5:
            a = np.asarray(hist[-20:], dtype=float)
            m = float(a.mean())
            s = float(a.std()) + 1e-6
            z20 = (u_mid - m) / s

        def clip(x: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, x))

        tilt = self.W_RET * clip(ret1 / self.TILT_RET_SCALE, -1.0, 1.0) + self.W_Z * clip(
            z20 / self.TILT_Z_SCALE, -1.0, 1.0
        )
        tilt_px = tilt * self.TILT_AMP

        T = t_years(csv_day, local_ts)
        step = ts // 100
        orders: dict[str, list[Order]] = {}

        if step >= self.WARMUP_STEPS:
            for v in VOUCHERS:
                if v not in depths:
                    continue
                bb, ba, wm = _book(depths[v])
                bb2, ba2, wm2 = _synth(bb, ba, wm)
                K = int(v.split("_")[1])
                theo, _de, vega = theo_delta_vega(u_mid, float(K), T)
                if not math.isfinite(theo):
                    continue
                spread = max(1, int(ba2 - bb2))
                half = max(1, spread // 3)
                wv = float(vega) if math.isfinite(vega) else 0.0
                scale = min(max(wv, 1.0), 80.0) / 40.0
                shift = tilt_px * scale
                center = theo + shift
                bid_p = int(round(center - half))
                ask_p = int(round(center + half))
                bid_p = max(bb2 + 1, min(bid_p, ba2 - 2))
                ask_p = min(ba2 - 1, max(ask_p, bb2 + 2))
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

            if HYDRO in depths:
                bbh, bah, _ = _book(depths[HYDRO])
                if bbh is not None and bah is not None:
                    hm = 0.5 * (bbh + bah)
                    spread_h = max(1, bah - bbh)
                    ph = int(round(hm))
                    ph_b = max(bbh + 1, ph - spread_h // 4)
                    ph_a = min(bah - 1, ph + spread_h // 4)
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

            p_u = int(pos.get(U, 0))
            edge = max(1, int(ba_u - bb_u) // 5)
            if abs(z20) > 0.8 and abs(ret1) > 0.15:
                if z20 > 0 and ret1 > 0 and p_u < LIMITS[U] - self.EXTRACT_SIZE:
                    orders[U] = [Order(U, int(ba_u), min(self.EXTRACT_SIZE, LIMITS[U] - p_u))]
                elif z20 < 0 and ret1 < 0 and p_u > -LIMITS[U] + self.EXTRACT_SIZE:
                    orders[U] = [Order(U, int(bb_u), -min(self.EXTRACT_SIZE, LIMITS[U] + p_u))]

        store["u_mid_hist"] = hist
        return orders, 0, json.dumps(store, separators=(",", ":"))
