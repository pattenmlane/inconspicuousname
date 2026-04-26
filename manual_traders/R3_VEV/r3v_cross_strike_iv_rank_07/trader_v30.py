"""
Round 3 — joint tight book (vouchers_final_strategy only, iteration v30).

Thesis: trade **only** when VEV_5200 and VEV_5300 both have L1 spread <= TH=2 (Sonic / STRATEGY.txt).
When the gate is on: passive join-bid long **VELVETFRUIT_EXTRACT** and the two anchor vouchers (flow
/ hedge surface). No HYDROGEL, no smile-residual / IV-rank / legacy r3v_cross_strike logic.

DTE: round3work/round3description.txt; csv_day from extract open-S sequence; intraday
dte_eff = 8 - csv_day - (timestamp//100)/10000 (for documentation only; this trader is spread+mid).
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TH_SPREAD = 2
# Execution: one-sided join bid on tight books (higher PnL focus on extract; anchors for surface)
EXTRACT_BUY_Q = 120
VEV_ANCHOR_Q = 100
# Optional small participation on other strikes when gate is on (inclineGod: every contract is its own)
VEV_WING_Q = 8

WARMUP = 5
VOUCHERS = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
EXTRACT_LIM = 200
VEV_LIM = 300

_EMA_KEY = "ema_S"
EMA_S_N = 12


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _sym_for_product(state: TradingState, product: str) -> str | None:
    for sym, lst in (getattr(state, "listings", {}) or {}).items():
        if getattr(lst, "product", None) == product:
            return sym
    return None


def _best_ba(depth: OrderDepth | None) -> tuple[int | None, int | None]:
    if depth is None:
        return None, None
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None, None
    return max(buys.keys()), min(sells.keys())


def _ema(prev: float | None, x: float, n: int) -> float:
    if prev is None:
        return x
    a = 2.0 / (n + 1.0)
    return a * x + (1.0 - a) * prev


def _update_csv_day(td: dict[str, Any], ts: int, s_raw: float) -> int:
    if ts != 0:
        return int(td.get("csv_day", 0))
    hist = td.get("open_S_hist")
    if not isinstance(hist, list):
        hist = []
    cur = round(float(s_raw), 2)
    if not hist or abs(float(hist[-1]) - cur) > 0.25:
        hist.append(cur)
    td["open_S_hist"] = hist[:4]
    return max(0, min(len(hist) - 1, 2))


def _tight_gate(depths: dict[str, OrderDepth], s520: str | None, s530: str | None) -> bool:
    if not s520 or not s530:
        return False
    b5, a5 = _best_ba(depths.get(s520))
    b3, a3 = _best_ba(depths.get(s530))
    if b5 is None or a5 is None or b3 is None or a3 is None:
        return False
    return (a5 - b5) <= TH_SPREAD and (a3 - b3) <= TH_SPREAD


def _add_buy(
    m: dict[str, list[Order]], sym: str, bb: int, ba: int, qty: int, pos: int, lim: int
) -> None:
    if qty <= 0:
        return
    qb = min(qty, lim - pos)
    if qb > 0 and ba > bb and bb + 1 < ba:
        m.setdefault(sym, []).append(Order(sym, bb + 1, qb))


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        ts = int(getattr(state, "timestamp", 0))
        pos: dict[str, int] = getattr(state, "position", None) or {}
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}

        sym_u = _sym_for_product(state, "VELVETFRUIT_EXTRACT")
        if sym_u is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        s520 = _sym_for_product(state, "VEV_5200")
        s530 = _sym_for_product(state, "VEV_5300")
        if not s520 or not s530:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        du = depths.get(sym_u)
        ubb, uba = _best_ba(du)
        if ubb is None or uba is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        s_raw = 0.5 * (ubb + uba)
        prev_e = td.get(_EMA_KEY)
        ema_s = _ema(float(prev_e) if isinstance(prev_e, (int, float)) else None, s_raw, EMA_S_N)
        td[_EMA_KEY] = ema_s
        csv_day = _update_csv_day(td, ts, s_raw)
        td["csv_day"] = csv_day

        if ts // 100 < WARMUP:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _tight_gate(depths, s520, s530):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        out: dict[str, list[Order]] = {}
        # Extract: short-hold long bias when surface is "hedgeable" (toy policy per STRATEGY.txt)
        pos_u = int(pos.get(sym_u, 0))
        _add_buy(out, sym_u, ubb, uba, EXTRACT_BUY_Q, pos_u, EXTRACT_LIM)

        for prod in VOUCHERS:
            sym = _sym_for_product(state, prod)
            if sym is None:
                continue
            d = depths.get(sym)
            bb, ba = _best_ba(d)
            if bb is None or ba is None:
                continue
            p = int(pos.get(sym, 0))
            if prod in ("VEV_5200", "VEV_5300"):
                _add_buy(out, sym, int(bb), int(ba), VEV_ANCHOR_Q, p, VEV_LIM)
            else:
                # Only touch other strikes when the joint surface is on (Sonic: cleaner t-stat)
                _add_buy(out, sym, int(bb), int(ba), VEV_WING_Q, p, VEV_LIM)

        return out, 0, json.dumps(td, separators=(",", ":"))
