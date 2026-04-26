"""Round 3 — vouchers_final_strategy/STRATEGY.txt + ORIGINAL_DISCORD_QUOTES (Sonic gate).

- Sonic / STRATEGY: only trade 5200/5300 **pair** ideas when L1 full spread of **both** legs
  <= 2 (t-stat / edge gated on book state; wide = different regime — exit).
- inclineGod: D = mid(5200)-mid(5300) is the **book-aligned** object (not a multi-strike smile).
- **No** naive “long extract every tick” — STRATEGY mid-forward edge is not bid/ask PnL; we
  ship **regime + pair** only (VEV_5200, VEV_5300, no other vouchers). VELVETFRUIT_EXTRACT
  in limits but **no** extract orders in this version.

D uses EWMA when gate on; z from pre-tick (mu, var). Open: |z|>=ZO. Close: |z|<ZC or gate off.
No HYDROGEL.
"""
from __future__ import annotations
import json
import math
from datamodel import Order, TradingState

S5200, S5300 = "VEV_5200", "VEV_5300"
SEX, SH = "VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"
LIMITS: dict[str, int] = {SEX: 200, SH: 200, S5200: 300, S5300: 300}

TH, ALPHA = 2.0, 0.05
WARMUP, ZO, ZC = 50, 1.4, 0.45
QMAX = 28


def _mid(d) -> float | None:
    if not d.buy_orders or not d.sell_orders:
        return None
    return 0.5 * (max(d.buy_orders) + min(d.sell_orders))


def _spr(d) -> float | None:
    if not d.buy_orders or not d.sell_orders:
        return None
    return float(min(d.sell_orders) - max(d.buy_orders))


def _gate(d0, d1) -> bool:
    a, b = _spr(d0), _spr(d1)
    if a is None or b is None or a < 0 or b < 0:
        return False
    return a <= TH and b <= TH


def _q(absz: float) -> int:
    return int(min(QMAX, max(4, 8 + 6 * (absz - ZO))))


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}
        o: dict[str, list] = {k: [] for k in LIMITS}
        dep = state.order_depths
        for s in (S5200, S5300, SEX):
            if s not in dep:
                return o, 0, json.dumps(td)

        pre, ts = td.get("prev_ts"), state.timestamp
        if pre is not None and ts < pre:
            td["day_idx"] = int(td.get("day_idx", 0)) + 1
            td["ticks"] = 0
        td["prev_ts"] = ts
        tick = int(td.get("ticks", 0)) + 1
        td["ticks"] = tick

        d0, d1 = dep[S5200], dep[S5300]
        a0, a1 = _mid(d0), _mid(d1)
        if a0 is None or a1 is None:
            return o, 0, json.dumps(td)
        diff = a0 - a1

        tight = _gate(d0, d1)
        mu0 = float(td.get("ew_mu", 0.0))
        v0 = float(td.get("ew_var", 100.0))
        z = 0.0
        if tight:
            z = (diff - mu0) / max(math.sqrt(v0), 1e-6)
            td["ew_mu"] = (1.0 - ALPHA) * mu0 + ALPHA * diff
            td["ew_var"] = max(
                (1.0 - ALPHA) * v0 + ALPHA * (diff - mu0) ** 2, 1.0
            )
        else:
            td["ew_mu"] = mu0
            td["ew_var"] = v0

        p0 = state.position.get(S5200, 0)
        p1 = state.position.get(S5300, 0)
        paired = p0 != 0 and p1 != 0 and p0 == -p1

        if not tight:
            if paired and d0.buy_orders and d0.sell_orders and d1.buy_orders and d1.sell_orders:
                if p0 < 0:
                    o[S5200].append(Order(S5200, min(d0.sell_orders), -p0))
                else:
                    o[S5200].append(Order(S5200, max(d0.buy_orders), -p0))
                if p1 < 0:
                    o[S5300].append(Order(S5300, min(d1.sell_orders), -p1))
                else:
                    o[S5300].append(Order(S5300, max(d1.buy_orders), -p1))
            return o, 0, json.dumps(td)

        if tick < WARMUP:
            return o, 0, json.dumps(td)

        if paired and abs(z) < ZC and d0.buy_orders and d0.sell_orders and d1.buy_orders and d1.sell_orders:
            if p0 < 0:
                o[S5200].append(Order(S5200, min(d0.sell_orders), -p0))
            else:
                o[S5200].append(Order(S5200, max(d0.buy_orders), -p0))
            if p1 < 0:
                o[S5300].append(Order(S5300, min(d1.sell_orders), -p1))
            else:
                o[S5300].append(Order(S5300, max(d1.buy_orders), -p1))
            return o, 0, json.dumps(td)

        if (not paired) and abs(z) >= ZO and p0 == 0 and p1 == 0:
            q = _q(abs(z))
            if z > 0 and d0.buy_orders and d1.sell_orders:
                o[S5200].append(Order(S5200, max(d0.buy_orders), -q))
                o[S5300].append(Order(S5300, min(d1.sell_orders), q))
            elif z < 0 and d0.sell_orders and d1.buy_orders:
                o[S5200].append(Order(S5200, min(d0.sell_orders), q))
                o[S5300].append(Order(S5300, max(d1.buy_orders), -q))

        return o, 0, json.dumps(td)
