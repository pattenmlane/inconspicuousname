"""Round 4: **v12 VEV_5300** + small **VELVETFRUIT_EXTRACT** leg on basket-only bursts.

Tape (r4_tight_burst_extract_fwd_by_5300_leg.json): at joint tight + M01→M22 burst,
extract fwd K20 mean **+0.525** when the M01→M22 leg is **not** on VEV_5300 vs **+0.046**
when it is on 5300 (n=160 vs 131). v9 showed basket-only *5300-only* sim underperforms v7;
here we keep **v12** 5300 entries on **any** burst and add extract only on **basket-only**
subset (counterparty-conditioned cross-asset).

Extract: EX_LOT=2, MAX_EX_POS=24, EX_COOLDOWN=40, MIN_HOLD_EX=10; join buy / ask-1-style
sell mirroring 5300. Flatten extract when not (tight ∧ basket_only_burst) after hold.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from datamodel import Order, TradingState
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TDIR = REPO / "Prosperity4Data" / "ROUND_4"
S5200, S5300 = "VEV_5200", "VEV_5300"
EX = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
LIMITS = {
    EX: 200,
    HYDRO: 200,
    **{f"VEV_{k}": 300 for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
}
TH = 2.0
BURST_MIN = 4
LOT = 6
COOLDOWN = 20
MAX_POS = 90
MIN_HOLD_TICKS = 10

EX_LOT = 2
EX_COOLDOWN = 40
MAX_EX_POS = 24
MIN_HOLD_EX = 10


def _build_trades_by_day_ts():
    by: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    for csv_day in (1, 2, 3):
        p = TDIR / f"trades_round_4_day_{csv_day}.csv"
        if not p.is_file():
            continue
        with open(p, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                ts = int(row["timestamp"])
                by[(csv_day, ts)].append(
                    (
                        str(row["symbol"]),
                        str(row["buyer"]).strip(),
                        str(row["seller"]).strip(),
                        float(row["price"]),
                        int(float(row["quantity"])),
                    )
                )
    return by


_TR = _build_trades_by_day_ts()


def _l1_spread(d) -> float | None:
    if not d.buy_orders or not d.sell_orders:
        return None
    return float(min(d.sell_orders) - max(d.buy_orders))


def _joint_tight(depth) -> bool:
    if S5200 not in depth or S5300 not in depth:
        return False
    a, b = _l1_spread(depth[S5200]), _l1_spread(depth[S5300])
    if a is None or b is None or a < 0 or b < 0:
        return False
    return a <= TH and b <= TH


def _burst_m01_m22_any(csv_day: int, ts: int) -> bool:
    rows = _TR.get((csv_day, ts), [])
    if len(rows) < BURST_MIN:
        return False
    return any(b == "Mark 01" and s == "Mark 22" for _sym, b, s, _p, _q in rows)


def _burst_m01_m22_basket_only(csv_day: int, ts: int) -> bool:
    rows = _TR.get((csv_day, ts), [])
    if len(rows) < BURST_MIN:
        return False
    if not any(b == "Mark 01" and s == "Mark 22" for _sym, b, s, _p, _q in rows):
        return False
    if any(sym == S5300 and b == "Mark 01" and s == "Mark 22" for sym, b, s, _p, _q in rows):
        return False
    return True


def _join_buy_price(dx) -> int | None:
    if not dx.buy_orders or not dx.sell_orders:
        return None
    bid = max(dx.buy_orders)
    ask = min(dx.sell_orders)
    if ask <= bid:
        return bid
    sp = ask - bid
    if sp >= 2:
        return min(bid + 1, ask)
    return bid


def _improve_sell_price(dx) -> int | None:
    if not dx.buy_orders or not dx.sell_orders:
        return None
    bid = max(dx.buy_orders)
    ask = min(dx.sell_orders)
    if ask <= bid:
        return bid
    sp = ask - bid
    if sp >= 2:
        return max(ask - 1, bid)
    return ask


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}
        o = {p: [] for p in LIMITS}
        ts = state.timestamp
        prev = td.get("prev_ts")
        if prev is not None and ts < prev:
            td["day_idx"] = int(td.get("day_idx", 0)) + 1
            td["last_active_tick"] = 0
            td["last_ex_active_tick"] = 0
        td["prev_ts"] = ts
        tick = int(td.get("ticks", 0)) + 1
        td["ticks"] = tick
        d_idx = int(td.get("day_idx", 0))
        csv_day = min(3, max(1, d_idx + 1))

        d = state.order_depths
        if S5200 not in d or S5300 not in d or EX not in d:
            return o, 0, json.dumps(td)
        d53 = d[S5300]
        dex = d[EX]
        if not d53.buy_orders or not d53.sell_orders:
            return o, 0, json.dumps(td)
        if not dex.buy_orders or not dex.sell_orders:
            return o, 0, json.dumps(td)

        tight = _joint_tight(d)
        burst_any = _burst_m01_m22_any(csv_day, ts)
        burst_basket = _burst_m01_m22_basket_only(csv_day, ts)
        active_53 = tight and burst_any
        active_ex = tight and burst_basket
        if active_53:
            td["last_active_tick"] = tick
        if active_ex:
            td["last_ex_active_tick"] = tick

        p53 = state.position.get(S5300, 0)
        pe = state.position.get(EX, 0)
        last = int(td.get("last_sig", 0))
        last_key = td.get("last_burst_key")
        last_active_tick = int(td.get("last_active_tick", 0))
        last_ex_sig = int(td.get("last_ex_sig", 0))
        last_ex_key = td.get("last_ex_burst_key")
        last_ex_active = int(td.get("last_ex_active_tick", 0))

        # --- exit VEV_5300 ---
        if not active_53 and p53 > 0 and tick - last_active_tick >= MIN_HOLD_TICKS:
            spx = _improve_sell_price(d53)
            if spx is not None:
                q = min(p53, 40, LIMITS[S5300] + p53)
                if q > 0:
                    o[S5300].append(Order(S5300, spx, -q))

        # --- exit extract ---
        if not active_ex and pe > 0 and tick - last_ex_active >= MIN_HOLD_EX:
            ex_px = _improve_sell_price(dex)
            if ex_px is not None:
                qe = min(pe, 40, LIMITS[EX] + pe)
                if qe > 0:
                    o[EX].append(Order(EX, ex_px, -qe))

        # --- enter VEV_5300 (any burst under tight) ---
        if active_53:
            burst_key = f"{csv_day}:{ts}"
            if burst_key != last_key or (tick - last) >= COOLDOWN:
                if p53 < MAX_POS:
                    jpx = _join_buy_price(d53)
                    if jpx is not None:
                        q = min(LOT, LIMITS[S5300] - p53, 12)
                        if q > 0:
                            o[S5300].append(Order(S5300, jpx, q))
                            td["last_sig"] = tick
                            td["last_burst_key"] = burst_key

        # --- enter extract (basket-only burst under tight) ---
        if active_ex:
            ex_key = f"ex:{csv_day}:{ts}"
            if ex_key != last_ex_key or (tick - last_ex_sig) >= EX_COOLDOWN:
                if pe < MAX_EX_POS:
                    jex = _join_buy_price(dex)
                    if jex is not None:
                        qe = min(EX_LOT, LIMITS[EX] - pe, 12)
                        if qe > 0:
                            o[EX].append(Order(EX, jex, qe))
                            td["last_ex_sig"] = tick
                            td["last_ex_burst_key"] = ex_key

        return o, 0, json.dumps(td)
