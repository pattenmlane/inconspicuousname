"""Tiny backtest harness for the Round 3 bot.

Walks the historical CSVs in Prosperity4Data/ROUND_3 tick-by-tick, builds a
TradingState that looks like the live one, calls Trader.run(), and fills any
order that crosses the *current* book (price-priority, partial fills against
posted volume; aggressively filled at quote price, not slipped).

This isn't a competition-perfect simulator (no opposing-bot reaction model)
but it's enough to prove the strategy isn't actively losing money.
"""
from __future__ import annotations

import json
import math
import sys
import importlib.util
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA = Path("Prosperity4Data/ROUND_3")
HERE = Path(__file__).resolve().parent

# Load the Trader from round3_bot (avoid having both a backtester datamodel and
# competition datamodel collide).
sys.path.insert(0, str(HERE.parent / "imc-prosperity-4-backtester"))
from prosperity4bt.datamodel import Order, OrderDepth, TradingState, Listing  # noqa: E402

# Ensure the bot uses the backtester datamodel
import prosperity4bt.datamodel as dm  # noqa: E402
sys.modules.setdefault("datamodel", dm)

spec = importlib.util.spec_from_file_location("round3_bot", HERE / "round3_bot.py")
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)  # type: ignore
Trader = mod.Trader  # type: ignore

PRODUCTS = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"] + [
    f"VEV_{k}" for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
]
POS_LIM = {
    "HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]},
}


def load_day(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
    return df[df["product"].isin(PRODUCTS)].copy()


def row_to_depth(rows: pd.DataFrame) -> dict[str, OrderDepth]:
    """rows = all product rows for a single timestamp."""
    out: dict[str, OrderDepth] = {}
    for _, r in rows.iterrows():
        d = OrderDepth()
        for i in (1, 2, 3):
            bp = r.get(f"bid_price_{i}")
            bv = r.get(f"bid_volume_{i}")
            ap = r.get(f"ask_price_{i}")
            av = r.get(f"ask_volume_{i}")
            if pd.notna(bp) and pd.notna(bv) and int(bv) > 0:
                d.buy_orders[int(bp)] = int(bv)
            if pd.notna(ap) and pd.notna(av) and int(av) > 0:
                # OrderDepth uses negative quantities for sells per IMC convention,
                # but the bot only inspects abs values via book_summary.
                d.sell_orders[int(ap)] = -int(av)
        out[str(r["product"])] = d
    return out


def fill_orders(orders: dict[str, list[Order]], depths: dict[str, OrderDepth],
                positions: dict[str, int], cash: dict[str, float], trade_log: list,
                ts: int, day: int) -> None:
    for sym, order_list in orders.items():
        depth = depths.get(sym)
        if depth is None:
            continue
        # snapshot the available volumes at each price
        avail_sell = {p: abs(q) for p, q in depth.sell_orders.items()}
        avail_buy = {p: abs(q) for p, q in depth.buy_orders.items()}

        for o in order_list:
            qty = int(o.quantity)
            px = int(o.price)
            if qty == 0:
                continue
            pos = positions.get(sym, 0)
            lim = POS_LIM.get(sym, 0)

            if qty > 0:
                # Buying — match against available asks at price <= px
                want = qty
                # also respect position limit
                want = min(want, lim - pos)
                if want <= 0:
                    continue
                for ap in sorted(list(avail_sell.keys())):
                    if ap > px or want <= 0:
                        break
                    fill = min(avail_sell[ap], want)
                    if fill <= 0:
                        continue
                    positions[sym] = pos + fill
                    pos = positions[sym]
                    cash[sym] -= fill * ap
                    avail_sell[ap] -= fill
                    if avail_sell[ap] == 0:
                        del avail_sell[ap]
                    want -= fill
                    trade_log.append((day, ts, sym, "BUY", ap, fill))
            else:
                want = -qty
                want = min(want, lim + pos)
                if want <= 0:
                    continue
                for bp in sorted(list(avail_buy.keys()), reverse=True):
                    if bp < px or want <= 0:
                        break
                    fill = min(avail_buy[bp], want)
                    if fill <= 0:
                        continue
                    positions[sym] = pos - fill
                    pos = positions[sym]
                    cash[sym] += fill * bp
                    avail_buy[bp] -= fill
                    if avail_buy[bp] == 0:
                        del avail_buy[bp]
                    want -= fill
                    trade_log.append((day, ts, sym, "SELL", bp, fill))

        # Persist depth changes back so subsequent products at the same tick see them
        depth.buy_orders = {p: q for p, q in depth.buy_orders.items() if avail_buy.get(p, 0) > 0}
        depth.sell_orders = {p: -avail_sell[p] for p in avail_sell}


def mark_to_market(positions: dict[str, int], cash: dict[str, float],
                   depths: dict[str, OrderDepth]) -> tuple[float, dict[str, float]]:
    pnl_by_sym: dict[str, float] = {}
    total = 0.0
    for sym in PRODUCTS:
        pos = positions.get(sym, 0)
        if pos == 0:
            pnl_by_sym[sym] = cash.get(sym, 0.0)
            total += pnl_by_sym[sym]
            continue
        depth = depths.get(sym)
        # mark at mid
        if depth is not None and depth.buy_orders and depth.sell_orders:
            mid = 0.5 * (max(depth.buy_orders) + min(depth.sell_orders))
        elif depth is not None and depth.buy_orders:
            mid = float(max(depth.buy_orders))
        elif depth is not None and depth.sell_orders:
            mid = float(min(depth.sell_orders))
        else:
            mid = 0.0
        pnl_by_sym[sym] = cash.get(sym, 0.0) + pos * mid
        total += pnl_by_sym[sym]
    return total, pnl_by_sym


def run_backtest(verbose: bool = False, trace_first_n: int = 0):
    trader = Trader()
    positions: dict[str, int] = defaultdict(int)
    cash: dict[str, float] = defaultdict(float)
    trade_log: list = []
    n_traded = 0

    # listings (for completeness; the bot doesn't currently use them)
    listings = {p: Listing(p, p, 1) for p in PRODUCTS}

    daily_pnl = {}
    for day in (0, 1, 2):
        df = load_day(day)
        # Group by timestamp
        ts_groups = df.groupby("timestamp", sort=True)
        prev_ts_pnl = sum(cash.values())
        traced = 0
        for ts, rows in ts_groups:
            depths = row_to_depth(rows)
            state = TradingState(
                traderData="",
                timestamp=int(ts),
                listings=listings,
                order_depths=depths,
                own_trades={},
                market_trades={},
                position=dict(positions),
                observations=None,
            )
            orders, _, _ = trader.run(state)
            if traced < trace_first_n:
                # print compact diagnostics
                snap = {sym: [(o.price, o.quantity) for o in ol] for sym, ol in orders.items()}
                if snap:
                    print(f"day {day} ts {ts} pos={dict(positions)} orders={snap}")
                    traced += 1
            fill_orders(orders, depths, positions, cash, trade_log, int(ts), day)
        # mark to market at end of day using the last depth snapshot
        final_total, final_per = mark_to_market(positions, cash, depths)
        daily_pnl[day] = final_total - prev_ts_pnl
        if verbose:
            print(f"\n=== End of day {day} ===")
            print(f"  cash sum:        {sum(cash.values()):>14.2f}")
            print(f"  mark-to-market:  {final_total:>14.2f}")
            print(f"  positions:       {dict(positions)}")

    final_total, final_per = mark_to_market(positions, cash, depths)
    print("\n=== Final per-symbol PnL (cash + mark-to-market) ===")
    for sym in PRODUCTS:
        print(f"  {sym:>30} {final_per.get(sym, 0):>12.2f}   pos={positions.get(sym, 0)}")
    print(f"\n  TOTAL: {final_total:>14.2f}    (n_trades={len(trade_log)})")
    # break down by symbol/side
    by = defaultdict(int)
    for day, ts, sym, side, px, qty in trade_log:
        by[(sym, side)] += qty
    print("\n=== Trade volumes ===")
    for (sym, side), q in sorted(by.items()):
        print(f"  {sym:>30} {side} {q:>8}")

    return final_total, trade_log


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    trace = 5 if "--trace" in sys.argv else 0
    run_backtest(verbose=verbose, trace_first_n=trace)
