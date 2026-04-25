"""More realistic backtest:
  - Fill against the standing book (just like IMC backtester) at touch price.
  - ALSO fill our passive orders at the next tick if the next tick's mid moves
    past our quote (someone's standing order would have crossed us).
  - This simulates the live trading environment where house bots actively cross
    quotes posted inside the wall.

PnL = cash + position * mid (same as IMC).
"""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA = Path("Prosperity4Data/ROUND_3")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "imc-prosperity-4-backtester"))
import prosperity4bt.datamodel as dm
sys.modules.setdefault("datamodel", dm)
from prosperity4bt.datamodel import Order, OrderDepth, TradingState, Listing  # noqa: E402

PRODUCTS_ALL = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"] + [
    f"VEV_{k}" for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
]
LIMITS = {
    "HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]},
}


def load_strat(path: str):
    spec = importlib.util.spec_from_file_location("strat_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Trader()


def row_to_depth(rows):
    out = {}
    for _, r in rows.iterrows():
        d = OrderDepth()
        for i in (1, 2, 3):
            bp = r.get(f"bid_price_{i}"); bv = r.get(f"bid_volume_{i}")
            ap = r.get(f"ask_price_{i}"); av = r.get(f"ask_volume_{i}")
            if pd.notna(bp) and pd.notna(bv) and int(bv) > 0:
                d.buy_orders[int(bp)] = int(bv)
            if pd.notna(ap) and pd.notna(av) and int(av) > 0:
                d.sell_orders[int(ap)] = -int(av)
        out[str(r["product"])] = d
    return out


def run(strat_path: str, fill_passive: bool = True, verbose: bool = False):
    trader = load_strat(strat_path)
    positions = defaultdict(int)
    cash = defaultdict(float)
    trade_log = []
    listings = {p: Listing(p, p, 1) for p in PRODUCTS_ALL}
    daily = {}
    trader_data = ""

    # Pre-load all 3 days
    all_rows = []
    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        df = df[df["product"].isin(PRODUCTS_ALL)]
        all_rows.append(df)

    grand_pnl = 0.0
    for day, df in enumerate(all_rows):
        prev = sum(cash.values()) + sum(positions[p] * 0 for p in positions)
        ts_groups = sorted(df.groupby("timestamp", sort=True))
        depths_history = []
        for i, (ts, rows) in enumerate(ts_groups):
            depths = row_to_depth(rows)
            depths_history.append((ts, depths))

        # Iterate
        for i, (ts, depths) in enumerate(depths_history):
            state = TradingState(
                traderData=trader_data, timestamp=int(ts), listings=listings,
                order_depths=depths, own_trades={}, market_trades={},
                position=dict(positions), observations=None,
            )
            orders, _, td = trader.run(state)
            trader_data = td or ""

            # Phase 1: fill against current standing book (IMC matcher)
            for sym, ord_list in orders.items():
                depth = depths.get(sym)
                if depth is None:
                    continue
                avail_sells = {p: abs(q) for p, q in depth.sell_orders.items()}
                avail_buys = {p: abs(q) for p, q in depth.buy_orders.items()}
                lim = LIMITS.get(sym, 0)
                for o in ord_list:
                    qty = int(o.quantity); px = int(o.price)
                    if qty == 0:
                        continue
                    pos = positions[sym]
                    if qty > 0:
                        want = min(qty, lim - pos)
                        for ap in sorted(avail_sells.keys()):
                            if ap > px or want <= 0:
                                break
                            fill = min(avail_sells[ap], want)
                            if fill <= 0:
                                continue
                            positions[sym] = pos + fill; pos = positions[sym]
                            cash[sym] -= fill * ap
                            avail_sells[ap] -= fill
                            if avail_sells[ap] == 0: del avail_sells[ap]
                            want -= fill
                            trade_log.append((day, ts, sym, "BUY_BOOK", ap, fill))
                            o.quantity -= fill
                    else:
                        want = min(-qty, lim + pos)
                        for bp in sorted(avail_buys.keys(), reverse=True):
                            if bp < px or want <= 0:
                                break
                            fill = min(avail_buys[bp], want)
                            if fill <= 0:
                                continue
                            positions[sym] = pos - fill; pos = positions[sym]
                            cash[sym] += fill * bp
                            avail_buys[bp] -= fill
                            if avail_buys[bp] == 0: del avail_buys[bp]
                            want -= fill
                            trade_log.append((day, ts, sym, "SELL_BOOK", bp, fill))
                            o.quantity += fill

                # Phase 2: passive fills against next-tick book (the realistic kicker)
                if fill_passive and i + 1 < len(depths_history):
                    next_depth = depths_history[i + 1][1].get(sym)
                    if next_depth is not None and next_depth.buy_orders and next_depth.sell_orders:
                        next_bb = max(next_depth.buy_orders)
                        next_ba = min(next_depth.sell_orders)
                        for o in ord_list:
                            if o.quantity == 0:
                                continue
                            qty = int(o.quantity); px = int(o.price)
                            pos = positions[sym]
                            # passive buy at price px gets filled if next-tick best ask <= px
                            if qty > 0 and next_ba <= px:
                                # how much volume at next tick? use next_depth.sell_orders volume at <=px
                                want = min(qty, lim - pos)
                                vol_avail = sum(abs(v) for p, v in next_depth.sell_orders.items() if p <= px)
                                # be conservative — fill at most min(want, half the visible vol)
                                fill = min(want, vol_avail // 2)
                                if fill > 0:
                                    positions[sym] = pos + fill
                                    cash[sym] -= fill * px  # we paid our quoted price
                                    trade_log.append((day, ts, sym, "BUY_PASS", px, fill))
                                    o.quantity -= fill
                            elif qty < 0 and next_bb >= px:
                                want = min(-qty, lim + pos)
                                vol_avail = sum(abs(v) for p, v in next_depth.buy_orders.items() if p >= px)
                                fill = min(want, vol_avail // 2)
                                if fill > 0:
                                    positions[sym] = pos - fill
                                    cash[sym] += fill * px
                                    trade_log.append((day, ts, sym, "SELL_PASS", px, fill))
                                    o.quantity += fill

        # End-of-day mark
        last_depths = depths_history[-1][1]
        total = 0.0
        for p in PRODUCTS_ALL:
            pos = positions.get(p, 0)
            d = last_depths.get(p)
            if d and d.buy_orders and d.sell_orders:
                mid = 0.5 * (max(d.buy_orders) + min(d.sell_orders))
            else:
                mid = 0.0
            total += cash[p] + pos * mid
        daily[day] = total - grand_pnl
        grand_pnl = total

    print(f"Per-day PnL: {daily}")
    # Per-symbol PnL
    final_per = {}
    for p in PRODUCTS_ALL:
        pos = positions.get(p, 0)
        d = depths_history[-1][1].get(p)
        mid = 0.5 * (max(d.buy_orders) + min(d.sell_orders)) if d and d.buy_orders and d.sell_orders else 0.0
        final_per[p] = cash[p] + pos * mid
    for p in PRODUCTS_ALL:
        if abs(final_per.get(p, 0)) > 1 or positions.get(p, 0) != 0:
            print(f"  {p:>30} {final_per[p]:>14.2f}  pos={positions.get(p, 0)}")
    print(f"  TOTAL: {sum(final_per.values()):.2f}    n_trades={len(trade_log)}")
    return sum(final_per.values()), trade_log


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "round3_analysis/strats/strat_inside_wall.py"
    no_pass = "--no-pass" in sys.argv
    run(path, fill_passive=not no_pass)
