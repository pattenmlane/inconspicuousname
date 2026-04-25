"""Realistic MM PnL using actual trade flow as the fill source.

For each trade in trades.csv:
  - if our standing bid >= trade_price (and it's a sell aggressor): we BUY at our bid
  - if our standing ask <= trade_price (and it's a buy aggressor): we SELL at our ask

We assume our orders have priority (since we're inside the book). This is
realistic for the IMC live simulator where house bots cross the wall.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("Prosperity4Data/ROUND_3")
STRIKES = [5100, 5200, 5300]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
LIM = 300
SOFT_CAP = 200


def simulate(day: int, voucher: str, soft_cap: int = SOFT_CAP, lim: int = LIM):
    prices = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
    trades = pd.read_csv(DATA / f"trades_round_3_day_{day}.csv", sep=";")

    p = prices[prices["product"] == voucher].sort_values("timestamp").reset_index(drop=True)
    t = trades[trades["symbol"] == voucher].sort_values("timestamp").reset_index(drop=True)

    if p.empty:
        return None

    # Build per-timestamp lookup of our standing quote prices
    p_idx = p.set_index("timestamp")
    pos = 0
    cash = 0.0
    fills = 0
    fill_log = []

    for _, tr in t.iterrows():
        ts = int(tr["timestamp"])
        tp = int(tr["price"])
        tq = int(tr["quantity"])
        if ts not in p_idx.index:
            continue
        row = p_idx.loc[ts]
        bb = row["bid_price_1"]; ba = row["ask_price_1"]
        if pd.isna(bb) or pd.isna(ba):
            continue
        spread = int(ba - bb)

        # Our quote prices (same logic as Trader)
        if spread >= 3:
            our_bid = int(bb) + 1
            our_ask = int(ba) - 1
        elif spread == 2:
            our_bid = int(bb)
            our_ask = int(ba)
        else:
            continue

        # Inventory skew: if very long, narrow the ask to flatten
        if pos > 100:
            our_ask = max(int(bb) + 1, our_ask - 1)
        elif pos < -100:
            our_bid = min(int(ba) - 1, our_bid + 1)

        # Determine if trade was a buy or sell aggressor.
        # If trade price >= our_ask => buyer aggressively lifted asks => we get filled SHORT at our_ask
        # If trade price <= our_bid => seller aggressively hit bids => we get filled LONG at our_bid
        # We have priority because we're INSIDE the wall.
        if tp >= our_ask and (lim + pos) > 0:
            qty = min(tq, lim + pos, max(0, soft_cap + pos))
            if qty > 0:
                cash += qty * our_ask
                pos -= qty
                fills += 1
                fill_log.append((ts, "SELL", our_ask, qty, pos))
        if tp <= our_bid and (lim - pos) > 0:
            qty = min(tq, lim - pos, max(0, soft_cap - pos))
            if qty > 0:
                cash -= qty * our_bid
                pos += qty
                fills += 1
                fill_log.append((ts, "BUY", our_bid, qty, pos))

    last_mid = float(p["mid_price"].iloc[-1])
    final_pnl = cash + pos * last_mid
    return {
        "day": day, "voucher": voucher, "n_fills": fills,
        "final_pnl": final_pnl, "final_pos": pos,
        "n_trades_in_csv": len(t),
    }


def main():
    rows = []
    for day in (0, 1, 2):
        for v in VOUCHERS:
            r = simulate(day, v)
            if r is None:
                continue
            rows.append(r)
    print(f"{'day':>3} {'voucher':>10} {'final_pnl':>12} {'pos':>6} {'fills':>7} {'csv_trades':>11}")
    for r in rows:
        print(f"{r['day']:>3} {r['voucher']:>10} {r['final_pnl']:>12.2f} {r['final_pos']:>6} "
              f"{r['n_fills']:>7} {r['n_trades_in_csv']:>11}")
    tot = sum(r["final_pnl"] for r in rows)
    print(f"\nTotal: {tot:,.2f}   per-day-strike avg: {tot/9:,.2f}")


if __name__ == "__main__":
    main()
