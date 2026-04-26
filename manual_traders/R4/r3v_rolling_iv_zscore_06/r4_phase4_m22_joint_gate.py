#!/usr/bin/env python3
"""Post–Phase 3: Mark 22 as passive seller on aggressive buys — same-symbol fwd K=5 vs Sonic joint gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
TH = 2
DAYS = [1, 2, 3]
K = 5
PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]


def load_px() -> pd.DataFrame:
    fs = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            fs.append(df)
    return pd.concat(fs, ignore_index=True)


def load_tr() -> pd.DataFrame:
    fs = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            fs.append(df)
    return pd.concat(fs, ignore_index=True)


def series_pair(px: pd.DataFrame, day: int, sym: str) -> tuple[np.ndarray, np.ndarray]:
    g = px[(px["day"] == day) & (px["product"] == sym)].sort_values("timestamp")
    return g["timestamp"].to_numpy(dtype=np.int64), g["mid_price"].astype(float).to_numpy()


def forward_delta(ts: np.ndarray, mid: np.ndarray, t0: int, k: int) -> float:
    i = int(np.searchsorted(ts, t0, side="left"))
    if i >= len(ts):
        return float("nan")
    if ts[i] != t0:
        i = int(np.searchsorted(ts, t0, side="right") - 1)
    i = max(0, min(i, len(ts) - 1))
    j = min(i + k, len(mid) - 1)
    if j <= i:
        return float("nan")
    return float(mid[j] - mid[i])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    px = load_px()
    tr = load_tr()
    sp = px[px["product"].isin(["VEV_5200", "VEV_5300"])].copy()
    sp["spread"] = sp["ask_price_1"] - sp["bid_price_1"]
    sp52 = sp[sp["product"] == "VEV_5200"][["day", "timestamp", "spread"]].rename(columns={"spread": "s52"})
    sp53 = sp[sp["product"] == "VEV_5300"][["day", "timestamp", "spread"]].rename(columns={"spread": "s53"})
    jt = sp52.merge(sp53, on=["day", "timestamp"])
    jt["joint_tight"] = (jt["s52"] <= TH) & (jt["s53"] <= TH)
    jmap = jt.set_index(["day", "timestamp"])["joint_tight"].to_dict()

    px2 = px.rename(columns={"product": "symbol"})
    fixed = []
    for d in DAYS:
        for sym in PRODUCTS:
            tsub = tr[(tr["day"] == d) & (tr["symbol"] == sym)].sort_values("timestamp")
            psub = px2[(px2["day"] == d) & (px2["symbol"] == sym)].sort_values("timestamp")
            if tsub.empty or psub.empty:
                continue
            mm = pd.merge_asof(
                tsub,
                psub[["timestamp", "bid_price_1", "ask_price_1"]],
                on="timestamp",
                direction="backward",
            )
            mm["day"] = d
            mm["symbol"] = sym
            fixed.append(mm)
    mtr = pd.concat(fixed, ignore_index=True)
    mtr["buy_aggr"] = mtr["price"].astype(float) >= mtr["ask_price_1"].astype(float) - 1e-9
    mtr["joint_tight"] = [jmap.get((int(r["day"]), int(r["timestamp"])), False) for _, r in mtr.iterrows()]

    sub = mtr[(mtr["seller"] == "Mark 22") & mtr["buy_aggr"]].copy()
    rows = []
    for _, r in sub.iterrows():
        d, sym, t0 = int(r["day"]), str(r["symbol"]), int(r["timestamp"])
        ts, mid = series_pair(px, d, sym)
        dlt = forward_delta(ts, mid, t0, K)
        if np.isnan(dlt):
            continue
        rows.append(
            {
                "day": d,
                "symbol": sym,
                "joint_tight": bool(r["joint_tight"]),
                f"fwd_same_K{K}": dlt,
            }
        )
    ev = pd.DataFrame(rows)
    if ev.empty:
        (OUT / "r4_p4_mark22_buyaggr_fwd5_joint_gate.txt").write_text("no_events\n")
        return

    def _summ(g: pd.DataFrame) -> str:
        if g.empty:
            return "n=0"
        x = g[f"fwd_same_K{K}"].to_numpy(dtype=float)
        return f"n={len(g)} mean={x.mean():.6g} median={np.median(x):.6g} frac_pos={(x > 0).mean():.4f}"

    lines = [
        "Mark 22 as seller on aggressive buy (price>=ask1); same-symbol forward mid K=5; R4 days 1-3.",
        f"ALL: {_summ(ev)}",
        f"JOINT_TIGHT (5200&5300 spread<={TH}): {_summ(ev[ev['joint_tight']])}",
        f"NOT_TIGHT: {_summ(ev[~ev['joint_tight']])}",
        "--- by symbol (ALL) ---",
    ]
    for sym, g in ev.groupby("symbol"):
        lines.append(f"{sym}: {_summ(g)}")
    lines.append("--- by day (ALL) ---")
    for d, g in ev.groupby("day"):
        lines.append(f"day_{d}: {_summ(g)}")
    lines.append("--- by day x joint_tight ---")
    for d in sorted(ev["day"].unique()):
        g = ev[ev["day"] == d]
        lines.append(f"day_{d} tight: {_summ(g[g['joint_tight']])} | loose: {_summ(g[~g['joint_tight']])}")

    (OUT / "r4_p4_mark22_buyaggr_fwd5_joint_gate.txt").write_text("\n".join(lines) + "\n")

    # Merged-timeline triggers for trader_v25 (same offset scheme as r4_phase2_analysis)
    m22_ev = sub.copy()
    day_list = sorted(DAYS)
    max_ts = {d: int(px[px["day"] == d]["timestamp"].max()) for d in day_list}
    cum: dict[int, int] = {}
    off = 0
    for d in day_list:
        cum[d] = off
        off += max_ts[d] + 100
    triggers_all = sorted(
        [int(cum[int(r["day"])] + int(r["timestamp"])) for _, r in m22_ev.iterrows()]
    )
    m22_tight = m22_ev[m22_ev["joint_tight"]]
    triggers_tight = sorted(
        [int(cum[int(r["day"])] + int(r["timestamp"])) for _, r in m22_tight.iterrows()]
    )
    sig_path = OUT / "r4_v25_signals.json"
    sig_path.write_text(
        json.dumps(
            {
                "mark22_buyaggr_passive_seller_merged_ts": triggers_all,
                "mark22_buyaggr_joint_tight_merged_ts": triggers_tight,
                "day_cum_offset": {str(k): v for k, v in cum.items()},
                "day_max_ts": {str(k): v for k, v in max_ts.items()},
                "window_ts": 50_000,
                "rule": "Phase1_Mark22_as_seller_on_aggressive_buy_any_symbol_merged_timeline",
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", OUT / "r4_p4_mark22_buyaggr_fwd5_joint_gate.txt", sig_path)


if __name__ == "__main__":
    main()
