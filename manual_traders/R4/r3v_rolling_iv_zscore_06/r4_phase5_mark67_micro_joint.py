#!/usr/bin/env python3
"""
Post–Phase 3: refine **Mark 67 aggressive buy on EXTRACT** using the **Sonic joint gate**
(5200+5300 spread<=2) at the tape print. **Microprice minus mid** is logged: on this
slice, aggressive buys at the ask imply **micro > mid for every** event (redundant
as a filter). Compare forward **VEV_5300** mid K=5 / K=20 vs the full pool; emit
merged timestamps for `trader_v26` (joint-tight **at Mark67 print time**).
"""
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
    px2 = px.rename(columns={"product": "symbol"})

    sp = px[px["product"].isin(["VEV_5200", "VEV_5300"])].copy()
    sp["spread"] = sp["ask_price_1"] - sp["bid_price_1"]
    sp52 = sp[sp["product"] == "VEV_5200"][["day", "timestamp", "spread"]].rename(columns={"spread": "s52"})
    sp53 = sp[sp["product"] == "VEV_5300"][["day", "timestamp", "spread"]].rename(columns={"spread": "s53"})
    jt = sp52.merge(sp53, on=["day", "timestamp"])
    jt["joint_tight"] = (jt["s52"] <= TH) & (jt["s53"] <= TH)
    jmap = jt.set_index(["day", "timestamp"])["joint_tight"].to_dict()

    sym = "VELVETFRUIT_EXTRACT"
    rows = []
    for d in DAYS:
        tsub = tr[(tr["day"] == d) & (tr["symbol"] == sym)].sort_values("timestamp")
        psub = px2[(px2["day"] == d) & (px2["symbol"] == sym)].sort_values("timestamp")
        if tsub.empty or psub.empty:
            continue
        mm = pd.merge_asof(
            tsub,
            psub[
                [
                    "timestamp",
                    "bid_price_1",
                    "ask_price_1",
                    "bid_volume_1",
                    "ask_volume_1",
                    "mid_price",
                ]
            ],
            on="timestamp",
            direction="backward",
        )
        mm["day"] = d
        for _, r in mm.iterrows():
            if r["buyer"] != "Mark 67":
                continue
            bp, ap = float(r["bid_price_1"]), float(r["ask_price_1"])
            if float(r["price"]) < ap - 1e-9:
                continue
            bv = float(r["bid_volume_1"])
            av = abs(float(r["ask_volume_1"]))
            mid = float(r["mid_price"])
            if bv + av <= 0:
                continue
            micro = (bp * av + ap * bv) / (bv + av)
            mic_bias = micro - mid
            jt0 = bool(jmap.get((d, int(r["timestamp"])), False))
            t0 = int(r["timestamp"])
            ts53, m53 = series_pair(px, d, "VEV_5300")
            f5 = forward_delta(ts53, m53, t0, 5)
            f20 = forward_delta(ts53, m53, t0, 20)
            rows.append(
                {
                    "day": d,
                    "timestamp": t0,
                    "joint_tight": jt0,
                    "micro_minus_mid": mic_bias,
                    "fwd5300_K5": f5,
                    "fwd5300_K20": f20,
                }
            )

    ev = pd.DataFrame(rows)
    if ev.empty:
        (OUT / "r4_p5_mark67_extract_refine.txt").write_text("no_events\n")
        return

    def block(name: str, mask: pd.Series) -> str:
        g = ev[mask]
        if g.empty:
            return f"{name}: n=0"
        a = g["fwd5300_K5"].to_numpy(dtype=float)
        b = g["fwd5300_K20"].to_numpy(dtype=float)
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        return (
            f"{name}: n={len(g)} "
            f"meanK5={a.mean():.4f} fracposK5={(a>0).mean():.3f} "
            f"meanK20={b.mean():.4f} fracposK20={(b>0).mean():.3f}"
        )

    n_all = len(ev)
    n_micro = int((ev["micro_minus_mid"] > 0).sum())
    lines = [
        "Mark 67 aggressive buy on VELVETFRUIT_EXTRACT; forward VEV_5300 mid; R4 days 1-3.",
        f"Note: micro_minus_mid>0 for {n_micro}/{n_all} events (buy at/through ask implies upward micro bias; not a selective filter on this slice).",
        block("ALL", pd.Series(True, index=ev.index)),
        block("joint_tight", ev["joint_tight"]),
        block("micro_gt_mid (micro-mid>0)", ev["micro_minus_mid"] > 0),
        block("joint_tight AND micro_gt_mid", ev["joint_tight"] & (ev["micro_minus_mid"] > 0)),
        "--- by day (joint & micro) ---",
    ]
    m = ev["joint_tight"] & (ev["micro_minus_mid"] > 0)
    for d in sorted(ev["day"].unique()):
        lines.append(block(f"day_{d}", (ev["day"] == d) & m))
    (OUT / "r4_p5_mark67_extract_refine.txt").write_text("\n".join(lines) + "\n")

    # Merged timeline triggers for trader_v26 (same offsets as r4_v23_signals.json)
    day_list = sorted(DAYS)
    max_ts = {d: int(px[px["day"] == d]["timestamp"].max()) for d in day_list}
    cum: dict[int, int] = {}
    off = 0
    for d in day_list:
        cum[d] = off
        off += max_ts[d] + 100

    pick = ev[m]
    triggers = sorted(int(cum[int(r["day"])] + int(r["timestamp"])) for _, r in pick.iterrows())
    (OUT / "r4_v26_signals.json").write_text(
        json.dumps(
            {
                "mark67_extract_buy_aggr_filtered_merged_ts": triggers,
                "filters": {
                    "joint_5200_5300_spread_le": TH,
                    "extract_micro_minus_mid_gt": 0,
                    "note_micro_gt_mid": "Vacuous on this slice (all Mark67 aggr extract prints have micro>mid).",
                },
                "day_cum_offset": {str(k): v for k, v in cum.items()},
                "day_max_ts": {str(k): v for k, v in max_ts.items()},
                "window_ts": 50_000,
                "rule": "Mark67_aggr_buy_EXTRACT_and_joint_tight_and_micro_gt_mid",
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", OUT / "r4_p5_mark67_extract_refine.txt", OUT / "r4_v26_signals.json", "n_triggers", len(triggers))


if __name__ == "__main__":
    main()
