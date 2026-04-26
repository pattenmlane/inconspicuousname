#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned forward mids (K in {5,20,100} ticks).

Horizon definition: **K ticks** = K steps forward on the **same (day, symbol)** price tape,
using the ordered sequence of timestamps present in `prices_round_4_day_*.csv` for that symbol
(typically 100 timestamp units per step in this dataset).

Aggressor inference: at trade (day, timestamp, symbol, price), compare `price` to concurrent
`bid_price_1` / `ask_price_1` from the price row with matching (day, timestamp, product).
  - buyer-aggressive if price >= ask1
  - seller-aggressive if price <= bid1
  - else unknown (both / mid)

Outputs under manual_traders/R4/r3v_volume_weighted_residual_05/analysis_outputs/phase1/
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# .../manual_traders/R4/<id>/this_file.py -> repo root is parents[3]
REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

FOCUS = ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK", "VEV_5200", "VEV_5300", "VEV_5400"]
KS = (5, 20, 100)
MIN_N_CELL = 30
MIN_N_MARK = 80


def load_prices() -> pd.DataFrame:
    frames = []
    for d in (1, 2, 3):
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = df["day"].astype(int)
        df["timestamp"] = df["timestamp"].astype(int)
        df["product"] = df["product"].astype(str)
        df["mid"] = pd.to_numeric(df["mid_price"], errors="coerce")
        df["bid1"] = pd.to_numeric(df["bid_price_1"], errors="coerce")
        df["ask1"] = pd.to_numeric(df["ask_price_1"], errors="coerce")
        df["spr"] = df["ask1"] - df["bid1"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_mid_paths(pr: pd.DataFrame) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(day, symbol) -> (timestamps int64, mids float64, spreads float64)."""
    out: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (day, sym), g in pr.groupby(["day", "product"]):
        g = g.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")
        ts = g["timestamp"].to_numpy(np.int64)
        mid = g["mid"].to_numpy(np.float64)
        spr = g["spr"].to_numpy(np.float64)
        out[(int(day), str(sym))] = (ts, mid, spr)
    return out


def fwd_mid(ts_arr: np.ndarray, mid_arr: np.ndarray, t: int, k: int) -> float:
    i = int(np.searchsorted(ts_arr, t, side="left"))
    if i >= len(ts_arr) or ts_arr[i] != t:
        return float("nan")
    j = i + k
    if j >= len(mid_arr):
        return float("nan")
    return float(mid_arr[j] - mid_arr[i])


def spread_at(pr_idx: dict, day: int, sym: str, t: int) -> float:
    key = (day, sym)
    if key not in pr_idx:
        return float("nan")
    ts_arr, _, spr_arr = pr_idx[key]
    i = int(np.searchsorted(ts_arr, t, side="left"))
    if i >= len(ts_arr) or ts_arr[i] != t:
        return float("nan")
    return float(spr_arr[i])


def aggressor(price: float, bid1: float, ask1: float) -> str:
    if not (np.isfinite(price) and np.isfinite(bid1) and np.isfinite(ask1)):
        return "unk"
    if price >= ask1:
        return "buy_agg"
    if price <= bid1:
        return "sell_agg"
    return "unk"


def main() -> None:
    pr = load_prices()
    pr_idx = build_mid_paths(pr)
    # fast lookup for BBO at (day, ts, sym)
    bbo = pr.set_index(["day", "timestamp", "product"])[["bid1", "ask1", "spr"]]

    trades: list[pd.DataFrame] = []
    for d in (1, 2, 3):
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        t = pd.read_csv(p, sep=";")
        t["day"] = d
        trades.append(t)
    tr = pd.concat(trades, ignore_index=True)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = tr["quantity"].astype(int)
    tr["buyer"] = tr["buyer"].astype(str)
    tr["seller"] = tr["seller"].astype(str)
    tr["symbol"] = tr["symbol"].astype(str)

    rows_out: list[dict] = []
    for _, r in tr.iterrows():
        day = int(r["day"])
        ts = int(r["timestamp"])
        sym = str(r["symbol"])
        px = float(r["price"])
        key = (day, ts, sym)
        if key not in bbo.index:
            bid1, ask1 = float("nan"), float("nan")
        else:
            bid1 = float(bbo.loc[key, "bid1"])
            ask1 = float(bbo.loc[key, "ask1"])
        ag = aggressor(px, bid1, ask1)
        spr = spread_at(pr_idx, day, sym, ts)
        spr_u = spread_at(pr_idx, day, "VELVETFRUIT_EXTRACT", ts)
        spr_bin = "tight_u" if (np.isfinite(spr_u) and spr_u <= 6) else ("wide_u" if np.isfinite(spr_u) else "na")
        # hour bucket: 4 equal-count bins on timestamp within day (coarse session)
        rows_out.append(
            {
                "day": day,
                "timestamp": ts,
                "symbol": sym,
                "buyer": r["buyer"],
                "seller": r["seller"],
                "qty": int(r["quantity"]),
                "notional": abs(px * r["quantity"]),
                "aggressor": ag,
                "spread_sym": spr,
                "spread_u": spr_u,
                "spr_bin_u": spr_bin,
            }
        )

    ev = pd.DataFrame(rows_out)
    # forward mids same symbol + extract
    for K in KS:
        col = f"fwd_mid_{K}"
        colx = f"fwd_u_{K}"
        colh = f"fwd_h_{K}"
        vals, vals_u, vals_h = [], [], []
        for _, r in ev.iterrows():
            d, ts, sym = int(r["day"]), int(r["timestamp"]), str(r["symbol"])
            key = (d, sym)
            if key not in pr_idx:
                vals.append(float("nan"))
            else:
                ts_a, mid_a, _ = pr_idx[key]
                vals.append(fwd_mid(ts_a, mid_a, ts, K))
            ku = (d, "VELVETFRUIT_EXTRACT")
            if ku in pr_idx:
                ts_a, mid_a, _ = pr_idx[ku]
                vals_u.append(fwd_mid(ts_a, mid_a, ts, K))
            else:
                vals_u.append(float("nan"))
            kh = (d, "HYDROGEL_PACK")
            if kh in pr_idx:
                ts_a, mid_a, _ = pr_idx[kh]
                vals_h.append(fwd_mid(ts_a, mid_a, ts, K))
            else:
                vals_h.append(float("nan"))
        ev[col] = vals
        ev[colx] = vals_u
        ev[colh] = vals_h

    ev.to_csv(OUT / "r4_trades_enriched.csv", index=False)

    # --- 1) Participant predictiveness: Mark U when buyer or seller, split aggressor ---
    summary_rows = []
    names = sorted(set(ev["buyer"]) | set(ev["seller"]))
    for U in names:
        if U == "nan":
            continue
        sub = ev[(ev["buyer"] == U) | (ev["seller"] == U)].copy()
        sub["side_role"] = np.where(
            (sub["buyer"] == U) & (sub["aggressor"] == "buy_agg"),
            "U_buy_agg",
            np.where(
                (sub["seller"] == U) & (sub["aggressor"] == "sell_agg"),
                "U_sell_agg",
                np.where(sub["buyer"] == U, "U_passive_buy", "U_passive_sell"),
            ),
        )
        for role, g in sub.groupby("side_role"):
            for sym in g["symbol"].unique():
                gg = g[g["symbol"] == sym]
                for K in KS:
                    c = f"fwd_mid_{K}"
                    x = pd.to_numeric(gg[c], errors="coerce").dropna()
                    if len(x) < MIN_N_MARK:
                        continue
                    m, s = float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0
                    se = s / math.sqrt(len(x)) if len(x) > 1 else float("nan")
                    tstat = m / se if se and se > 0 and math.isfinite(se) else float("nan")
                    summary_rows.append(
                        {
                            "Mark": U,
                            "role": role,
                            "symbol": sym,
                            "K": K,
                            "n": len(x),
                            "mean_fwd_mid": m,
                            "t_stat": tstat,
                            "frac_pos": float((x > 0).mean()),
                        }
                    )

    s1 = pd.DataFrame(summary_rows)
    s1.to_csv(OUT / "r4_participant_predictiveness.csv", index=False)
    top1 = s1[s1["n"] >= MIN_N_MARK].sort_values("mean_fwd_mid", ascending=False).head(25)
    top1.to_csv(OUT / "r4_participant_predictiveness_top25.csv", index=False)

    # Stratify: product x spr_bin_u x K for Mark 01 when seller aggressive (example high-flow)
    strat_rows = []
    for sym_focus in FOCUS:
        for sprb in ["tight_u", "wide_u", "na"]:
            g = ev[(ev["symbol"] == sym_focus) & (ev["spr_bin_u"] == sprb)]
            for K in KS:
                c = f"fwd_mid_{K}"
                x = pd.to_numeric(g[c], errors="coerce").dropna()
                if len(x) < MIN_N_CELL:
                    continue
                strat_rows.append(
                    {
                        "slice": f"all|sym={sym_focus}|u_spread={sprb}",
                        "K": K,
                        "n": len(x),
                        "mean": float(x.mean()),
                        "frac_pos": float((x > 0).mean()),
                    }
                )
    pd.DataFrame(strat_rows).to_csv(OUT / "r4_stratify_symbol_spreadbin.csv", index=False)

    # --- 2) Baseline: cell mean by (buyer, seller, symbol) for K=20 ---
    K0 = 20
    c0 = f"fwd_mid_{K0}"
    cell = (
        ev.groupby(["buyer", "seller", "symbol"])[c0]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "cell_mean_fwd", "count": "cell_n"})
    )
    cell.to_csv(OUT / "r4_cell_mean_buyer_seller_symbol_k20.csv", index=False)
    ev2 = ev.merge(
        cell,
        on=["buyer", "seller", "symbol"],
        how="left",
    )
    ev2["residual_k20"] = pd.to_numeric(ev2[c0], errors="coerce") - ev2["cell_mean_fwd"]
    ev2.to_csv(OUT / "r4_trades_with_residual_k20.csv", index=False)
    res_summary = (
        ev2.groupby(["buyer", "seller", "symbol"])["residual_k20"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    res_summary = res_summary[res_summary["count"] >= MIN_N_CELL].sort_values(
        "mean", key=abs, ascending=False
    )
    res_summary.head(40).to_csv(OUT / "r4_residual_top_absmean.csv", index=False)

    # --- 3) Graph buyer -> seller ---
    pair_c = Counter(zip(ev["buyer"], ev["seller"]))
    pair_n = defaultdict(float)
    for _, r in ev.iterrows():
        pair_n[(r["buyer"], r["seller"])] += float(r["notional"])
    lines = ["=== buyer -> seller (count, notional) ==="]
    for (b, s), n in pair_c.most_common(25):
        lines.append(f"{b} -> {s}: count={n} notional={pair_n[(b,s)]:.1f}")
    (OUT / "r4_graph_top_pairs.txt").write_text("\n".join(lines) + "\n")

    # 2-hop: Mark 01 -> X -> Y counts (simple)
    edges = [(b, s) for b, s in pair_c if pair_c[(b, s)] >= 20]
    succ: dict[str, Counter] = defaultdict(Counter)
    for b, s in pair_c:
        succ[b][s] += pair_c[(b, s)]
    hop2 = Counter()
    for a in succ:
        for b, w1 in succ[a].most_common(8):
            for c, w2 in succ[b].most_common(8):
                hop2[(a, b, c)] += min(w1, w2)
    hlines = ["=== top 2-hop chains A->B->C (min weight heuristic) ==="]
    for chain, w in hop2.most_common(20):
        hlines.append(f"{chain[0]} -> {chain[1]} -> {chain[2]}: score={w}")
    (OUT / "r4_graph_2hop.txt").write_text("\n".join(hlines) + "\n")

    # --- 4) Bursts ---
    burst = ev.groupby(["day", "timestamp"]).agg(
        n=("symbol", "count"),
        symbols=("symbol", lambda x: ",".join(sorted(set(x)))),
        buyers=("buyer", lambda x: ",".join(sorted(set(x)))),
        sellers=("seller", lambda x: ",".join(sorted(set(x)))),
    )
    burst = burst[burst["n"] >= 3].reset_index()
    burst.to_csv(OUT / "r4_bursts_ge3.csv", index=False)
    # orchestrator = mode of buyer across burst rows
    orch = []
    for (day, ts), g in ev.groupby(["day", "timestamp"]):
        if len(g) < 3:
            continue
        bc = g["buyer"].value_counts().idxmax()
        orch.append({"day": day, "timestamp": ts, "n": len(g), "orch_buyer": bc})
    orch_df = pd.DataFrame(orch)
    orch_df.to_csv(OUT / "r4_burst_orchestrator_buyer_mode.csv", index=False)

    # Event study: after burst (day,ts), forward U mid at +K vs random control (same day, random ts with n=1 trade)
    rng = np.random.default_rng(0)
    single = ev.groupby(["day", "timestamp"]).filter(lambda x: len(x) == 1)
    burst_keys = set(zip(burst["day"], burst["timestamp"]))
    u_key = (1, "VELVETFRUIT_EXTRACT")  # placeholder; use pr_idx per day
    ev_study = []
    for (d, ts) in burst_keys:
        if (d, "VELVETFRUIT_EXTRACT") not in pr_idx:
            continue
        ts_a, mid_a, _ = pr_idx[(d, "VELVETFRUIT_EXTRACT")]
        for K in (5, 20):
            fu = fwd_mid(ts_a, mid_a, int(ts), K)
            if not np.isfinite(fu):
                continue
            # control: one random single-trade timestamp same day
            cand = single[single["day"] == d]["timestamp"].to_numpy()
            if len(cand) == 0:
                continue
            ts2 = int(rng.choice(cand))
            fu2 = fwd_mid(ts_a, mid_a, ts2, K)
            if np.isfinite(fu2):
                ev_study.append({"day": d, "K": K, "burst_fwd_u": fu, "control_fwd_u": fu2})
    pd.DataFrame(ev_study).to_csv(OUT / "r4_burst_vs_control_extract_fwd.csv", index=False)

    # --- 5) Adverse selection proxy: after buy_agg / sell_agg, same symbol fwd ---
    adv = []
    for ag in ["buy_agg", "sell_agg"]:
        g = ev[ev["aggressor"] == ag]
        for K in KS:
            c = f"fwd_mid_{K}"
            x = pd.to_numeric(g[c], errors="coerce").dropna()
            adv.append(
                {
                    "aggressor": ag,
                    "K": K,
                    "n": len(x),
                    "mean_fwd": float(x.mean()),
                    "frac_pos": float((x > 0).mean()),
                }
            )
    pd.DataFrame(adv).to_csv(OUT / "r4_adverse_proxy_aggressor_fwd.csv", index=False)

    # Day-stability: Mark 01 -> Mark 22 on VEV_5300, K=20
    stab = []
    pair_mask = (ev["buyer"] == "Mark 01") & (ev["seller"] == "Mark 22") & (ev["symbol"] == "VEV_5300")
    for d in (1, 2, 3):
        x = pd.to_numeric(ev.loc[pair_mask & (ev["day"] == d), "fwd_mid_20"], errors="coerce").dropna()
        if len(x) == 0:
            continue
        stab.append(
            {
                "slice": "Mark01->Mark22|VEV_5300|K=20",
                "day": d,
                "n": len(x),
                "mean": float(x.mean()),
                "frac_pos": float((x > 0).mean()),
            }
        )
    pd.DataFrame(stab).to_csv(OUT / "r4_day_stability_m01_m22_vev5300_k20.csv", index=False)

    print("Wrote", OUT)


if __name__ == "__main__":
    main()
