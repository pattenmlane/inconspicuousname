#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned forward mids (tape evidence).

Reads Prosperity4Data/ROUND_4 prices + trades for days 1–3 (backtester layout).
Horizon K: forward timestamp = trade_ts + K * 100 (same 100-tick grid as prices).

Outputs under manual_traders/R4/r3v_jump_gap_filter_17/outputs/phase1/
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
HORIZONS = (5, 20, 100)
PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]
CROSS = ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"]
TICK = 100
MAX_TS = 999900


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        df = pd.read_csv(p, sep=";")
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        df = pd.read_csv(p, sep=";")
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def spread_tob(row: pd.Series) -> float:
    ba = row["ask_price_1"]
    bb = row["bid_price_1"]
    if pd.isna(ba) or pd.isna(bb):
        return float("nan")
    return float(ba) - float(bb)


def forward_mid(
    mid_index: dict[tuple[int, str, int], float], day: int, sym: str, ts: int
) -> float | None:
    key = (day, sym, ts)
    v = mid_index.get(key)
    return None if v is None else float(v)


def main() -> None:
    pr = load_prices()
    pr["spread"] = pr.apply(spread_tob, axis=1)
    pr["mid"] = pr["mid_price"].astype(float)

    # Per (day, product) spread quantiles for regime tag at each timestamp
    q50 = pr.groupby(["day", "product"])["spread"].transform("median")
    pr["spread_regime"] = np.where(pr["spread"] <= q50, "tight", "wide")

    keys = list(zip(pr["day"].astype(int), pr["product"].astype(str), pr["timestamp"].astype(int)))
    mid_index = dict(zip(keys, pr["mid"].astype(float)))
    regime_index = dict(zip(keys, pr["spread_regime"].astype(str)))

    tr = load_trades()
    tr["price"] = tr["price"].astype(float)
    tr["quantity"] = tr["quantity"].astype(int)

    # Merge BBO at trade timestamp
    bb = pr.rename(
        columns={
            "bid_price_1": "bb",
            "ask_price_1": "ba",
            "spread": "bbo_spread",
            "spread_regime": "sym_spread_regime",
        }
    )[["day", "timestamp", "product", "bb", "ba", "mid", "bbo_spread", "sym_spread_regime"]]
    tr = tr.merge(
        bb,
        left_on=["day", "timestamp", "symbol"],
        right_on=["day", "timestamp", "product"],
        how="left",
    )
    tr.drop(columns=["product"], inplace=True, errors="ignore")

    def aggressor(row: pd.Series) -> str:
        if pd.isna(row["ba"]) or pd.isna(row["bb"]):
            return "unknown"
        p, ba, bb = row["price"], float(row["ba"]), float(row["bb"])
        if p >= ba:
            return "aggr_buy"
        if p <= bb:
            return "aggr_sell"
        return "unknown"

    tr["aggressor"] = tr.apply(aggressor, axis=1)

    # Burst: (day, timestamp) with >= 2 symbols
    burst_size = tr.groupby(["day", "timestamp"])["symbol"].nunique()
    burst_ts = set(burst_size[burst_size >= 2].index)
    tr["burst"] = tr.apply(lambda r: (int(r["day"]), int(r["timestamp"])) in burst_ts, axis=1)

    # Session bucket: coarse decile of timestamp within day
    tr["session_bin"] = (tr["timestamp"] // 100000).astype(int).clip(0, 9)

    rows_out: list[dict] = []
    for r in tr.itertuples(index=False):
        day = int(r.day)
        ts = int(r.timestamp)
        sym = str(r.symbol)
        mid0 = mid_index.get((day, sym, ts))
        if mid0 is None or mid0 <= 0:
            continue
        regime = regime_index.get((day, sym, ts), "unknown")
        for K in HORIZONS:
            fts = min(ts + K * TICK, MAX_TS)
            m1 = forward_mid(mid_index, day, sym, fts)
            if m1 is None:
                continue
            d_same = m1 - mid0
            row = {
                "day": day,
                "timestamp": ts,
                "symbol": sym,
                "buyer": str(r.buyer),
                "seller": str(r.seller),
                "pair": f"{r.buyer}->{r.seller}",
                "aggressor": r.aggressor,
                "burst": bool(r.burst),
                "session_bin": int(r.session_bin),
                "spread_regime": regime,
                "K": K,
                "fwd_same": d_same,
            }
            for cs in CROSS:
                m0c = mid_index.get((day, cs, ts))
                m1c = forward_mid(mid_index, day, cs, fts)
                if m0c is not None and m1c is not None:
                    row[f"fwd_{cs}"] = m1c - m0c
                else:
                    row[f"fwd_{cs}"] = float("nan")
            rows_out.append(row)

    ev = pd.DataFrame(rows_out)

    # --- 1) Participant-level tables ---
    def summarize_group(gdf: pd.DataFrame, col: str) -> pd.DataFrame:
        gdf = gdf[np.isfinite(gdf[col])]
        if len(gdf) < 5:
            return pd.DataFrame()
        mu = float(gdf[col].mean())
        med = float(gdf[col].median())
        pos = float((gdf[col] > 0).mean())
        n = int(len(gdf))
        s = float(gdf[col].std(ddof=1)) if n > 1 else float("nan")
        tstat = (mu / (s / math.sqrt(n))) if n > 1 and s > 0 and math.isfinite(s) else float("nan")
        return pd.DataFrame(
            [{"n": n, "mean": mu, "median": med, "frac_pos": pos, "t_stat": tstat}]
        )

    part_rows = []
    for U in sorted(set(ev["buyer"]) | set(ev["seller"])):
        for role in ("buyer", "seller"):
            sub = ev[ev["buyer"] == U] if role == "buyer" else ev[ev["seller"] == U]
            if len(sub) < 10:
                continue
            for ag in ["all", "aggr_buy", "aggr_sell"]:
                s0 = sub if ag == "all" else sub[sub["aggressor"] == ag]
                if len(s0) < 10:
                    continue
                for K in HORIZONS:
                    sk = s0[s0["K"] == K]
                    for sym in PRODUCTS:
                        ss = sk[sk["symbol"] == sym]
                        if len(ss) < 8:
                            continue
                        for reg in ["all", "tight", "wide"]:
                            s1 = ss if reg == "all" else ss[ss["spread_regime"] == reg]
                            if len(s1) < 8:
                                continue
                            for br in [False, True]:
                                s2 = s1[s1["burst"] == br] if br else s1[~s1["burst"]]
                                if len(s2) < 8:
                                    continue
                                st = summarize_group(s2, "fwd_same")
                                if st.empty:
                                    continue
                                part_rows.append(
                                    {
                                        "participant": U,
                                        "side_role": role,
                                        "aggressor_filter": ag,
                                        "K": K,
                                        "symbol": sym,
                                        "spread_regime": reg,
                                        "burst_only": br,
                                        **st.iloc[0].to_dict(),
                                    }
                                )

    part_df = pd.DataFrame(part_rows)
    part_df.sort_values(["n", "t_stat"], ascending=[False, False]).to_csv(
        OUT / "participant_forward_stats.csv", index=False
    )

    # Pool across days: top by |t| with n>=30
    top_part = (
        part_df[part_df["n"] >= 30]
        .assign(abs_t=lambda d: d["t_stat"].abs())
        .sort_values("abs_t", ascending=False)
        .head(80)
    )
    top_part.to_csv(OUT / "participant_forward_stats_top.csv", index=False)

    # --- 2) Pair baseline cell means + residuals ---
    cell = (
        ev.groupby(["buyer", "seller", "symbol", "spread_regime", "K"])["fwd_same"]
        .agg(["mean", "count"])
        .reset_index()
    )
    cell.rename(columns={"mean": "cell_mean_fwd", "count": "cell_n"}, inplace=True)
    ev2 = ev.merge(
        cell,
        on=["buyer", "seller", "symbol", "spread_regime", "K"],
        how="left",
    )
    ev2["residual_fwd"] = ev2["fwd_same"] - ev2["cell_mean_fwd"]
    ev2.to_csv(OUT / "events_with_cell_residual.csv", index=False)

    resid_sum = (
        ev2.groupby(["buyer", "seller", "symbol", "K"])["residual_fwd"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    resid_sum = resid_sum[resid_sum["count"] >= 15].sort_values(
        "mean", key=lambda s: s.abs(), ascending=False
    )
    resid_sum.head(100).to_csv(OUT / "pair_symbol_residual_summary.csv", index=False)

    # Leave-one-day-out cell mean (lighter overfit check)
    loo_rows = []
    for holdout in DAYS:
        train = ev[ev["day"] != holdout]
        test = ev[ev["day"] == holdout]
        cell_tr = (
            train.groupby(["buyer", "seller", "symbol", "spread_regime", "K"])["fwd_same"]
            .mean()
            .reset_index(name="cell_mean_loo")
        )
        te = test.merge(
            cell_tr,
            on=["buyer", "seller", "symbol", "spread_regime", "K"],
            how="left",
        )
        te["res_loo"] = te["fwd_same"] - te["cell_mean_loo"]
        for (b, s, sym, K), g in te.groupby(["buyer", "seller", "symbol", "K"]):
            if len(g) < 10:
                continue
            loo_rows.append(
                {
                    "holdout_day": holdout,
                    "buyer": b,
                    "seller": s,
                    "symbol": sym,
                    "K": int(K),
                    "n": len(g),
                    "mean_res_loo": float(g["res_loo"].mean()),
                }
            )
    pd.DataFrame(loo_rows).to_csv(OUT / "leave_one_day_residuals.csv", index=False)

    # --- 3) Graph buyer -> seller ---
    pair_counts = tr.groupby(["buyer", "seller"]).size().reset_index(name="count")
    pair_notional = (
        tr.assign(notional=tr["price"] * tr["quantity"])
        .groupby(["buyer", "seller"])["notional"]
        .sum()
        .reset_index(name="notional")
    )
    edges = pair_counts.merge(pair_notional, on=["buyer", "seller"])
    edges.sort_values("count", ascending=False).to_csv(OUT / "graph_edges.csv", index=False)

    rev_edges = edges.assign(rb=edges["seller"], rs=edges["buyer"])[["rb", "rs", "count", "notional"]].rename(
        columns={"count": "count_rev", "notional": "notional_rev"}
    )
    recip = edges.merge(rev_edges, left_on=["buyer", "seller"], right_on=["rb", "rs"], how="left")
    recip["count_rev"] = recip["count_rev"].fillna(0).astype(int)
    recip["notional_rev"] = recip["notional_rev"].fillna(0.0)
    recip["reciprocity_ratio"] = np.where(
        recip["count"] > 0, recip["count_rev"].astype(float) / recip["count"].astype(float), np.nan
    )
    recip.sort_values("count", ascending=False)[
        ["buyer", "seller", "count", "notional", "count_rev", "notional_rev", "reciprocity_ratio"]
    ].head(50).to_csv(OUT / "graph_reciprocity_top.csv", index=False)

    lines = ["=== Directed pair summary ==="]
    lines.append(f"Unique directed edges: {len(edges)}")
    top = edges.head(15)
    for _, r in top.iterrows():
        lines.append(f"  {r['buyer']} -> {r['seller']}: count={int(r['count'])} notional={r['notional']:.0f}")
    # hub score: in+out degree by count
    inc = edges.groupby("seller")["count"].sum()
    outc = edges.groupby("buyer")["count"].sum()
    names = set(inc.index) | set(outc.index)
    hub = sorted(
        ((n, int(inc.get(n, 0) + outc.get(n, 0))) for n in names),
        key=lambda x: -x[1],
    )[:10]
    lines.append("Hub (total incident trade count on directed edges):")
    for n, c in hub:
        lines.append(f"  {n}: {c}")
    lines.append("")
    lines.append("Reciprocity (reverse edge count / forward count); see graph_reciprocity_top.csv")
    rt = recip.sort_values("count", ascending=False).head(8)
    for _, r in rt.iterrows():
        lines.append(
            f"  {r['buyer']}->{r['seller']}: fwd={int(r['count'])} rev={int(r['count_rev'])} ratio={r['reciprocity_ratio']:.3f}"
        )
    (OUT / "graph_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # Weak 2-hop: consecutive rows (time order) with seller_i == buyer_{i+1}
    trs = tr.sort_values(["day", "timestamp"])
    chains: list[tuple[str, str, str]] = []
    for _, g in trs.groupby("day"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            a1, s1 = str(g.loc[i, "buyer"]), str(g.loc[i, "seller"])
            a2, s2 = str(g.loc[i + 1, "buyer"]), str(g.loc[i + 1, "seller"])
            if s1 == a2:
                chains.append((a1, s1, s2))
    if chains:
        cnt = Counter(chains)
        topc = sorted(cnt.items(), key=lambda kv: -kv[1])[:40]
        pd.DataFrame(
            [{"A": a, "B": b, "C": c, "count": n} for (a, b, c), n in topc]
        ).to_csv(OUT / "two_hop_chain_counts.csv", index=False)

    # --- 4) Bursts ---
    burst_df = (
        tr.groupby(["day", "timestamp"])
        .agg(
            n_prints=("symbol", "size"),
            n_syms=("symbol", "nunique"),
            buyer_mode=("buyer", lambda x: x.mode().iloc[0] if len(x) else ""),
            seller_mode=("seller", lambda x: x.mode().iloc[0] if len(x) else ""),
        )
        .reset_index()
    )
    burst_df = burst_df[burst_df["n_syms"] >= 2].sort_values("n_prints", ascending=False)
    burst_df.to_csv(OUT / "burst_multi_symbol.csv", index=False)

    # Forward extract mid after burst vs random control (same n, matched timestamp count)
    burst_ts_set = set(zip(burst_df["day"], burst_df["timestamp"]))
    ext_fwd = []
    for _, b in burst_df.head(500).iterrows():
        d, ts = int(b["day"]), int(b["timestamp"])
        for K in (5, 20):
            fts = min(ts + K * TICK, MAX_TS)
            m0 = mid_index.get((d, "VELVETFRUIT_EXTRACT", ts))
            m1 = mid_index.get((d, "VELVETFRUIT_EXTRACT", fts))
            if m0 is None or m1 is None:
                continue
            ext_fwd.append({"K": K, "burst": True, "fwd": m1 - m0})
    # controls: random timestamps with same day distribution
    all_ts = tr[["day", "timestamp"]].drop_duplicates()
    ctrl = all_ts.sample(min(2000, len(all_ts)), random_state=1)
    for _, c in ctrl.iterrows():
        d, ts = int(c["day"]), int(c["timestamp"])
        if (d, ts) in burst_ts_set:
            continue
        for K in (5, 20):
            fts = min(ts + K * TICK, MAX_TS)
            m0 = mid_index.get((d, "VELVETFRUIT_EXTRACT", ts))
            m1 = mid_index.get((d, "VELVETFRUIT_EXTRACT", fts))
            if m0 is None or m1 is None:
                continue
            ext_fwd.append({"K": K, "burst": False, "fwd": m1 - m0})
    bf = pd.DataFrame(ext_fwd)
    burst_study = bf.groupby(["K", "burst"])["fwd"].agg(["mean", "median", "count"]).reset_index()
    burst_study.to_csv(OUT / "burst_vs_control_extract_fwd.csv", index=False)

    # --- 5) Adverse selection proxy: after aggr_buy/aggr_sell, same-symbol forward ---
    adv = []
    for ag in ["aggr_buy", "aggr_sell"]:
        sub = ev[(ev["aggressor"] == ag) & (ev["K"] == 20)]
        for (b, s), g in sub.groupby(["buyer", "seller"]):
            if len(g) < 10:
                continue
            adv.append(
                {
                    "aggressor": ag,
                    "buyer": b,
                    "seller": s,
                    "n": len(g),
                    "mean_fwd_same_k20": float(g["fwd_same"].mean()),
                    "median_fwd": float(g["fwd_same"].median()),
                }
            )
    pd.DataFrame(adv).sort_values("mean_fwd_same_k20").to_csv(
        OUT / "adverse_proxy_k20_by_pair.csv", index=False
    )

    # --- 1b) Every distinct name: trade-row coverage (spec: tag each U)
    tr_names = tr["buyer"].astype(str), tr["seller"].astype(str)
    all_u = sorted(set(tr_names[0].unique()) | set(tr_names[1].unique()))
    cov_rows = []
    for U in all_u:
        msk = (tr["buyer"] == U) | (tr["seller"] == U)
        ev_msk = (ev["buyer"] == U) | (ev["seller"] == U)
        cov_rows.append(
            {
                "participant": U,
                "n_trade_prints": int(msk.sum()),
                "n_event_horizon_rows": int(ev_msk.sum()),
            }
        )
    pd.DataFrame(cov_rows).sort_values("n_trade_prints", ascending=False).to_csv(
        OUT / "participant_name_coverage.csv", index=False
    )

    # --- 1c) Per-day mean fwd (same symbol) for top Phase-1 extract K=5 aggr-buy signals
    stab_specs: list[tuple[str, str, str, int]] = [
        ("Mark 67", "buyer", "aggr_buy", 5),
        ("Mark 22", "seller", "aggr_buy", 5),
        ("Mark 49", "seller", "aggr_buy", 5),
    ]
    stab_rows: list[dict] = []
    for U, role, ag, K in stab_specs:
        if role == "buyer":
            base = ev[(ev["buyer"] == U) & (ev["aggressor"] == ag) & (ev["symbol"] == "VELVETFRUIT_EXTRACT") & (ev["K"] == K)]
        else:
            base = ev[(ev["seller"] == U) & (ev["aggressor"] == ag) & (ev["symbol"] == "VELVETFRUIT_EXTRACT") & (ev["K"] == K)]
        for d in DAYS:
            g = base[base["day"] == d]
            n = int(len(g))
            if n == 0:
                continue
            v = g["fwd_same"].astype(float)
            stab_rows.append(
                {
                    "signal_key": f"{U}_{role}_{ag}_EXTRACT_K{K}",
                    "day": d,
                    "n": n,
                    "mean_fwd_same": float(v.mean()),
                    "median_fwd_same": float(v.median()),
                    "frac_pos": float((v > 0).mean()),
                }
            )
    pd.DataFrame(stab_rows).to_csv(OUT / "extract_aggrbuy_top3_per_day_stability.csv", index=False)

    # Narrative summary for humans
    summ = []
    summ.append("Also: participant_name_coverage.csv (all names), extract_aggrbuy_top3_per_day_stability.csv")
    summ.append("Round 4 Phase 1 — automated summary (see CSVs in same folder)")
    summ.append(f"Trade rows: {len(tr)} | Event horizons rows: {len(ev)}")
    summ.append("")
    summ.append("Burst vs control extract forward (mean):")
    summ.append(burst_study.to_string(index=False))
    summ.append("")
    summ.append("Top participant cells by |t| (n>=40, aggr_buy/sell, tight regime, K=20):")
    view = part_df[
        (part_df["n"] >= 40)
        & (part_df["K"] == 20)
        & (part_df["spread_regime"] == "tight")
        & (part_df["aggressor_filter"].isin(["aggr_buy", "aggr_sell"]))
    ].assign(abs_t=lambda d: d["t_stat"].abs())
    summ.append(view.sort_values("abs_t", ascending=False).head(25).to_string(index=False))
    (OUT / "phase1_automated_summary.txt").write_text("\n".join(summ), encoding="utf-8")

    print("Wrote outputs to", OUT)


if __name__ == "__main__":
    main()
