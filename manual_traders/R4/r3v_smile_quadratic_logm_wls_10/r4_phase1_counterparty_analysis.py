#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned forward mids (tape evidence).

Reads Prosperity4Data/ROUND_4 prices + trades days 1–3 (semicolon CSV).
Horizon K = K *next observation rows* per symbol after sorting by (day, timestamp)
(same bar index convention as round4 ping: ticks = tape rows for that product).

Writes under this folder:
  - r4_p1_forward_by_mark.csv
  - r4_p1_participant_by_day.csv     — per distinct name U×day×role×K (day-stability; min n per cell)
  - r4_p1_name_flow_balance.csv     — per-name trade counts, qty balance, notional
  - r4_p1_name_universe.txt         — how many unique buyer/seller strings (R4 d1-3: 7, all Mark *)
  - r4_p1_mark_product_cross.csv   — Mark×role×traded symbol×K×spread regime: same/extract/hydro fwd + bootstrap CI
  - r4_p1_mark_burst_same_sym.csv    — same-symbol fwd stratified burst vs isolated
  - r4_p1_2hop_motifs.csv            — 2-hop buyer→seller→seller2 counts (structural)
  - r4_p1_pair_baseline_residuals.csv
  - r4_p1_graph_edges.csv
  - r4_p1_burst_events.csv
  - r4_p1_phase1_summary.txt

Note: does not overwrite r4_phase1_gate.json (refined in-repo); re-run that block by hand or keep analysis.json
as the source of truth for the Phase 1 completion object.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]  # .../manual_traders/R4/<id>/file -> repo root
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent
DAYS = [1, 2, 3]
KS = (5, 20, 100)
PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]


def load_prices() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = df["day"].astype(int)
        df["timestamp"] = df["timestamp"].astype(int)
        df["symbol"] = df["product"].astype(str)
        bid = pd.to_numeric(df["bid_price_1"], errors="coerce")
        ask = pd.to_numeric(df["ask_price_1"], errors="coerce")
        mid = pd.to_numeric(df["mid_price"], errors="coerce")
        df = df.assign(bid1=bid, ask1=ask, mid=mid)
        df["spread"] = (ask - bid).where(bid.notna() & ask.notna())
        frames.append(df[["day", "timestamp", "symbol", "bid1", "ask1", "mid", "spread"]])
    return pd.concat(frames, ignore_index=True)


def load_trades() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = int(d)
        df["timestamp"] = df["timestamp"].astype(int)
        df["symbol"] = df["symbol"].astype(str)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["quantity"] = df["quantity"].astype(int)
        df["buyer"] = df["buyer"].astype(str)
        df["seller"] = df["seller"].astype(str)
        frames.append(df[["day", "timestamp", "buyer", "seller", "symbol", "price", "quantity"]])
    return pd.concat(frames, ignore_index=True)


def add_forward_mids(px: pd.DataFrame) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []
    for sym, g in px.groupby("symbol", sort=False):
        g = g.sort_values(["day", "timestamp"]).reset_index(drop=True)
        for k in KS:
            g[f"fwd_mid_{k}"] = g["mid"].shift(-k) - g["mid"]
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)


def classify_side_vec(p: pd.Series, bid: pd.Series, ask: pd.Series) -> pd.Series:
    out = pd.Series("unknown", index=p.index, dtype=object)
    ok = p.notna() & bid.notna() & ask.notna()
    out.loc[ok & (p >= ask)] = "buyer_aggressive"
    out.loc[ok & (p <= bid)] = "seller_aggressive"
    return out


def tstat_welch(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    n1, n2 = len(a), len(b)
    se = math.sqrt(v1 / n1 + v2 / n2) if v1 >= 0 and v2 >= 0 else float("nan")
    if se <= 0 or not math.isfinite(se):
        return float("nan")
    return (m1 - m2) / se


def hour_bucket(ts: int) -> int:
    # timestamps step in 100ms units per prosperity style; map coarsely
    return (ts // 100) % 24


def bootstrap_mean_ci(vals: np.ndarray, *, rng: np.random.Generator, n_boot: int = 800) -> tuple[float, float]:
    v = vals[np.isfinite(vals)]
    if len(v) < 10:
        return (float("nan"), float("nan"))
    if len(v) == 1:
        return (float(v[0]), float(v[0]))
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    px = load_prices()
    px = add_forward_mids(px)
    tr = load_trades()
    m = tr.merge(
        px,
        on=["day", "timestamp", "symbol"],
        how="left",
        suffixes=("", "_bbo"),
    )
    m["side"] = classify_side_vec(m["price"], m["bid1"], m["ask1"])
    for k in KS:
        m[f"fwd_mid_{k}"] = m[f"fwd_mid_{k}"].astype(float)

    # Cross-asset forwards at trade timestamp (extract + hydro)
    ex = px.loc[px["symbol"] == "VELVETFRUIT_EXTRACT", ["day", "timestamp"] + [f"fwd_mid_{k}" for k in KS]].rename(
        columns={f"fwd_mid_{k}": f"ex_fwd_{k}" for k in KS}
    )
    hy = px.loc[px["symbol"] == "HYDROGEL_PACK", ["day", "timestamp"] + [f"fwd_mid_{k}" for k in KS]].rename(
        columns={f"fwd_mid_{k}": f"hy_fwd_{k}" for k in KS}
    )
    m = m.merge(ex, on=["day", "timestamp"], how="left")
    m = m.merge(hy, on=["day", "timestamp"], how="left")

    all_names = sorted({str(x) for x in m["buyer"]} | {str(x) for x in m["seller"]})
    (OUT / "r4_p1_name_universe.txt").write_text(
        "Round 4 Phase 1 — distinct counterparty names (buyer ∪ seller) over tape days 1–3\n"
        f"count: {len(all_names)}\n"
        f"names: {all_names!r}\n"
        "Horizon K: next K price rows for that symbol after (day, timestamp), sorted by time (see script docstring).\n",
        encoding="utf-8",
    )

    # Notional buy/sell from each name's side (aggressor-agnostic: signed flow by initiator of trade)
    name_rows = []
    for u in all_names:
        bmask = m["buyer"] == u
        smask = m["seller"] == u
        nb, ns = int(bmask.sum()), int(smask.sum())
        if nb + ns == 0:
            continue
        q_b = m.loc[bmask, "quantity"].to_numpy()
        q_s = m.loc[smask, "quantity"].to_numpy()
        p_b = m.loc[bmask, "price"].to_numpy()
        p_s = m.loc[smask, "price"].to_numpy()
        notional_b = float(np.nansum(p_b * q_b)) if len(q_b) else 0.0
        notional_s = float(np.nansum(p_s * q_s)) if len(q_s) else 0.0
        name_rows.append(
            {
                "name": u,
                "n_as_buyer": nb,
                "n_as_seller": ns,
                "n_prints": nb + ns,
                "net_prints_buy_minus_sell": nb - ns,
                "signed_qty_imbalance": float(np.nansum(q_b) - np.nansum(q_s)),
                "notional_as_buyer": notional_b,
                "notional_as_seller": notional_s,
            }
        )
    pd.DataFrame(name_rows).sort_values("n_prints", ascending=False).to_csv(OUT / "r4_p1_name_flow_balance.csv", index=False)

    # --- 1) Participant-level: name U (every distinct string) as buyer/seller aggressor; pooled + per-day
    rows = []
    marks = all_names
    for u in marks:
        for role, mask in (
            ("buyer_agg", (m["buyer"] == u) & (m["side"] == "buyer_aggressive")),
            ("seller_agg", (m["seller"] == u) & (m["side"] == "seller_aggressive")),
            ("any_touch", (m["buyer"] == u) | (m["seller"] == u)),
        ):
            sub = m.loc[mask].copy()
            if len(sub) < 10:
                continue
            for k in KS:
                col = f"fwd_mid_{k}"
                vals = sub[col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                n = len(vals)
                if n < 10:
                    continue
                spr = sub["spread"].to_numpy(dtype=float)
                spr_q = np.nanquantile(spr, [0.33, 0.66]) if np.any(np.isfinite(spr)) else (np.nan, np.nan)
                for regime, rmask in (
                    ("all", np.ones(len(sub), dtype=bool)),
                    ("tight_spread", sub["spread"].notna() & (sub["spread"] <= spr_q[0])),
                    ("wide_spread", sub["spread"].notna() & (sub["spread"] >= spr_q[1])),
                ):
                    v = sub.loc[rmask, col].to_numpy(dtype=float)
                    v = v[np.isfinite(v)]
                    if len(v) < 8:
                        continue
                    rows.append(
                        {
                            "name": u,
                            "role": role,
                            "horizon_K": k,
                            "regime": regime,
                            "n": len(v),
                            "mean_fwd": float(np.mean(v)),
                            "median_fwd": float(np.median(v)),
                            "frac_pos": float(np.mean(v > 0)),
                            "t_vs_zero": float(np.mean(v) / (np.std(v, ddof=1) / math.sqrt(len(v)))) if len(v) > 1 and np.std(v, ddof=1) > 0 else float("nan"),
                        }
                    )
    df_mark = pd.DataFrame(rows)
    # Back-compat alias column for older references
    if len(df_mark):
        df_mark["mark"] = df_mark["name"]
    df_mark.to_csv(OUT / "r4_p1_forward_by_mark.csv", index=False)

    # Per-day same-symbol forward (stability: sign across days; min n)
    day_stab: list[dict] = []
    for u in all_names:
        for role, mask in (
            ("buyer_agg", (m["buyer"] == u) & (m["side"] == "buyer_aggressive")),
            ("seller_agg", (m["seller"] == u) & (m["side"] == "seller_aggressive")),
            ("any_touch", (m["buyer"] == u) | (m["seller"] == u)),
        ):
            for d in DAYS:
                sub = m.loc[mask & (m["day"] == d)]
                if len(sub) < 5:
                    continue
                for k in KS:
                    col = f"fwd_mid_{k}"
                    v = sub[col].to_numpy(dtype=float)
                    v = v[np.isfinite(v)]
                    if len(v) < 5:
                        continue
                    day_stab.append(
                        {
                            "name": u,
                            "day": d,
                            "role": role,
                            "horizon_K": k,
                            "n": len(v),
                            "mean_fwd": float(np.mean(v)),
                            "median_fwd": float(np.median(v)),
                            "frac_pos": float(np.mean(v > 0)),
                            "t_vs_zero": float(
                                np.mean(v) / (np.std(v, ddof=1) / math.sqrt(len(v)))
                            )
                            if len(v) > 1 and np.std(v, ddof=1) > 0
                            else float("nan"),
                        }
                    )
    pd.DataFrame(day_stab).to_csv(OUT / "r4_p1_participant_by_day.csv", index=False)

    # --- 1b) Burst vs isolated (same-symbol fwd) for Mark flows
    ts_n = m.groupby(["day", "timestamp"])["symbol"].transform("count")
    m["burst_multi"] = ts_n >= 4

    rng = np.random.default_rng(0)
    burst_rows: list[dict] = []
    marks = sorted(set(m["buyer"]) | set(m["seller"]))
    marks = [x for x in marks if str(x).startswith("Mark ")]
    for u in marks:
        for role, mask in (
            ("buyer_agg", (m["buyer"] == u) & (m["side"] == "buyer_aggressive")),
            ("seller_agg", (m["seller"] == u) & (m["side"] == "seller_aggressive")),
        ):
            sub = m.loc[mask]
            if len(sub) < 15:
                continue
            for burst_lab, bmask in (
                ("multi_ts_burst", sub["burst_multi"]),
                ("isolated", ~sub["burst_multi"]),
            ):
                bb = sub.loc[bmask]
                for k in KS:
                    col = f"fwd_mid_{k}"
                    vals = bb[col].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) < 8:
                        continue
                    lo, hi = bootstrap_mean_ci(vals, rng=rng)
                    burst_rows.append(
                        {
                            "mark": u,
                            "role": role,
                            "burst_stratum": burst_lab,
                            "horizon_K": k,
                            "n": len(vals),
                            "mean_fwd_same": float(np.mean(vals)),
                            "t_vs_zero": float(np.mean(vals) / (np.std(vals, ddof=1) / math.sqrt(len(vals))))
                            if len(vals) > 1 and np.std(vals, ddof=1) > 0
                            else float("nan"),
                            "ci95_low": lo,
                            "ci95_high": hi,
                        }
                    )
    pd.DataFrame(burst_rows).to_csv(OUT / "r4_p1_mark_burst_same_sym.csv", index=False)

    # --- 1c) Mark×role×traded product×K×spread regime: same / extract / hydro forwards
    cross_rows: list[dict] = []
    for u in marks:
        for role, mask in (
            ("buyer_agg", (m["buyer"] == u) & (m["side"] == "buyer_aggressive")),
            ("seller_agg", (m["seller"] == u) & (m["side"] == "seller_aggressive")),
        ):
            sub = m.loc[mask]
            if len(sub) < 12:
                continue
            for traded in PRODUCTS:
                sp = sub.loc[sub["symbol"] == traded]
                if len(sp) < 10:
                    continue
                spr = sp["spread"].to_numpy(dtype=float)
                spr_q = np.nanquantile(spr, [0.33, 0.66]) if np.any(np.isfinite(spr)) else (np.nan, np.nan)
                for regime, rmask in (
                    ("all", np.ones(len(sp), dtype=bool)),
                    ("tight_spread", sp["spread"].notna() & (sp["spread"] <= spr_q[0])),
                    ("wide_spread", sp["spread"].notna() & (sp["spread"] >= spr_q[1])),
                ):
                    bb = sp.loc[rmask]
                    if len(bb) < 8:
                        continue
                    for k in KS:
                        same_c = f"fwd_mid_{k}"
                        ex_c = f"ex_fwd_{k}"
                        hy_c = f"hy_fwd_{k}"
                        for tgt, col in (("same_symbol", same_c), ("VELVETFRUIT_EXTRACT", ex_c), ("HYDROGEL_PACK", hy_c)):
                            vals = bb[col].to_numpy(dtype=float)
                            vals = vals[np.isfinite(vals)]
                            if len(vals) < 8:
                                continue
                            lo, hi = bootstrap_mean_ci(vals, rng=rng)
                            mu = float(np.mean(vals))
                            t0 = (
                                float(mu / (np.std(vals, ddof=1) / math.sqrt(len(vals))))
                                if len(vals) > 1 and np.std(vals, ddof=1) > 0
                                else float("nan")
                            )
                            cross_rows.append(
                                {
                                    "mark": u,
                                    "role": role,
                                    "traded_symbol": traded,
                                    "spread_regime": regime,
                                    "horizon_K": k,
                                    "fwd_target": tgt,
                                    "n": len(vals),
                                    "mean_fwd": mu,
                                    "median_fwd": float(np.median(vals)),
                                    "frac_pos": float(np.mean(vals > 0)),
                                    "t_vs_zero": t0,
                                    "ci95_low": lo,
                                    "ci95_high": hi,
                                }
                            )
    pd.DataFrame(cross_rows).to_csv(OUT / "r4_p1_mark_product_cross.csv", index=False)

    # --- 2) Pair baseline: cell mean fwd20 by (buyer, seller, symbol), residual
    cell = m.groupby(["buyer", "seller", "symbol"], as_index=False).agg(
        cell_mean_fwd20=("fwd_mid_20", "mean"),
        cell_n=("fwd_mid_20", "count"),
    )
    glob = float(m["fwd_mid_20"].mean())
    m2 = m.merge(cell, on=["buyer", "seller", "symbol"], how="left")
    m2["residual_fwd20"] = m2["fwd_mid_20"] - m2["cell_mean_fwd20"].fillna(glob)
    m2[["day", "timestamp", "buyer", "seller", "symbol", "side", "fwd_mid_20", "cell_mean_fwd20", "residual_fwd20"]].to_csv(
        OUT / "r4_p1_pair_baseline_residuals.csv", index=False
    )

    # --- 3) Graph edges
    m["notional"] = m["price"] * m["quantity"]
    eg = m.groupby(["buyer", "seller"], as_index=False).agg(n=("symbol", "count"), notional=("notional", "sum"))
    eg = eg.sort_values("n", ascending=False)
    eg.to_csv(OUT / "r4_p1_graph_edges.csv", index=False)

    # --- 3b) 2-hop motifs A→B→C (chain counts where hop1 seller == hop2 buyer)
    hop_rows: list[dict] = []
    for _, r1 in eg.head(20).iterrows():
        b1, s1 = str(r1["buyer"]), str(r1["seller"])
        sub2 = eg[eg["buyer"] == s1]
        for _, r2 in sub2.head(25).iterrows():
            s2 = str(r2["seller"])
            hop_rows.append(
                {
                    "hop1_buyer": b1,
                    "hop1_seller": s1,
                    "hop2_seller": s2,
                    "n_hop1": int(r1["n"]),
                    "n_hop2": int(r2["n"]),
                    "min_n": int(min(r1["n"], r2["n"])),
                }
            )
    pd.DataFrame(hop_rows).to_csv(OUT / "r4_p1_2hop_motifs.csv", index=False)

    # --- 4) Bursts
    def _symset(s: pd.Series) -> str:
        return ",".join(sorted({str(x) for x in s}))

    burst = (
        m.groupby(["day", "timestamp"], as_index=False)
        .agg(n_trades=("symbol", "count"), symbols=("symbol", _symset))
        .sort_values("n_trades", ascending=False)
    )
    burst_big = burst[burst["n_trades"] >= 4].copy()
    burst_big.to_csv(OUT / "r4_p1_burst_events.csv", index=False)

    # Burst forward extract: merge burst timestamps to extract fwd
    ext = px.loc[px["symbol"] == "VELVETFRUIT_EXTRACT", ["day", "timestamp", "fwd_mid_20"]].rename(columns={"fwd_mid_20": "ext_fwd20"})
    burst_big = burst_big.merge(ext, on=["day", "timestamp"], how="left")
    ctrl = burst.sample(min(500, len(burst)), random_state=0)
    ctrl = ctrl.merge(ext, on=["day", "timestamp"], how="left")
    burst_mean = float(burst_big["ext_fwd20"].dropna().mean()) if len(burst_big) else float("nan")
    ctrl_mean = float(ctrl["ext_fwd20"].dropna().mean()) if len(ctrl) else float("nan")

    # --- 5) Summary text + top edges scan
    lines: list[str] = []
    lines.append("Round 4 Phase 1 — automated summary (days 1–3)")
    lines.append(f"Trade rows: {len(m):,} | price BBO matches: {m['mid'].notna().sum():,}")
    lines.append(f"Burst rows (>=4 trades same ts): {len(burst_big)}")
    lines.append(f"Mean extract fwd20 after big bursts: {burst_mean:.4f} | random control: {ctrl_mean:.4f}")
    lines.append("")
    lines.append("Top directed pairs (count):")
    for _, r in eg.head(8).iterrows():
        lines.append(f"  {r['buyer']} → {r['seller']}: n={int(r['n'])} notional={r['notional']:.0f}")
    lines.append("")
    lines.append("Strongest Mark×role×K×all (|t|>2, n>=30) on same-symbol fwd:")
    if len(df_mark):
        sub = df_mark[(df_mark["regime"] == "all") & (df_mark["n"] >= 30)].copy()
        sub["abs_t"] = sub["t_vs_zero"].abs()
        sub = sub.sort_values("abs_t", ascending=False).head(15)
        for _, r in sub.iterrows():
            lines.append(
                f"  {r['mark']} {r['role']} K={int(r['horizon_K'])} mean={r['mean_fwd']:.4f} n={int(r['n'])} t={r['t_vs_zero']:.2f} frac+={r['frac_pos']:.2f}"
            )
    (OUT / "r4_p1_phase1_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # r4_phase1_gate.json is maintained manually to match analysis.json (round4_phase1_complete) — not overwritten here.


if __name__ == "__main__":
    main()
