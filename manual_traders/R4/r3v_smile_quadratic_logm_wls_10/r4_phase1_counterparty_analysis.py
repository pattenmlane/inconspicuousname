#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned forward mids (tape evidence).

Reads Prosperity4Data/ROUND_4 prices + trades days 1–3 (semicolon CSV).
Horizon K = K *next observation rows* per symbol after sorting by (day, timestamp)
(same bar index convention as round4 ping: ticks = tape rows for that product).

Writes under this folder:
  - r4_p1_forward_by_mark.csv
  - r4_p1_pair_baseline_residuals.csv
  - r4_p1_graph_edges.csv
  - r4_p1_burst_events.csv
  - r4_p1_phase1_summary.txt
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]  # .../manual_traders/R4/<id>/file -> repo root
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent
REL_PREFIX = "manual_traders/R4/r3v_smile_quadratic_logm_wls_10"


def _rel(p: Path) -> str:
    return f"{REL_PREFIX}/{p.name}"
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

    # --- 1) Participant-level: Mark U as buyer aggressor vs seller aggressor
    rows = []
    marks = sorted(set(m["buyer"]) | set(m["seller"]))
    marks = [x for x in marks if x.startswith("Mark ")]
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
                            "mark": u,
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
    df_mark.to_csv(OUT / "r4_p1_forward_by_mark.csv", index=False)

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

    # Phase 1 gate JSON fragment
    gate = {
        "round4_phase1_complete": {
            "phase": 1,
            "tape_days": DAYS,
            "outputs": {
                "participant_forward": _rel(OUT / "r4_p1_forward_by_mark.csv"),
                "pair_residuals": _rel(OUT / "r4_p1_pair_baseline_residuals.csv"),
                "graph_edges": _rel(OUT / "r4_p1_graph_edges.csv"),
                "bursts": _rel(OUT / "r4_p1_burst_events.csv"),
                "summary_txt": _rel(OUT / "r4_p1_phase1_summary.txt"),
                "script": f"{REL_PREFIX}/r4_phase1_counterparty_analysis.py",
            },
            "bullets": [
                {"bullet": "1 participant-level", "conclusion": "See r4_p1_forward_by_mark.csv; Mark-conditioned fwd mids vary by role/K; many cells low-n.", "paths": ["r4_p1_forward_by_mark.csv"]},
                {"bullet": "2 bot baseline / residuals", "conclusion": "Per (buyer,seller,symbol) mean fwd20 and trade-level residuals in r4_p1_pair_baseline_residuals.csv.", "paths": ["r4_p1_pair_baseline_residuals.csv"]},
                {"bullet": "3 graph motifs", "conclusion": "Mark 01→Mark 22 dominates edge list (r4_p1_graph_edges.csv).", "paths": ["r4_p1_graph_edges.csv"]},
                {"bullet": "4 bursts", "conclusion": "Multi-symbol same-ts bursts listed; vs random-time extract fwd20 compare in summary.", "paths": ["r4_p1_burst_events.csv", "r4_p1_phase1_summary.txt"]},
                {"bullet": "5 adverse selection", "conclusion": "Proxy: aggressive-side fwd on same symbol; detailed in forward_by_mark for buyer_agg vs seller_agg.", "paths": ["r4_p1_forward_by_mark.csv"]},
            ],
            "top_5_tradeable_edges": [
                {
                    "rank": 1,
                    "hypothesis": "Document Mark 01→Mark 22 basket prints as structural (high n); exploit as regime / avoid fading without sim",
                    "effect": "Pair frequency >> others",
                    "n": int(eg.iloc[0]["n"]) if len(eg) else 0,
                    "days": DAYS,
                },
                {
                    "rank": 2,
                    "hypothesis": "Conditional forward mids after bursts vs control (extract) — see summary means",
                    "effect": f"burst_ext_fwd20_mean={burst_mean:.4f} vs_ctrl={ctrl_mean:.4f}",
                    "n": int(len(burst_big)),
                    "days": DAYS,
                },
                {
                    "rank": 3,
                    "hypothesis": "Per-Mark aggressive buy/sell fwd tables for horizon selection (K=5/20/100)",
                    "effect": "See CSV; pick cells with n>=30 and stable sign across days (manual follow-up)",
                    "n": int(len(df_mark)),
                    "days": DAYS,
                },
                {
                    "rank": 4,
                    "hypothesis": "Residual outliers after (buyer,seller,symbol) mean — second-order fade/lean candidates",
                    "effect": "Distribution in residuals CSV",
                    "n": int(len(m2)),
                    "days": DAYS,
                },
                {
                    "rank": 5,
                    "hypothesis": "Spread-regime split (tight vs wide quantiles) on Mark flows",
                    "effect": "See forward_by_mark regime column",
                    "n": int(len(df_mark)),
                    "days": DAYS,
                },
            ],
            "negative_results": [
                "No single Mark×horizon cell yet meets institutional 'edge' without out-of-sample sim; tape is 3 days only.",
                "Trade count O(1e3) vs price rows O(1e5): many BBO timestamps have no prints — forward joins are sparse for some symbols.",
            ],
        }
    }
    (OUT / "r4_phase1_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
