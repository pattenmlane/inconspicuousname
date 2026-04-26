#!/usr/bin/env python3
"""
Round 4 Phase 1 **exhaustion** (ping spec) — *all* counterparty names + lead–lag sketch.

1) The Round 4 trade tape here has only **7** distinct `buyer`∪`seller` strings. For **each** name
   `U`, every row with `buyer==U` or `seller==U` is tagged with aggressor side role (same logic as
   `phase1_r4_counterparty_analysis.py`).

2) For each (name, role, symbol) cell with n ≥ `MIN_N_CELL`, report mean, **median**, t-stat, frac_pos,
   bootstrap 95% CI for **same-symbol** fwd and **VELVETFRUIT_EXTRACT** / **HYDROGEL_PACK** fwd at
   K ∈ {5, 20, 100}. Horizon = K steps on the (day, symbol) price tape.

3) **Lead–lag (simple):** per `day`, align extract **fwd_u_20** at every U timestamp with indicator
   `any_trade M01→M22` at the **same** timestamp, and with **1-step lag** in the ordered timestamp
   series (next tick). Report Pearson r (nan if var 0).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
INP = Path(__file__).resolve().parent / "analysis_outputs" / "phase1" / "r4_trades_enriched.csv"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

KS = (5, 20, 100)
MIN_N_CELL = 5
BOOT = 2000
RNG = np.random.default_rng(0)


def aggressor_from_row(r: pd.Series) -> str:
    return str(r.get("aggressor", "unk"))


def role_for_name(r: pd.Series, name: str) -> str:
    b, s = str(r["buyer"]), str(r["seller"])
    ag = aggressor_from_row(r)
    if b == name and ag == "buy_agg":
        return "U_buy_agg"
    if s == name and ag == "sell_agg":
        return "U_sell_agg"
    if b == name:
        return "U_passive_buy"
    if s == name:
        return "U_passive_sell"
    return "unk"


def bootstrap_ci(x: np.ndarray) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(x))
    if len(x) < 4:
        return m, m, m
    idx = RNG.integers(0, len(x), size=(BOOT, len(x)))
    mus = x[idx].mean(axis=1)
    return m, float(np.percentile(mus, 2.5)), float(np.percentile(mus, 97.5))


def t_stat_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    se = s / math.sqrt(len(x)) if s > 0 else float("nan")
    return m / se if se and math.isfinite(se) and se > 0 else float("nan")


def build_u_panel() -> dict[int, pd.DataFrame]:
    """day -> long df with columns timestamp, m01m22, fwd_u_20 from prices."""
    from collections import defaultdict

    # load U prices and compute forward for each day
    out: dict[int, pd.DataFrame] = {}
    for day in (1, 2, 3):
        p = DATA / f"prices_round_4_day_{day}.csv"
        if not p.is_file():
            continue
        pr = pd.read_csv(p, sep=";")
        u = pr[pr["product"] == "VELVETFRUIT_EXTRACT"].sort_values("timestamp")
        u = u.drop_duplicates("timestamp", keep="first")
        ts = u["timestamp"].to_numpy(int)
        bid = pd.to_numeric(u["bid_price_1"], errors="coerce")
        ask = pd.to_numeric(u["ask_price_1"], errors="coerce")
        mid = (bid + ask) / 2.0
        midv = mid.to_numpy(float)
        k = 20
        fwd = np.full(len(midv), np.nan)
        for i in range(len(midv) - k):
            fwd[i] = midv[i + k] - midv[i]
        m01m22 = np.zeros(len(ts), dtype=int)
        tp = DATA / f"trades_round_4_day_{day}.csv"
        if tp.is_file():
            tr = pd.read_csv(tp, sep=";")
            hit = (tr["buyer"] == "Mark 01") & (tr["seller"] == "Mark 22")
            hit_ts = set(tr.loc[hit, "timestamp"].astype(int).unique())
            for j, t in enumerate(ts):
                if int(t) in hit_ts:
                    m01m22[j] = 1
        out[day] = pd.DataFrame(
            {"timestamp": ts, "m01m22": m01m22, "fwd_u_20": fwd, "mid": midv}
        )
    return out


def leadlag_text(panels: dict[int, pd.DataFrame]) -> str:
    lines = [
        "Pearson r between (M01→M22 trade at same timestamp) and extract fwd_u_20 at that tick.",
        "Also r with m01m22 **lagged one tick forward** (next row in U tape) vs fwd_u_20 (same t).",
        "",
    ]
    for day, df in sorted(panels.items()):
        m = np.asarray(df["m01m22"], float)
        y = np.asarray(df["fwd_u_20"], float)
        ok = np.isfinite(y)
        m, y = m[ok], y[ok]
        if len(y) < 10:
            continue
        r0 = np.corrcoef(m, y)[0, 1] if m.std() > 0 and y.std() > 0 else float("nan")
        # lag: use previous tick's m01m22 to predict y at t (i.e. m shifted forward so m[t-1] aligns with y[t])
        m1 = m[:-1]
        y1 = y[1:]
        rlag = (
            np.corrcoef(m1, y1)[0, 1] if len(m1) > 2 and m1.std() > 0 and y1.std() > 0 else float("nan")
        )
        lines.append(
            f"day {day}: n={len(y)} r(same_ts, m01m22 vs fwd_u_20)={r0:.4f}  r(m01m22[t-1] vs fwd_u_20[t])={rlag:.4f}  m01m22 rate={m.mean():.4f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ev = pd.read_csv(INP)
    ev["buyer"] = ev["buyer"].astype(str)
    ev["seller"] = ev["seller"].astype(str)
    names = sorted(set(ev["buyer"]) | set(ev["seller"]))

    rows: list[dict] = []
    for name in names:
        if name in ("nan", "None"):
            continue
        sub = ev[(ev["buyer"] == name) | (ev["seller"] == name)].copy()
        if sub.empty:
            continue
        sub["side_role"] = [role_for_name(sub.iloc[i], name) for i in range(len(sub))]
        for role, g in sub.groupby("side_role"):
            if role == "unk":
                continue
            for sym in g["symbol"].unique():
                gg = g[g["symbol"] == str(sym)]
                n = len(gg)
                if n < MIN_N_CELL:
                    continue
                for K in KS:
                    for col_fwd, target in [
                        (f"fwd_mid_{K}", f"sym@{K}"),
                        (f"fwd_u_{K}", f"u@{K}"),
                        (f"fwd_h_{K}", f"h@{K}"),
                    ]:
                        if col_fwd not in gg.columns:
                            continue
                        x = pd.to_numeric(gg[col_fwd], errors="coerce").dropna().to_numpy()
                        if len(x) < MIN_N_CELL:
                            continue
                        m, lo, hi = bootstrap_ci(x)
                        rows.append(
                            {
                                "name": name,
                                "side_role": role,
                                "symbol": sym,
                                "K": K,
                                "fwd_field": col_fwd,
                                "target": target,
                                "n": len(x),
                                "mean": float(np.mean(x)),
                                "median": float(np.median(x)),
                                "t_stat": t_stat_mean(x),
                                "frac_pos": float((x > 0).mean()),
                                "ci95_lo": lo,
                                "ci95_hi": hi,
                            }
                        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["name", "symbol", "side_role", "K", "target"])
    df.to_csv(OUT / "r4_phase1c_exhaustive_per_name.csv", index=False)
    coverage = f"Distinct names: {len(names)} = {names}\nRows in exhaustive table (sub-rows per K×target): {len(df)}\n"
    (OUT / "r4_phase1c_name_coverage.txt").write_text(
        coverage + f"All names appear in r4_trades_enriched: Phase 1 'each U' is fully enumerated for this dataset.\n"
    )

    panels = build_u_panel()
    txt = leadlag_text(panels)
    (OUT / "r4_phase1c_m01_m22_leadlag_extract_k20.txt").write_text(
        "Horizon: fwd_u_20 = mid(U at t+20 ticks) - mid(U at t) on ordered U price tape per day.\n\n"
        + txt
    )
    print("Wrote", OUT, "exhaustive rows", len(df))


if __name__ == "__main__":
    main()
