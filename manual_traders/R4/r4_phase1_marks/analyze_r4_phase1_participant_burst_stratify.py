#!/usr/bin/env python3
"""
Phase 1 — **bullet 1 supplement:** participant markouts **× burst bucket** at print time.

**Burst bucket** (same ``day``, ``timestamp`` as the print):
  - ``burst``: **n_prints >= 4** (same as ``burst_ge4``)
  - ``isolated``: **n_prints == 1**
  - ``small_multi``: **2 <= n_prints <= 3**

**Horizons K** ∈ {5, 20, 100}: ``fwd_same_K``, ``fwd_EXTRACT_K``, ``fwd_HYDRO_K`` (same
definition as ``analyze_phase1.py`` — unique timestamp index steps per day).

**Aggressor side:** ``aggr_buy`` (buyer lifts ask) / ``aggr_sell`` (seller hits bid);
``at_mid`` rows excluded (same convention as ``participant_tables``).

**Welch:** For each (mark, side, symbol, K) where **burst** and **isolated** both have
``n >= 12``, test mean(fwd_same_K) burst minus isolated (and same for fwd_EXTRACT_K).

Outputs (``outputs/``):
  - ``phase17_participant_burst_bucket_cells.csv``
  - ``phase17_burst_vs_isolated_welch.csv``
  - ``phase17_small_multi_vs_isolated_welch.csv`` (often more power than burst vs isolated)
  - ``phase17_participant_burst_stratify_summary.txt``

Run:
  python3 manual_traders/R4/r4_phase1_marks/analyze_r4_phase1_participant_burst_stratify.py
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
DAYS = [1, 2, 3]
KS = [5, 20, 100]
BUCKETS = ("burst", "isolated", "small_multi")
MIN_CELL = 10
MIN_WELCH = 8


def t_stat_welch(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    if va == 0 and vb == 0:
        return float("nan")
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / se)


def load_p1():
    spec = importlib.util.spec_from_file_location("p1", HERE / "analyze_phase1.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def attach_burst_bucket(te: pd.DataFrame, tr_raw: pd.DataFrame) -> pd.DataFrame:
    cnt = tr_raw.groupby(["day", "timestamp"]).size().reset_index(name="n_prints_ts")
    m = te.merge(cnt, on=["day", "timestamp"], how="left")
    n = m["n_prints_ts"].fillna(1).astype(int)
    m["burst_bucket"] = np.where(
        n >= 4,
        "burst",
        np.where(n == 1, "isolated", np.where((n >= 2) & (n <= 3), "small_multi", "other")),
    )
    return m


def participant_burst_cells(m: pd.DataFrame) -> pd.DataFrame:
    m = m[m["side"].isin(("aggr_buy", "aggr_sell"))].copy()
    rows = []
    marks = sorted(set(m["buyer"].astype(str)) | set(m["seller"].astype(str)))
    for U in marks:
        buy_u = m[(m["side"] == "aggr_buy") & (m["buyer"] == U)]
        sell_u = m[(m["side"] == "aggr_sell") & (m["seller"] == U)]
        for side, sub0 in (("aggr_buy", buy_u), ("aggr_sell", sell_u)):
            for sym in sub0["symbol"].astype(str).unique():
                for bucket in BUCKETS:
                    sub = sub0[(sub0["symbol"] == sym) & (sub0["burst_bucket"] == bucket)]
                    if len(sub) < MIN_CELL:
                        continue
                    for K in KS:
                        fs = f"fwd_same_{K}"
                        fe = f"fwd_EXTRACT_{K}"
                        fh = f"fwd_HYDRO_{K}"
                        xs = sub[fs].astype(float).values
                        xs = xs[np.isfinite(xs)]
                        xe = sub[fe].astype(float).values
                        xe = xe[np.isfinite(xe)]
                        xh = sub[fh].astype(float).values
                        xh = xh[np.isfinite(xh)]
                        if len(xs) < MIN_CELL:
                            continue
                        days = sorted(sub["day"].unique().tolist())
                        rows.append(
                            {
                                "mark": U,
                                "side": side,
                                "symbol": sym,
                                "burst_bucket": bucket,
                                "K": K,
                                "n": len(xs),
                                "mean_fwd_same": float(np.mean(xs)),
                                "median_fwd_same": float(np.median(xs)),
                                "frac_pos_same": float(np.mean(xs > 0)),
                                "mean_fwd_extract": float(np.mean(xe)) if len(xe) else float("nan"),
                                "mean_fwd_hydro": float(np.mean(xh)) if len(xh) else float("nan"),
                                "days_present": ",".join(str(d) for d in days),
                                "n_days": len(days),
                            }
                        )
    return pd.DataFrame(rows)


def welch_two_buckets(
    m: pd.DataFrame, bucket_a: str, bucket_b: str, label: str
) -> pd.DataFrame:
    m = m[m["side"].isin(("aggr_buy", "aggr_sell"))].copy()
    rows = []
    marks = sorted(set(m["buyer"].astype(str)) | set(m["seller"].astype(str)))
    for U in marks:
        buy_u = m[(m["side"] == "aggr_buy") & (m["buyer"] == U)]
        sell_u = m[(m["side"] == "aggr_sell") & (m["seller"] == U)]
        for side, sub0 in (("aggr_buy", buy_u), ("aggr_sell", sell_u)):
            for sym in sub0["symbol"].astype(str).unique():
                b = sub0[(sub0["symbol"] == sym) & (sub0["burst_bucket"] == bucket_a)]
                i = sub0[(sub0["symbol"] == sym) & (sub0["burst_bucket"] == bucket_b)]
                if len(b) < MIN_WELCH or len(i) < MIN_WELCH:
                    continue
                for K in KS:
                    fs = f"fwd_same_{K}"
                    fe = f"fwd_EXTRACT_{K}"
                    xb, xi = b[fs].astype(float).values, i[fs].astype(float).values
                    xb, xi = xb[np.isfinite(xb)], xi[np.isfinite(xi)]
                    if len(xb) < MIN_WELCH or len(xi) < MIN_WELCH:
                        continue
                    yb, yi = b[fe].astype(float).values, i[fe].astype(float).values
                    yb, yi = yb[np.isfinite(yb)], yi[np.isfinite(yi)]
                    rows.append(
                        {
                            "comparison": label,
                            "mark": U,
                            "side": side,
                            "symbol": sym,
                            "K": K,
                            "n_a": len(xb),
                            "n_b": len(xi),
                            "mean_same_a": float(np.mean(xb)),
                            "mean_same_b": float(np.mean(xi)),
                            "welch_same_a_minus_b": t_stat_welch(xb, xi),
                            "mean_ext_a": float(np.mean(yb)) if len(yb) else float("nan"),
                            "mean_ext_b": float(np.mean(yi)) if len(yi) else float("nan"),
                            "welch_ext_a_minus_b": t_stat_welch(yb, yi)
                            if len(yb) >= MIN_WELCH and len(yi) >= MIN_WELCH
                            else float("nan"),
                        }
                    )
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("welch_same_a_minus_b", key=np.abs, ascending=False)
    return df


def write_summary(
    cells: pd.DataFrame, welch_bi: pd.DataFrame, welch_smi: pd.DataFrame
) -> None:
    lines = [
        "Participant × burst_bucket × K (aggressor-only). MIN_CELL for cells:",
        str(MIN_CELL),
        "\n\nLargest |Welch_same| **burst vs isolated** (both n>=",
        str(MIN_WELCH),
        "):\n",
    ]
    if len(welch_bi):
        for _, r in welch_bi.head(20).iterrows():
            lines.append(
                f"  {r['mark']} {r['side']} {r['symbol']} K={int(r['K'])}  "
                f"same: a_mean={r['mean_same_a']:.4g} b_mean={r['mean_same_b']:.4g} "
                f"Welch={r['welch_same_a_minus_b']:.3f} (n {int(r['n_a'])}/{int(r['n_b'])})  "
                f"ext_Welch={r['welch_ext_a_minus_b']:.3f}\n"
            )
    lines.append("\nCells with |Welch_same|>=2 (burst vs isolated, same-asset):\n")
    if len(welch_bi):
        sig = welch_bi[np.abs(welch_bi["welch_same_a_minus_b"]) >= 2.0]
        if len(sig) == 0:
            lines.append("  (none)\n")
        else:
            for _, r in sig.iterrows():
                lines.append(
                    f"  {r['mark']} {r['side']} {r['symbol']} K={int(r['K'])} Welch={r['welch_same_a_minus_b']:.3f}\n"
                )
    lines.append("\nTop |Welch_same| **small_multi vs isolated** (extract + hydro focus):\n")
    if len(welch_smi):
        subw = welch_smi[
            welch_smi["symbol"].isin(("VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"))
        ].head(15)
        for _, r in subw.iterrows():
            lines.append(
                f"  {r['mark']} {r['side']} {r['symbol']} K={int(r['K'])} "
                f"Welch_same={r['welch_same_a_minus_b']:.3f} n={int(r['n_a'])}/{int(r['n_b'])}\n"
            )
    lines.append("\nExtract-focused: top mean_fwd_extract in **burst** bucket (extract symbol, K=20, n>=30):\n")
    ext = cells[(cells["symbol"] == "VELVETFRUIT_EXTRACT") & (cells["K"] == 20) & (cells["burst_bucket"] == "burst")]
    ext = ext[ext["n"] >= 30].sort_values("mean_fwd_extract", ascending=False).head(12)
    for _, r in ext.iterrows():
        lines.append(
            f"  {r['mark']} {r['side']} n={int(r['n'])} mean_ext={r['mean_fwd_extract']:.4g} mean_same={r['mean_fwd_same']:.4g} days={r['days_present']}\n"
        )
    (OUT / "phase17_participant_burst_stratify_summary.txt").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    p1 = load_p1()
    tr_raw = pd.concat(
        [
            pd.read_csv(p1.DATA / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d)
            for d in DAYS
        ],
        ignore_index=True,
    )
    te = p1.build_trade_enriched()
    m = attach_burst_bucket(te, tr_raw)
    cells = participant_burst_cells(m)
    cells.to_csv(OUT / "phase17_participant_burst_bucket_cells.csv", index=False)
    w_bi = welch_two_buckets(m, "burst", "isolated", "burst_vs_isolated")
    w_bi.to_csv(OUT / "phase17_burst_vs_isolated_welch.csv", index=False)
    w_smi = welch_two_buckets(m, "small_multi", "isolated", "small_multi_vs_isolated")
    w_smi.to_csv(OUT / "phase17_small_multi_vs_isolated_welch.csv", index=False)
    write_summary(cells, w_bi, w_smi)
    print(
        "Wrote phase17_* n_cells=",
        len(cells),
        "n_welch_bi=",
        len(w_bi),
        "n_welch_smi=",
        len(w_smi),
    )


if __name__ == "__main__":
    main()
