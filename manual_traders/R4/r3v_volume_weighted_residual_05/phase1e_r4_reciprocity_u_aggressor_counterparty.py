#!/usr/bin/env python3
"""
Round 4 Phase 1e — **graph reciprocity / hubs** + **U aggressor × passive counterparty** markouts.

1) Undirected pair table: for each unordered {A,B}, counts and notionals for A→B and B→A,
   reciprocity ratio min/max(count), hub summary.

2) **U buy_agg:** passive seller = `seller`. Per **passive_counterparty** pooled and **per day**,
   fwd_u @ K∈{5,20} with mean/median/n/t/bootstrap CI (n≥5 per cell).

3) **U sell_agg:** passive buyer = `buyer`, same stats (n≥15 for pooled summary table only).
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

INP = Path(__file__).resolve().parent / "analysis_outputs" / "phase1" / "r4_trades_enriched.csv"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

U = "VELVETFRUIT_EXTRACT"
RNG = np.random.default_rng(11)
BOOT = 2000


def bootstrap_ci(x: np.ndarray) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(x))
    idx = RNG.integers(0, len(x), size=(BOOT, len(x)))
    mus = x[idx].mean(axis=1)
    return m, float(np.percentile(mus, 2.5)), float(np.percentile(mus, 97.5))


def t_stat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    se = s / math.sqrt(len(x)) if s > 0 else float("nan")
    return m / se if se and math.isfinite(se) else float("nan")


def main() -> None:
    ev = pd.read_csv(INP)
    ev["buyer"] = ev["buyer"].astype(str)
    ev["seller"] = ev["seller"].astype(str)

    cnt: dict[tuple[str, str], int] = defaultdict(int)
    notional: dict[tuple[str, str], float] = defaultdict(float)
    for _, r in ev.iterrows():
        b, s = r["buyer"], r["seller"]
        cnt[(b, s)] += 1
        notional[(b, s)] += float(r.get("notional", 0) or 0)

    # undirected pair aggregation
    pair_dat: dict[tuple[str, str], list[int, int, float, float]] = {}
    for (a, b), c in cnt.items():
        if a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        if (lo, hi) not in pair_dat:
            pair_dat[(lo, hi)] = [0, 0, 0.0, 0.0]
        arr = pair_dat[(lo, hi)]
        if a == lo and b == hi:
            arr[0] += c
            arr[2] += notional[(a, b)]
        elif a == hi and b == lo:
            arr[1] += c
            arr[3] += notional[(a, b)]

    rec_rows = []
    for (lo, hi), (c_lo_hi, c_hi_lo, n_lo_hi, n_hi_lo) in pair_dat.items():
        tot = c_lo_hi + c_hi_lo
        rmin, rmax = (c_hi_lo / c_lo_hi) if c_lo_hi else float("inf"), (c_lo_hi / c_hi_lo) if c_hi_lo else float("inf")
        rec_rows.append(
            {
                "A": lo,
                "B": hi,
                "count_lo_to_hi": c_lo_hi,
                "count_hi_to_lo": c_hi_lo,
                "notional_lo_to_hi": n_lo_hi,
                "notional_hi_to_lo": n_hi_lo,
                "undirected_count": tot,
                "reciprocity_ratio_hi_over_lo": (c_hi_lo / c_lo_hi) if c_lo_hi else float("nan"),
            }
        )
    rec_df = pd.DataFrame(rec_rows).sort_values("undirected_count", ascending=False)
    rec_df.to_csv(OUT / "r4_phase1e_graph_reciprocity_pairs.csv", index=False)

    names = sorted(set(ev["buyer"]) | set(ev["seller"]))
    out_d = defaultdict(int)
    in_d = defaultdict(int)
    for (a, b), c in cnt.items():
        out_d[a] += c
        in_d[b] += c
    hub = [{"name": n, "deg_out": out_d[n], "deg_in": in_d[n], "deg_total": out_d[n] + in_d[n]} for n in names]
    pd.DataFrame(hub).sort_values("deg_total", ascending=False).to_csv(
        OUT / "r4_phase1e_graph_hub_degrees.csv", index=False
    )

    uu = ev[ev["symbol"] == U].copy()
    u_buy = uu[uu["aggressor"] == "buy_agg"].copy()
    u_sell = uu[uu["aggressor"] == "sell_agg"].copy()

    summ_buy = []
    for pc in u_buy["seller"].unique():
        for K in (5, 20):
            col = f"fwd_u_{K}"
            x = pd.to_numeric(u_buy.loc[u_buy["seller"] == pc, col], errors="coerce").dropna().to_numpy()
            if len(x) < 15:
                continue
            m, lo, hi = bootstrap_ci(x)
            summ_buy.append(
                {
                    "role": "buy_agg_passive_seller",
                    "passive_counterparty": pc,
                    "K": K,
                    "n": len(x),
                    "mean": float(np.mean(x)),
                    "median": float(np.median(x)),
                    "t_stat": t_stat(x),
                    "frac_pos": float((x > 0).mean()),
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }
            )
    pd.DataFrame(summ_buy).sort_values(["K", "n"], ascending=[True, False]).to_csv(
        OUT / "r4_phase1e_u_aggressor_by_passive_counterparty.csv", index=False
    )

    summ_sell = []
    for pc in u_sell["buyer"].unique():
        for K in (5, 20):
            col = f"fwd_u_{K}"
            x = pd.to_numeric(u_sell.loc[u_sell["buyer"] == pc, col], errors="coerce").dropna().to_numpy()
            if len(x) < 15:
                continue
            m, lo, hi = bootstrap_ci(x)
            summ_sell.append(
                {
                    "role": "sell_agg_passive_buyer",
                    "passive_counterparty": pc,
                    "K": K,
                    "n": len(x),
                    "mean": float(np.mean(x)),
                    "median": float(np.median(x)),
                    "t_stat": t_stat(x),
                    "frac_pos": float((x > 0).mean()),
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }
            )
    pd.DataFrame(summ_sell).sort_values(["K", "n"], ascending=[True, False]).to_csv(
        OUT / "r4_phase1e_u_sell_agg_passive_buyer_fwd.csv", index=False
    )

    day_rows = []
    for pc in sorted(u_buy["seller"].unique()):
        for d in (1, 2, 3):
            for K in (5, 20):
                col = f"fwd_u_{K}"
                g = u_buy[(u_buy["day"] == d) & (u_buy["seller"] == pc)]
                x = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy()
                if len(x) < 5:
                    continue
                m, lo, hi = bootstrap_ci(x)
                day_rows.append(
                    {
                        "passive_seller": pc,
                        "day": d,
                        "K": K,
                        "n": len(x),
                        "mean": float(np.mean(x)),
                        "median": float(np.median(x)),
                        "t_stat": t_stat(x),
                        "frac_pos": float((x > 0).mean()),
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                    }
                )
    pd.DataFrame(day_rows).sort_values(["passive_seller", "day", "K"]).to_csv(
        OUT / "r4_phase1e_u_buy_agg_passive_seller_by_day.csv", index=False
    )

    top = rec_df.head(8)
    lines = [
        "Top undirected pairs by total directed count (see CSV for full table).",
        "Reciprocity_ratio_hi_over_lo = count_hi_to_lo / count_lo_to_hi (canonical lex order).",
        "",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"{r['A']} <-> {r['B']}: undirected_count={int(r['undirected_count'])} "
            f"lo->hi={int(r['count_lo_to_hi'])} hi->lo={int(r['count_hi_to_lo'])} "
            f"recip={r['reciprocity_ratio_hi_over_lo']:.3f}"
        )
    (OUT / "r4_phase1e_graph_summary_top_pairs.txt").write_text("\n".join(lines) + "\n")
    print("Wrote phase1e to", OUT)


if __name__ == "__main__":
    main()
