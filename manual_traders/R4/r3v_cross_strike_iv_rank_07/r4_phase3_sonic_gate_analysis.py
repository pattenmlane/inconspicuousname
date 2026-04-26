#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate (5200+5300 spread<=2) on Round 4 tapes.

Aligns with round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py:
inner join timestamps where VEV_5200, VEV_5300, and VELVETFRUIT_EXTRACT price rows exist;
tight = (s5200<=TH) & (s5300<=TH).

Outputs:
- r4_p3_spread_spread_corr.csv — corr(s5200,s5300), corr(s5200,s_ext), corr(s5300,s_ext), corr(s5200,s5000), ...
- r4_p3_gate_extract_fwd_k20.csv — replicate STRATEGY-style extract forward k=20 tight vs loose on R4 panel
- r4_p3_trade_enriched_with_gate.csv — Phase1 enriched trades + sonic_tight (for audit)
- r4_p3_three_way_pair_symbol_k20.csv — (sonic_tight, pair, symbol) cells for dm_ex_k20
- r4_p3_phase1_compare_mark22_vev5300_sell.csv — Mark22 sell_agg on VEV_5300: gated vs all
- r4_p3_burst_extract_fwd_k20_by_gate.csv — burst × gate interaction on extract k=20
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.join(os.path.dirname(__file__), "analysis_outputs")
ENRICHED = os.path.join(OUT, "r4_p1_trade_enriched.csv")
PRICE_GLOB = "Prosperity4Data/ROUND_4/prices_round_4_day_{d}.csv"
DAYS = (1, 2, 3)
TH = 2
K_FWD = 20


def load_day_prices(day: int) -> pd.DataFrame:
    p = pd.read_csv(PRICE_GLOB.format(d=day), sep=";")
    p["day"] = day
    return p


def one_prod(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = df[df["product"] == product].drop_duplicates("timestamp", keep="first").sort_values("timestamp")
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return pd.DataFrame(
        {
            "day": v["day"].values,
            "timestamp": v["timestamp"].values,
            "spread": (ask - bid).astype(float).values,
            "mid": mid.astype(float).values,
        }
    )


def aligned_panel(day: int) -> pd.DataFrame:
    df = load_day_prices(day)
    a = one_prod(df, "VEV_5200").rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = one_prod(df, "VEV_5300").rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = one_prod(df, "VELVETFRUIT_EXTRACT").rename(columns={"spread": "s_ext", "mid": "m_ext"})
    v5 = one_prod(df, "VEV_5000").rename(columns={"spread": "s5000", "mid": "mid5000"})
    m = a.merge(b, on=["day", "timestamp"], how="inner").merge(e, on=["day", "timestamp"], how="inner")
    m = m.merge(v5, on=["day", "timestamp"], how="inner")
    m = m.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return m


def add_gate_fwd(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy()
    out["sonic_tight"] = (out["s5200"] <= TH) & (out["s5300"] <= TH)
    out["m_ext_f"] = out.groupby("day")["m_ext"].shift(-K_FWD)
    out["fwd_k"] = out["m_ext_f"] - out["m_ext"]
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    panels = [add_gate_fwd(aligned_panel(d)) for d in DAYS]
    panel = pd.concat(panels, ignore_index=True)

    # Spread–spread / spread correlations (pooled valid rows)
    cols = ["s5200", "s5300", "s_ext", "s5000"]
    sub = panel[cols].dropna()
    corrs = {}
    for i, c1 in enumerate(cols):
        for c2 in cols[i:]:
            corrs[f"corr_{c1}_{c2}"] = float(sub[c1].corr(sub[c2])) if len(sub) > 50 else float("nan")
    corrs["n_rows"] = int(len(sub))
    pd.DataFrame([corrs]).to_csv(os.path.join(OUT, "r4_p3_spread_spread_corr.csv"), index=False)

    # STRATEGY-style tight vs loose extract forward
    tight = panel.loc[panel["sonic_tight"], "fwd_k"].dropna()
    loose = panel.loc[~panel["sonic_tight"], "fwd_k"].dropna()
    if len(tight) > 10 and len(loose) > 10:
        tt = stats.ttest_ind(tight.values, loose.values, equal_var=False)
        tstat, pval = float(tt.statistic), float(tt.pvalue)
    else:
        tstat, pval = float("nan"), float("nan")
    pd.DataFrame(
        [
            {
                "tight_n": int(len(tight)),
                "tight_mean": float(tight.mean()) if len(tight) else float("nan"),
                "loose_n": int(len(loose)),
                "loose_mean": float(loose.mean()) if len(loose) else float("nan"),
                "welch_t": tstat,
                "welch_p": pval,
                "horizon_k_rows": K_FWD,
            }
        ]
    ).to_csv(os.path.join(OUT, "r4_p3_gate_extract_fwd_k20.csv"), index=False)

    # Merge gate onto trades
    gate_df = panel[["day", "timestamp", "sonic_tight"]].drop_duplicates()
    if not os.path.isfile(ENRICHED):
        raise SystemExit("Missing r4_p1_trade_enriched.csv — run Phase 1 script first")
    tr = pd.read_csv(ENRICHED)
    tr = tr.merge(gate_df, on=["day", "timestamp"], how="left")
    tr["sonic_tight"] = tr["sonic_tight"].fillna(False)
    tr.to_csv(os.path.join(OUT, "r4_p3_trade_enriched_with_gate.csv"), index=False)

    burst_ts = (
        tr.groupby(["day", "timestamp"])
        .size()
        .reset_index(name="n")
    )
    burst_set = set(zip(burst_ts.loc[burst_ts["n"] > 1, "day"], burst_ts.loc[burst_ts["n"] > 1, "timestamp"]))
    tr["burst"] = [(int(a), int(b)) in burst_set for a, b in zip(tr["day"], tr["timestamp"])]

    # Three-way: sonic_tight × pair × symbol on dm_ex_k20
    rows = []
    for (st, pair, sym), g in tr.groupby(["sonic_tight", "pair", "symbol"]):
        x = g["dm_ex_k20"].dropna()
        n = len(x)
        if n < 15:
            continue
        m = float(x.mean())
        s = float(x.std(ddof=1)) if n > 1 else float("nan")
        tstat = float(m / (s / np.sqrt(n))) if s and s > 1e-12 and s == s else float("nan")
        rows.append(
            {
                "sonic_tight": bool(st),
                "pair": pair,
                "symbol": sym,
                "n": n,
                "mean_dm_ex_k20": m,
                "t": tstat,
                "pos_frac": float((x > 0).mean()),
            }
        )
    tw = pd.DataFrame(rows).sort_values("n", ascending=False)
    tw.to_csv(os.path.join(OUT, "r4_p3_three_way_pair_symbol_k20.csv"), index=False)

    # Mark22 sell VEV_5300 aggressive: gated vs not
    m22 = tr[(tr["seller"] == "Mark 22") & (tr["symbol"] == "VEV_5300") & (tr["agg"] == "sell_agg")]
    summ = []
    for st, lab in [(True, "sonic_tight"), (False, "sonic_loose"), (None, "all")]:
        if st is None:
            g = m22
        else:
            g = m22[m22["sonic_tight"] == st]
        x = g["dm_self_k20"].dropna()
        summ.append({"slice": lab, "n": len(x), "mean_dm_self_k20": float(x.mean()) if len(x) else float("nan")})
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "r4_p3_mark22_vev5300_sell_by_gate.csv"), index=False)

    # Burst × gate extract k=20
    br = []
    for b in [True, False]:
        for st in [True, False]:
            g = tr[(tr["burst"] == b) & (tr["sonic_tight"] == st)]
            x = g["dm_ex_k20"].dropna()
            if len(x) < 20:
                continue
            br.append(
                {
                    "burst": b,
                    "sonic_tight": st,
                    "n": len(x),
                    "mean_dm_ex_k20": float(x.mean()),
                    "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if x.std(ddof=1) > 0 else float("nan"),
                }
            )
    pd.DataFrame(br).to_csv(os.path.join(OUT, "r4_p3_burst_extract_fwd_k20_by_gate.csv"), index=False)

    # Mark67 buy extract: compare all vs tight-only (phase1/2 style)
    m67 = tr[(tr["buyer"] == "Mark 67") & (tr["symbol"] == "VELVETFRUIT_EXTRACT") & (tr["agg"] == "buy_agg")]
    cmp = []
    for st, lab in [(None, "all"), (True, "tight_only"), (False, "loose_only")]:
        g = m67 if st is None else m67[m67["sonic_tight"] == st]
        x = g["dm_ex_k20"].dropna()
        cmp.append({"slice": lab, "n": len(x), "mean_k20": float(x.mean()) if len(x) else float("nan")})
    pd.DataFrame(cmp).to_csv(os.path.join(OUT, "r4_p3_mark67_extract_buy_k20_by_gate.csv"), index=False)

    # Overlap: Mark67 aggressive buy on extract vs sonic gate (same row as enriched trades)
    m67b = tr[(tr["buyer"] == "Mark 67") & (tr["symbol"] == "VELVETFRUIT_EXTRACT") & (tr["agg"] == "buy_agg")]
    ov = m67b.groupby("sonic_tight").size().reset_index(name="n_prints")
    ov.to_csv(os.path.join(OUT, "r4_p3_mark67_extract_agg_buy_count_by_gate.csv"), index=False)

    with open(os.path.join(OUT, "r4_phase3_machine_summary.json"), "w") as f:
        json.dump(
            {
                "panel_rows": int(len(panel)),
                "p_tight": float(panel["sonic_tight"].mean()),
                "three_way_cells_written": int(len(tw)),
                "note": "Backtester now populates state.market_trades in test_runner.py; prior v1 PnL used empty counterparty state.",
            },
            f,
            indent=2,
        )

    print("Phase 3 ->", OUT)


if __name__ == "__main__":
    main()
