#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate + inclineGod spread panels (tape).

Matches R3 script logic: inner-join timestamps for VEV_5200, VEV_5300, VELVETFRUIT_EXTRACT;
tight = (s5200 <= TH) & (s5300 <= TH); forward extract mid K steps (default K=20).

Also: spread–spread correlations, spread vs extract mid; trade-level Phase-1 events merged
with joint_tight at print time; Mark 01→Mark 22 × gate × symbol summaries.

Outputs: manual_traders/R4/r3v_jump_gap_filter_17/outputs/phase3/

Run: python3 manual_traders/R4/r3v_jump_gap_filter_17/r4_phase3_joint_gate_r4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = (1, 2, 3)
TH = 2
K_FWD = 20
TICK = 100
VEV_5200, VEV_5300 = "VEV_5200", "VEV_5300"
EXTRACT = "VELVETFRUIT_EXTRACT"


def _strip(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return v.assign(
        spread=(ask - bid).astype(float),
        mid=mid,
    )[["timestamp", "spread", "mid"]].copy()


def aligned_panel(pr_day: pd.DataFrame) -> pd.DataFrame:
    a = _strip(pr_day, VEV_5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = _strip(pr_day, VEV_5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = _strip(pr_day, EXTRACT).rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(e, on="timestamp", how="inner")
    return m.sort_values("timestamp").reset_index(drop=True)


def add_forward(m: pd.DataFrame, *, th: int, k: int) -> pd.DataFrame:
    out = m.copy()
    out["tight"] = (out["s5200"] <= th) & (out["s5300"] <= th)
    out["fwd_k"] = out["m_ext"].shift(-k) - out["m_ext"]
    return out


def welch(tight: np.ndarray, loose: np.ndarray) -> tuple[float, float, float, float, int, int]:
    t = np.asarray(tight, dtype=float)
    n = np.asarray(loose, dtype=float)
    t = t[np.isfinite(t)]
    n = n[np.isfinite(n)]
    if len(t) < 2 or len(n) < 2:
        return (np.nan,) * 4 + (len(t), len(n))
    r = stats.ttest_ind(t, n, equal_var=False, nan_policy="omit")
    return (float(t.mean()), float(n.mean()), float(r.statistic), float(r.pvalue), len(t), len(n))


def main() -> None:
    pr = pd.concat(
        [pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";").assign(day=d) for d in DAYS],
        ignore_index=True,
    )

    gate_rows = []
    corr_rows = []
    fig_scatters = []

    for d in DAYS:
        p = pr[pr["day"] == d]
        m = aligned_panel(p)
        m = add_forward(m, th=TH, k=K_FWD)
        m["day"] = d
        gate_rows.append(m)
        valid = m["fwd_k"].notna()
        mv = m.loc[valid]
        tmask = mv["tight"]
        ft, fn = mv.loc[tmask, "fwd_k"], mv.loc[~tmask, "fwd_k"]
        mt, mn, tst, pv, nt, nn = welch(ft.values, fn.values)
        corr_rows.append(
            {
                "day": d,
                "mean_fwd_tight": mt,
                "mean_fwd_loose": mn,
                "welch_t": tst,
                "welch_p": pv,
                "n_tight": nt,
                "n_loose": nn,
                "corr_s5200_s5300": float(mv["s5200"].corr(mv["s5300"])),
                "corr_s5200_m_ext": float(mv["s5200"].corr(mv["m_ext"])),
                "corr_s5300_m_ext": float(mv["s5300"].corr(mv["m_ext"])),
                "corr_s5200_s_ext": float(mv["s5200"].corr(mv["s_ext"])),
                "corr_s5300_s_ext": float(mv["s5300"].corr(mv["s_ext"])),
                "frac_tight": float(tmask.mean()),
            }
        )

        # inclineGod: scatter s5200 vs s5300 colored by tight
        fig, ax = plt.subplots(figsize=(6.5, 6))
        lo = mv.loc[~tmask]
        ti = mv.loc[tmask]
        ax.scatter(lo["s5200"], lo["s5300"], s=4, alpha=0.25, c="gray", label="not tight")
        ax.scatter(ti["s5200"], ti["s5300"], s=6, alpha=0.5, c="crimson", label="tight")
        ax.axvline(TH, color="k", lw=0.8, ls="--")
        ax.axhline(TH, color="k", lw=0.8, ls="--")
        ax.set_xlabel("VEV_5200 spread")
        ax.set_ylabel("VEV_5300 spread")
        ax.set_title(f"R4 day {d} — spread vs spread (joint box ≤{TH})")
        ax.legend(markerscale=2)
        fig.tight_layout()
        pth = OUT / f"r4_spread_spread_scatter_day{d}.png"
        fig.savefig(pth, dpi=140)
        plt.close(fig)
        fig_scatters.append(str(pth))

    all_m = pd.concat(gate_rows, ignore_index=True)
    valid_all = all_m["fwd_k"].notna()
    mv_all = all_m.loc[valid_all]
    tmask = mv_all["tight"]
    mt, mn, tst, pv, nt, nn = welch(
        mv_all.loc[tmask, "fwd_k"].values, mv_all.loc[~tmask, "fwd_k"].values
    )
    corr_rows.append(
        {
            "day": "pooled",
            "mean_fwd_tight": mt,
            "mean_fwd_loose": mn,
            "welch_t": tst,
            "welch_p": pv,
            "n_tight": nt,
            "n_loose": nn,
            "corr_s5200_s5300": float(mv_all["s5200"].corr(mv_all["s5300"])),
            "corr_s5200_m_ext": float(mv_all["s5200"].corr(mv_all["m_ext"])),
            "corr_s5300_m_ext": float(mv_all["s5300"].corr(mv_all["m_ext"])),
            "corr_s5200_s_ext": float(mv_all["s5200"].corr(mv_all["s_ext"])),
            "corr_s5300_s_ext": float(mv_all["s5300"].corr(mv_all["s_ext"])),
            "frac_tight": float(tmask.mean()),
        }
    )
    pd.DataFrame(corr_rows).to_csv(OUT / "r4_joint_gate_forward_extract_k20_by_day.csv", index=False)

    # Forward distribution comparison (pooled valid)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        mv_all.loc[~tmask, "fwd_k"],
        bins=60,
        alpha=0.55,
        density=True,
        label="not tight",
        color="gray",
    )
    ax.hist(
        mv_all.loc[tmask, "fwd_k"],
        bins=40,
        alpha=0.65,
        density=True,
        label="tight",
        color="crimson",
    )
    ax.set_xlabel(f"Forward Δ extract mid (K={K_FWD} steps)")
    ax.set_ylabel("density")
    ax.set_title("R4 pooled (days 1–3): forward extract mid — tight vs not")
    ax.legend()
    fig.tight_layout()
    fp = OUT / "r4_fwd_extract_hist_tight_vs_loose.png"
    fig.savefig(fp, dpi=140)
    plt.close(fig)

    # --- Merge trades with joint_tight at (day, timestamp) ---
    jt_map = {}
    for _, r in all_m.iterrows():
        jt_map[(int(r["day"]), int(r["timestamp"]))] = bool(r["tight"])

    keys = list(
        zip(pr["day"].astype(int), pr["product"].astype(str), pr["timestamp"].astype(int))
    )
    mid_ix = dict(zip(keys, pr["mid_price"].astype(float)))
    tr = pd.concat(
        [pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d) for d in DAYS],
        ignore_index=True,
    )
    tr["joint_tight"] = [jt_map.get((int(d), int(t)), False) for d, t in zip(tr["day"], tr["timestamp"])]
    tr["price"] = tr["price"].astype(float)

    bb = pr.rename(columns={"product": "symbol"})[
        ["day", "timestamp", "symbol", "bid_price_1", "ask_price_1"]
    ]
    te = tr.merge(bb, on=["day", "timestamp", "symbol"], how="left")
    te["aggr_buy"] = te["price"] >= te["ask_price_1"]
    te["aggr_sell"] = te["price"] <= te["bid_price_1"]

    def fwd_sym(row: pd.Series, k: int) -> float | None:
        d, sym, ts = int(row["day"]), str(row["symbol"]), int(row["timestamp"])
        t2 = ts + k * TICK
        if t2 > 999900:
            t2 = 999900
        a, b = mid_ix.get((d, sym, ts)), mid_ix.get((d, sym, t2))
        if a is None or b is None:
            return None
        return float(b - a)

    # Phase-1 style: K=5 forward on traded symbol
    te["fwd5"] = te.apply(lambda r: fwd_sym(r, 5), axis=1)
    te = te[np.isfinite(te["fwd5"])]

    # Mark 01 → 22 interaction with gate
    m122 = te[(te["buyer"] == "Mark 01") & (te["seller"] == "Mark 22")]
    if len(m122) > 0:
        g = (
            m122.groupby(["day", "joint_tight", "symbol"])["fwd5"]
            .agg(["mean", "count"])
            .reset_index()
        )
        g.to_csv(OUT / "mark01_mark22_fwd5_by_gate_and_symbol.csv", index=False)

    # Aggressive extract buy × seller × gate (Phase 1 echo, full table)
    exb = te[(te["symbol"] == EXTRACT) & te["aggr_buy"]]
    if len(exb) > 0:
        exb.groupby(["day", "joint_tight", "seller"])["fwd5"].agg(["mean", "count"]).reset_index().to_csv(
            OUT / "r4_aggr_extract_buy_fwd5_by_gate_seller.csv", index=False
        )

    # Three-way: top pairs × gate × symbol (min n)
    te["pair"] = te["buyer"].astype(str) + "->" + te["seller"].astype(str)
    vc = te["pair"].value_counts()
    top_pairs = vc.head(6).index.tolist()
    tw = []
    for pair in top_pairs:
        sub = te[te["pair"] == pair]
        for (d, jt, sym), g in sub.groupby(["day", "joint_tight", "symbol"]):
            if len(g) < 8:
                continue
            tw.append(
                {
                    "pair": pair,
                    "day": int(d),
                    "joint_tight": bool(jt),
                    "symbol": str(sym),
                    "n": len(g),
                    "mean_fwd5": float(g["fwd5"].mean()),
                    "frac_aggr_buy": float(g["aggr_buy"].mean()),
                }
            )
    pd.DataFrame(tw).sort_values(["pair", "day", "symbol", "joint_tight"]).to_csv(
        OUT / "three_way_pair_gate_symbol_fwd5.csv", index=False
    )

    # Summary text for humans
    cr = pd.DataFrame(corr_rows)
    lines = [
        "Round 4 Phase 3 — Sonic joint gate (inner join 5200+5300+extract), TH=%d, K_fwd=%d"
        % (TH, K_FWD),
        "",
        cr.to_string(index=False),
        "",
        "Plots:",
        str(OUT / "r4_fwd_extract_hist_tight_vs_loose.png"),
        *fig_scatters,
    ]
    (OUT / "phase3_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
