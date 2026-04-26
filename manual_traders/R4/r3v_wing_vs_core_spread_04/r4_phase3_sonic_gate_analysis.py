#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate on **price** tape (inner join 5200, 5300, extract),
matching round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py:
  spread = ask1 - bid1; tight = (s5200 <= TH) & (s5300 <= TH); forward extract K steps.

Also: inclineGod-style spread–spread / spread–mid correlations; counterparty stats
**at print timestamp** split by tight; three-way (pair × tight × symbol) Mark67 focus.

Outputs: manual_traders/R4/.../outputs/phase3/

Run: python3 manual_traders/R4/r3v_wing_vs_core_spread_04/r4_phase3_sonic_gate_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
VEV_5200, VEV_5300 = "VEV_5200", "VEV_5300"
EXTRACT = "VELVETFRUIT_EXTRACT"
TH = 2
K = 20
PRODUCTS = [
    "HYDROGEL_PACK",
    EXTRACT,
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]


def _one_product(px: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        px[px["product"] == product]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return pd.DataFrame(
        {
            "timestamp": v["timestamp"].astype(int),
            "spread": (ask - bid).astype(float),
            "mid": mid.astype(float),
        }
    )


def aligned_panel(px: pd.DataFrame) -> pd.DataFrame:
    a = _one_product(px, VEV_5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = _one_product(px, VEV_5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = _one_product(px, EXTRACT).rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(e[["timestamp", "m_ext", "s_ext"]], on="timestamp", how="inner")
    return m.sort_values("timestamp").reset_index(drop=True)


def add_forward_tight(m: pd.DataFrame, *, th: int = TH, k: int = K) -> pd.DataFrame:
    out = m.copy()
    out["tight"] = (out["s5200"] <= th) & (out["s5300"] <= th)
    out["fwd_k"] = out["m_ext"].shift(-k) - out["m_ext"]
    return out


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (float("nan"),) * 4
    r = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(a.mean()), float(b.mean()), float(r.statistic), float(r.pvalue)


def classify_aggression(price: float, bid1: float, ask1: float) -> str:
    if price >= ask1:
        return "buy_aggr"
    if price <= bid1:
        return "sell_aggr"
    return "inside"


def main() -> None:
    rows_gate: list[dict] = []
    corr_rows: list[dict] = []
    pooled_tight_fwd: list[float] = []
    pooled_loose_fwd: list[float] = []

    for tape_day in DAYS:
        px = pd.read_csv(DATA / f"prices_round_4_day_{tape_day}.csv", sep=";")
        p = aligned_panel(px)
        p = add_forward_tight(p, th=TH, k=K)
        valid = p["fwd_k"].notna()
        pv = p.loc[valid].copy()
        tight_m = pv["tight"]
        f_t = pv.loc[tight_m, "fwd_k"].values
        f_n = pv.loc[~tight_m, "fwd_k"].values
        mt, mn, tst, pvl = welch(f_t, f_n)
        pooled_tight_fwd.extend(f_t.tolist())
        pooled_loose_fwd.extend(f_n.tolist())
        p_tight = float(tight_m.mean())
        rows_gate.append(
            {
                "tape_day": tape_day,
                "n_rows": int(len(pv)),
                "p_tight": p_tight,
                "mean_fwd_tight": mt,
                "mean_fwd_not_tight": mn,
                "welch_t": tst,
                "welch_p": pvl,
                "n_tight": int(tight_m.sum()),
                "n_loose": int((~tight_m).sum()),
            }
        )
        # inclineGod: correlations full sample
        c_52 = float(pv["s5200"].corr(pv["s5300"]))
        c_5e = float(pv["s5200"].corr(pv["s_ext"]))
        c_3e = float(pv["s5300"].corr(pv["s_ext"]))
        c_5m = float(pv["s5200"].corr(pv["m_ext"]))
        c_3m = float(pv["s5300"].corr(pv["m_ext"]))
        for subset, label in ((pv, "all"), (pv.loc[tight_m], "tight_only"), (pv.loc[~tight_m], "not_tight_only")):
            if len(subset) < 30:
                continue
            corr_rows.append(
                {
                    "tape_day": tape_day,
                    "subset": label,
                    "corr_s5200_s5300": float(subset["s5200"].corr(subset["s5300"])),
                    "corr_s5200_s_ext": float(subset["s5200"].corr(subset["s_ext"])),
                    "corr_s5300_s_ext": float(subset["s5300"].corr(subset["s_ext"])),
                    "corr_s5200_m_ext": float(subset["s5200"].corr(subset["m_ext"])),
                    "corr_s5300_m_ext": float(subset["s5300"].corr(subset["m_ext"])),
                    "n": len(subset),
                }
            )

    pd.DataFrame(rows_gate).to_csv(OUT / "sonic_gate_fwd_extract_k20_by_day.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(OUT / "spread_correlations_by_gate_subset.csv", index=False)

    mt, mn, tst, pvl = welch(np.array(pooled_tight_fwd), np.array(pooled_loose_fwd))
    (OUT / "sonic_gate_fwd_extract_k20_pooled.txt").write_text(
        f"pooled_all_days: mean_fwd_tight={mt:.6g} mean_fwd_not_tight={mn:.6g} welch_t={tst:.6g} p={pvl:.6g} "
        f"n_tight={np.sum(np.isfinite(pooled_tight_fwd))} n_loose={np.sum(np.isfinite(pooled_loose_fwd))}\n",
        encoding="utf-8",
    )

    # --- Trades merged with tight at print time (inner join panel) ---
    book_parts = []
    for d in DAYS:
        px = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        b = px[px["product"].isin(PRODUCTS)].copy()
        b["tape_day"] = d
        b = b.rename(columns={"product": "symbol"})
        book_parts.append(b)
    book = pd.concat(book_parts, ignore_index=True)

    tr_parts = []
    for d in DAYS:
        t = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        t["tape_day"] = d
        tr_parts.append(t)
    tr = pd.concat(tr_parts, ignore_index=True)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")

    m = tr.merge(
        book,
        on=["tape_day", "timestamp", "symbol"],
        how="left",
    )
    m["aggression"] = [
        classify_aggression(float(p), float(b), float(a))
        if pd.notna(p) and pd.notna(b) and pd.notna(a)
        else "unknown"
        for p, b, a in zip(m["price"], m["bid_price_1"], m["ask_price_1"], strict=True)
    ]

    tight_at_ts: dict[tuple[int, int], bool] = {}
    mid_ext_by_ts: dict[int, tuple[dict[int, int], np.ndarray]] = {}
    for tape_day in DAYS:
        px = pd.read_csv(DATA / f"prices_round_4_day_{tape_day}.csv", sep=";")
        p = aligned_panel(px)
        p = add_forward_tight(p, th=TH, k=K)
        ts_idx = {int(t): i for i, t in enumerate(p["timestamp"])}
        mids = p["m_ext"].astype(float).values
        for _, r in p.iterrows():
            tight_at_ts[(tape_day, int(r["timestamp"]))] = bool(r["tight"])
        mid_ext_by_ts[tape_day] = (ts_idx, mids)

    def fwd_ext(d: int, ts: int, k: int) -> float:
        ts_idx, mids = mid_ext_by_ts[d]
        ti = ts_idx.get(int(ts))
        if ti is None or ti + k >= len(mids):
            return float("nan")
        a, b = mids[ti], mids[ti + k]
        if np.isnan(a) or np.isnan(b):
            return float("nan")
        return float(b - a)

    m["joint_tight_print"] = [tight_at_ts.get((int(r.tape_day), int(r.timestamp)), False) for r in m.itertuples()]
    m["fwd_extract_k20"] = [fwd_ext(int(r.tape_day), int(r.timestamp), K) for r in m.itertuples()]

    sub67 = m[
        (m["symbol"] == EXTRACT)
        & (m["buyer"] == "Mark 67")
        & (m["aggression"] == "buy_aggr")
    ]
    g67 = (
        sub67.groupby(["joint_tight_print"])
        .agg(n=("fwd_extract_k20", "count"), mean=("fwd_extract_k20", "mean"), med=("fwd_extract_k20", "median"))
        .reset_index()
    )
    g67.to_csv(OUT / "mark67_buy_aggr_extract_fwd_k20_by_joint_tight.csv", index=False)

    # Three-way: Mark01->Mark22 on extract × tight
    sub01 = m[(m["buyer"] == "Mark 01") & (m["seller"] == "Mark 22") & (m["symbol"] == EXTRACT)]
    g01 = (
        sub01.groupby(["joint_tight_print"])
        .agg(n=("fwd_extract_k20", "count"), mean=("fwd_extract_k20", "mean"))
        .reset_index()
    )
    g01.to_csv(OUT / "mark01_to_mark22_extract_fwd_k20_by_joint_tight.csv", index=False)

    # Pair × tight for top pairs (extract)
    top_pairs = (
        m[m["symbol"] == EXTRACT]
        .assign(pair=lambda x: x["buyer"].astype(str) + "->" + x["seller"].astype(str))
        .groupby(["pair", "joint_tight_print"])
        .agg(n=("fwd_extract_k20", "count"), mean=("fwd_extract_k20", "mean"))
        .reset_index()
    )
    top_pairs = top_pairs[top_pairs["n"] >= 8].sort_values("n", ascending=False)
    top_pairs.to_csv(OUT / "extract_pair_by_joint_tight_fwd_k20.csv", index=False)

    # Extended signals for trader_v2: Mark67 buy_aggr extract + joint_tight at PRINT time
    sig = sub67[["tape_day", "timestamp", "joint_tight_print"]].drop_duplicates()
    sig_list = [[int(r.tape_day), int(r.timestamp), bool(r.joint_tight_print)] for r in sig.itertuples()]
    (OUT / "signals_mark67_buy_aggr_extract_with_tight.json").write_text(json.dumps(sig_list), encoding="utf-8")

    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        ax = axes[0]
        # Day 3 has highest P(tight); scatter tight vs all for readability
        tape_day = 3
        px = pd.read_csv(DATA / f"prices_round_4_day_{tape_day}.csv", sep=";")
        p = add_forward_tight(aligned_panel(px), th=TH, k=K)
        v = p.loc[p["fwd_k"].notna()]
        ax.scatter(v.loc[~v["tight"], "s5200"], v.loc[~v["tight"], "s5300"], s=3, alpha=0.12, c="gray", label="not tight")
        ax.scatter(v.loc[v["tight"], "s5200"], v.loc[v["tight"], "s5300"], s=5, alpha=0.45, c="tab:blue", label="tight")
        ax.axhline(TH, color="red", lw=0.8, alpha=0.6)
        ax.axvline(TH, color="red", lw=0.8, alpha=0.6)
        ax.set_xlabel("VEV_5200 spread")
        ax.set_ylabel("VEV_5300 spread")
        ax.set_title(f"Day {tape_day}: spread–spread (Sonic box)")
        ax.legend(loc="upper right", fontsize=7)

        ax2 = axes[1]
        td = pd.DataFrame(rows_gate)
        x = td["tape_day"].astype(str)
        ax2.bar(x, td["mean_fwd_tight"], alpha=0.85, label="tight")
        ax2.bar(x, td["mean_fwd_not_tight"], alpha=0.5, label="not tight")
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_ylabel(f"Mean fwd extract (K={K})")
        ax2.set_title("Tight vs not (inner-join tape)")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(OUT / "r4_phase3_gate_overview.png", dpi=120)
        plt.close(fig)

    summary = {
        "TH": TH,
        "K": K,
        "pooled_welch": {"mean_tight": mt, "mean_not_tight": mn, "t": tst, "p": pvl},
        "n_signals_mark67_with_tight_flag": len(sig_list),
        "outputs": sorted(p.name for p in OUT.iterdir() if p.is_file()),
    }
    (OUT / "phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote", OUT)


if __name__ == "__main__":
    main()
