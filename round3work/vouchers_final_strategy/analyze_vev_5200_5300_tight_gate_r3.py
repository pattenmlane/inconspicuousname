"""
Reproduce the "Sonic / inclineGod" style analysis: **joint tight book** on VEV_5200 & VEV_5300.

Run from repo root or this directory (uses ``round3work/tipworkflow`` for data loaders):

  python3 round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

_PKG = Path(__file__).resolve().parent
_TIP = _PKG.parent / "tipworkflow"
if str(_TIP) not in sys.path:
    sys.path.insert(0, str(_TIP))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_DIR
from data import load_day_raw

# Outputs live under this strategy folder
OUT_DIR = _PKG / "outputs"

VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
EXTRACT = "VELVETFRUIT_EXTRACT"
TH = 2  # "spreads at or below 2"
K = 20  # forward steps (bars)
N_TS_PANEL1 = 1000  # "first 1000 timestamps" for top-left
MA_WIN = 100  # signal frequency smoothing
SEED = 0


def _one_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    v = v.assign(
        spread=(ask - bid).astype(float),
        mid=mid,
    )
    return v[["timestamp", "spread", "mid"]].copy()


def aligned_panel(day: int) -> pd.DataFrame:
    """One row per timestamp with s5200, s5300, m_ext; inner join = all three present."""
    df = load_day_raw(day)
    a = _one_product(df, VEV_5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = _one_product(df, VEV_5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = _one_product(df, EXTRACT).rename(columns={"spread": "s_ext", "mid": "m_ext"})

    m = a.merge(b, on="timestamp", how="inner").merge(
        e[["timestamp", "m_ext", "s_ext"]], on="timestamp", how="inner"
    )
    m = m.sort_values("timestamp").reset_index(drop=True)
    return m


def add_forward_and_tight(
    m: pd.DataFrame, *, th: int = TH, k: int = K
) -> pd.DataFrame:
    out = m.copy()
    out["tight"] = (out["s5200"] <= th) & (out["s5300"] <= th)
    out["m_ext_f"] = out["m_ext"].shift(-k)
    out["fwd_k"] = out["m_ext_f"] - out["m_ext"]
    return out


def ttest_fwd(tight: np.ndarray, loose: np.ndarray) -> tuple[float, float, float, float]:
    """Welch t-test on K-step forward return: tight vs not-tight (drop nan)."""
    t = np.asarray(tight, dtype=float)
    n = np.asarray(loose, dtype=float)
    t = t[np.isfinite(t)]
    n = n[np.isfinite(n)]
    if len(t) < 2 or len(n) < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    a = stats.ttest_ind(t, n, equal_var=False, nan_policy="omit")
    return (float(t.mean()), float(n.mean()), float(a.statistic), float(a.pvalue))


def plot_six_panels(
    days: tuple[int, ...] = (0, 1, 2),
    out_png: Path | None = None,
) -> None:
    out_png = out_png or (OUT_DIR / "r3_tight_spread_6panel.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # --- day 0 full panel (main stats) ---
    p0 = add_forward_and_tight(aligned_panel(0))
    valid0 = p0["fwd_k"].notna()
    p0v = p0.loc[valid0]
    t_mask = p0v["tight"] & p0v["s5200"].notna() & p0v["s5300"].notna()
    f_t = p0v.loc[t_mask, "fwd_k"]
    f_n = p0v.loc[~t_mask, "fwd_k"]
    m_t, m_n, tstat, pval = ttest_fwd(f_t.values, f_n.values)
    c_sp = p0v["s5200"].corr(p0v["s5300"])
    c5m = p0v["s5200"].corr(p0v["m_ext"])
    c3m = p0v["s5300"].corr(p0v["m_ext"])

    # First N timestamps for (1)
    u_ts = p0["timestamp"].unique()[:N_TS_PANEL1]
    p0_head = p0[p0["timestamp"].isin(u_ts)]

    # --- build figure (2×3: top: spreads, price, forward; bottom: cumulative, signal, scatter) ---
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 9.0))

    # (0,0) Spreads over time (first 1000 timestamps), day 0
    ax = axes[0, 0]
    ax.plot(p0_head["timestamp"], p0_head["s5200"], color="#1f77b4", lw=0.8, alpha=0.85, label="VEV_5200")
    ax.plot(p0_head["timestamp"], p0_head["s5300"], color="#ff7f0e", lw=0.8, alpha=0.85, label="VEV_5300")
    ax.axhline(TH, color="red", ls="--", lw=1.0, label=f"threshold = {TH}")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("spread (ask − bid, price points)")
    ax.set_title(f"VEV_5200 & VEV_5300 spreads (day 0, first {len(u_ts)} timestamps)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    # (0,1) VELVET mid: tight vs not
    ax = axes[0, 1]
    s_t = p0v.loc[p0v["tight"], "m_ext"]
    s_n = p0v.loc[~p0v["tight"], "m_ext"]
    ax.hist(
        s_t, bins=60, alpha=0.55, color="C0", density=True, label="Tight (both ≤2)", histtype="stepfilled"
    )
    ax.hist(
        s_n, bins=60, alpha=0.45, color="C1", density=True, label="Not tight", histtype="stepfilled"
    )
    ax.set_xlabel(f"{EXTRACT} mid (same rows as {K}-step fwd sample)")
    ax.set_ylabel("density")
    ax.set_title("Price distribution: tight vs not (day 0)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # (0,2) Forward return K-step
    ax = axes[0, 2]
    ax.hist(
        f_t, bins=80, alpha=0.55, color="C0", density=True, label=f"Tight: mean={m_t:.3f} (n={len(f_t):,})"
    )
    ax.hist(
        f_n, bins=80, alpha=0.45, color="C1", density=True, label=f"Not tight: mean={m_n:.3f} (n={len(f_n):,})"
    )
    ax.set_xlabel(f"{K}-step forward Δmid (extract)")
    ax.set_ylabel("density")
    ax.set_title(f"Forward return ({K} steps); Welch t={tstat:.2f}, p={pval:.2e}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # (1,0) Cumulative: cumsum of fwd_k * 1_tight
    ax = axes[1, 0]
    for d, c in zip(days, ("C0", "C1", "C2")):
        try:
            pd_ = add_forward_and_tight(aligned_panel(d))
        except FileNotFoundError:
            continue
        w = np.where(pd_["tight"] & pd_["fwd_k"].notna().values, pd_["fwd_k"].values, 0.0)
        csum = np.cumsum(w)
        ax.plot(np.arange(len(csum)), csum, color=c, lw=1.0, label=f"day {d}")
    ax.set_xlabel("bar index (aligned timestamps within day)")
    ax.set_ylabel("cumulative sum of 1_tight · fwd_k (overlapping)")
    ax.set_title(f"Cumulative sum of {K}-step move when both tight (per day)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)

    # (1,1) Signal frequency: rolling mean of tight, day 0
    ax = axes[1, 1]
    sig = p0["tight"].astype(float)
    if len(sig) >= MA_WIN:
        ax.plot(
            p0["timestamp"],
            sig.rolling(MA_WIN, min_periods=1).mean(),
            color="purple",
            lw=1.0,
            label=f"tight share (MA {MA_WIN})",
        )
    else:
        ax.plot(p0["timestamp"], sig, color="purple", lw=0.5)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("P(both tight) rolling")
    ax.set_ylim(0, 1.05)
    ax.set_title("Signal frequency over time (day 0, smoothed)")
    ax.grid(True, alpha=0.2)

    # (1,2) scatter s5200 vs s5300
    ax = axes[1, 2]
    rng = np.random.default_rng(SEED)
    idx = p0.index
    if len(idx) > 8000:
        idx = rng.choice(idx, size=8000, replace=False)
    s5 = p0.loc[sorted(idx), "s5200"]
    s3 = p0.loc[sorted(idx), "s5300"]
    ax.scatter(s5, s3, s=2, c="0.3", alpha=0.25, rasterized=True)
    ax.axvline(TH, color="r", ls="--", alpha=0.6)
    ax.axhline(TH, color="r", ls="--", alpha=0.6)
    both = p0["tight"].astype(bool)
    ax.scatter(
        p0.loc[both, "s5200"].iloc[::20],
        p0.loc[both, "s5300"].iloc[::20],
        s=6,
        c="g",
        alpha=0.5,
        label="both tight (subsample)",
    )
    ax.set_xlabel("VEV_5200 spread")
    ax.set_ylabel("VEV_5300 spread")
    ax.set_title("5200 vs 5300 spread (day 0; green = tight)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        "Round 3 — joint VEV_5200 & VEV_5300 spread gate "
        f"(tight: both ≤{TH} price pts) + extract mid forward move",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # text summary
    txt = dedent(
        f"""
    Round 3 tight-spread gate replication (our tape)
    -----------------------------------------------
    Data: {DATA_DIR} | TH = {TH} (both s5200 and s5300 <= TH)
    Forward horizon K = {K} aligned rows (one timestamp per row after join)

    day 0 (valid {K}-step fwd rows, n = {len(p0v):,}):
      P(both tight)  ≈ {p0v['tight'].mean():.4f}
      mean(fwd) | tight     = {m_t:.6g}
      mean(fwd) | not-tight  = {m_n:.6g}
      Welch t-stat = {tstat:.4f},  p = {pval:.4e}

    Correlations (day 0, same rows) — "corr of spreads" / vs extract mid:
      corr(s5200, s5300)  = {c_sp:.4f}
      corr(s5200, m_ext)  = {c5m:.4f}
      corr(s5300, m_ext)  = {c3m:.4f}

    inclineGod: spreads / book state; 5200 vs 5300 spread co-move ≈ {c_sp:.2f}.
    Sonic: BOTH ≤{TH} simultaneously; P(tight)≈{p0v['tight'].mean():.2f} but forward mean much higher when on.
    """
    )
    ptxt = out_png.parent / "r3_tight_spread_summary.txt"
    ptxt.write_text(txt, encoding="utf-8")
    print(txt)
    print("Wrote", out_png)
    print("Wrote", ptxt)


def main() -> None:
    plot_six_panels()


if __name__ == "__main__":
    main()
