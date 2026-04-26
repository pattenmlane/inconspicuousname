"""
Round 4 Phase 2 tape analysis (orthogonal tables).

- Burst-conditioned: trades within ±W of (day, ts) where Mark 01→Mark 22 has >=3 VEV prints.
- Microprice vs mid; spread change vs |fwd_5| (compression → vol proxy).
- Extract signed flow lags vs forward extract mid (distributed lag OLS).
- Mark 22 on VEV_5300 × spread tertile interaction.

Run: python3 manual_traders/R4/r3v_velvet_beta_hedge_02/analyze_r4_phase2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
W = 300
KS = [5, 20]


def load_px_full() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = df["day"].astype(int)
        frames.append(df)
    px = pd.concat(frames, ignore_index=True)
    bid1 = px["bid_price_1"].astype(float)
    ask1 = px["ask_price_1"].astype(float)
    bv1 = px["bid_volume_1"].astype(float).clip(lower=1e-9)
    av1 = px["ask_volume_1"].astype(float).clip(lower=1e-9)
    mid = px["mid_price"].astype(float)
    px["spread"] = ask1 - bid1
    px["microprice"] = (bid1 * av1 + ask1 * bv1) / (bv1 + av1)
    px["micro_minus_mid"] = px["microprice"] - mid
    px = px.sort_values(["day", "product", "timestamp"])
    for (dy, pr), g in px.groupby(["day", "product"], sort=False):
        idx = g.index
        m = g["mid_price"].astype(float).values
        for k in KS:
            fwd = np.full(len(g), np.nan)
            for i in range(len(g) - k):
                fwd[i] = m[i + k] - m[i]
            px.loc[idx, f"fwd_{k}"] = fwd
            px.loc[idx, f"abs_fwd_{k}"] = np.abs(fwd)
    px["dspread"] = px.groupby(["day", "product"])["spread"].diff()
    return px


def load_trades() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        t = pd.read_csv(p, sep=";")
        t["day"] = d
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def burst_centers_01_22(tr: pd.DataFrame) -> pd.DataFrame:
    sc = "product" if "product" in tr.columns else "symbol"
    vev = tr[tr[sc].str.startswith("VEV_", na=False)]
    bc = (
        vev[(vev["buyer"] == "Mark 01") & (vev["seller"] == "Mark 22")]
        .groupby(["day", "timestamp"])
        .size()
        .reset_index(name="n_vev")
    )
    return bc[bc["n_vev"] >= 3]


def near_burst_mask(days: np.ndarray, ts: np.ndarray, centers: dict[int, np.ndarray]) -> np.ndarray:
    out = np.zeros(len(ts), dtype=bool)
    for i in range(len(ts)):
        d = int(days[i])
        t = int(ts[i])
        arr = centers.get(d)
        if arr is None or len(arr) == 0:
            continue
        j = np.searchsorted(arr, t)
        for cand in (j - 1, j, j + 1):
            if 0 <= cand < len(arr) and abs(arr[cand] - t) <= W:
                out[i] = True
                break
    return out


def main() -> None:
    px = load_px_full()
    tr = load_trades()
    tr = tr.rename(columns={"symbol": "product"})
    cols = [
        "day",
        "timestamp",
        "product",
        "spread",
        "microprice",
        "mid_price",
        "micro_minus_mid",
        "fwd_5",
        "fwd_20",
        "abs_fwd_5",
        "abs_fwd_20",
        "dspread",
    ]
    merged = tr.merge(px[cols], on=["day", "timestamp", "product"], how="left")

    bc = burst_centers_01_22(tr)
    bc.to_csv(OUT / "r4_phase2_burst_centers_01_22.csv", index=False)
    centers: dict[int, np.ndarray] = {}
    for d in DAYS:
        sub = bc.loc[bc["day"] == d, "timestamp"].astype(int).sort_values().unique()
        centers[d] = np.asarray(sub, dtype=int)

    merged["near_01_22_burst"] = near_burst_mask(
        merged["day"].values, merged["timestamp"].values.astype(int), centers
    )
    (
        merged[merged["near_01_22_burst"]]
        .groupby(["product", "buyer", "seller"])
        .agg(n=("fwd_5", "count"), m5=("fwd_5", "mean"), m20=("fwd_20", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
        .to_csv(OUT / "r4_phase2_near_burst_forward_by_pair_product.csv", index=False)
    )

    cor_rows: list[dict] = []
    for pr in ["VELVETFRUIT_EXTRACT", "VEV_5300", "VEV_5200", "HYDROGEL_PACK"]:
        subp = px[(px["product"] == pr)].dropna(subset=["dspread", "abs_fwd_5"])
        if len(subp) > 100:
            cor_rows.append(
                {
                    "product": pr,
                    "corr_dspread_absfwd5": float(subp["dspread"].corr(subp["abs_fwd_5"])),
                    "n": int(len(subp)),
                }
            )
    pd.DataFrame(cor_rows).to_csv(OUT / "r4_phase2_spread_compression_vs_vol.csv", index=False)

    ut = tr[tr["product"] == "VELVETFRUIT_EXTRACT"].copy()
    ut = ut.sort_values(["day", "timestamp"])
    ut["flow"] = np.where(ut["buyer"].astype(str).str.startswith("Mark"), ut["quantity"].astype(int), 0) - np.where(
        ut["seller"].astype(str).str.startswith("Mark"), ut["quantity"].astype(int), 0
    )
    u_px = px[px["product"] == "VELVETFRUIT_EXTRACT"][["day", "timestamp", "fwd_5"]].drop_duplicates(
        ["day", "timestamp"]
    )
    ut = ut.merge(u_px, on=["day", "timestamp"], how="left")
    for lag in [0, 1, 2, 3, 5]:
        ut[f"f{lag}"] = ut.groupby("day")["flow"].shift(lag)
    u2 = ut.dropna(subset=["fwd_5", "f0", "f1", "f2", "f3", "f5"])
    if len(u2) > 40:
        X = u2[["f0", "f1", "f2", "f3", "f5"]].values.astype(float)
        y = u2["fwd_5"].values.astype(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=1e-6)
        (OUT / "r4_phase2_extract_flow_lags_fwd5.json").write_text(
            json.dumps({"coefs_l0_l1_l2_l3_l5": [float(x) for x in beta], "n": len(u2)}), encoding="utf-8"
        )

    m22 = merged[(merged["product"] == "VEV_5300") & (merged["seller"] == "Mark 22")].dropna(subset=["spread", "fwd_5"])
    if len(m22) > 20:
        try:
            m22["spr_q"] = pd.qcut(
                m22["spread"].astype(float), q=3, labels=["tight", "mid", "wide"], duplicates="drop"
            )
        except ValueError:
            m22["spr_q"] = pd.cut(
                m22["spread"].astype(float), bins=[-0.1, 2, 5, 200], labels=["tight_le2", "mid3_5", "wide"]
            )
        m22.groupby("spr_q", observed=True)["fwd_5"].agg(["count", "mean", "std"]).reset_index().to_csv(
            OUT / "r4_phase2_mark22_on_5300_fwd5_by_spread_tertile.csv", index=False
        )

    (OUT / "r4_phase2_run_done.txt").write_text("ok\n", encoding="utf-8")
    print("Wrote phase2 outputs to", OUT)


if __name__ == "__main__":
    main()
