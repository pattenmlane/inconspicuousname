#!/usr/bin/env python3
"""
Round 4 Phase 2 — burst-conditioned markouts, microprice, spread–vol, signed-flow lags, IV proxy.

Inputs: analysis_outputs/r4_trade_markouts_wide.csv (Phase 1).

Run: python3 manual_traders/R4/r3v_smile_cubic_spline_logk_12/r4_phase2_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
P1 = OUT
DAYS = (1, 2, 3)
EX = "VELVETFRUIT_EXTRACT"
HY = "HYDROGEL_PACK"
BURST_WIN = 500


def load_prices(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    df["day"] = day
    for c in df.columns:
        if "price" in c or "volume" in c or c == "mid_price":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_trades(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
    df["day"] = day
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    return df


def microprice(bb, ba, bv, av) -> float | None:
    if any(pd.isna(x) for x in (bb, ba, bv, av)):
        return None
    bv, av = float(bv), abs(float(av))
    den = bv + av
    if den <= 0:
        return None
    return (float(bb) * av + float(ba) * bv) / den


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wide = pd.read_csv(P1 / "r4_trade_markouts_wide.csv")

    tr = pd.concat([load_trades(d) for d in DAYS], ignore_index=True)
    burst = (
        tr.groupby(["day", "timestamp"])
        .agg(
            n=("symbol", "count"),
            b=("buyer", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
            s=("seller", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
        )
        .reset_index()
    )
    m01m22 = burst[(burst["n"] >= 4) & (burst["b"] == "Mark 01") & (burst["s"] == "Mark 22")]
    centers = list(zip(m01m22["day"].astype(int), m01m22["timestamp"].astype(int)))

    def near_burst(d: int, t: int) -> bool:
        for bd, bt in centers:
            if int(bd) == int(d) and abs(int(bt) - int(t)) <= BURST_WIN:
                return True
        return False

    wide["near_m01_m22_burst"] = [near_burst(r["day"], r["timestamp"]) for _, r in wide.iterrows()]
    for k in (5, 20):
        col = f"mark_{k}_same"
        g = (
            wide.groupby(["symbol", "near_m01_m22_burst"])[col]
            .agg(n="count", mean="mean", std="std")
            .reset_index()
        )
        g.to_csv(OUT / f"r4_phase2_burst_window_mark{k}.csv", index=False)

    w530 = wide[(wide["symbol"] == "VEV_5300") & (wide["near_m01_m22_burst"])].copy()
    note530 = {
        "n": len(w530),
        "mean_m5": float(w530["mark_5_same"].mean()) if len(w530) else None,
        "mean_m20": float(w530["mark_20_same"].mean()) if len(w530) else None,
    }
    (OUT / "r4_phase2_vev5300_burst_meanrevert.txt").write_text(str(note530) + "\n", encoding="utf-8")

    # Microprice deviation (sample: last row per ts per symbol per day, EX + VEV_5300)
    micro_stats = []
    for d in DAYS:
        pr = load_prices(d)
        for sym in (EX, "VEV_5300", HY):
            sub = pr[pr["product"] == sym].sort_values("timestamp")
            if sub.empty:
                continue
            sub = sub.groupby("timestamp", as_index=False).last()
            sub["mp"] = sub.apply(
                lambda r: microprice(r["bid_price_1"], r["ask_price_1"], r["bid_volume_1"], r["ask_volume_1"]),
                axis=1,
            )
            sub["mid"] = sub["mid_price"]
            sub["dev"] = sub["mp"] - sub["mid"]
            sub["spr"] = sub["ask_price_1"] - sub["bid_price_1"]
            micro_stats.append(
                {
                    "day": d,
                    "symbol": sym,
                    "mean_abs_dev": float(sub["dev"].abs().mean()),
                    "median_spr": float(sub["spr"].median()),
                    "n": len(sub),
                }
            )
    pd.DataFrame(micro_stats).to_csv(OUT / "r4_phase2_microprice_summary_by_day_symbol.csv", index=False)

    # Extract: spread change vs subsequent 20-tick realized |mid| variation
    vol_rows = []
    for d in DAYS:
        pr = load_prices(d)
        ex = pr[pr["product"] == EX].sort_values("timestamp").groupby("timestamp", as_index=False).last()
        spr = (ex["ask_price_1"] - ex["bid_price_1"]).astype(float).to_numpy()
        mid = ex["mid_price"].astype(float).to_numpy()
        if len(spr) < 50:
            continue
        dspr = np.diff(spr)
        dmid = np.abs(np.diff(mid))
        for i in range(len(dspr) - 25):
            vol_rows.append({"day": d, "dspr": dspr[i], "rv20": float(np.sum(dmid[i + 1 : i + 21]))})
    vdf = pd.DataFrame(vol_rows)
    if len(vdf) > 100:
        vdf.to_csv(OUT / "r4_phase2_extract_spreadchange_vs_rv20.csv", index=False)
        vdf["q"] = pd.qcut(vdf["dspr"], q=5, duplicates="drop")
        vdf.groupby("q")["rv20"].agg(["mean", "count"]).reset_index().to_csv(
            OUT / "r4_phase2_extract_dspr_quintile_rv20.csv", index=False
        )

    # Signed flow per (day, ts, symbol) then lag-correlation with extract mid returns
    def signed_trade(r: pd.Series, bb: float, ba: float) -> int:
        p = float(r["price"])
        q = int(r["quantity"])
        if p >= float(ba) - 1e-9:
            return q
        if p <= float(bb) + 1e-9:
            return -q
        return 0

    sf_list = []
    for d in DAYS:
        pr = load_prices(d)
        t = load_trades(d)
        for _, r in t.iterrows():
            sym = str(r["symbol"])
            ts = int(r["timestamp"])
            sub = pr[(pr["product"] == sym) & (pr["timestamp"] <= ts)]
            if sub.empty:
                continue
            row = sub.sort_values("timestamp").iloc[-1]
            bb, ba = row["bid_price_1"], row["ask_price_1"]
            if pd.isna(bb) or pd.isna(ba):
                continue
            sg = signed_trade(r, float(bb), float(ba))
            if sg == 0:
                continue
            sf_list.append({"day": d, "timestamp": ts, "symbol": sym, "signed": sg})
    sf = pd.DataFrame(sf_list)
    sf = sf.groupby(["day", "timestamp", "symbol"], as_index=False)["signed"].sum()

    def mid_series(day: int, sym: str) -> pd.DataFrame:
        pr = load_prices(day)
        sub = pr[pr["product"] == sym].sort_values("timestamp").groupby("timestamp", as_index=False).last()
        return sub[["timestamp", "mid_price"]].rename(columns={"mid_price": "mid"})

    lag_out = []
    for d in DAYS:
        exs = mid_series(d, EX).sort_values("timestamp")
        exs["ret"] = exs["mid"].diff()
        for other in ("VEV_5300", HY):
            oth = mid_series(d, other).sort_values("timestamp")
            m = pd.merge_asof(exs, oth.rename(columns={"mid": "mid_o"}), on="timestamp", direction="backward")
            m["ret_o"] = m["mid_o"].diff()
            m = m.dropna(subset=["ret", "ret_o"])
            if len(m) < 50:
                continue
            for lag in (-3, -1, 0, 1, 3):
                if lag == 0:
                    x, y = m["ret"].to_numpy(), m["ret_o"].to_numpy()
                elif lag > 0:
                    x, y = m["ret"].iloc[lag:].to_numpy(), m["ret_o"].iloc[:-lag].to_numpy()
                else:
                    L = -lag
                    x, y = m["ret"].iloc[:-L].to_numpy(), m["ret_o"].iloc[L:].to_numpy()
                if len(x) < 40:
                    continue
                c = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan
                lag_out.append({"day": d, "other": other, "lag": lag, "corr": float(c), "n": len(x)})
    pd.DataFrame(lag_out).to_csv(OUT / "r4_phase2_extract_ret_lag_corr_other.csv", index=False)

    # IV proxy: (C - max(S-K,0))/S at each ts for VEV_5200 vs extract mid; residual vs daily median; |res| vs active marks
    iv_rows = []
    for d in DAYS:
        pr = load_prices(d)
        ex = pr[pr["product"] == EX].groupby("timestamp", as_index=False).last()
        v = pr[pr["product"] == "VEV_5200"].groupby("timestamp", as_index=False).last()
        m = pd.merge(ex[["timestamp", "mid_price"]], v[["timestamp", "mid_price"]], on="timestamp", suffixes=("_S", "_C"))
        m = m.dropna()
        S = m["mid_price_S"].astype(float)
        C = m["mid_price_C"].astype(float)
        K = 5200.0
        intr = (S - K).clip(lower=0.0)
        proxy = (C - intr) / S.replace(0, np.nan)
        m["iv_proxy"] = proxy
        med = float(m["iv_proxy"].median())
        m["iv_res"] = m["iv_proxy"] - med
        trd = load_trades(d)
        nmarks = trd.groupby("timestamp").apply(lambda g: len(set(g["buyer"]) | set(g["seller"])), include_groups=False)
        nm = nmarks.reindex(m["timestamp"]).fillna(1)
        m["n_marks"] = nm.values
        m["day"] = d
        iv_rows.append(m[["day", "timestamp", "iv_proxy", "iv_res", "n_marks"]])
    ivdf = pd.concat(iv_rows, ignore_index=True)
    ivdf.to_csv(OUT / "r4_phase2_iv_proxy_residual.csv", index=False)
    ivdf["abs_res"] = ivdf["iv_res"].abs()
    ivdf.groupby("n_marks")["abs_res"].agg(["mean", "count"]).reset_index().to_csv(
        OUT / "r4_phase2_iv_absres_by_active_mark_count.csv", index=False
    )

    # Adverse selection extract: buy_aggr vs seller 22/49
    sub = wide[(wide["aggressor"] == "buy_aggr") & (wide["seller"].isin(["Mark 22", "Mark 49"])) & (wide["symbol"] == EX)]
    if len(sub):
        sub.groupby("seller").agg(n=("mark_20_same", "count"), m5=("mark_5_same", "mean"), m20=("mark_20_same", "mean")).to_csv(
            OUT / "r4_phase2_adverse_extract_buyaggr_seller22_49.csv"
        )

    lines = ["=== Round 4 Phase 2 (automated) ===", ""]
    lines.append("VEV_5300 near M01→M22 burst window (±%d):" % BURST_WIN)
    for _, r in wide[(wide["symbol"] == "VEV_5300")].groupby("near_m01_m22_burst")["mark_20_same"].agg(["count", "mean"]).iterrows():
        lines.append(f"  near_burst={_} n={int(r['count'])} mean_m20={r['mean']:.4f}")
    lines.append("")
    lines.append("See r4_phase2_extract_ret_lag_corr_other.csv for lead-lag.")
    (OUT / "r4_phase2_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
