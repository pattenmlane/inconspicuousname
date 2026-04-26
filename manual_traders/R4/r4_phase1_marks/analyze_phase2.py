#!/usr/bin/env python3
"""
Round 4 Phase 2 — orthogonal edges (named-bot × burst proximity, microstructure,
cross-instrument same-timestamp flow, regime splits, lite smile proxy, pair adverse).

Prerequisite: Phase 1 complete. Reuses ``analyze_phase1.build_trade_enriched`` etc.

Run from repo root:
  python3 manual_traders/R4/r4_phase1_marks/analyze_phase2.py
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# HERE is the ``r4_phase1_marks`` directory → parents[2] == /workspace
REPO = HERE.parents[2]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
BURST_TIME_WINDOW = 500  # clock units: ±500 around burst timestamp


def load_p1():
    spec = importlib.util.spec_from_file_location("p1", HERE / "analyze_phase1.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def t_stat_welch(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0 or not np.isfinite(se):
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / se)


def burst_aggregates(tr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (day, ts), sub in tr.groupby(["day", "timestamp"]):
        syms = sub["symbol"].astype(str).tolist()
        rows.append(
            {
                "day": day,
                "timestamp": int(ts),
                "n_prints": len(sub),
                "n_sym": len(set(syms)),
                "buyer0": str(sub["buyer"].iloc[0]),
                "seller0": str(sub["seller"].iloc[0]),
                "bu_n": int(sub["buyer"].nunique()),
                "se_n": int(sub["seller"].nunique()),
            }
        )
    g = pd.DataFrame(rows)
    g["burst_ge4"] = g["n_prints"] >= 4
    g["type_01_22_multi_vev"] = (
        g["burst_ge4"]
        & (g["buyer0"] == "Mark 01")
        & (g["seller0"] == "Mark 22")
        & (g["n_sym"] >= 3)
    )
    return g


def attach_burst_proximity_fast(te: pd.DataFrame, bursts: pd.DataFrame, window: int) -> pd.DataFrame:
    te = te.copy()
    te["near_burst4"] = False
    te["near_01_22_vev_burst"] = False
    for day in DAYS:
        sub = te[te["day"] == day]
        b4_ts = bursts.loc[(bursts["day"] == day) & bursts["burst_ge4"], "timestamp"].astype(int).values
        b01_ts = bursts.loc[(bursts["day"] == day) & bursts["type_01_22_multi_vev"], "timestamp"].astype(
            int
        ).values
        if len(b4_ts):
            t = sub["timestamp"].astype(int).values
            te.loc[sub.index, "near_burst4"] = (np.abs(t[:, None] - b4_ts[None, :]) <= window).any(axis=1)
        if len(b01_ts):
            t = sub["timestamp"].astype(int).values
            te.loc[sub.index, "near_01_22_vev_burst"] = (np.abs(t[:, None] - b01_ts[None, :]) <= window).any(
                axis=1
            )
    return te


def mean_revert_vs_trend_5300(te: pd.DataFrame) -> None:
    sub = te[(te["symbol"] == "VEV_5300") & te["near_01_22_vev_burst"]]
    rest = te[(te["symbol"] == "VEV_5300") & ~te["near_01_22_vev_burst"]]
    rows = []
    for name, g in [("near_01_22_burst", sub), ("control", rest)]:
        x5 = g["fwd_same_5"].astype(float).values
        x20 = g["fwd_same_20"].astype(float).values
        ok = np.isfinite(x5) & np.isfinite(x20)
        if ok.sum() < 10:
            continue
        rows.append(
            {
                "bucket": name,
                "n": int(ok.sum()),
                "mean_fwd5": float(np.mean(x5[ok])),
                "mean_fwd20": float(np.mean(x20[ok])),
                "frac_same_sign_5_20": float(np.mean((x5[ok] * x20[ok]) > 0)),
                "frac_opp_sign_5_20": float(np.mean((x5[ok] * x20[ok]) < 0)),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "phase2_vev5300_mr_vs_trend_near_01_22_burst.csv", index=False)


def leave_one_day_burst_extract(te: pd.DataFrame, tr: pd.DataFrame) -> None:
    cnt = tr.groupby(["day", "timestamp"]).size().reset_index(name="burst_n")
    m = te.merge(cnt, on=["day", "timestamp"], how="left")
    rows = []
    for holdout in DAYS:
        test = m[m["day"] == holdout]
        x1 = test.loc[test["burst_n"] >= 4, "fwd_EXTRACT_20"].astype(float).dropna().values
        x0 = test.loc[test["burst_n"] < 4, "fwd_EXTRACT_20"].astype(float).dropna().values
        if len(x1) >= 5 and len(x0) >= 5:
            rows.append(
                {
                    "holdout_day": holdout,
                    "n_burst": len(x1),
                    "n_non": len(x0),
                    "mean_burst": float(np.mean(x1)),
                    "mean_non": float(np.mean(x0)),
                    "welch_t_burst_minus_non": t_stat_welch(x1, x0),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "phase2_leave_one_day_burst_extract_welch.csv", index=False)


def microprice_spread_vol(p1) -> None:
    """Bar-level VEV_5300: corr(spread, sum of |Δmid| over next 5 bar-to-bar moves)."""
    rows = []
    for day in DAYS:
        _, pack = p1.load_prices_day(day)
        ts = pack["ts"]
        L = len(ts)
        df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
        sub = df[df["product"] == "VEV_5300"].set_index("timestamp").reindex(ts)
        bid = sub["bid_price_1"].astype(float).values
        ask = sub["ask_price_1"].astype(float).values
        vb = sub["bid_volume_1"].fillna(0).astype(float).values
        va = sub["ask_volume_1"].fillna(0).astype(float).values
        mid = sub["mid_price"].astype(float).values
        den = vb + va
        mp = np.where(den > 0, (bid * va + ask * vb) / den, mid)
        mp_skew = mp - mid
        sp = ask - bid
        abs_d = np.abs(np.diff(mid))
        # sum(abs_d[i:i+5]) for i = 0 .. L-6
        if len(abs_d) >= 5:
            vol5 = np.convolve(abs_d, np.ones(5), mode="valid")
            for i in range(len(vol5)):
                if i + 5 >= L:
                    break
                rows.append(
                    {
                        "day": day,
                        "spread": float(sp[i]) if np.isfinite(sp[i]) else np.nan,
                        "mp_skew": float(mp_skew[i]) if np.isfinite(mp_skew[i]) else np.nan,
                        "sum_abs_dmid_next5": float(vol5[i]),
                    }
                )
    bar = pd.DataFrame(rows).dropna(subset=["spread", "sum_abs_dmid_next5"])
    if len(bar) > 50:
        c = np.corrcoef(bar["spread"], bar["sum_abs_dmid_next5"])[0, 1]
        (OUT / "phase2_spread_vs_fwd_vol_vev5300_corr.txt").write_text(
            f"corr(spread_ticks, sum_abs_dmid_next5 on VEV_5300 bars): {c:.4f} n={len(bar)}\n",
            encoding="utf-8",
        )


def cross_lead_lag_bar_grid(p1, te: pd.DataFrame) -> None:
    """Aggregate signed trade flow onto price-bar index; correlate extract flow with
    future VEV_5300 mid change (and vice versa) at bar lags 0..10."""
    te = te.copy()
    te["signed"] = np.where(
        te["side"] == "aggr_buy",
        te["qty"],
        np.where(te["side"] == "aggr_sell", -te["qty"], 0),
    )
    rows = []
    for day in DAYS:
        _, pack = p1.load_prices_day(day)
        ts = pack["ts"]
        ts_to_i = pack["ts_to_i"]
        L = len(ts)
        mids = pack["mids"]
        j_ex = p1.ALL_PRODUCTS.index("VELVETFRUIT_EXTRACT")
        j_53 = p1.ALL_PRODUCTS.index("VEV_5300")
        dmid_ex = np.diff(mids["VELVETFRUIT_EXTRACT"], prepend=np.nan)
        dmid_53 = np.diff(mids["VEV_5300"], prepend=np.nan)

        f_ex = np.zeros(L)
        f_53 = np.zeros(L)
        sub = te[te["day"] == day]
        for _, r in sub.iterrows():
            sym = str(r["symbol"])
            ts_ = int(r["timestamp"])
            if ts_ not in ts_to_i:
                continue
            ix = int(ts_to_i[ts_])
            if sym == "VELVETFRUIT_EXTRACT":
                f_ex[ix] += float(r["signed"])
            elif sym == "VEV_5300":
                f_53[ix] += float(r["signed"])

        for lag in range(0, 11):
            if lag >= L:
                break
            tgt = dmid_53[lag:]
            src = f_ex[: len(tgt)]
            ok = np.isfinite(tgt) & np.isfinite(src)
            if ok.sum() > 30:
                c = np.corrcoef(src[ok], tgt[ok])[0, 1]
                rows.append(
                    {
                        "day": day,
                        "lead": "extract_flow_bar",
                        "target": "VEV_5300_dmid",
                        "lag_bars": lag,
                        "n": int(ok.sum()),
                        "corr": float(c),
                    }
                )
            tgt = dmid_ex[lag:]
            src = f_53[: len(tgt)]
            ok = np.isfinite(tgt) & np.isfinite(src)
            if ok.sum() > 30:
                c = np.corrcoef(src[ok], tgt[ok])[0, 1]
                rows.append(
                    {
                        "day": day,
                        "lead": "vev5300_flow_bar",
                        "target": "EXTRACT_dmid",
                        "lag_bars": lag,
                        "n": int(ok.sum()),
                        "corr": float(c),
                    }
                )
    pd.DataFrame(rows).to_csv(OUT / "phase2_cross_leadlag_bar_grid_flow.csv", index=False)


def attach_5300_spread_at_ts(p1, te: pd.DataFrame) -> pd.DataFrame:
    """BBO spread ticks for VEV_5300 at each trade's (day, timestamp) from prices."""
    rows = []
    for day in DAYS:
        _, pack = p1.load_prices_day(day)
        ts = pack["ts"]
        bid1 = pack["bids"]["VEV_5300"]
        ask1 = pack["asks"]["VEV_5300"]
        sp = ask1 - bid1
        for t, s in zip(ts.astype(int), sp):
            rows.append({"day": day, "timestamp": int(t), "spread5300": float(s)})
    spdf = pd.DataFrame(rows)
    out = te.merge(spdf, on=["day", "timestamp"], how="left")
    out["tight5300_book"] = out["spread5300"] <= 2
    return out


def regime_tight5300_x_burst(te: pd.DataFrame, tr: pd.DataFrame, p1) -> None:
    cnt = tr.groupby(["day", "timestamp"]).size().reset_index(name="burst_n")
    m = attach_5300_spread_at_ts(p1, te).merge(cnt, on=["day", "timestamp"], how="left")
    rows = []
    for tight in (True, False):
        for burst in (True, False):
            g = m[(m["tight5300_book"] == tight) & ((m["burst_n"] >= 4) == burst)]
            x = g["fwd_EXTRACT_20"].astype(float).dropna().values
            if len(x) < 10:
                continue
            rows.append(
                {
                    "tight5300_book_at_ts": tight,
                    "burst_ge4_same_ts": burst,
                    "n": len(x),
                    "mean_fwd_EXTRACT_20": float(np.mean(x)),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "phase2_regime_tight5300_x_burst_extract_fwd20.csv", index=False)


def smile_lite_tension_by_pair(tr: pd.DataFrame, p1) -> None:
    rows_vol = []
    vevs = [c for c in p1.ALL_PRODUCTS if c.startswith("VEV_")]
    for day in DAYS:
        df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
        piv = df[df["product"].isin(vevs)].pivot_table(
            index="timestamp", columns="product", values="mid_price", aggfunc="first"
        )
        tension = piv.std(axis=1).rolling(20, min_periods=5).mean()
        tension = tension.reset_index()
        tension.columns = ["timestamp", "smile_tension_proxy"]
        tension["day"] = day
        dtr = tr[tr["day"] == day].merge(tension, on=["day", "timestamp"], how="left")
        for (b, s), g in dtr.groupby(["buyer", "seller"]):
            if len(g) < 15:
                continue
            rows_vol.append(
                {
                    "day": day,
                    "buyer": b,
                    "seller": s,
                    "n": len(g),
                    "mean_smile_tension_proxy": float(np.nanmean(g["smile_tension_proxy"].values)),
                }
            )
    pd.DataFrame(rows_vol).to_csv(OUT / "phase2_smile_lite_tension_by_pair.csv", index=False)


def pair_01_22_passive_seller_fwd(te: pd.DataFrame) -> None:
    sub = te[(te["buyer"] == "Mark 01") & (te["seller"] == "Mark 22") & (te["side"] == "aggr_buy")]
    rows = []
    for sym in sub["symbol"].unique():
        g = sub[sub["symbol"] == sym]
        for K in (5, 20):
            col = f"fwd_same_{K}"
            x = g[col].astype(float).dropna().values
            if len(x) < 10:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "K": K,
                    "n": len(x),
                    "mean_minus_fwd": float(np.mean(-x)),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "phase2_01_22_aggr_buy_passive_seller_fwd.csv", index=False)


def m22_pressure_extract(te: pd.DataFrame) -> None:
    ex = te[te["symbol"] == "VELVETFRUIT_EXTRACT"].sort_values(["day", "timestamp"])
    ex["m22_signed"] = np.where(
        ex["buyer"] == "Mark 22",
        ex["qty"],
        np.where(ex["seller"] == "Mark 22", -ex["qty"], 0),
    )
    rows = []
    for day in DAYS:
        g = ex[ex["day"] == day]
        if len(g) < 30:
            continue
        win = 10
        s = g["m22_signed"].values
        cs = np.cumsum(np.insert(s, 0, 0))
        roll = cs[win:] - cs[:-win]
        fwd = g["fwd_EXTRACT_20"].astype(float).values[win:]
        m = min(len(roll), len(fwd))
        roll, fwd = roll[:m], fwd[:m]
        ok = np.isfinite(fwd)
        if ok.sum() < 20:
            continue
        c = np.corrcoef(roll[ok], fwd[ok])[0, 1]
        rows.append({"day": day, "corr_roll10_m22_net_vs_fwdEXTRACT20": float(c), "n": int(ok.sum())})
    pd.DataFrame(rows).to_csv(OUT / "phase2_m22_inventory_pressure_extract_corr.csv", index=False)


def pair_x_burst_markouts(te: pd.DataFrame, bursts: pd.DataFrame) -> None:
    te = attach_burst_proximity_fast(te, bursts, BURST_TIME_WINDOW)
    rows = []
    for b, s in [("Mark 01", "Mark 22"), ("Mark 14", "Mark 38"), ("Mark 55", "Mark 01")]:
        for tag, mask in [
            ("near_any_ge4", te["near_burst4"]),
            ("near_01_22_vev", te["near_01_22_vev_burst"]),
            ("far_from_ge4", ~te["near_burst4"]),
        ]:
            sub = te[(te["buyer"] == b) & (te["seller"] == s) & mask]
            if len(sub) < 10:
                continue
            for sym in sorted(sub["symbol"].unique())[:8]:
                g = sub[sub["symbol"] == sym]
                x = g["fwd_same_20"].astype(float).dropna().values
                if len(x) < 6:
                    continue
                rows.append(
                    {
                        "buyer": b,
                        "seller": s,
                        "burst_bucket": tag,
                        "symbol": sym,
                        "n": len(x),
                        "mean_fwd20": float(np.mean(x)),
                        "frac_pos": float(np.mean(x > 0)),
                    }
                )
    pd.DataFrame(rows).to_csv(OUT / "phase2_pair_x_burst_bucket_markout.csv", index=False)


def main() -> None:
    p1 = load_p1()
    print("Phase 2: building enriched trades...", flush=True)
    te = p1.build_trade_enriched()
    tr = pd.concat(
        [pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d) for d in DAYS],
        ignore_index=True,
    )
    bursts = burst_aggregates(tr)
    bursts.to_csv(OUT / "phase2_burst_metadata_by_timestamp.csv", index=False)

    pair_x_burst_markouts(te.copy(), bursts)
    mean_revert_vs_trend_5300(attach_burst_proximity_fast(te.copy(), bursts, BURST_TIME_WINDOW))
    leave_one_day_burst_extract(te, tr)
    microprice_spread_vol(p1)
    cross_lead_lag_bar_grid(p1, te)
    regime_tight5300_x_burst(te, tr, p1)
    smile_lite_tension_by_pair(tr, p1)
    pair_01_22_passive_seller_fwd(te)
    m22_pressure_extract(te)

    (OUT / "phase2_README.txt").write_text(
        f"BURST_TIME_WINDOW={BURST_TIME_WINDOW} clock units for proximity flags.\n",
        encoding="utf-8",
    )
    print(f"Done. Outputs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
