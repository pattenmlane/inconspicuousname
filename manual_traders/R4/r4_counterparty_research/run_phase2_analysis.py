#!/usr/bin/env python3
"""Round 4 Phase 2 — orthogonal to Phase 1 summaries: burst typing, microprice/spread vs U,
signed-flow lead–lag, per-day stability, simple BBO IV + Mark conditioning on VEV_5300."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
DAYS = (1, 2, 3)
TICK = 100
KS = (5, 20, 100)


def ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(s: float, k: float, t: float, sig: float) -> float:
    if t <= 1e-12 or sig <= 1e-12:
        return max(s - k, 0.0)
    v = sig * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sig * sig * t) / v
    d2 = d1 - v
    return s * ncdf(d1) - k * ncdf(d2)


def iv(mid: float, s: float, k: float, t: float) -> float | None:
    intr = max(s - k, 0.0)
    if mid <= intr + 1e-6 or mid >= s - 1e-6:
        return None
    lo, hi = 1e-4, 12.0
    if bs_call(s, k, t, lo) - mid > 0 or bs_call(s, k, t, hi) - mid < 0:
        return None
    for _ in range(30):
        m = 0.5 * (lo + hi)
        if bs_call(s, k, t, m) >= mid:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def dte_years(day: int, ts: int) -> float:
    d_open = 7.0 - float(day - 1)
    intra = (int(ts) // 100) / 10_000.0
    return max(d_open - intra, 1e-6) / 365.0


def load_prices(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    df = df.loc[df["day"] == day].copy()
    b = pd.to_numeric(df["bid_price_1"], errors="coerce")
    a = pd.to_numeric(df["ask_price_1"], errors="coerce")
    bv = pd.to_numeric(df["bid_volume_1"], errors="coerce").fillna(0)
    av = pd.to_numeric(df["ask_volume_1"], errors="coerce").fillna(0)
    df["spread"] = (a - b).astype(float)
    df["mid"] = pd.to_numeric(df["mid_price"], errors="coerce")
    den = bv + av
    df["micro"] = np.where(
        den > 0,
        (b * av + a * bv) / den,
        df["mid"],
    )
    df["imb"] = (bv - av) / (den + 1e-9)
    return df


def load_trades(day: int) -> pd.DataFrame:
    return pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")


def aggro(p: float, bid: float, ask: float) -> str:
    if p >= ask:
        return "buy"
    if p <= bid:
        return "sell"
    return "mid"


def main() -> None:
    pr_parts = []
    tr_parts = []
    for d in DAYS:
        pr = load_prices(d)
        pr["day"] = d
        pr_parts.append(pr)
        tr = load_trades(d)
        tr["day"] = d
        tr_parts.append(tr)
    pr = pd.concat(pr_parts, ignore_index=True)
    tr = pd.concat(tr_parts, ignore_index=True)

    # --- U timeline: micro, spread, fwd abs return ---
    u = pr[pr["product"] == "VELVETFRUIT_EXTRACT"].sort_values(["day", "timestamp"])
    u["d_mid"] = u.groupby("day")["mid"].diff()
    u["abs_d1"] = u.groupby("day")["d_mid"].transform(lambda s: s.shift(-1).abs())
    u["spr_chg"] = u.groupby("day")["spread"].diff()
    lines_u = ["U microprice/spread vs forward |d_mid| (1 tick ahead, pooled days 1-3)"]
    for reg, lab in [("all", "all")]:
        sub = u
        if len(sub) > 100:
            c1 = sub["spread"].corr(sub["abs_d1"])
            c2 = sub["micro"].corr(sub["mid"])
            c3 = sub["spread"].corr(sub["abs_d1"].shift(1))
            lines_u.append(
                f"corr(spread_t, |dU|_t+1)={c1:.4f}  corr(micro-mid, |dU|_t+1)={(sub['micro']-sub['mid']).corr(sub['abs_d1']):.4f} n={len(sub)}"
            )
    (OUT / "r4_phase2_microprice_spread_vs_U.txt").write_text("\n".join(lines_u), encoding="utf-8")

    # --- Signed aggressive flow on extract ---
    m = tr.merge(
        pr[pr["product"] == "VELVETFRUIT_EXTRACT"][
            ["day", "timestamp", "bid_price_1", "ask_price_1"]
        ].rename(columns={"timestamp": "timestamp"}),
        on=["day", "timestamp"],
        how="inner",
        validate="m:1",
    )
    m = m[m["symbol"] == "VELVETFRUIT_EXTRACT"]
    m["ag"] = [
        aggro(float(p), float(b), float(a))
        for p, b, a in zip(m["price"], m["bid_price_1"], m["ask_price_1"])
    ]
    m["sgn"] = np.where(m["ag"] == "buy", m["quantity"], np.where(m["ag"] == "sell", -m["quantity"], 0))
    flow = m.groupby(["day", "timestamp"])["sgn"].sum().reset_index(name="flow_U")

    u2 = u.merge(flow, on=["day", "timestamp"], how="left").fillna({"flow_U": 0.0})
    u2["fwd_d"] = u2.groupby("day")["d_mid"].shift(-1)

    lag_rows = []
    for L in range(0, 16):
        u2[f"du_L{L}"] = u2.groupby("day")["d_mid"].shift(-L)
        sub = u2[["flow_U", f"du_L{L}"]].dropna()
        if len(sub) > 200:
            lag_rows.append(
                {
                    "lag_ticks": L,
                    "corr_flow_future_dU": float(sub["flow_U"].corr(sub[f"du_L{L}"])),
                    "n": int(len(sub)),
                }
            )
    pd.DataFrame(lag_rows).to_csv(OUT / "r4_phase2_signed_flow_leadlag_U.csv", index=False)

    # --- Cross: hydro d_mid vs lagged U d_mid ---
    h = pr[pr["product"] == "HYDROGEL_PACK"].sort_values(["day", "timestamp"])
    h["d_h"] = h.groupby("day")["mid"].diff()
    u3 = u[["day", "timestamp", "d_mid"]].rename(columns={"d_mid": "d_u"})
    xh = h.merge(u3, on=["day", "timestamp"], how="inner")
    xrows = []
    for L in range(0, 11):
        xh[f"du_l{L}"] = xh.groupby("day")["d_u"].shift(L)
        sub = xh[["d_h", f"du_l{L}"]].dropna()
        if len(sub) > 500:
            xrows.append(
                {
                    "lag": L,
                    "corr_dHydro_dUlag": float(sub["d_h"].corr(sub[f"du_l{L}"])),
                    "n": len(sub),
                }
            )
    pd.DataFrame(xrows).to_csv(OUT / "r4_phase2_hydro_vs_U_lagcorr.csv", index=False)

    # --- Burst typing + post-burst U markout ---
    burst_n = tr.groupby(["day", "timestamp"]).size().rename("n").reset_index()
    burst_n = burst_n[burst_n["n"] >= 4]
    tr_b = tr.merge(burst_n, on=["day", "timestamp"])
    tr_b["is_01_22"] = ((tr_b["buyer"] == "Mark 01") & (tr_b["seller"] == "Mark 22")).astype(int)
    gburst = tr_b.groupby(["day", "timestamp"], as_index=False)
    has_01_22 = gburst["is_01_22"].max().rename(columns={"is_01_22": "has_01_22"})
    all_01_22 = gburst["is_01_22"].min().rename(columns={"is_01_22": "all_rows_01_22"})
    all_01_22["all_rows_01_22"] = all_01_22["all_rows_01_22"].astype(bool)
    has_01_22["has_01_22"] = has_01_22["has_01_22"].astype(bool)
    burst_type = burst_n.merge(has_01_22, on=["day", "timestamp"]).merge(all_01_22, on=["day", "timestamp"])

    arrs: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for (d, sym), g in pr.groupby(["day", "product"]):
        g = g.sort_values("timestamp")
        arrs[(int(d), str(sym))] = (
            g["timestamp"].to_numpy(dtype=np.int64),
            g["mid"].to_numpy(dtype=np.float64),
        )

    def fwd_u(day: int, ts: int, k: int) -> float | None:
        tss, mids = arrs.get((day, "VELVETFRUIT_EXTRACT"), (np.array([]), np.array([])))
        if tss.size == 0:
            return None
        tgt = ts + k * TICK
        i = np.searchsorted(tss, tgt, side="left")
        if i >= len(tss):
            return None
        i0 = np.searchsorted(tss, ts, side="left")
        if i0 >= len(tss):
            return None
        return float(mids[i] - mids[i0])

    def burst_stats(bt: pd.DataFrame, label: str) -> dict:
        mus = {k: [] for k in KS}
        for _, r in bt.iterrows():
            d, ts = int(r["day"]), int(r["timestamp"])
            for k in KS:
                v = fwd_u(d, ts, k)
                if v is not None:
                    mus[k].append(v)
        return {
            "label": label,
            "n_events": len(bt),
            **{f"mean_fwd_U_{k}": float(np.mean(mus[k])) if mus[k] else None for k in KS},
        }

    stats = [
        burst_stats(burst_type[burst_type["all_rows_01_22"]], "burst>=4 AND all prints are Mark01->Mark22"),
        burst_stats(
            burst_type[burst_type["has_01_22"] & (~burst_type["all_rows_01_22"])],
            "burst>=4 mixed (has 01->22 but not exclusively)",
        ),
        burst_stats(burst_type[~burst_type["has_01_22"]], "burst>=4 no Mark01->Mark22 print"),
    ]
    # control: random timestamps sample same n as burst - use non-burst ts from U grid
    ctrl_ts = u.merge(burst_n[["day", "timestamp"]], on=["day", "timestamp"], how="left", indicator=True)
    ctrl_ts = ctrl_ts[ctrl_ts["_merge"] == "left_only"].drop(columns="_merge")
    ctrl_sample = ctrl_ts.sample(min(len(burst_type), max(500, len(burst_type))), random_state=0)
    ctrl_bt = ctrl_sample[["day", "timestamp"]].drop_duplicates()
    stats.append(burst_stats(ctrl_bt, "control sample non-burst U rows"))

    pd.DataFrame(stats).to_csv(OUT / "r4_phase2_burst_type_U_fwd.csv", index=False)

    # --- Per-day stability: burst mean U fwd20 ---
    stab = []
    for d in DAYS:
        bt_d = burst_type[burst_type["day"] == d]
        vals = []
        for _, r in bt_d.iterrows():
            v = fwd_u(int(r["day"]), int(r["timestamp"]), 20)
            if v is not None:
                vals.append(v)
        stab.append({"day": d, "n_burst": len(bt_d), "mean_U_fwd20_burst": float(np.mean(vals)) if vals else None})
    pd.DataFrame(stab).to_csv(OUT / "r4_phase2_burst_U_fwd20_by_day.csv", index=False)

    # --- VEV_5300 IV when Mark01->Mark22 print vs other prints (same ts merge U mid) ---
    tr53 = tr[tr["symbol"] == "VEV_5300"].merge(
        pr[pr["product"] == "VELVETFRUIT_EXTRACT"][["day", "timestamp", "mid"]].rename(
            columns={"mid": "u_mid"}
        ),
        on=["day", "timestamp"],
    )
    tr53 = tr53.merge(
        pr[pr["product"] == "VEV_5300"][["day", "timestamp", "mid"]].rename(columns={"mid": "v_mid"}),
        on=["day", "timestamp"],
    )
    ivs_01 = []
    ivs_ot = []
    for _, r in tr53.iterrows():
        t = dte_years(int(r["day"]), int(r["timestamp"]))
        sig = iv(float(r["v_mid"]), float(r["u_mid"]), 5300.0, t)
        if sig is None:
            continue
        if r["buyer"] == "Mark 01" and r["seller"] == "Mark 22":
            ivs_01.append(sig)
        else:
            ivs_ot.append(sig)
    ivsum = {
        "n_01_22": len(ivs_01),
        "median_iv_01_22": float(np.median(ivs_01)) if ivs_01 else None,
        "n_other": len(ivs_ot),
        "median_iv_other": float(np.median(ivs_ot)) if ivs_ot else None,
    }
    (OUT / "r4_phase2_vev5300_iv_by_mark.json").write_text(json.dumps(ivsum, indent=2), encoding="utf-8")

    # --- Mark22 sell markout stratified by burst (option symbols) ---
    tr22 = tr[(tr["seller"] == "Mark 22") & (tr["symbol"].str.startswith("VEV_"))].merge(
        burst_n[["day", "timestamp", "n"]], on=["day", "timestamp"], how="left"
    )
    tr22["in_burst"] = tr22["n"].fillna(0) >= 4
    # merge mids for symbol
    tr22 = tr22.merge(
        pr.rename(columns={"product": "symbol", "mid": "m0", "bid_price_1": "bd", "ask_price_1": "ak"}),
        on=["day", "timestamp", "symbol"],
    )
    out_m = []
    for _, r in tr22.iterrows():
        d, sym, ts = int(r["day"]), str(r["symbol"]), int(r["timestamp"])
        tss, mids = arrs.get((d, sym), (np.array([]), np.array([])))
        i0 = np.searchsorted(tss, ts, side="left")
        tgt = ts + 20 * TICK
        i1 = np.searchsorted(tss, tgt, side="left")
        if i0 < len(tss) and i1 < len(tss):
            out_m.append(
                {
                    "in_burst": bool(r["in_burst"]),
                    "mark20": float(mids[i1] - mids[i0]),
                }
            )
    mm = pd.DataFrame(out_m)
    if len(mm):
        g = mm.groupby("in_burst")["mark20"].agg(["mean", "count"]).reset_index()
        g.to_csv(OUT / "r4_phase2_mark22_sell_vev_mark20_burst_vs_not.csv", index=False)

    print("phase2 outputs written to", OUT)


if __name__ == "__main__":
    main()
