"""
Round 4 Phase 2 — burst-conditioned stats, lead–lag signed flow, microprice/regime splits,
Mark-conditioned residuals (orthogonal to unconditioned Phase-1 tables).

Outputs: manual_traders/R4/r4_counterparty_gated/analysis_outputs/r4_phase2_*.csv/json/txt
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"

# Reuse phase1 helpers by importing the module
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_p1", Path(__file__).resolve().parent / "run_r4_phase1_analysis.py"
)
_p1 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_p1)


def burst_type(row: pd.Series) -> str:
    if int(row.get("burst_n", 1)) < 4:
        return "no_burst"
    b, s = str(row["buyer"]), str(row["seller"])
    if b == "Mark 01" and s == "Mark 22":
        return "burst_M01_M22"
    return "burst_other"


def signed_flow(tr: pd.DataFrame, fwd: dict) -> pd.DataFrame:
    """Per (day, timestamp, symbol): sum qty*(+1 buy aggr, -1 sell aggr)."""
    rows = []
    for _, r in tr.iterrows():
        day, sym, ts = int(r["day"]), str(r["symbol"]), int(r["timestamp"])
        key = (day, sym)
        if key not in fwd:
            continue
        st = fwd[key]
        i0 = _p1.idx_at_or_before(st["ts"], ts)
        bid1, ask1 = float(st["bid1"][i0]), float(st["ask1"][i0])
        pr = float(r["price"]) if pd.notna(r["price"]) else float("nan")
        ag = _p1.classify_aggression(pr, bid1, ask1)
        q = int(r["quantity"])
        sgn = 0
        if ag == "aggr_buy":
            sgn = q
        elif ag == "aggr_sell":
            sgn = -q
        rows.append({"day": day, "timestamp": ts, "symbol": sym, "signed": sgn})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["day", "timestamp", "symbol"], as_index=False)["signed"].sum()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    px_raw = _p1.load_prices()
    px = _p1.price_features(px_raw)
    tr = _p1.load_trades()
    tr["qty"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)
    burst_sz = tr.groupby(["day", "timestamp"]).size().rename("burst_n")
    tr = tr.merge(burst_sz, on=["day", "timestamp"], how="left")

    fwd_idx = _p1.build_forward_index(px)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")

    # --- Phase 2.1: burst-conditioned participant stats (reuse phase1 event table logic) ---
    ev_rows = []
    for _, r in tr.iterrows():
        day, sym, ts = int(r["day"]), str(r["symbol"]), int(r["timestamp"])
        key = (day, sym)
        if key not in fwd_idx:
            continue
        st = fwd_idx[key]
        i0 = _p1.idx_at_or_before(st["ts"], ts)
        mid0 = float(st["mid"][i0])
        j = i0 + 20
        if j >= len(st["mid"]):
            continue
        fwd20 = float(st["mid"][j]) - mid0
        bid1, ask1 = float(st["bid1"][i0]), float(st["ask1"][i0])
        ag = _p1.classify_aggression(float(r["price"]), bid1, ask1)
        ev_rows.append(
            {
                "day": day,
                "timestamp": ts,
                "symbol": sym,
                "buyer": str(r["buyer"]),
                "seller": str(r["seller"]),
                "pair": f"{r['buyer']}->{r['seller']}",
                "aggression": ag,
                "burst_n": int(r["burst_n"]),
                "burst_type": burst_type(r),
                "fwd_mid_20": fwd20,
            }
        )
    ev = pd.DataFrame(ev_rows)

    burst_cond = (
        ev.groupby(["burst_type", "buyer", "seller", "symbol"])["fwd_mid_20"]
        .agg(n="count", mean="mean")
        .reset_index()
        .query("n >= 10")
        .sort_values("mean", ascending=False)
    )
    burst_cond.to_csv(OUT / "r4_phase2_burst_conditioned_fwd20.csv", index=False)

    # Mark 67 extract aggr buy: burst vs no burst
    m67 = ev[
        (ev["buyer"] == "Mark 67")
        & (ev["symbol"] == "VELVETFRUIT_EXTRACT")
        & (ev["aggression"] == "aggr_buy")
    ]
    m67_sum = m67.groupby(m67["burst_type"].apply(lambda x: "burst" if x != "no_burst" else "no_burst"))[
        "fwd_mid_20"
    ].agg(["mean", "count"])
    (OUT / "r4_phase2_mark67_extract_burst_split.json").write_text(
        m67_sum.to_json(indent=2), encoding="utf-8"
    )

    # --- Phase 2.3: lead-lag signed flow vs extract fwd ---
    sf = signed_flow(tr, fwd_idx)
    ext = sf[sf["symbol"] == "VELVETFRUIT_EXTRACT"].rename(columns={"signed": "sf_ext"})
    lags = []
    for lag in (0, 1, 2, 3, 5):
        ext_sorted = ext.sort_values(["day", "timestamp"])
        ext_sorted["sf_ext_lag"] = ext_sorted.groupby("day")["sf_ext"].shift(lag)
        m2 = ev[ev["symbol"] == "VELVETFRUIT_EXTRACT"].merge(
            ext_sorted[["day", "timestamp", "sf_ext_lag"]], on=["day", "timestamp"], how="inner"
        )
        x = m2["sf_ext_lag"].to_numpy(dtype=float)
        y = m2["fwd_mid_20"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 50:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(x[m], y[m])[0, 1])
        lags.append({"lag_ticks": lag, "corr_signed_flow_ext_fwd20": corr, "n": int(m.sum())})
    pd.DataFrame(lags).to_csv(OUT / "r4_phase2_leadlag_signed_flow_extract.csv", index=False)

    # --- Phase 2.2 + 2.4: microprice vs mid, spread regime ---
    px_u = px[px["product"] == "VELVETFRUIT_EXTRACT"].copy()
    sub = px_u.merge(
        px_raw[px_raw["product"] == "VELVETFRUIT_EXTRACT"][
            ["day", "timestamp", "bid_volume_1", "ask_volume_1"]
        ],
        on=["day", "timestamp"],
        how="left",
    )
    bv = pd.to_numeric(sub["bid_volume_1"], errors="coerce").fillna(0)
    av = pd.to_numeric(sub["ask_volume_1"], errors="coerce").fillna(0)
    den = bv + av
    sub["micro"] = np.where(
        den > 0, (sub["ask1"] * bv + sub["bid1"] * av) / den, sub["mid"]
    )
    sub["micro_minus_mid"] = sub["micro"] - sub["mid"]
    sub["spread"] = sub["spread"]
    sub["spread_tight"] = (sub["spread"] <= 2).astype(int)
    sub = sub.sort_values(["day", "timestamp"])
    sub["fwd_mid_20"] = sub.groupby("day")["mid"].transform(lambda s: s.shift(-20) - s)
    reg = (
        sub.groupby(["spread_tight"])["fwd_mid_20"]
        .agg(n="count", mean="mean", std="std")
        .reset_index()
    )
    reg.to_csv(OUT / "r4_phase2_extract_fwd20_by_spread_tight.csv", index=False)
    sub[["day", "timestamp", "micro_minus_mid", "spread", "spread_tight"]].to_csv(
        OUT / "r4_phase2_extract_microprice_timeseries.csv", index=False
    )

    # --- Phase 2.5: residuals within burst_M01_M22 only ---
    base = (
        ev.groupby(["buyer", "seller", "symbol"])["fwd_mid_20"]
        .mean()
        .rename("global_mean")
        .reset_index()
    )
    evb = ev[ev["burst_type"] == "burst_M01_M22"].merge(base, on=["buyer", "seller", "symbol"], how="left")
    evb["resid"] = evb["fwd_mid_20"] - evb["global_mean"]
    evb.groupby(["buyer", "seller", "symbol"])["resid"].agg(["mean", "count"]).reset_index().to_csv(
        OUT / "r4_phase2_residual_fwd20_within_M01_M22_burst.csv", index=False
    )

    # Summary
    lines = [
        "Round 4 Phase 2 summary",
        "=======================",
        burst_cond.head(20).to_string(index=False),
        "",
        "Mark67 extract aggr_buy burst split:",
        (OUT / "r4_phase2_mark67_extract_burst_split.json").read_text(),
        "",
        "Lead-lag signed flow extract:",
        pd.DataFrame(lags).to_string(index=False),
        "",
        "Extract fwd20 by spread<=2:",
        reg.to_string(index=False),
    ]
    (OUT / "r4_phase2_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote phase2 outputs to", OUT)


if __name__ == "__main__":
    main()
