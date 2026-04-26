"""
Round 4 Phase 3 — Sonic joint gate (VEV_5200 & VEV_5300 BBO spread <= 2 same timestamp)
on R4 tape + spread–spread / spread–price correlations + counterparty stats gated vs loose.

Convention matches round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py:
inner join 5200, 5300, extract on timestamp; tight = (s5200<=TH)&(s5300<=TH); forward extract
mid = K rows forward on aligned index (same as R3 shift(-K) on panel sorted by time).

Outputs: manual_traders/R4/r4_counterparty_gated/analysis_outputs/r4_phase3_*
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
TH = 2
K_LIST = (5, 20, 100)
VEV_5200, VEV_5300 = "VEV_5200", "VEV_5300"
EXTRACT = "VELVETFRUIT_EXTRACT"

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_p1", Path(__file__).resolve().parent / "run_r4_phase1_analysis.py"
)
_p1 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_p1)


def one_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
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


def aligned_panel_day(day: int, df: pd.DataFrame) -> pd.DataFrame:
    a = one_product(df, VEV_5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = one_product(df, VEV_5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = one_product(df, EXTRACT).rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(
        e[["timestamp", "m_ext", "s_ext"]], on="timestamp", how="inner"
    )
    m = m.sort_values("timestamp").reset_index(drop=True)
    m["day"] = day
    return m


def add_gate_and_fwd(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy()
    out["joint_tight"] = (out["s5200"] <= TH) & (out["s5300"] <= TH)
    for k in K_LIST:
        out[f"m_ext_f{k}"] = out["m_ext"].shift(-k)
        out[f"fwd_ext_{k}"] = out[f"m_ext_f{k}"] - out["m_ext"]
    out["dm_ext"] = out["m_ext"].diff()
    return out


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return None
    x, y = a[m], b[m]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = []
    for p in sorted(DATA.glob("prices_round_4_day_*.csv")):
        day = int(p.stem.replace("prices_round_4_day_", ""))
        df = pd.read_csv(p, sep=";")
        panels.append(aligned_panel_day(day, df))
    pan = pd.concat(panels, ignore_index=True)

    pan = add_gate_and_fwd(pan)

    # --- Sonic-style: tight vs loose extract forward (per day + pooled) ---
    sonic_rows = []
    for d in sorted(pan["day"].unique()):
        sub = pan[pan["day"] == d].dropna(subset=["fwd_ext_20"])
        if sub.empty:
            continue
        t = sub[sub["joint_tight"]]["fwd_ext_20"].to_numpy(dtype=float)
        n = sub[~sub["joint_tight"]]["fwd_ext_20"].to_numpy(dtype=float)
        sonic_rows.append(
            {
                "day": int(d),
                "n_tight": int(len(t)),
                "mean_fwd20_tight": float(np.nanmean(t)) if len(t) else float("nan"),
                "n_loose": int(len(n)),
                "mean_fwd20_loose": float(np.nanmean(n)) if len(n) else float("nan"),
            }
        )
    sonic_df = pd.DataFrame(sonic_rows)
    pooled_t = pan[pan["joint_tight"]]["fwd_ext_20"].dropna().to_numpy(dtype=float)
    pooled_n = pan[~pan["joint_tight"]]["fwd_ext_20"].dropna().to_numpy(dtype=float)
    sonic_summary = {
        "TH": TH,
        "K": 20,
        "pooled_mean_fwd20_tight": float(np.mean(pooled_t)) if len(pooled_t) else None,
        "pooled_mean_fwd20_loose": float(np.mean(pooled_n)) if len(pooled_n) else None,
        "n_tight": int(len(pooled_t)),
        "n_loose": int(len(pooled_n)),
        "per_day": sonic_df.to_dict(orient="records"),
    }
    sonic_df.to_csv(OUT / "r4_phase3_sonic_fwd_extract_by_day.csv", index=False)
    (OUT / "r4_phase3_sonic_gate_summary.json").write_text(
        json.dumps(sonic_summary, indent=2), encoding="utf-8"
    )

    # --- inclineGod: spread–spread and spread vs dm_ext (full vs tight) ---
    corr_rows = []
    for label, sub in [("all", pan), ("joint_tight", pan[pan["joint_tight"]])]:
        if len(sub) < 50:
            continue
        corr_rows.append(
            {
                "subset": label,
                "n": len(sub),
                "corr_s5200_s5300": safe_corr(sub["s5200"].to_numpy(), sub["s5300"].to_numpy()),
                "corr_s5200_s_ext": safe_corr(sub["s5200"].to_numpy(), sub["s_ext"].to_numpy()),
                "corr_s5300_s_ext": safe_corr(sub["s5300"].to_numpy(), sub["s_ext"].to_numpy()),
                "corr_s5200_dm_ext": safe_corr(sub["s5200"].to_numpy(), sub["dm_ext"].to_numpy()),
                "corr_s5300_dm_ext": safe_corr(sub["s5300"].to_numpy(), sub["dm_ext"].to_numpy()),
            }
        )
    pd.DataFrame(corr_rows).to_csv(OUT / "r4_phase3_spread_spread_and_spread_price_corr.csv", index=False)

    # --- Merge trades to gate at (day, timestamp) ---
    tr = _p1.load_trades()
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["qty"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)
    g = pan[["day", "timestamp", "joint_tight", "s5200", "s5300"]].drop_duplicates()
    trg = tr.merge(g, on=["day", "timestamp"], how="inner")

    px = _p1.price_features(_p1.load_prices())
    fwd_idx = _p1.build_forward_index(px)

    ev = []
    for _, r in trg.iterrows():
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
        ev.append(
            {
                "day": day,
                "timestamp": ts,
                "symbol": sym,
                "buyer": str(r["buyer"]),
                "seller": str(r["seller"]),
                "pair": f"{r['buyer']}->{r['seller']}",
                "aggression": ag,
                "joint_tight": bool(r["joint_tight"]),
                "fwd_mid_20": fwd20,
            }
        )
    ev_df = pd.DataFrame(ev)

    # Mark01->Mark22 on VEV_5300: tight vs loose
    m01 = ev_df[(ev_df["pair"] == "Mark 01->Mark 22") & (ev_df["symbol"] == "VEV_5300")]
    split = (
        m01.groupby("joint_tight")["fwd_mid_20"]
        .agg(n="count", mean="mean", std="std")
        .reset_index()
    )
    split.to_csv(OUT / "r4_phase3_M01_M22_VEV5300_fwd20_gate_split.csv", index=False)

    m67 = ev_df[
        (ev_df["buyer"] == "Mark 67")
        & (ev_df["symbol"] == EXTRACT)
        & (ev_df["aggression"] == "aggr_buy")
    ]
    if not m67.empty:
        m67.groupby("joint_tight")["fwd_mid_20"].agg(n="count", mean="mean").reset_index().to_csv(
            OUT / "r4_phase3_mark67_extract_aggr_buy_gate_split.csv", index=False
        )

    # Top pairs × symbol × gate: n>=8
    trip = (
        ev_df.groupby(["pair", "symbol", "joint_tight"])["fwd_mid_20"]
        .agg(n="count", mean="mean")
        .reset_index()
        .query("n >= 8")
        .sort_values("mean", ascending=False)
    )
    trip.to_csv(OUT / "r4_phase3_triplet_pair_symbol_gate_fwd20.csv", index=False)

    # Compare Phase1-style: same cell without gate (from ev_df aggregate) — mean for M01->22 5300 all
    lines = [
        "Round 4 Phase 3 — Sonic joint gate on R4",
        "==========================================",
        json.dumps(sonic_summary, indent=2),
        "",
        "Spread correlations (see r4_phase3_spread_spread_and_spread_price_corr.csv)",
        pd.DataFrame(corr_rows).to_string(index=False),
        "",
        "Mark01->Mark22 VEV_5300 gate split:",
        split.to_string(index=False),
    ]
    (OUT / "r4_phase3_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
