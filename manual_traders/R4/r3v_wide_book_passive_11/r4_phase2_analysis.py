#!/usr/bin/env python3
"""
Round 4 Phase 2 — orthogonal edges (named-bot+burst, microstructure, lead-lag,
regimes incl. Sonic joint gate, simple smile proxy, passive adverse selection).

Reuses same tape convention as Phase 1: forward mid = K price rows ahead per (day, product).

Outputs: manual_traders/R4/r3v_wide_book_passive_11/analysis_outputs_r4_phase2/

Run from repo root:
  python3 manual_traders/R4/r3v_wide_book_passive_11/r4_phase2_analysis.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs_r4_phase2"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
K_LIST = [5, 20, 100]
EXTRACT = "VELVETFRUIT_EXTRACT"
GATE_5200 = "VEV_5200"
GATE_5300 = "VEV_5300"
W_BURST_NEIGH = 500  # timestamp units: "near burst" window for conditioning


def load_prices_long() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = d
        frames.append(df)
    pr = pd.concat(frames, ignore_index=True)
    pr["mid"] = pd.to_numeric(pr["mid_price"], errors="coerce")
    pr["bid1"] = pd.to_numeric(pr["bid_price_1"], errors="coerce")
    pr["ask1"] = pd.to_numeric(pr["ask_price_1"], errors="coerce")
    bv = pd.to_numeric(pr["bid_volume_1"], errors="coerce").fillna(0)
    av = pd.to_numeric(pr["ask_volume_1"], errors="coerce").fillna(0)
    pr["bid_vol1"] = bv
    pr["ask_vol1"] = av
    pr["spread"] = pr["ask1"] - pr["bid1"]
    denom = bv + av
    pr["microprice"] = np.where(
        denom > 0,
        (pr["bid1"] * av + pr["ask1"] * bv) / denom,
        pr["mid"],
    )
    pr["micro_minus_mid"] = pr["microprice"] - pr["mid"]
    return pr


def add_row_fwd(pr_long: pd.DataFrame, K: int) -> pd.DataFrame:
    pr = pr_long.sort_values(["day", "product", "timestamp"]).copy()
    col = f"fwd_mid_{K}"
    pr[col] = pr.groupby(["day", "product"], group_keys=False)["mid"].transform(
        lambda s: s.shift(-K) - s
    )
    return pr


def pivot_panel(pr: pd.DataFrame) -> pd.DataFrame:
    """Wide mids + spreads per (day, timestamp). Inner: timestamps where all core symbols exist."""
    core = [
        EXTRACT,
        "HYDROGEL_PACK",
        "VEV_4000",
        "VEV_4500",
        "VEV_5000",
        "VEV_5100",
        GATE_5200,
        GATE_5300,
        "VEV_5400",
        "VEV_5500",
        "VEV_6000",
        "VEV_6500",
    ]
    sub = pr[pr["product"].isin(core)]
    mids = sub.pivot_table(
        index=["day", "timestamp"],
        columns="product",
        values="mid",
        aggfunc="first",
    )
    s5200 = sub[sub["product"] == GATE_5200][["day", "timestamp", "spread"]].rename(
        columns={"spread": "s5200"}
    )
    s5300 = sub[sub["product"] == GATE_5300][["day", "timestamp", "spread"]].rename(
        columns={"spread": "s5300"}
    )
    panel = mids.reset_index().merge(s5200, on=["day", "timestamp"], how="inner")
    panel = panel.merge(s5300, on=["day", "timestamp"], how="inner")
    panel["sonic_tight"] = (panel["s5200"] <= 2) & (panel["s5300"] <= 2)
    return panel


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        t = pd.read_csv(p, sep=";")
        t["day"] = d
        frames.append(t)
    tr = pd.concat(frames, ignore_index=True)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)
    for c in ("symbol", "buyer", "seller"):
        tr[c] = tr[c].astype(str)
    return tr


def mark_basket_bursts(tr: pd.DataFrame) -> pd.DataFrame:
    """Mark 01→22 same-(day,ts) with >=3 distinct voucher symbols (basket ladder)."""
    g = tr.groupby(["day", "timestamp", "buyer", "seller"])
    burst_key = []
    for (day, ts, b, s), grp in g:
        syms = set(grp["symbol"])
        if b == "Mark 01" and s == "Mark 22" and len(syms) >= 3:
            burst_key.append((day, ts))
    burst_set = set(burst_key)
    tr = tr.copy()
    tr["basket_burst"] = tr.apply(
        lambda r: (r["day"], r["timestamp"]) in burst_set, axis=1
    )
    return tr


def aggressor(row, bid1, ask1) -> str:
    if pd.isna(bid1) or pd.isna(ask1) or pd.isna(row["price"]):
        return "unknown"
    p = row["price"]
    if p >= ask1:
        return "buy_aggr"
    if p <= bid1:
        return "sell_aggr"
    return "mid_passive"


def main() -> None:
    pr0 = load_prices_long()
    pr = pr0.copy()
    for K in K_LIST:
        pr = add_row_fwd(pr, K)

    tr = load_trades()
    tr = mark_basket_bursts(tr)

    px = pr.rename(columns={"product": "symbol"})
    sym_cols = ["day", "timestamp", "symbol", "mid", "spread", "bid1", "ask1", "micro_minus_mid"]
    for K in K_LIST:
        sym_cols.append(f"fwd_mid_{K}")
    m = tr.merge(px[sym_cols], on=["day", "timestamp", "symbol"], how="left")
    ex = px.loc[px["symbol"] == EXTRACT, ["day", "timestamp"] + [f"fwd_mid_{K}" for K in K_LIST]]
    ex = ex.rename(columns={f"fwd_mid_{K}": f"extract_fwd_{K}" for K in K_LIST})
    m = m.merge(ex, on=["day", "timestamp"], how="left")

    m["aggressor"] = m.apply(lambda r: aggressor(r, r["bid1"], r["ask1"]), axis=1)

    # --- Sonic joint gate at trade time (from merged row spread is symbol's spread; need 5200/5300 at ts)
    g52 = px.loc[px["symbol"] == GATE_5200, ["day", "timestamp", "spread"]].rename(
        columns={"spread": "gate_s52"}
    )
    g53 = px.loc[px["symbol"] == GATE_5300, ["day", "timestamp", "spread"]].rename(
        columns={"spread": "gate_s53"}
    )
    m = m.merge(g52, on=["day", "timestamp"], how="left").merge(
        g53, on=["day", "timestamp"], how="left"
    )
    m["sonic_tight"] = (m["gate_s52"] <= 2) & (m["gate_s53"] <= 2)
    m["sonic_tight"] = m["sonic_tight"].fillna(False)

    burst_ts = set(zip(tr.loc[tr["basket_burst"], "day"], tr.loc[tr["basket_burst"], "timestamp"]))

    def near_burst(row) -> bool:
        d, t = row["day"], row["timestamp"]
        for bd, bt in burst_ts:
            if bd != d:
                continue
            if abs(int(t) - int(bt)) <= W_BURST_NEIGH:
                return True
        return False

    m["near_basket_burst"] = m.apply(near_burst, axis=1)

    lines = []

    # 1) Mark 01→22 basket burst: post-event extract fwd (first trade row per burst timestamp)
    burst_first = (
        m[m["basket_burst"]]
        .drop_duplicates(subset=["day", "timestamp"])
        .copy()
    )
    ctrl = m[~m["basket_burst"]].drop_duplicates(subset=["day", "timestamp"])
    ctrl = ctrl.sample(min(len(burst_first), max(1, len(ctrl))), random_state=42)
    rows_burst = []
    for K in K_LIST:
        c = f"extract_fwd_{K}"
        a = pd.to_numeric(burst_first[c], errors="coerce").dropna()
        b = pd.to_numeric(ctrl[c], errors="coerce").dropna()
        ma, mb = float(a.mean()) if len(a) else float("nan"), float(b.mean()) if len(b) else float("nan")
        rows_burst.append(
            {
                "horizon_K": K,
                "n_burst_ts": len(a),
                "n_control_ts": len(b),
                "mean_burst": ma,
                "mean_control": mb,
            }
        )
    pd.DataFrame(rows_burst).to_csv(OUT / "p2_01_basket_burst_vs_control_extract_fwd.csv", index=False)

    # Mark 67 prints: sonic_tight vs not (interaction Phase1)
    sub67 = m[(m["buyer"] == "Mark 67") | (m["seller"] == "Mark 67")]
    rows67 = []
    for tight in (True, False):
        s = sub67[sub67["sonic_tight"] == tight]
        x = pd.to_numeric(s["fwd_mid_20"], errors="coerce").dropna()
        rows67.append(
            {
                "sonic_tight": tight,
                "n": len(x),
                "mean_fwd20_same_sym": float(x.mean()) if len(x) else float("nan"),
                "frac_pos": float((x > 0).mean()) if len(x) else float("nan"),
            }
        )
    pd.DataFrame(rows67).to_csv(OUT / "p2_02_mark67_fwd20_by_sonic_gate.csv", index=False)

    # 2) Microstructure: extract — corr(spread, abs(next mid change))
    ex_pr = pr0[pr0["product"] == EXTRACT].sort_values(["day", "timestamp"])
    ex_pr["next_dmid"] = ex_pr.groupby("day")["mid"].diff().abs()
    ex_pr["spread_chg"] = ex_pr.groupby("day")["spread"].diff()
    c_sp_abs = ex_pr["spread"].corr(ex_pr["next_dmid"])
    c_micro_abs = ex_pr["micro_minus_mid"].corr(ex_pr["next_dmid"])
    Path(OUT / "p2_03_extract_microstructure_corr.txt").write_text(
        f"EXTRACT: corr(spread, |Δmid next row|) = {c_sp_abs:.4f}\n"
        f"EXTRACT: corr(microprice-mid, |Δmid next row|) = {c_micro_abs:.4f}\n"
        f"(next row = consecutive timestamp row per day)\n"
    )

    # 3) Lead-lag panel (inner join timestamps): corr(extract Δmid, wing Δmid lagged)
    panel = pivot_panel(pr0)
    panel = panel.sort_values(["day", "timestamp"]).reset_index(drop=True)
    ext_ret = panel[EXTRACT].diff()
    lags = list(range(-5, 6))
    ll_rows = []
    wing = "VEV_5300"
    if wing in panel.columns:
        w_ret = panel[wing].diff()
        for L in lags:
            shifted = w_ret.shift(L)
            r = float(ext_ret.corr(shifted)) if len(ext_ret.dropna()) > 5 else float("nan")
            ll_rows.append({"wing": wing, "lag_wing_lead_extract": L, "corr_ext_dmid_wing_dmid": r})
    pd.DataFrame(ll_rows).to_csv(OUT / "p2_04_leadlag_ext_vs_vev5300.csv", index=False)

    # 4) Mark 55→14 on extract: stratify sonic
    sub5514 = m[(m["buyer"] == "Mark 55") & (m["seller"] == "Mark 14") & (m["symbol"] == EXTRACT)]
    rows5514 = []
    for tight in (True, False):
        s = sub5514[sub5514["sonic_tight"] == tight]
        x = pd.to_numeric(s["fwd_mid_20"], errors="coerce").dropna()
        rows5514.append(
            {
                "sonic_tight": tight,
                "n": len(x),
                "mean_fwd20": float(x.mean()) if len(x) else float("nan"),
            }
        )
    pd.DataFrame(rows5514).to_csv(OUT / "p2_05_mark55_14_extract_fwd20_by_sonic.csv", index=False)

    # 5) Smile proxy: IV skew from mids (same BS as Phase3-lite: not full — use wing spread IV rank)
    # Proxy: at each (day,ts) IV_rank = mid_4000 - mid_6500 (not true IV — document)
    p4000 = px[px["symbol"] == "VEV_4000"][["day", "timestamp", "mid"]].rename(columns={"mid": "m4000"})
    p6500 = px[px["symbol"] == "VEV_6500"][["day", "timestamp", "mid"]].rename(columns={"mid": "m6500"})
    skew = p4000.merge(p6500, on=["day", "timestamp"], how="inner")
    skew["wing_skew"] = skew["m4000"] - skew["m6500"]
    m2 = m.merge(skew[["day", "timestamp", "wing_skew"]], on=["day", "timestamp"], how="left")
    med_skew = m2["wing_skew"].median()
    m2["skew_hi"] = m2["wing_skew"] > med_skew

    rows_sm = []
    for hi in (True, False):
        s = m2[(m2["buyer"] == "Mark 01") & (m2["seller"] == "Mark 22") & (m2["skew_hi"] == hi)]
        x = pd.to_numeric(s["fwd_mid_20"], errors="coerce").dropna()
        rows_sm.append(
            {
                "high_wing_skew_proxy": hi,
                "n": len(x),
                "mean_fwd20_same_sym": float(x.mean()) if len(x) else float("nan"),
            }
        )
    pd.DataFrame(rows_sm).to_csv(OUT / "p2_06_mark01_22_fwd20_by_wing_skew_median.csv", index=False)

    # 6) Passive (mid) prints: mean fwd by (buyer, seller) top pairs
    pas = m[m["aggressor"] == "mid_passive"]
    top_pairs = (
        pas.groupby(["buyer", "seller"])
        .size()
        .sort_values(ascending=False)
        .head(15)
        .reset_index(name="n")
    )
    adv_rows = []
    for _, r in top_pairs.iterrows():
        sub = pas[(pas["buyer"] == r["buyer"]) & (pas["seller"] == r["seller"])]
        x = pd.to_numeric(sub["fwd_mid_20"], errors="coerce").dropna()
        if len(x) < 10:
            continue
        adv_rows.append(
            {
                "buyer": r["buyer"],
                "seller": r["seller"],
                "n": len(x),
                "mean_fwd20": float(x.mean()),
                "frac_pos": float((x > 0).mean()),
            }
        )
    pd.DataFrame(adv_rows).to_csv(OUT / "p2_07_passive_prints_fwd20_top_pairs.csv", index=False)

    # 7) Inventory proxy: cumulative signed qty per Mark vs extract fwd after their print
    marks = sorted(set(m["buyer"]) | set(m["seller"]))
    inv_rows = []
    for d in DAYS:
        td = m[m["day"] == d].sort_values("timestamp")
        for mk in marks:
            signed = 0
            fwd_list = []
            for _, r in td.iterrows():
                if r["buyer"] == mk:
                    signed += int(r["quantity"])
                if r["seller"] == mk:
                    signed -= int(r["quantity"])
                f = r.get("extract_fwd_20")
                if pd.notna(f):
                    fwd_list.append((signed, float(f)))
            if len(fwd_list) < 50:
                continue
            arr = np.array(fwd_list)
            corr = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1] if np.std(arr[:, 0]) > 0 else float("nan")
            inv_rows.append({"day": d, "mark": mk, "n": len(fwd_list), "corr_signed_cum_extract_fwd20": corr})
    pd.DataFrame(inv_rows).to_csv(OUT / "p2_08_cumulative_signed_qty_vs_extract_fwd20.csv", index=False)

    # README
    readme = []
    readme.append("Round 4 Phase 2 summary (see CSV/txt files)\n\n")
    readme.append(Path(OUT / "p2_03_extract_microstructure_corr.txt").read_text())
    readme.append("\nMark 67 fwd20 by Sonic gate:\n")
    readme.append(Path(OUT / "p2_02_mark67_fwd20_by_sonic_gate.csv").read_text())
    readme.append("\nMark55→14 extract by Sonic:\n")
    readme.append(Path(OUT / "p2_05_mark55_14_extract_fwd20_by_sonic.csv").read_text())
    readme.append("\nBasket burst vs control (extract):\n")
    readme.append(Path(OUT / "p2_01_basket_burst_vs_control_extract_fwd.csv").read_text())
    Path(OUT / "00_README_PHASE2.txt").write_text("".join(readme))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
