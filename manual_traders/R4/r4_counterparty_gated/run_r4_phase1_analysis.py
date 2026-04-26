"""
Round 4 Phase 1 — counterparty-conditioned forward mids (tape evidence).

Reads Prosperity4Data/ROUND_4 prices + trades (days present).
Horizon K = K steps in the *price tape* for the same (day, product): next K rows by timestamp.

Outputs under manual_traders/R4/r4_counterparty_gated/analysis_outputs/
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
HORIZONS = (5, 20, 100)
KEY_SYMS = [
    "VELVETFRUIT_EXTRACT",
    "HYDROGEL_PACK",
    "VEV_5200",
    "VEV_5300",
    "VEV_4000",
]


def load_prices() -> pd.DataFrame:
    frames = []
    for p in sorted(DATA.glob("prices_round_4_day_*.csv")):
        day = int(p.stem.replace("prices_round_4_day_", ""))
        df = pd.read_csv(p, sep=";")
        df["day"] = day
        frames.append(df)
    if not frames:
        raise SystemExit("No price files")
    return pd.concat(frames, ignore_index=True)


def load_trades() -> pd.DataFrame:
    frames = []
    for p in sorted(DATA.glob("trades_round_4_day_*.csv")):
        part = p.stem.replace("trades_round_4_day_", "")
        day = int(part)
        df = pd.read_csv(p, sep=";")
        df["day"] = day
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def price_features(px: pd.DataFrame) -> pd.DataFrame:
    """One row per (day, timestamp, product) with mid, spread, bid1, ask1."""
    out = px[
        ["day", "timestamp", "product", "mid_price", "bid_price_1", "ask_price_1"]
    ].copy()
    out["mid"] = pd.to_numeric(out["mid_price"], errors="coerce")
    bp = pd.to_numeric(out["bid_price_1"], errors="coerce")
    ap = pd.to_numeric(out["ask_price_1"], errors="coerce")
    out["spread"] = ap - bp
    out["bid1"] = bp
    out["ask1"] = ap
    return out.drop(columns=["mid_price"])


def build_forward_index(px: pd.DataFrame) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """For each (day, product): sorted timestamps, mids, spreads — index lookup for forward K."""
    store: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for (day, sym), g in px.groupby(["day", "product"]):
        g = g.sort_values("timestamp")
        ts = g["timestamp"].to_numpy(dtype=np.int64)
        mid = g["mid"].to_numpy(dtype=float)
        spr = g["spread"].to_numpy(dtype=float)
        bid1 = g["bid1"].to_numpy(dtype=float)
        ask1 = g["ask1"].to_numpy(dtype=float)
        store[(int(day), str(sym))] = {
            "ts": ts,
            "mid": mid,
            "spread": spr,
            "bid1": bid1,
            "ask1": ask1,
        }
    return store


def idx_at_or_before(ts_arr: np.ndarray, t: int) -> int:
    i = int(np.searchsorted(ts_arr, t, side="right") - 1)
    return max(0, min(i, len(ts_arr) - 1))


def classify_aggression(price: float, bid1: float, ask1: float) -> str:
    if not (np.isfinite(price) and np.isfinite(bid1) and np.isfinite(ask1)):
        return "unknown"
    if ask1 - bid1 <= 0:
        return "unknown"
    if price >= ask1 - 1e-9:
        return "aggr_buy"
    if price <= bid1 + 1e-9:
        return "aggr_sell"
    return "passive_mid"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    px_raw = load_prices()
    px = price_features(px_raw)
    tr = load_trades()
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["qty"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)

    fwd_idx = build_forward_index(px)

    # Burst: count rows per (day, timestamp)
    burst_sz = tr.groupby(["day", "timestamp"]).size().rename("burst_n")
    tr = tr.merge(burst_sz, on=["day", "timestamp"], how="left")

    rows: list[dict] = []
    for _, r in tr.iterrows():
        day = int(r["day"])
        sym = str(r["symbol"])
        ts = int(r["timestamp"])
        key = (day, sym)
        if key not in fwd_idx:
            continue
        st = fwd_idx[key]
        ts_arr = st["ts"]
        i0 = idx_at_or_before(ts_arr, ts)
        bid1 = float(st["bid1"][i0])
        ask1 = float(st["ask1"][i0])
        spr = float(st["spread"][i0])
        mid0 = float(st["mid"][i0])
        ag = classify_aggression(float(r["price"]), bid1, ask1)

        rec = {
            "day": day,
            "timestamp": ts,
            "symbol": sym,
            "buyer": str(r["buyer"]),
            "seller": str(r["seller"]),
            "pair": f"{r['buyer']}->{r['seller']}",
            "aggression": ag,
            "burst_n": int(r["burst_n"]),
            "spread": spr,
            "mid0": mid0,
        }
        n = len(ts_arr)
        for K in HORIZONS:
            j = i0 + K
            if j < n:
                rec[f"fwd_mid_{K}"] = float(st["mid"][j]) - mid0
            else:
                rec[f"fwd_mid_{K}"] = float("nan")
        # cross: extract forward same K
        for K in HORIZONS:
            k2 = (day, "VELVETFRUIT_EXTRACT")
            if k2 in fwd_idx:
                ts_e = fwd_idx[k2]["ts"]
                mid_e = fwd_idx[k2]["mid"]
                ie = idx_at_or_before(ts_e, ts)
                je = ie + K
                if je < len(mid_e):
                    rec[f"fwd_ext_{K}"] = float(mid_e[je]) - float(mid_e[ie])
                else:
                    rec[f"fwd_ext_{K}"] = float("nan")
            else:
                rec[f"fwd_ext_{K}"] = float("nan")
        rows.append(rec)

    ev = pd.DataFrame(rows)

    # Session bucket: quartile of timestamp within each day
    ev["ts_bucket"] = ev.groupby("day")["timestamp"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False, duplicates="drop")
    )

    # Spread quantile within (day, symbol)
    def spread_q(g: pd.Series) -> pd.Series:
        return pd.qcut(
            g.rank(method="first"),
            4,
            labels=["q0_tight", "q1", "q2", "q3_wide"],
            duplicates="drop",
        )

    ev["spread_bucket"] = ev.groupby(["day", "symbol"])["spread"].transform(
        lambda g: spread_q(g) if g.notna().sum() > 20 else "q_na"
    )

    burst_flag = (ev["burst_n"] >= 4).astype(int)
    ev["burst_ge4"] = burst_flag

    def safe_tstat(x: np.ndarray) -> tuple[float, int]:
        x = x[np.isfinite(x)]
        n = len(x)
        if n < 30:
            return float("nan"), n
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1))
        if s < 1e-12:
            return float("nan"), n
        return m / (s / math.sqrt(n)), n

    # --- 1) Participant-level: aggressive buy (buyer lifts) vs aggressive sell ---
    part_rows = []
    names = sorted(set(ev["buyer"].astype(str)).union(set(ev["seller"].astype(str))))
    for U in names:
        if not U or U == "nan":
            continue
        sub_b = ev[(ev["buyer"] == U) & (ev["aggression"] == "aggr_buy")]
        sub_s = ev[(ev["seller"] == U) & (ev["aggression"] == "aggr_sell")]
        for role, sub in [("U_aggr_buy", sub_b), ("U_aggr_sell", sub_s)]:
            if sub.empty:
                continue
            for sym in sub["symbol"].unique():
                s2 = sub[sub["symbol"] == sym]
                for K in HORIZONS:
                    col = f"fwd_mid_{K}"
                    x = s2[col].to_numpy(dtype=float)
                    tstat, n = safe_tstat(x)
                    part_rows.append(
                        {
                            "participant": U,
                            "role": role,
                            "symbol": sym,
                            "K": K,
                            "n": n,
                            "mean_fwd": float(np.nanmean(x)) if n else float("nan"),
                            "t_stat": tstat,
                            "frac_pos": float(np.mean(x > 0)) if n else float("nan"),
                        }
                    )

    part_df = pd.DataFrame(part_rows)
    part_df.to_csv(OUT / "r4_participant_fwd_by_symbol.csv", index=False)

    # Per-day cell means for (buyer, seller, symbol) n>=5
    cell = (
        ev.groupby(["day", "buyer", "seller", "symbol"])["fwd_mid_20"]
        .agg(["mean", "count"])
        .reset_index()
    )
    cell = cell[cell["count"] >= 5]
    cell.to_csv(OUT / "r4_cell_mean_fwd20_by_day_pair_symbol.csv", index=False)

    # --- 2) Baseline: global mean fwd20 by (buyer, seller, symbol), residual ---
    base = (
        ev.groupby(["buyer", "seller", "symbol"])["fwd_mid_20"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "baseline_fwd20", "count": "n_cell"})
    )
    base = base[base["n_cell"] >= 15]
    ev2 = ev.merge(
        base,
        on=["buyer", "seller", "symbol"],
        how="left",
    )
    ev2["resid_fwd20"] = ev2["fwd_mid_20"] - ev2["baseline_fwd20"]
    ev2[["day", "timestamp", "buyer", "seller", "symbol", "fwd_mid_20", "baseline_fwd20", "resid_fwd20"]].to_csv(
        OUT / "r4_trade_level_residual_fwd20.csv", index=False
    )

    # --- 3) Graph edges ---
    edge = tr.groupby(["buyer", "seller"]).agg(n=("symbol", "size"), notional=("price", lambda x: float(np.sum(x)))).reset_index()
    edge = edge.sort_values("n", ascending=False)
    edge.to_csv(OUT / "r4_directed_edges_buyer_seller.csv", index=False)

    # --- 4) Burst event study: first row per (day, ts) burst>=4 ---
    burst_keys = ev.loc[ev["burst_n"] >= 4, ["day", "timestamp"]].drop_duplicates()
    ext_series = {d: fwd_idx[(d, "VELVETFRUIT_EXTRACT")] for d in ev["day"].unique() if (d, "VELVETFRUIT_EXTRACT") in fwd_idx}

    def fwd_ext_after(day: int, ts: int, K: int) -> float:
        st = ext_series.get(int(day))
        if not st:
            return float("nan")
        ts_arr, mid = st["ts"], st["mid"]
        i0 = idx_at_or_before(ts_arr, ts)
        j = i0 + K
        if j >= len(mid):
            return float("nan")
        return float(mid[j]) - float(mid[i0])

    burst_effects = []
    for _, bk in burst_keys.iterrows():
        d, t = int(bk["day"]), int(bk["timestamp"])
        burst_effects.append(
            {
                "day": d,
                "timestamp": t,
                "fwd_ext_20": fwd_ext_after(d, t, 20),
                "fwd_ext_100": fwd_ext_after(d, t, 100),
            }
        )
    burst_df = pd.DataFrame(burst_effects)
    # control: random same-n timestamps without burst
    non = ev.loc[ev["burst_n"] == 1, ["day", "timestamp"]].drop_duplicates()
    if len(non) > len(burst_df):
        ctrl = non.sample(n=min(len(burst_df) * 3, len(non)), random_state=0)
    else:
        ctrl = non
    ctrl_eff = []
    for _, bk in ctrl.iterrows():
        d, t = int(bk["day"]), int(bk["timestamp"])
        ctrl_eff.append({"day": d, "timestamp": t, "fwd_ext_20": fwd_ext_after(d, t, 20)})
    ctrl_df = pd.DataFrame(ctrl_eff)

    burst_summary = {
        "n_bursts": int(len(burst_df)),
        "mean_fwd_ext_20_burst": float(np.nanmean(burst_df["fwd_ext_20"])),
        "mean_fwd_ext_20_control": float(np.nanmean(ctrl_df["fwd_ext_20"])),
        "n_control": int(len(ctrl_df)),
    }
    burst_df.to_csv(OUT / "r4_burst_event_rows.csv", index=False)
    (OUT / "r4_burst_vs_control_summary.json").write_text(json.dumps(burst_summary, indent=2), encoding="utf-8")

    # --- 5) Adverse selection proxy: aggr_buy then fwd (passive seller hurt?) ---
    ag = ev[ev["aggression"] == "aggr_buy"].copy()
    adv = (
        ag.groupby(["seller", "symbol"])["fwd_mid_20"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_fwd20_after_aggr_buy", "count": "n"})
    )
    adv = adv[adv["n"] >= 20].sort_values("mean_fwd20_after_aggr_buy")
    adv.to_csv(OUT / "r4_passive_seller_markout_aggr_buy.csv", index=False)

    # --- 3b) Two-hop chains: consecutive trades same day with seller(t)==buyer(t+1) ---
    trs = tr.sort_values(["day", "timestamp"]).reset_index(drop=True)
    hops: list[dict] = []
    for day in sorted(trs["day"].unique()):
        sub = trs[trs["day"] == day].reset_index(drop=True)
        for i in range(len(sub) - 1):
            r0, r1 = sub.iloc[i], sub.iloc[i + 1]
            if str(r0["seller"]) != str(r1["buyer"]):
                continue
            ts1 = int(r1["timestamp"])
            hops.append(
                {
                    "day": int(day),
                    "timestamp": ts1,
                    "motif": f"{r0['buyer']}->{r0['seller']}->{r1['seller']}",
                    "sym0": str(r0["symbol"]),
                    "sym1": str(r1["symbol"]),
                }
            )
    hop_df = pd.DataFrame(hops)
    if not hop_df.empty:
        hop_df["fwd_ext_20"] = [
            fwd_ext_after(int(r["day"]), int(r["timestamp"]), 20) for _, r in hop_df.iterrows()
        ]
        hop_sum = hop_df.groupby("motif")["fwd_ext_20"].agg(["count", "mean"]).reset_index()
        hop_sum = hop_sum.rename(columns={"count": "n", "mean": "mean_fwd_ext_20"}).sort_values(
            "n", ascending=False
        )
        hop_sum.to_csv(OUT / "r4_twohop_motif_fwd_extract20.csv", index=False)
    else:
        pd.DataFrame(columns=["motif", "n", "mean_fwd_ext_20"]).to_csv(
            OUT / "r4_twohop_motif_fwd_extract20.csv", index=False
        )

    # Stability: Mark 67 aggressive extract buys by day
    m67 = ev[
        (ev["buyer"] == "Mark 67")
        & (ev["aggression"] == "aggr_buy")
        & (ev["symbol"] == "VELVETFRUIT_EXTRACT")
    ]
    if not m67.empty:
        m67.groupby("day")["fwd_mid_5"].agg(["mean", "count"]).reset_index().to_csv(
            OUT / "r4_mark67_aggr_extract_fwd5_by_day.csv", index=False
        )

    # Summary text for humans
    top_part = part_df.dropna(subset=["t_stat"]).sort_values("t_stat", key=abs, ascending=False).head(25)
    lines = [
        "Round 4 Phase 1 — summary (automated)",
        "====================================",
        f"Trades analyzed: {len(ev):,} | price rows: {len(px):,}",
        "Horizon K = steps in price tape for same (day, symbol); fwd_mid_K = mid(t+K)-mid(t) at same ts index.",
        "",
        "Top |t-stat| participant×symbol×K (aggr_buy / aggr_sell roles): see r4_participant_fwd_by_symbol.csv",
        top_part.to_string(index=False),
        "",
        "Burst vs control (extract forward 20 steps):",
        json.dumps(burst_summary, indent=2),
        "",
        "Worst passive sellers after aggressive buys on them (mean fwd_mid_20, n>=20):",
        adv.head(8).to_string(index=False),
    ]
    (OUT / "r4_phase1_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
