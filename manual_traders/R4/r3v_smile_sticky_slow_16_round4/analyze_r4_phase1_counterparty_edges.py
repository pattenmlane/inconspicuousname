"""
Round 4 Phase 1 — counterparty-conditioned forward mids (tape tick horizons).

Convention: **K ticks** = move **K rows forward** in the price tape for (day, symbol),
starting from the row with **exact** matching `timestamp` (Round 4 prices include `day`).

Outputs: analysis_outputs/*.json, *.csv (see round4work ping_followup_phases Phase 1).

Run from repo root:
  python3 manual_traders/R4/r3v_smile_sticky_slow_16_round4/analyze_r4_phase1_counterparty_edges.py
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

KS = (5, 20, 100)


def list_days() -> list[int]:
    return sorted(int(p.stem.split("_")[-1]) for p in DATA.glob("prices_round_4_day_*.csv"))


def load_prices() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in list_days():
        p = DATA / f"prices_round_4_day_{d}.csv"
        df = pd.read_csv(p, sep=";")
        df["day"] = int(d)
        df["bb"] = pd.to_numeric(df.get("bid_price_1"), errors="coerce")
        df["ba"] = pd.to_numeric(df.get("ask_price_1"), errors="coerce")
        df["mid"] = pd.to_numeric(df.get("mid_price"), errors="coerce")
        frames.append(df[["day", "timestamp", "product", "mid", "bb", "ba"]])
    return pd.concat(frames, ignore_index=True)


def load_trades() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in list_days():
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = int(d)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_index(px: pd.DataFrame) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for (d, sym), g in px.groupby(["day", "product"]):
        g = g.sort_values("timestamp")
        out[(int(d), str(sym))] = (
            g["timestamp"].astype(int).to_numpy(),
            g["mid"].astype(float).to_numpy(),
            g["bb"].astype(float).to_numpy(),
            g["ba"].astype(float).to_numpy(),
        )
    return out


def row_pos(tss: np.ndarray, ts: int) -> int | None:
    pos = int(np.searchsorted(tss, ts, side="left"))
    if pos < len(tss) and tss[pos] == ts:
        return pos
    return None


def mid_forward_k(idx: dict, day: int, sym: str, ts: int, k: int) -> float | None:
    key = (int(day), str(sym))
    if key not in idx:
        return None
    tss, mids, _, _ = idx[key]
    pos = row_pos(tss, ts)
    if pos is None:
        return None
    j = pos + k
    if j >= len(tss):
        return None
    v = mids[j]
    if not math.isfinite(float(v)):
        return None
    return float(v)


def aggressor(price: float, bb: float | None, ba: float | None) -> str:
    if bb is None or ba is None or not (math.isfinite(bb) and math.isfinite(ba)):
        return "unk"
    if price >= float(ba):
        return "buy_agg"
    if price <= float(bb):
        return "sell_agg"
    return "unk"


def hour_bucket(ts: int) -> int:
    return (int(ts) // 100) % 24


def spread_q(bb: float, ba: float) -> str:
    sp = float(ba - bb)
    if sp <= 2.0:
        return "tight"
    if sp <= 6.0:
        return "mid"
    return "wide"


def main() -> None:
    px = load_prices()
    tr = load_trades()
    idx = build_index(px)
    bbo = px.drop_duplicates(subset=["day", "timestamp", "product"], keep="first").set_index(
        ["day", "timestamp", "product"]
    )[["bb", "ba"]]

    burst_n = tr.groupby(["day", "timestamp"]).size()
    burst_keys = set(burst_n[burst_n >= 2].index.tolist())

    rows: list[dict] = []
    pair_rows: list[dict] = []

    for _, r in tr.iterrows():
        d = int(r["day"])
        ts = int(r["timestamp"])
        sym = str(r["symbol"])
        buyer = str(r["buyer"])
        seller = str(r["seller"])
        pair = f"{buyer}|{seller}"
        pr = float(r["price"])
        qty = int(r["quantity"])
        key = (d, ts, sym)
        if key in bbo.index:
            bb = float(bbo.loc[key, "bb"])
            ba = float(bbo.loc[key, "ba"])
        else:
            bb = ba = float("nan")
        agg = aggressor(pr, bb if math.isfinite(bb) else None, ba if math.isfinite(ba) else None)
        sq = spread_q(bb, ba) if math.isfinite(bb) and math.isfinite(ba) else "unk"
        hb = hour_bucket(ts)
        burst = 1 if (d, ts) in burst_keys else 0

        m0 = mid_forward_k(idx, d, sym, ts, 0)
        if m0 is None:
            continue
        for k in KS:
            mk = mid_forward_k(idx, d, sym, ts, k)
            if mk is None:
                continue
            dmid = mk - m0
            rows.append(
                {
                    "day": d,
                    "timestamp": ts,
                    "symbol": sym,
                    "buyer": buyer,
                    "seller": seller,
                    "pair": pair,
                    "qty": qty,
                    "price": pr,
                    "agg": agg,
                    "spread_q": sq,
                    "hour": hb,
                    "burst": burst,
                    "K": k,
                    "fwd_mid": dmid,
                }
            )
            pair_rows.append(
                {
                    "day": d,
                    "pair": pair,
                    "symbol": sym,
                    "K": k,
                    "spread_q": sq,
                    "burst": burst,
                    "fwd_mid": dmid,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows")

    marks = sorted(set(df["buyer"]) | set(df["seller"]))
    by_mark: dict[str, list] = defaultdict(list)
    for U in marks:
        for label, sub in (("buy_agg", df[(df["buyer"] == U) & (df["agg"] == "buy_agg")]), ("sell_agg", df[(df["seller"] == U) & (df["agg"] == "sell_agg")])):
            if sub.empty:
                continue
            for (sym, k, sq, br), g in sub.groupby(["symbol", "K", "spread_q", "burst"]):
                arr = g["fwd_mid"].astype(float).to_numpy()
                if len(arr) < 30:
                    continue
                tt = stats.ttest_1samp(arr, 0.0, nan_policy="omit")
                by_mark[U].append(
                    {
                        "role": label,
                        "symbol": str(sym),
                        "K": int(k),
                        "spread_q": str(sq),
                        "burst": int(br),
                        "n": int(len(arr)),
                        "mean": float(np.mean(arr)),
                        "median": float(np.median(arr)),
                        "frac_pos": float(np.mean(arr > 0)),
                        "t_stat": float(tt.statistic),
                        "p_value": float(tt.pvalue),
                    }
                )

    stab = []
    for d in list_days():
        g = df[
            (df["buyer"] == "Mark 01")
            & (df["agg"] == "buy_agg")
            & (df["symbol"] == "VEV_5200")
            & (df["K"] == 20)
            & (df["spread_q"] == "tight")
            & (df["day"] == d)
        ]
        stab.append({"day": d, "n": len(g), "mean_fwd": float(g["fwd_mid"].mean()) if len(g) else None})

    (OUT / "r4_phase1_forward_by_mark.json").write_text(
        json.dumps({"per_mark_cells": {k: v for k, v in sorted(by_mark.items())}, "stability_mark01_vev5200_k20_tight_by_day": stab}, indent=2),
        encoding="utf-8",
    )

    pdf = pd.DataFrame(pair_rows)
    cells = (
        pdf.groupby(["pair", "symbol", "K", "spread_q", "burst"])
        .agg(n=("fwd_mid", "count"), mean_fwd=("fwd_mid", "mean"))
        .reset_index()
    )
    cells.to_csv(OUT / "r4_phase1_pair_cell_means.csv", index=False)
    cell_lookup = {(r.pair, r.symbol, int(r.K), r.spread_q, int(r.burst)): float(r.mean_fwd) for r in cells.itertuples()}

    residuals = []
    for r in df.itertuples():
        key = (r.pair, r.symbol, int(r.K), r.spread_q, int(r.burst))
        mu = cell_lookup.get(key)
        if mu is None:
            continue
        residuals.append({"pair": r.pair, "symbol": r.symbol, "K": r.K, "res": float(r.fwd_mid - mu)})
    resdf = pd.DataFrame(residuals)
    top_res = (
        resdf.groupby(["pair", "symbol", "K"])
        .agg(n=("res", "count"), mean_res=("res", "mean"))
        .reset_index()
        .assign(abs_mean=lambda x: x["mean_res"].abs())
        .sort_values("abs_mean", ascending=False)
        .head(50)
        .drop(columns=["abs_mean"])
    )
    top_res.to_csv(OUT / "r4_phase1_residuals_top.csv", index=False)

    ec = tr.groupby(["buyer", "seller"]).agg(n=("quantity", "count"), lots=("quantity", "sum")).reset_index().sort_values("n", ascending=False)
    ec.to_csv(OUT / "r4_phase1_graph_edges.csv", index=False)

    burst_fwd = []
    for (d, ts), g in tr.groupby(["day", "timestamp"]):
        if len(g) < 2:
            continue
        orch = str(g["buyer"].mode().iloc[0])
        m0 = mid_forward_k(idx, int(d), "VELVETFRUIT_EXTRACT", int(ts), 0)
        if m0 is None:
            continue
        for k in (5, 20):
            mk = mid_forward_k(idx, int(d), "VELVETFRUIT_EXTRACT", int(ts), k)
            if mk is None:
                continue
            burst_fwd.append({"day": int(d), "timestamp": int(ts), "n_prints": len(g), "mode_buyer": orch, "K": k, "fwd": float(mk - m0)})
    bf = pd.DataFrame(burst_fwd)
    rng = np.random.default_rng(0)
    ctrl = []
    all_ts_by_day = {d: tr.loc[tr["day"] == d, "timestamp"].astype(int).unique() for d in list_days()}
    for _, r in bf.iterrows():
        d = int(r["day"])
        k = int(r["K"])
        pool = [t for t in all_ts_by_day[d] if (d, int(t)) not in burst_keys]
        if len(pool) < 10:
            continue
        for _ in range(3):
            ts2 = int(rng.choice(pool))
            m0 = mid_forward_k(idx, d, "VELVETFRUIT_EXTRACT", ts2, 0)
            mk = mid_forward_k(idx, d, "VELVETFRUIT_EXTRACT", ts2, k)
            if m0 is None or mk is None:
                continue
            ctrl.append({"K": k, "fwd": float(mk - m0)})
    cdf = pd.DataFrame(ctrl)
    (OUT / "r4_phase1_bursts_forward.json").write_text(
        json.dumps(
            {
                "burst_mean_by_K": bf.groupby("K")["fwd"].mean().to_dict() if len(bf) else {},
                "control_mean_by_K": cdf.groupby("K")["fwd"].mean().to_dict() if len(cdf) else {},
                "n_burst_events": len(bf),
                "n_control_samples": len(cdf),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    adv = []
    for agg in ("buy_agg", "sell_agg"):
        sub = df[df["agg"] == agg]
        for U in marks:
            gpart = sub[sub["buyer"] == U] if agg == "buy_agg" else sub[sub["seller"] == U]
            g20 = gpart[gpart["K"] == 20]["fwd_mid"]
            if len(g20) < 15:
                continue
            adv.append(
                {
                    "mark": U,
                    "agg": agg,
                    "n": int(len(g20)),
                    "mean_fwd20": float(g20.mean()),
                    "median_fwd20": float(g20.median()),
                }
            )
    pd.DataFrame(adv).sort_values("mean_fwd20").to_csv(OUT / "r4_phase1_adverse_aggressor.csv", index=False)

    # --- 3b) Consecutive-trade two-hop: buyer->seller on row j matches buyer on row j+1 (same day, time-sorted)
    trs = tr.sort_values(["day", "timestamp"]).reset_index(drop=True)
    trip = Counter()
    for d in list_days():
        sub = trs[trs["day"] == d]
        for i in range(len(sub) - 1):
            a, b = str(sub.iloc[i]["buyer"]), str(sub.iloc[i]["seller"])
            nxt = sub.iloc[i + 1]
            if str(nxt["buyer"]) != b:
                continue
            c = str(nxt["seller"])
            trip[(a, b, c)] += 1
    top_triples = trip.most_common(25)
    (OUT / "r4_phase1_twohop_top.json").write_text(json.dumps({"top_buyer_seller_seller_triples": [{"a": a, "b": b, "c": c, "n": n} for (a, b, c), n in top_triples]}, indent=2), encoding="utf-8")

    print("Wrote", OUT)


if __name__ == "__main__":
    main()
