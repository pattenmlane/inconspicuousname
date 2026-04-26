#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned markouts (suggested direction.txt).

Horizon K: forward **K price ticks** = next row at timestamp + K*100 (tape step 100).
Merges each trade with same (day, timestamp, symbol) BBO row from prices CSV.
Aggressor: buy if price >= ask1; sell if price <= bid1; else mid (ambiguous).

Outputs under manual_traders/R4/r4_counterparty_research/outputs/
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = (1, 2, 3)
KS = (5, 20, 100)
TICK = 100
FOCUS = (
    "VELVETFRUIT_EXTRACT",
    "HYDROGEL_PACK",
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
)


def load_prices(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    df["day"] = day
    b1 = pd.to_numeric(df["bid_price_1"], errors="coerce")
    a1 = pd.to_numeric(df["ask_price_1"], errors="coerce")
    df["spread"] = (a1 - b1).astype(float)
    df["mid"] = pd.to_numeric(df["mid_price"], errors="coerce")
    return df


def load_trades(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
    df["day"] = day
    return df


def aggressor(row: pd.Series) -> str:
    p, bid, ask = float(row["price"]), float(row["bid_price_1"]), float(row["ask_price_1"])
    if p >= ask:
        return "buy"
    if p <= bid:
        return "sell"
    return "mid"


def session_bucket(ts: int, day_ts_min: int, day_ts_max: int) -> str:
    if day_ts_max <= day_ts_min:
        return "0"
    t = (ts - day_ts_min) / max(day_ts_max - day_ts_min, 1)
    if t < 1 / 3:
        return "early"
    if t < 2 / 3:
        return "mid"
    return "late"


def build_mid_arrays(pr: pd.DataFrame) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """(day, product) -> (sorted timestamps, mids)."""
    out: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for (d, sym), g in pr.groupby(["day", "product"]):
        g = g.sort_values("timestamp")
        out[(int(d), str(sym))] = (
            g["timestamp"].to_numpy(dtype=np.int64),
            g["mid"].to_numpy(dtype=np.float64),
        )
    return out


def forward_mid(
    arrs: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    day: int,
    sym: str,
    ts: int,
    k_ticks: int,
) -> float | None:
    tss, mids = arrs.get((day, sym), (np.array([]), np.array([])))
    if tss.size == 0:
        return None
    target = ts + k_ticks * TICK
    i = np.searchsorted(tss, target, side="left")
    if i >= len(tss):
        return None
    return float(mids[i])


def main() -> None:
    all_tr: list[pd.DataFrame] = []
    pr_parts: list[pd.DataFrame] = []
    for d in DAYS:
        pr = load_prices(d)
        pr_parts.append(pr)
        tr = load_trades(d)
        all_tr.append(tr)
    pr_all = pd.concat(pr_parts, ignore_index=True)
    tr_all = pd.concat(all_tr, ignore_index=True)

    # merge BBO at trade time
    m = tr_all.merge(
        pr_all[
            [
                "day",
                "timestamp",
                "product",
                "mid",
                "spread",
                "bid_price_1",
                "ask_price_1",
            ]
        ].rename(columns={"product": "symbol"}),
        on=["day", "timestamp", "symbol"],
        how="left",
        validate="m:1",
    )
    if m["mid"].isna().any():
        bad = int(m["mid"].isna().sum())
        raise SystemExit(f"merge missing mids: {bad}")

    m["aggressor"] = m.apply(aggressor, axis=1)
    m["notional"] = m["price"].astype(float) * m["quantity"].astype(float)

    # spread decile per (day, symbol) at trade rows
    m["spr_q"] = (
        m.groupby(["day", "symbol"])["spread"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
        .astype("float")
    )
    m["spr_regime"] = np.where(m["spr_q"] >= 7, "wide", np.where(m["spr_q"] <= 2, "tight", "mid"))

    # session bucket per day
    bounds = m.groupby("day")["timestamp"].agg(["min", "max"])
    m["session"] = m.apply(
        lambda r: session_bucket(int(r["timestamp"]), int(bounds.loc[r["day"], "min"]), int(bounds.loc[r["day"], "max"])),
        axis=1,
    )

    arrs = build_mid_arrays(pr_all)

    rows = []
    for _, r in m.iterrows():
        d = int(r["day"])
        sym = str(r["symbol"])
        ts = int(r["timestamp"])
        rec = {
            "day": d,
            "timestamp": ts,
            "symbol": sym,
            "buyer": r["buyer"],
            "seller": r["seller"],
            "aggressor": r["aggressor"],
            "price": float(r["price"]),
            "qty": int(r["quantity"]),
            "notional": float(r["notional"]),
            "mid0": float(r["mid"]),
            "spread0": float(r["spread"]),
            "spr_regime": r["spr_regime"],
            "session": r["session"],
        }
        for K in KS:
            m_sym = forward_mid(arrs, d, sym, ts, K)
            m_u = forward_mid(arrs, d, "VELVETFRUIT_EXTRACT", ts, K)
            m_h = forward_mid(arrs, d, "HYDROGEL_PACK", ts, K)
            rec[f"fwd_{K}_sym"] = m_sym
            rec[f"fwd_{K}_u"] = m_u
            rec[f"fwd_{K}_h"] = m_h
            if m_sym is not None:
                rec[f"mark_{K}_sym"] = m_sym - rec["mid0"]
            else:
                rec[f"mark_{K}_sym"] = np.nan
            if m_u is not None:
                u0 = float(
                    pr_all.loc[
                        (pr_all["day"] == d)
                        & (pr_all["timestamp"] == ts)
                        & (pr_all["product"] == "VELVETFRUIT_EXTRACT"),
                        "mid",
                    ].iloc[0]
                )
                rec[f"mark_{K}_u"] = m_u - u0
            else:
                rec[f"mark_{K}_u"] = np.nan
            if m_h is not None:
                h0 = float(
                    pr_all.loc[
                        (pr_all["day"] == d)
                        & (pr_all["timestamp"] == ts)
                        & (pr_all["product"] == "HYDROGEL_PACK"),
                        "mid",
                    ].iloc[0]
                )
                rec[f"mark_{K}_h"] = m_h - h0
            else:
                rec[f"mark_{K}_h"] = np.nan
        rows.append(rec)

    enr = pd.DataFrame(rows)
    enr_path = OUT / "r4_trades_enriched_markouts.csv"
    enr.to_csv(enr_path, index=False)

    # --- burst detection ---
    burst_size = (
        m.groupby(["day", "timestamp"]).size().rename("n_prints").reset_index()
    )
    burst_big = burst_size[burst_size["n_prints"] >= 4]
    burst_join = m.merge(burst_big, on=["day", "timestamp"])
    orch = (
        burst_join.groupby(["day", "timestamp"])["buyer"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")
        .reset_index(name="orchestrator_buyer")
    )
    m2 = m.merge(burst_big, on=["day", "timestamp"], how="left")
    m2["burst"] = m2["n_prints"].fillna(0) >= 4
    m2 = m2.merge(orch, on=["day", "timestamp"], how="left")

    # --- graph edges ---
    edge_c = Counter()
    edge_n = Counter()
    for _, r in m.iterrows():
        key = (str(r["buyer"]), str(r["seller"]))
        edge_c[key] += 1
        edge_n[key] += float(r["price"]) * int(r["quantity"])

    graph_lines = ["buyer\tseller\tcount\tnotional"]
    for (b, s), c in edge_c.most_common(40):
        graph_lines.append(f"{b}\t{s}\t{c}\t{edge_n[(b,s)]:.1f}")
    (OUT / "r4_graph_top_edges.txt").write_text("\n".join(graph_lines), encoding="utf-8")

    # --- baseline cell mean: (buyer, seller, symbol) mark U @ K=20 ---
    cell = (
        enr.groupby(["buyer", "seller", "symbol"])["mark_20_u"]
        .agg(["mean", "count"])
        .reset_index()
    )
    cell = cell.rename(columns={"mean": "baseline_mean_u20", "count": "cell_n"})
    enr2 = enr.merge(
        cell,
        on=["buyer", "seller", "symbol"],
        how="left",
    )
    enr2["resid_u20"] = enr2["mark_20_u"] - enr2["baseline_mean_u20"]
    enr2.to_csv(OUT / "r4_trades_with_baseline_residual.csv", index=False)

    # --- stratified stats: participant U as buyer when aggressor buy ---
    def summarize(df: pd.DataFrame, col: str, label: str) -> list[str]:
        lines = [label] if label else []
        for name, sub in df.groupby(col):
            if str(name) in ("nan", "None"):
                continue
            n = len(sub)
            if n < 30:
                continue
            row = {"name": str(name), "n": n}
            for K in KS:
                mk = sub[f"mark_{K}_sym"].dropna()
                mu = sub[f"mark_{K}_u"].dropna()
                if len(mk) >= 20:
                    row[f"m_sym_{K}_mean"] = float(mk.mean())
                    row[f"m_sym_{K}_pos"] = float((mk > 0).mean())
                if len(mu) >= 20:
                    row[f"m_u_{K}_mean"] = float(mu.mean())
            lines.append(json.dumps(row))
        return lines

    buy_side = enr[enr["aggressor"] == "buy"]
    sell_side = enr[enr["aggressor"] == "sell"]

    rep: list[str] = []
    rep.append("=== Aggressive BUY: forward markout by buyer (sym + U) ===")
    rep.extend(summarize(buy_side, "buyer", ""))
    rep.append("")
    rep.append("=== Aggressive SELL: forward markout by seller ===")
    rep.extend(summarize(sell_side, "seller", ""))

    # Pair-conditioned: Mark 01 -> Mark 22 on VEV_5300
    sub = enr[(enr["buyer"] == "Mark 01") & (enr["seller"] == "Mark 22") & (enr["symbol"] == "VEV_5300")]
    rep.append("")
    rep.append(f"Mark 01 -> Mark 22, VEV_5300, n={len(sub)}")
    for K in KS:
        mk = sub[f"mark_{K}_sym"].dropna()
        mu = sub[f"mark_{K}_u"].dropna()
        if len(mk):
            rep.append(
                f"  K={K} sym mean={mk.mean():.4f} frac+={(mk>0).mean():.3f} | U mean={mu.mean():.4f} n_sym={len(mk)} n_u={len(mu)}"
            )

    # Burst event study: extract K=20 after burst vs non-burst
    enr_burst = enr.merge(
        m2[["day", "timestamp", "burst"]].drop_duplicates(),
        on=["day", "timestamp"],
    )
    b_yes = enr_burst[enr_burst["burst"]]["mark_20_u"].dropna()
    b_no = enr_burst[~enr_burst["burst"]]["mark_20_u"].dropna()
    rep.append("")
    rep.append(f"Burst vs non: U mark20 mean burst={b_yes.mean():.4f} n={len(b_yes)} | non={b_no.mean():.4f} n={len(b_no)}")

    (OUT / "r4_phase1_summary.txt").write_text("\n".join(rep), encoding="utf-8")

    # Top residual buckets
    top_res = (
        enr2.groupby(["buyer", "seller", "symbol"])["resid_u20"]
        .agg(["mean", "count"])
        .reset_index()
        .query("count >= 25")
        .sort_values("mean", key=np.abs, ascending=False)
        .head(15)
    )
    top_res.to_csv(OUT / "r4_top_residual_cells_u20.csv", index=False)

    print("Wrote", enr_path)
    print("Wrote", OUT / "r4_phase1_summary.txt")


if __name__ == "__main__":
    main()
