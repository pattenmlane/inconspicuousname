"""
Round 4 Phase 1 — counterparty-conditioned forward mids (manual_traders/R4/r3v_velvet_beta_hedge_02).

Horizon: K = next K price rows for the same (day, product) after the trade timestamp
(sorted by timestamp ascending within day).

Aggression: at merge (day, symbol, timestamp) — aggressive buy if price >= ask1;
aggressive sell if price <= bid1; else unknown.

Outputs under outputs/ — see analysis.json round4_phase1_complete for paths.
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
KS = [5, 20, 100]
PRODUCTS_FOCUS = [
    "VELVETFRUIT_EXTRACT",
    "HYDROGEL_PACK",
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]


def load_prices() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(
            p,
            sep=";",
            usecols=[
                "day",
                "timestamp",
                "product",
                "bid_price_1",
                "ask_price_1",
                "mid_price",
            ],
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_forward_mids(px: pd.DataFrame) -> pd.DataFrame:
    """Per (day, product), sort by timestamp, add fwd_K = mid at row i+K minus mid at i."""
    parts: list[pd.DataFrame] = []
    for (d, sym), g in px.groupby(["day", "product"], sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        mid = g["mid_price"].astype(float)
        for k in KS:
            g[f"fwd_mid_delta_{k}"] = mid.shift(-k) - mid
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


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


def main() -> None:
    px = load_prices()
    px = add_forward_mids(px)
    px["spread"] = (px["ask_price_1"] - px["bid_price_1"]).astype(float)

    tr = load_trades()
    tr = tr.rename(columns={"symbol": "product"})
    tr["price"] = tr["price"].astype(float)
    tr["quantity"] = tr["quantity"].astype(int)

    merged = tr.merge(
        px,
        on=["day", "timestamp", "product"],
        how="left",
        suffixes=("", "_px"),
    )

    # aggression
    merged["aggr_buy"] = merged["price"] >= merged["ask_price_1"]
    merged["aggr_sell"] = merged["price"] <= merged["bid_price_1"]
    sp = merged["spread"].replace([np.inf, -np.inf], np.nan)
    try:
        merged["spread_q"] = pd.qcut(sp, q=4, labels=["q1_tight", "q2", "q3", "q4_wide"], duplicates="drop")
    except ValueError:
        merged["spread_q"] = pd.cut(sp, bins=[-np.inf, 2, 5, 10, np.inf], labels=["tight_le2", "m3_5", "m6_10", "wide"])
    merged["hour_bin"] = (merged["timestamp"] // 10_000).astype(int) % 24

    burst = (
        merged.groupby(["day", "timestamp"])
        .agg(n_prints=("product", "size"), products=("product", lambda x: ",".join(sorted(x.unique()))))
        .reset_index()
    )
    burst["is_burst"] = burst["n_prints"] >= 4
    burst.to_csv(OUT / "r4_burst_calendar.csv", index=False)

    # orchestrator: most common buyer on multi-symbol bursts
    burst_keys = merged.groupby(["day", "timestamp"]).size()
    multi = burst_keys[burst_keys >= 2].index
    orch_rows: list[dict] = []
    for key in list(multi)[:5000]:  # cap for speed
        d, ts = key
        sub = merged[(merged["day"] == d) & (merged["timestamp"] == ts)]
        if sub["product"].nunique() < 2:
            continue
        bc = sub["buyer"].value_counts().head(1)
        sc = sub["seller"].value_counts().head(1)
        orch_rows.append(
            {
                "day": d,
                "timestamp": ts,
                "n": len(sub),
                "n_sym": sub["product"].nunique(),
                "top_buyer": bc.index[0] if len(bc) else "",
                "top_buyer_n": int(bc.iloc[0]) if len(bc) else 0,
                "top_seller": sc.index[0] if len(sc) else "",
                "top_seller_n": int(sc.iloc[0]) if len(sc) else 0,
            }
        )
    pd.DataFrame(orch_rows).to_csv(OUT / "r4_burst_orchestrator_sample.csv", index=False)

    # directed graph
    edge = merged.groupby(["buyer", "seller"], as_index=False).agg(
        n=("product", "size"), notional=("quantity", lambda s: float((s * merged.loc[s.index, "price"]).sum()))
    )
    edge = edge.sort_values("n", ascending=False)
    edge.to_csv(OUT / "r4_buyer_seller_edges.csv", index=False)

    # participant-level: expand each trade to buyer-event and seller-event with role
    rows_b: list[dict] = []
    for _, r in merged.iterrows():
        for k in KS:
            col = f"fwd_mid_delta_{k}"
            if col not in r or pd.isna(r[col]):
                continue
            dv = float(r[col])
            base = {
                "day": r["day"],
                "product": r["product"],
                "spread_q": str(r["spread_q"]) if pd.notna(r.get("spread_q")) else "na",
                "hour_bin": int(r["hour_bin"]),
                "k": k,
                "dv": dv,
            }
            if bool(r["aggr_buy"]):
                rows_b.append({**base, "U": r["buyer"], "role": "aggr_buyer"})
            if bool(r["aggr_sell"]):
                rows_b.append({**base, "U": r["seller"], "role": "aggr_seller"})
    ev = pd.DataFrame(rows_b)
    # stratify participant events by spread bucket (from merge)
    strat_cols = ["U", "role", "product", "k", "spread_q"]
    if len(ev) and "spread_q" in merged.columns:
        ev_s = []
        for _, r in merged.iterrows():
            sq = r.get("spread_q")
            if pd.isna(sq):
                continue
            for k in KS:
                col = f"fwd_mid_delta_{k}"
                if col not in r or pd.isna(r[col]):
                    continue
                dv = float(r[col])
                if bool(r["aggr_buy"]):
                    ev_s.append(
                        {
                            "U": r["buyer"],
                            "role": "aggr_buyer",
                            "product": r["product"],
                            "k": k,
                            "spread_q": str(sq),
                            "dv": dv,
                        }
                    )
                if bool(r["aggr_sell"]):
                    ev_s.append(
                        {
                            "U": r["seller"],
                            "role": "aggr_seller",
                            "product": r["product"],
                            "k": k,
                            "spread_q": str(sq),
                            "dv": dv,
                        }
                    )
        ev_sq = pd.DataFrame(ev_s)
        if len(ev_sq):
            ev_sq.groupby(["U", "role", "product", "k", "spread_q"])["dv"].agg(["count", "mean", "std"]).reset_index().to_csv(
                OUT / "r4_participant_forward_by_spreadq.csv", index=False
            )

    if len(ev):
        g = (
            ev.groupby(["U", "role", "product", "k", "day"])
            .agg(n=("dv", "size"), mean_dv=("dv", "mean"), std_dv=("dv", "std"))
            .reset_index()
        )
        g["t_approx"] = g["mean_dv"] / (g["std_dv"] / np.sqrt(g["n"]).replace(0, np.nan))
        g.to_csv(OUT / "r4_participant_forward_by_day.csv", index=False)
        g2 = (
            ev.groupby(["U", "role", "product", "k"])
            .agg(n=("dv", "size"), mean_dv=("dv", "mean"), std_dv=("dv", "std"))
            .reset_index()
        )
        g2["t_approx"] = g2["mean_dv"] / (g2["std_dv"] / np.sqrt(g2["n"]).replace(0, np.nan))
        g2.to_csv(OUT / "r4_participant_forward_pooled.csv", index=False)

    # cross-asset: VELVET mid forward when trade is on VEV_* (merge extract row same ts)
    u_px = px[px["product"] == "VELVETFRUIT_EXTRACT"][["day", "timestamp", "mid_price"]].rename(
        columns={"mid_price": "u_mid"}
    )
    vev_tr = merged[merged["product"].str.startswith("VEV_", na=False)].copy()
    vev_tr = vev_tr.merge(u_px, on=["day", "timestamp"], how="left")
    # forward extract: shift on extract-only series
    ux = px[px["product"] == "VELVETFRUIT_EXTRACT"].sort_values(["day", "timestamp"])
    ux = ux.assign(
        fwd_u_20=ux.groupby("day")["mid_price"].shift(-20) - ux["mid_price"]
    )
    vev_tr = vev_tr.merge(
        ux[["day", "timestamp", "fwd_u_20"]],
        on=["day", "timestamp"],
        how="left",
    )
    if len(vev_tr):
        vev_tr.to_csv(OUT / "r4_vev_trades_with_u_forward20.csv", index=False)

    # baseline cell means forward20 same symbol
    if "fwd_mid_delta_20" in merged.columns:
        cell = merged.groupby(["buyer", "seller", "product"], dropna=False)["fwd_mid_delta_20"].mean().reset_index(
            name="baseline_fwd20"
        )
        m2 = merged.merge(cell, on=["buyer", "seller", "product"], how="left")
        m2["resid_fwd20"] = m2["fwd_mid_delta_20"] - m2["baseline_fwd20"]
        m2.groupby(["buyer", "seller", "product"], dropna=False)["resid_fwd20"].agg(["mean", "count"]).reset_index().sort_values(
            "count", ascending=False
        ).head(50).to_csv(OUT / "r4_residual_fwd20_top_pairs.csv", index=False)

    # burst event study: forward U mid after burst vs random control (same n, random ts)
    burst_ts = burst.loc[burst["is_burst"], ["day", "timestamp"]].drop_duplicates()
    u_sorted = ux.sort_values(["day", "timestamp"]).reset_index(drop=True)

    def fwd_u_at(d: int, ts: int, k: int = 20) -> float:
        sub = u_sorted[(u_sorted["day"] == d) & (u_sorted["timestamp"] == ts)]
        if sub.empty:
            return float("nan")
        idx = sub.index[0]
        j = idx + k
        if j >= len(u_sorted) or u_sorted.loc[j, "day"] != d:
            return float("nan")
        return float(u_sorted.loc[j, "mid_price"] - sub["mid_price"].iloc[0])

    burst_fwd: list[float] = []
    for _, r in burst_ts.head(200).iterrows():
        burst_fwd.append(fwd_u_at(int(r["day"]), int(r["timestamp"]), 20))
    rng = np.random.default_rng(0)
    ctrl_fwd: list[float] = []
    for d in DAYS:
        uts = u_sorted.loc[u_sorted["day"] == d, "timestamp"].values
        if len(uts) < 50:
            continue
        pick = rng.choice(uts, size=min(200, len(uts)), replace=False)
        for ts in pick[:200]:
            ctrl_fwd.append(fwd_u_at(d, int(ts), 20))
    summary_burst = {
        "n_burst_sample": len([x for x in burst_fwd if np.isfinite(x)]),
        "mean_fwd_u20_after_burst": float(np.nanmean(burst_fwd)),
        "n_ctrl": len([x for x in ctrl_fwd if np.isfinite(x)]),
        "mean_fwd_u20_random_ts": float(np.nanmean(ctrl_fwd)),
    }
    (OUT / "r4_burst_vs_control_u_forward20.json").write_text(json.dumps(summary_burst, indent=2), encoding="utf-8")

    # Top candidate edges: pool n>=30, abs t_approx > 2 for aggr roles
    candidates: list[str] = []
    if len(ev):
        pool = (
            ev.groupby(["U", "role", "product", "k"])
            .agg(n=("dv", "size"), mean_dv=("dv", "mean"), std_dv=("dv", "std"))
            .reset_index()
        )
        pool["t"] = pool["mean_dv"] / (pool["std_dv"] / np.sqrt(pool["n"]).replace(0, np.nan))
        top = pool[(pool["n"] >= 30) & (pool["t"].abs() >= 1.5)].sort_values("t", key=abs, ascending=False)
        top.head(40).to_csv(OUT / "r4_candidate_edges_t_ge_1.5_n30.csv", index=False)
        for _, r in top.head(10).iterrows():
            candidates.append(
                f"{r['U']} {r['role']} {r['product']} K={int(r['k'])} n={int(r['n'])} mean={r['mean_dv']:.4f} t~{r['t']:.2f}"
            )

    (OUT / "r4_phase1_run_summary.txt").write_text(
        "\n".join(
            [
                "Round 4 Phase 1 script complete.",
                f"Trade rows merged: {len(merged)}",
                f"Participant-event rows (aggr only): {len(ev)}",
                "Key outputs:",
                "  r4_participant_forward_pooled.csv",
                "  r4_participant_forward_by_day.csv",
                "  r4_buyer_seller_edges.csv",
                "  r4_burst_calendar.csv",
                "  r4_burst_orchestrator_sample.csv",
                "  r4_burst_vs_control_u_forward20.json",
                "  r4_residual_fwd20_top_pairs.csv",
                "  r4_candidate_edges_t_ge_1.5_n30.csv",
                "Top candidate strings:",
                *candidates[:15],
            ]
        ),
        encoding="utf-8",
    )
    print("Wrote outputs to", OUT)


if __name__ == "__main__":
    main()
