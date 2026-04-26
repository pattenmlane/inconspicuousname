"""
Phase-2 style cross-instrument lead–lag: signed trade flow vs forward extract mid, × Sonic gate.

Per (day, timestamp): merge joint gate (5200+5300 inner join, tight = both spr<=2).
Signed flow per product: sum over trades at that (day, ts) of (+qty) if aggressive buy
(price>=ask1), (-qty) if aggressive sell (price<=bid1), else 0.

Panel: extract price rows (unique timestamp per day) with fwd_5 from mid shift(-5)-mid.
Inner-merge signed flows from trades at **exact same** timestamp only (sparse; zeros where
no trade on that product that tick).

For each gate regime {all, tight, loose}, lag L in 0..8: Pearson corr between
flow_P shifted by L rows within day (lead: flow at t-L vs fwd at t) and fwd_5.

Outputs JSON + long CSV for inspection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
TH = 2
DAYS = [1, 2, 3]
EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
V5200, V5300 = "VEV_5200", "VEV_5300"
LEGS = [EXTRACT, HYDRO, V5200, V5300, "VEV_4000", "VEV_6500"]


def gate_frame() -> pd.DataFrame:
    rows = []
    for day in DAYS:
        df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
        for p in (V5200, V5300):
            v = df[df["product"] == p].drop_duplicates("timestamp", keep="first")
            bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
            ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
            rows.append(
                pd.DataFrame(
                    {
                        "day": day,
                        "timestamp": v["timestamp"].values,
                        "product": p,
                        "spr": (ask - bid).astype(float).values,
                    }
                )
            )
    x = pd.concat(rows, ignore_index=True)
    a = x[x["product"] == V5200][["day", "timestamp", "spr"]].rename(columns={"spr": "s5200"})
    b = x[x["product"] == V5300][["day", "timestamp", "spr"]].rename(columns={"spr": "s5300"})
    m = a.merge(b, on=["day", "timestamp"], how="inner")
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    return m[["day", "timestamp", "tight"]]


def extract_panel() -> pd.DataFrame:
    px = []
    for day in DAYS:
        df = pd.read_csv(
            DATA / f"prices_round_4_day_{day}.csv",
            sep=";",
            usecols=["day", "timestamp", "product", "mid_price"],
        )
        px.append(df)
    u = pd.concat(px, ignore_index=True)
    u = u[u["product"] == EXTRACT].drop_duplicates(["day", "timestamp"]).sort_values(["day", "timestamp"])
    u["mid"] = pd.to_numeric(u["mid_price"], errors="coerce")
    u["fwd_5"] = u.groupby("day")["mid"].transform(lambda s: s.shift(-5) - s)
    return u[["day", "timestamp", "fwd_5"]]


def signed_flow_table() -> pd.DataFrame:
    px_list = []
    for d in DAYS:
        px_list.append(
            pd.read_csv(
                DATA / f"prices_round_4_day_{d}.csv",
                sep=";",
                usecols=["day", "timestamp", "product", "bid_price_1", "ask_price_1"],
            )
        )
    px = pd.concat(px_list, ignore_index=True)

    tr_list = []
    for d in DAYS:
        t = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        t["day"] = d
        tr_list.append(t)
    tr = pd.concat(tr_list, ignore_index=True).rename(columns={"symbol": "product"})
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr = tr.merge(
        px[["day", "timestamp", "product", "bid_price_1", "ask_price_1"]],
        on=["day", "timestamp", "product"],
        how="left",
    )
    bid1 = pd.to_numeric(tr["bid_price_1"], errors="coerce")
    ask1 = pd.to_numeric(tr["ask_price_1"], errors="coerce")
    qty = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0)
    agb = tr["price"] >= ask1
    ags = tr["price"] <= bid1
    tr["signed"] = 0.0
    tr.loc[agb, "signed"] = qty[agb]
    tr.loc[ags, "signed"] = -qty[ags]

    g = tr.groupby(["day", "timestamp", "product"], as_index=False)["signed"].sum()
    return g


def panel_wide() -> pd.DataFrame:
    gate = gate_frame()
    u = extract_panel().merge(gate, on=["day", "timestamp"], how="left")
    sf = signed_flow_table()
    for p in LEGS:
        sub = sf[sf["product"] == p][["day", "timestamp", "signed"]].rename(columns={"signed": f"sf_{p}"})
        u = u.merge(sub, on=["day", "timestamp"], how="left")
        u[f"sf_{p}"] = u[f"sf_{p}"].fillna(0.0)
    return u


def lag_corr(df: pd.DataFrame, col: str, target: str, lag: int) -> tuple[float, int]:
    parts = []
    for _, g in df.groupby("day"):
        if len(g) <= lag + 5:
            continue
        x = g[col].shift(lag).values
        y = g[target].values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 30:
            continue
        parts.append((x[m], y[m]))
    if not parts:
        return float("nan"), 0
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    if len(x) < 30 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan"), len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    return r, len(x)


def main() -> None:
    u = panel_wide()
    u = u.dropna(subset=["fwd_5", "tight"])
    rows = []
    max_lag = 8
    for regime, mask in [
        ("all", slice(None)),
        ("tight", u["tight"] == True),
        ("loose", u["tight"] == False),
    ]:
        sub = u.loc[mask] if mask is not slice(None) else u
        for lag in range(0, max_lag + 1):
            for p in LEGS:
                col = f"sf_{p}"
                r, n = lag_corr(sub, col, "fwd_5", lag)
                rows.append({"regime": regime, "lag": lag, "leg": p, "corr_fwd5": r, "n": n})

    long_df = pd.DataFrame(rows)
    long_df.to_csv(OUT / "r4_signed_flow_lag_corr_fwd5_x_gate.csv", index=False)

    best = long_df[np.isfinite(long_df["corr_fwd5"])].copy()
    best["absr"] = best["corr_fwd5"].abs()
    top = best.sort_values("absr", ascending=False).head(25)
    summary = {
        "description": "Corr(sf_leg.shift(lag), extract_fwd_5) on extract price timestamps; exact-ts signed flow merge.",
        "n_rows_panel": int(len(u)),
        "top_abs_corr": top.to_dict(orient="records"),
    }
    with open(OUT / "r4_signed_flow_lags_x_gate.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", OUT / "r4_signed_flow_lags_x_gate.json")


if __name__ == "__main__":
    main()
