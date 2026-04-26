#!/usr/bin/env python3
"""
Round 4 Phase 1 — Burst **orchestrator** attribution (suggested direction bullet 4).

For each (day, timestamp) with >=3 trade rows (any product):
- Sum **buy** quantity by buyer name; sum **sell** quantity by seller name at that tick.
- **buy_orchestrator** = argmax buyer by gross buy qty; **sell_orchestrator** = argmax seller by gross sell qty.
- **dominant_pair** = (buyer, seller) with max total notional (sum price*qty) across rows at tick.
- Merge **VELVETFRUIT_EXTRACT** forward mid K=20 from price tape (same convention as Phase 1).

Outputs:
- r4_burst_orchestrator_rows.csv — one row per burst tick with orchestrator fields + fwd20_ex
- r4_burst_orchestrator_fwd20_summary.csv — mean fwd20_ex by (day, buy_orchestrator, sell_orchestrator) with n>=10
- r4_burst_orchestrator_index.json — definitions + top pair counts
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
DAYS = (1, 2, 3)
K = 20
EXTRACT = "VELVETFRUIT_EXTRACT"
MIN_BURST = 3
MIN_CELL = 10


def load_prices_all() -> pd.DataFrame:
    parts = []
    for d in DAYS:
        df = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        df["day"] = d
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_mid_index(prices: pd.DataFrame) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    out: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for (d, sym), g in prices.groupby(["day", "product"]):
        g = g.sort_values("timestamp")
        out[(int(d), str(sym))] = {
            "ts": g["timestamp"].to_numpy(dtype=np.int64),
            "mid": pd.to_numeric(g["mid_price"], errors="coerce").to_numpy(dtype=float),
        }
    return out


def forward_mid_delta(
    idx: dict[tuple[int, str], dict[str, np.ndarray]],
    day: int,
    symbol: str,
    ts: int,
    k: int,
) -> float:
    key = (day, symbol)
    if key not in idx:
        return float("nan")
    ts_arr = idx[key]["ts"]
    mid_arr = idx[key]["mid"]
    j = int(np.searchsorted(ts_arr, ts, side="left"))
    if j >= len(ts_arr) or ts_arr[j] != ts:
        return float("nan")
    j2 = min(j + k, len(mid_arr) - 1)
    return float(mid_arr[j2] - mid_arr[j])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices_all()
    idx = build_mid_index(prices)

    trades = pd.concat([pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d) for d in DAYS], ignore_index=True)
    trades["product"] = trades["symbol"].astype(str)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce").fillna(0)

    gsz = trades.groupby(["day", "timestamp"]).size().rename("n_trades").reset_index()
    bursts = gsz[gsz["n_trades"] >= MIN_BURST]

    rows = []
    for _, br in bursts.iterrows():
        d, ts = int(br["day"]), int(br["timestamp"])
        sub = trades[(trades["day"] == d) & (trades["timestamp"] == ts)]
        buy_vol = Counter()
        sell_vol = Counter()
        pair_notional: defaultdict[tuple[str, str], float] = defaultdict(float)
        for _, r in sub.iterrows():
            q = float(r["quantity"])
            b, s = str(r["buyer"]), str(r["seller"])
            buy_vol[b] += q
            sell_vol[s] += q
            pair_notional[(b, s)] += abs(q) * float(r["price"])

        buy_orch = buy_vol.most_common(1)[0][0] if buy_vol else ""
        sell_orch = sell_vol.most_common(1)[0][0] if sell_vol else ""
        dom_pair = max(pair_notional, key=pair_notional.get) if pair_notional else ("", "")
        n_sym = sub["product"].nunique()
        fwd20 = forward_mid_delta(idx, d, EXTRACT, ts, K)
        rows.append(
            {
                "day": d,
                "timestamp": ts,
                "n_trades": int(len(sub)),
                "n_products": int(n_sym),
                "buy_orchestrator": buy_orch,
                "sell_orchestrator": sell_orch,
                "dominant_pair_buyer": dom_pair[0],
                "dominant_pair_seller": dom_pair[1],
                "fwd20_ex": fwd20,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r4_burst_orchestrator_rows.csv", index=False)

    summ = []
    for (d, bo, so), g in df.groupby(["day", "buy_orchestrator", "sell_orchestrator"]):
        v = g["fwd20_ex"].dropna()
        n = len(v)
        if n < MIN_CELL:
            continue
        summ.append(
            {
                "day": int(d),
                "buy_orchestrator": bo,
                "sell_orchestrator": so,
                "n_bursts": n,
                "mean_fwd20_ex": float(v.mean()),
                "median_fwd20_ex": float(v.median()),
            }
        )
    pd.DataFrame(summ).sort_values(["n_bursts", "mean_fwd20_ex"], ascending=[False, False]).to_csv(
        OUT / "r4_burst_orchestrator_fwd20_summary.csv", index=False
    )

    # Pooled Mark01->Mark22 as orchestrator pair on burst ticks
    m122 = df[(df["buy_orchestrator"] == "Mark 01") & (df["sell_orchestrator"] == "Mark 22")]
    v = m122["fwd20_ex"].dropna()
    top_pairs = (
        df.groupby(["dominant_pair_buyer", "dominant_pair_seller"])
        .size()
        .reset_index(name="n_burst_ticks")
        .sort_values("n_burst_ticks", ascending=False)
        .head(15)
    )

    meta = {
        "n_burst_timestamps": int(len(df)),
        "mark01_buy_mark22_sell_orchestrator_ticks": int(len(m122)),
        "pooled_mean_fwd20_ex_when_M01_buy_orch_M22_sell_orch": float(v.mean()) if len(v) else float("nan"),
        "median_fwd20_ex_same": float(v.median()) if len(v) else float("nan"),
        "top_dominant_pairs": top_pairs.to_dict(orient="records"),
        "definition": "Orchestrator = party with max gross traded qty on that side at the timestamp across all products; burst = >=3 rows same (day,timestamp).",
    }
    (OUT / "r4_burst_orchestrator_index.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2)[:2000])


if __name__ == "__main__":
    main()
