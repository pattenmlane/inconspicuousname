"""
Cross-section (per timestamp): VEV top-of-book spread vs BS vega at model_iv,
same timing as trader_v0. Pooled stats inform optional per-strike half-spread offsets.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
U = "VELVETFRUIT_EXTRACT"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOU = [f"VEV_{k}" for k in STRIKES]
_COEFFS = (0.14215151147708086, -0.0016298611395181932, 0.23576325646627055)


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def model_iv(S: float, K: float, T: float) -> float:
    if S <= 0 or K <= 0 or T <= 0:
        return 0.25
    m_t = math.log(K / S) / math.sqrt(T)
    a, b, c = _COEFFS
    return max(((a * m_t) + b) * m_t + c, 1e-4)


def vega(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12 or S <= 0 or K <= 0:
        return 0.0
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    return S * _pdf(d1) * math.sqrt(T)


def t_years(day: int, ts: int) -> float:
    dte = max(float(8 - int(day)) - (int(ts) // 100) / 10_000.0, 1e-6)
    return dte / 365.0


def top_spread(df: pd.DataFrame, product: str, ts: int) -> float | None:
    r = df[(df["timestamp"] == ts) & (df["product"] == product)]
    if r.empty:
        return None
    row = r.iloc[0]
    bps, aps = [], []
    for i in (1, 2, 3):
        bp, ap = row.get(f"bid_price_{i}"), row.get(f"ask_price_{i}")
        bv, av = row.get(f"bid_volume_{i}"), row.get(f"ask_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and int(bv) > 0:
            bps.append(float(bp))
        if pd.notna(ap) and pd.notna(av) and int(av) > 0:
            aps.append(float(ap))
    if not bps or not aps:
        return None
    return min(aps) - max(bps)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    step = 50
    corrs: list[float] = []
    rows_vega: list[tuple[str, float]] = []
    rows_spread: list[tuple[str, float]] = []

    for day in (0, 1, 2):
        raw = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        ts_list = sorted(raw["timestamp"].unique())[::step]
        for ts in ts_list:
            u = raw[(raw["timestamp"] == ts) & (raw["product"] == U)]
            if u.empty:
                continue
            mid_u = float(u.iloc[0]["mid_price"])
            if not np.isfinite(mid_u) or mid_u <= 0:
                continue
            T = t_years(day, int(ts))
            spreads, vegas, names = [], [], []
            for v in VOU:
                sp = top_spread(raw, v, int(ts))
                if sp is None or sp < 0:
                    continue
                K = float(v.split("_")[1])
                sig = model_iv(mid_u, K, T)
                vg = vega(mid_u, K, T, sig)
                spreads.append(float(sp))
                vegas.append(float(vg))
                names.append(v)
                rows_vega.append((v, float(vg)))
                rows_spread.append((v, float(sp)))
            if len(spreads) < 6:
                continue
            x, y = np.array(vegas), np.array(spreads)
            if float(np.std(x)) < 1e-9 or float(np.std(y)) < 1e-9:
                continue
            corrs.append(float(np.corrcoef(x, y)[0, 1]))

    mean_by_v: dict[str, list[float]] = {v: [] for v in VOU}
    for v, s in rows_spread:
        mean_by_v[v].append(s)
    pooled_mean = float(np.mean([s for _, s in rows_spread])) if rows_spread else 1.0
    strike_offsets = {}
    for v in VOU:
        xs = mean_by_v[v]
        if not xs:
            strike_offsets[v] = 0.0
            continue
        strike_offsets[v] = float(np.mean(xs) - pooled_mean)

    summ = {
        "method": f"Subsample every {step} timestamps; per tick corr(vega_model, top_spread) across strikes with valid book; vega at model_iv.",
        "mean_within_tick_corr_vega_spread": float(np.mean(corrs)) if corrs else None,
        "std_within_tick_corr": float(np.std(corrs)) if corrs else None,
        "n_tick_corrs": int(len(corrs)),
        "mean_spread_minus_pool_by_voucher": strike_offsets,
        "pooled_mean_spread": pooled_mean,
        "shock_spread_note": "VEV_5200 mean_dspread_shock - mean_dspread_calm ~ +0.011 from leadlag_shock script (signed shock widening focal).",
    }
    pd.DataFrame(rows_spread, columns=["voucher", "spread"]).groupby("voucher").mean().reset_index().to_csv(
        OUT / "mean_top_spread_by_voucher.csv", index=False
    )
    (OUT / "spread_vega_crosssection_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("Wrote", OUT / "spread_vega_crosssection_summary.json", "mean corr", summ.get("mean_within_tick_corr_vega_spread"))


if __name__ == "__main__":
    main()
