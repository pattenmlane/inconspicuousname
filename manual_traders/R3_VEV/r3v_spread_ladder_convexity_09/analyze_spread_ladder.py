#!/usr/bin/env python3
"""Offline tape analysis for r3v_spread_ladder_convexity_09 (Round 3 CSVs only)."""
from __future__ import annotations

import csv
import json
import math
import statistics as stats
from collections import defaultdict, deque
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[3] / "Prosperity4Data" / "ROUND_3"

STRIKES_FLY = (5000, 5100, 5200)
SYMS = tuple(f"VEV_{k}" for k in STRIKES_FLY)
U = "VELVETFRUIT_EXTRACT"


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def implied_vol(S: float, K: float, T: float, r: float, price: float) -> float | None:
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-6:
        return None
    lo, hi = 1e-4, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_call_price(S, K, T, r, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    return norm_cdf(d1)


def tte_days_for_csv_day(csv_day: int) -> int:
    # round3description.txt: tape day 0 -> TTE 8d, day 1 -> 7d, day 2 -> 6d at open of that history slice
    return 8 - csv_day


def load_day(csv_day: int) -> list[dict]:
    path = DATA / f"prices_round_3_day_{csv_day}.csv"
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def aggregate_by_timestamp(rows: list[dict]) -> dict[int, dict[str, tuple[float, float, float]]]:
    """timestamp -> product -> (bid, ask, mid)."""
    by_ts: dict[int, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    for row in rows:
        ts = int(row["timestamp"])
        prod = row["product"]
        try:
            b1 = float(row["bid_price_1"] or 0)
            a1 = float(row["ask_price_1"] or 0)
        except (TypeError, ValueError):
            continue
        if b1 <= 0 or a1 <= 0:
            continue
        mid = 0.5 * (b1 + a1)
        by_ts[ts][prod] = (b1, a1, mid)
    return by_ts


def butterfly_mid(by_ts: dict[int, dict[str, tuple[float, float, float]]]) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for ts in sorted(by_ts):
        d = by_ts[ts]
        if U not in d:
            continue
        if not all(s in d for s in SYMS):
            continue
        m5, m51, m52 = d[SYMS[0]][2], d[SYMS[1]][2], d[SYMS[2]][2]
        out.append((ts, m5 + m52 - 2.0 * m51))
    return out


def rolling_z(series: list[float], win: int) -> list[float | None]:
    out: list[float | None] = []
    dq: deque[float] = deque()
    s = 0.0
    ss = 0.0
    for x in series:
        dq.append(x)
        s += x
        ss += x * x
        if len(dq) > win:
            y = dq.popleft()
            s -= y
            ss -= y * y
        if len(dq) < win:
            out.append(None)
            continue
        n = len(dq)
        mean = s / n
        var = max(ss / n - mean * mean, 0.0)
        std = math.sqrt(var)
        out.append(None if std < 1e-9 else (x - mean) / std)
    return out


def z_threshold_stats(series: list[float], z_grid: list[float], win: int) -> dict[str, dict[str, float]]:
    """When rolling z of butterfly exceeds thr, mean next-tick change in bf (fade = expect negative)."""
    zseq = rolling_z(series, win)
    out: dict[str, dict[str, float]] = {}
    for thr in z_grid:
        nxt: list[float] = []
        for i in range(len(series) - 1):
            z0 = zseq[i]
            if z0 is None or z0 <= thr:
                continue
            nxt.append(series[i + 1] - series[i])
        out[f"z_gt_{thr}"] = {
            "count": float(len(nxt)),
            "mean_next_bf_delta": float(stats.mean(nxt)) if nxt else float("nan"),
        }
    return out


def main() -> None:
    summary: dict = {
        "tte_mapping": "From round3work/round3description.txt: historical tape day d uses TTE = 8 - d days at the start of that day slice (example: day 1 -> 8d, day 2 -> 7d, day 3 -> 6d in doc; our CSVs are day_0, day_1, day_2 so TTE = 8,7,6).",
        "iv_method": "Black-Scholes European call; bisection implied vol from mid; r=0; T = (TTE_days)/365 with TTE from mapping above.",
        "greeks": "Analytical BS delta per strike using implied vol at each timestamp.",
    }

    rows_by_day: dict[int, list] = {}
    for d in (0, 1, 2):
        rows_by_day[d] = load_day(d)

    per_day_rows: list[dict] = []
    iv_bump_samples: list[dict] = []
    for csv_day in (0, 1, 2):
        by_ts = aggregate_by_timestamp(rows_by_day[csv_day])
        tte_d = tte_days_for_csv_day(csv_day)
        T = max(tte_d, 1) / 365.0
        bf_ts = butterfly_mid(by_ts)
        series = [x for _, x in bf_ts]
        zseq = rolling_z(series, 500)
        z_valid = [z for z in zseq if z is not None]
        per_day_rows.append(
            {
                "csv_day": csv_day,
                "tte_days_open": tte_d,
                "n_ticks": len(series),
                "bf_mid_mean": round(stats.mean(series), 4),
                "bf_mid_std": round(stats.pstdev(series), 4) if len(series) > 1 else 0.0,
                "z_mean": round(stats.mean(z_valid), 4) if z_valid else None,
                "z_p99": round(sorted(z_valid)[int(0.99 * (len(z_valid) - 1))], 4) if len(z_valid) > 100 else None,
            }
        )
        # Small sample: timestamps with |z| > 2.5 for audit trail
        for (ts, bf), z in zip(bf_ts, zseq):
            if z is not None and abs(z) > 2.5:
                iv_bump_samples.append({"csv_day": csv_day, "timestamp": ts, "bf_mid": round(bf, 3), "bf_z": round(z, 3)})
        # IV bump mean + BS delta means (single pass)
        bumps: list[float] = []
        delta_samples: dict[str, list[float]] = {s: [] for s in SYMS}
        for ts in sorted(by_ts):
            d = by_ts[ts]
            if U not in d or not all(s in d for s in SYMS):
                continue
            S = d[U][2]
            iv_triple = []
            ok = True
            for lab, strike in zip(SYMS, STRIKES_FLY):
                mid = d[lab][2]
                iv = implied_vol(S, float(strike), T, 0.0, mid)
                if iv is None:
                    ok = False
                    break
                iv_triple.append(iv)
            if ok:
                bumps.append(iv_triple[1] - 0.5 * (iv_triple[0] + iv_triple[2]))
                for lab, strike, iv in zip(SYMS, STRIKES_FLY, iv_triple):
                    delta_samples[lab].append(bs_delta(S, float(strike), T, 0.0, iv))
        per_day_rows[-1]["iv_bump_mean"] = round(stats.mean(bumps), 6) if bumps else None
        per_day_rows[-1]["iv_bump_std"] = round(stats.pstdev(bumps), 6) if len(bumps) > 1 else None
        for lab in SYMS:
            ds = delta_samples[lab]
            per_day_rows[-1][f"delta_{lab}_mean"] = round(stats.mean(ds), 5) if ds else None

    bf_path = OUT_DIR / "bf_mid_zscore_summary_by_day.csv"
    with bf_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_day_rows[0].keys()))
        w.writeheader()
        w.writerows(per_day_rows)

    iv_path = OUT_DIR / "bf_high_z_sample_ticks.csv"
    with iv_path.open("w", newline="") as fp:
        cols = ["csv_day", "timestamp", "bf_mid", "bf_z"]
        w_iv = csv.DictWriter(fp, fieldnames=cols)
        w_iv.writeheader()
        for row in iv_bump_samples[:800]:
            w_iv.writerow(row)

    grid_results = {}
    for csv_day in (0, 1, 2):
        by_ts = aggregate_by_timestamp(rows_by_day[csv_day])
        series = [x for _, x in butterfly_mid(by_ts)]
        grid_results[str(csv_day)] = z_threshold_stats(series, [1.5, 2.0, 2.5, 3.0], 500)

    summary["butterfly_mid_z"] = {
        "rolling_window": 500,
        "conditional_mean_next_bf_delta_when_z_gt_threshold": grid_results,
        "note": "If convexity is overstated at high z, next bf delta tends negative (short fly profits).",
    }

    meta_path = OUT_DIR / "analysis_summary_meta.json"
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote", bf_path, "rows", len(per_day_rows))
    print("Wrote", iv_path, "rows", min(len(iv_bump_samples), 800))
    print("Wrote", meta_path)


if __name__ == "__main__":
    main()
