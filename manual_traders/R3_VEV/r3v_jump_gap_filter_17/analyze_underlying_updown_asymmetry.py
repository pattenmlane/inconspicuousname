"""
Tape analysis: OTM/ATM voucher mid response asymmetry to extract up vs down moves.

Method: for each day CSV, per-timestamp forward differences of wall_mid(VEV) and extract mid.
Regress dV on dS separately for dS>0 and dS<0 (WLS optional); report slope ratio and IV proxy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
VEVS = [
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
]
EX = "VELVETFRUIT_EXTRACT"


def wall_mid_from_row(r: pd.Series) -> float | None:
    b1, a1 = r.get("bid_price_1"), r.get("ask_price_1")
    try:
        bb, ba = int(b1), int(a1)
    except (TypeError, ValueError):
        return None
    if bb <= 0 or ba <= 0 or ba < bb:
        return None
    return 0.5 * (bb + ba)


def day_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    return df[df["product"].isin(VEVS + [EX])].copy()


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows = []
    for d in (0, 1, 2):
        path = DATA / f"prices_round_3_day_{d}.csv"
        df = day_frame(path)
        # pivot: timestamp -> product -> mid
        mids: dict[tuple[int, str], float] = {}
        for _, r in df.iterrows():
            sym = str(r["product"])
            ts = int(r["timestamp"])
            m = wall_mid_from_row(r)
            if m is None:
                continue
            mids[(ts, sym)] = float(m)
        times = sorted({t for t, _ in mids.keys()})
        if len(times) < 2:
            continue
        for i in range(1, len(times)):
            t0, t1 = times[i - 1], times[i]
            if t1 - t0 > 5_000_000:  # ignore huge gaps (session breaks)
                continue
            if (t0, EX) not in mids or (t1, EX) not in mids:
                continue
            dS = mids[(t1, EX)] - mids[(t0, EX)]
            if abs(dS) < 1e-6:
                continue
            for sym in VEVS:
                if (t0, sym) not in mids or (t1, sym) not in mids:
                    continue
                dV = mids[(t1, sym)] - mids[(t0, sym)]
                K = int(sym.split("_")[1])
                rows.append(
                    {
                        "day": d,
                        "strike": K,
                        "dS": dS,
                        "dV": dV,
                        "up": 1 if dS > 0 else 0,
                    }
                )

    pool = pd.DataFrame(rows)
    if pool.empty:
        raise SystemExit("no rows")

    summary: dict = {"by_strike_pooled": {}, "meta": {}}
    for K in sorted(pool["strike"].unique()):
        sub = pool[pool["strike"] == K]
        up = sub[sub["dS"] > 0]
        dn = sub[sub["dS"] < 0]
        for name, part in (("up", up), ("down", dn)):
            if len(part) < 30:
                beta = float("nan")
            else:
                X = part["dS"].to_numpy(dtype=float).reshape(-1, 1)
                y = part["dV"].to_numpy(dtype=float)
                b, _, _, _ = lstsq(X, y, rcond=None)
                beta = float(b[0])
            summary["by_strike_pooled"].setdefault(str(int(K)), {})[f"beta_{name}"] = beta
            summary["by_strike_pooled"][str(int(K))][f"n_{name}"] = int(len(part))

    # Pooled OLS per strike: asymmetry ratio |beta_up| / |beta_down| (avoid div0)
    asym: dict[str, float] = {}
    for kstr, dct in summary["by_strike_pooled"].items():
        bu = dct.get("beta_up", float("nan"))
        bd = dct.get("beta_down", float("nan"))
        if bu == 0 or not np.isfinite(bu) or not np.isfinite(bd):
            asym[kstr] = 1.0
        else:
            asym[kstr] = float(abs(bu) / (abs(bd) + 1e-9))

    summary["asymmetry_ratio_abs_beta_up_over_down"] = asym

    # Large-move window: |dS| above pooled 80th percentile (approx jumpy regime)
    thr = float(np.quantile(np.abs(pool["dS"].to_numpy()), 0.80))
    big = pool[np.abs(pool["dS"]) >= thr]
    summary["meta"]["large_move_absdS_threshold_p80"] = thr
    summary["by_strike_large_move"] = {}
    for K in sorted(big["strike"].unique()):
        sub = big[big["strike"] == K]
        up = sub[sub["dS"] > 0]
        dn = sub[sub["dS"] < 0]
        bpool = 0.5 * (
            summary["by_strike_pooled"][str(int(K))]["beta_up"]
            + summary["by_strike_pooled"][str(int(K))]["beta_down"]
        )
        dct: dict = {}
        for label, part in (("up", up), ("down", dn)):
            if len(part) < 5:
                dct[f"mean_resid_{label}"] = float("nan")
                dct[f"n_{label}"] = int(len(part))
                continue
            r = float((part["dV"] - bpool * part["dS"]).mean())
            dct[f"mean_resid_{label}"] = r
            dct[f"n_{label}"] = int(len(part))
        dct["bpool_ols"] = float(bpool)
        summary["by_strike_large_move"][str(int(K))] = dct

    out_json = out_dir / "analysis_underlying_updown_asymmetry.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(out_json, "written")


if __name__ == "__main__":
    main()
