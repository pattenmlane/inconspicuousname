"""
Load hold-1 `true_fv` series from round3work/fairs and merge on timestamp (day 39 probes).

Each product was uploaded separately; timestamps 0..99900 align across all Round-3 probes
in this repo (inner-join intersection = 1000 rows).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]


def _repo_root() -> Path:
    # .../round3work/plotting/truemethod/common/this_file → repo = parents[4]
    return Path(__file__).resolve().parents[4]


def _first_csv(fair_root: Path, glob_rel: str) -> Path:
    matches = sorted(fair_root.glob(glob_rel))
    if not matches:
        raise FileNotFoundError(f"No CSV under {fair_root}: {glob_rel}")
    return matches[0]


def load_true_fv_wide(repo: Path | None = None) -> pd.DataFrame:
    """
    Columns: timestamp (index), S (extract true_fv), VEV_* (true_fv), day=39, dte=5 (Round 3 final TTE).
    Also attaches mid_price gaps: mid_<v> from each probe CSV for diagnostics.
    """
    repo = repo or _repo_root()
    fair_root = repo / "round3work" / "fairs"

    ex_path = _first_csv(fair_root, "VELVETFRUIT_EXTRACTfair/**/*VELVETFRUIT_EXTRACT_true_fv_day39.csv")
    ex = pd.read_csv(ex_path, sep=";")
    base = ex[["timestamp", "true_fv", "mid_price"]].copy()
    base = base.rename(columns={"true_fv": "S", "mid_price": "mid_S"})
    base["timestamp"] = base["timestamp"].astype(int)

    for v in VOUCHERS:
        k = v.split("_")[1]
        path = _first_csv(fair_root, f"{k}fair/**/*{v}_true_fv_day39.csv")
        d = pd.read_csv(path, sep=";")
        sub = d[["timestamp", "true_fv", "mid_price"]].copy()
        sub = sub.rename(columns={"true_fv": v, "mid_price": f"mid_{v}"})
        sub["timestamp"] = sub["timestamp"].astype(int)
        base = base.merge(sub, on="timestamp", how="inner")

    base["day"] = 39
    base["dte"] = 5
    return base.set_index("timestamp").sort_index()


def fv_mid_gap_summary(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "product": "VELVETFRUIT_EXTRACT",
            "mean_abs_fv_minus_mid": float((wide["S"] - wide["mid_S"]).abs().mean()),
        }
    )
    for v in VOUCHERS:
        colm = f"mid_{v}"
        if colm in wide.columns:
            rows.append(
                {
                    "product": v,
                    "mean_abs_fv_minus_mid": float((wide[v] - wide[colm]).abs().mean()),
                }
            )
    return pd.DataFrame(rows)
