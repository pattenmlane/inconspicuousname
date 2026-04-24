"""
VEV_5200 only: switch_mean / current_theo_diff distributions for each of the 12 parabola
coefficient sets (six branches × all_strikes vs near_5000_5500).

- Historical days 0–2: original_method (Brent T from wind/no-wind plot_iv) and
  test_implementation (notebook T from wind/no-wind nb_method_core).
- Fair day 39: truemethod (merged extract + VEV_5200 prices books, DTE=5 probe T).

Does not edit calibration.json. See each folder's threshold_suggestions.txt.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
_WORK = _ROOT.parent
_REPO = _WORK.parent.parent.parent
_DATA = _REPO / "Prosperity4Data" / "ROUND_3"
_PARABOLA_JSON = _REPO / "round3work" / "plotting" / "parabola_fits_six_branches.json"
_CAL = _WORK / "calibration.json"

sys.path.insert(0, str(_WORK))
from frankfurt_iv_scalp_core import (  # noqa: E402
    book_from_row,
    compute_option_indicators,
    load_calibration,
    synthetic_walls_if_missing,
)

K = 5200
OPT = "VEV_5200"
U = "VELVETFRUIT_EXTRACT"
STEP = 50


def _load_plot_iv(combined_dir: Path) -> Any:
    p = combined_dir / "plot_iv_smile_round3.py"
    name = f"piv_{abs(hash(str(p))) % 1_000_000}"
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise RuntimeError(str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_nb(nb_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"nb_{nb_path.parent.name}", nb_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(str(nb_path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _make_t_probe(winding: bool) -> Callable[[int], float]:
    """DTE=5 Round-3 probe session (same as truemethod)."""
    import math

    def intraday_progress(ts: int) -> float:
        return (int(ts) // 100) / 10_000.0

    d0 = 5.0

    def t_years(ts: int) -> float:
        if winding:
            d_eff = max(d0 - intraday_progress(ts), 1e-6)
        else:
            d_eff = d0
        return d_eff / 365.0

    return t_years


def _fair_day39_merged() -> pd.DataFrame:
    """timestamp-indexed: S from extract mid, VEV_5200 full book rows as dicts via groupby."""
    fairs = _REPO / "round3work" / "fairs"

    def _first(glob_pat: str) -> Path:
        m = sorted(fairs.glob(glob_pat))
        if not m:
            raise FileNotFoundError(glob_pat)
        return m[0]

    ex = pd.read_csv(_first("VELVETFRUIT_EXTRACTfair/**/prices_round_3_day_39.csv"), sep=";")
    ex = ex[ex["product"] == U][["timestamp", "mid_price"]].copy()
    ex = ex.rename(columns={"mid_price": "S_mid"})
    ex["timestamp"] = ex["timestamp"].astype(int)

    v = pd.read_csv(_first("5200fair/**/prices_round_3_day_39.csv"), sep=";")
    v = v[v["product"] == OPT].copy()
    v["timestamp"] = v["timestamp"].astype(int)

    m = ex.merge(v, on="timestamp", how="inner").sort_values("timestamp")
    return m


def collect_historical_plot_iv(
    piv: Any, coeffs: list[float], cal_base: dict[str, Any]
) -> tuple[list[float], list[float]]:
    cal = dict(cal_base)
    cal["coeffs_high_to_low"] = coeffs
    sw_list: list[float] = []
    diff_list: list[float] = []

    for day in (0, 1, 2):
        ema: dict[str, float] = {}
        df = pd.read_csv(_DATA / f"prices_round_3_day_{day}.csv", sep=";")
        ts_list = sorted(df["timestamp"].unique())[::STEP]
        for ts in ts_list:
            g = df[df["timestamp"] == ts]
            if OPT not in g["product"].values or U not in g["product"].values:
                continue
            ro = g[g["product"] == OPT].iloc[0].to_dict()
            ru = g[g["product"] == U].iloc[0].to_dict()
            _, _, bid_w, ask_w, bb, ba, wm = book_from_row(ro)
            _, _, _, _, ubb, uba, _ = book_from_row(ru)
            if ubb is None or uba is None:
                continue
            u_mid = 0.5 * float(ubb) + 0.5 * float(uba)
            bid_w, ask_w, wm, bb, ba = synthetic_walls_if_missing(bid_w, ask_w, bb, ba)
            if wm is None or bb is None or ba is None:
                continue
            T = float(piv.t_years_effective(day, int(ts)))
            ind = compute_option_indicators(cal, ema, u_mid, K, T, float(wm), float(bb), float(ba), OPT)
            if ind.get("switch_mean") is not None:
                sw_list.append(float(ind["switch_mean"]))
            if ind.get("current_theo_diff") is not None:
                diff_list.append(float(ind["current_theo_diff"]))
    return sw_list, diff_list


def collect_historical_nb(
    nb: Any, coeffs: list[float], cal_base: dict[str, Any]
) -> tuple[list[float], list[float]]:
    cal = dict(cal_base)
    cal["coeffs_high_to_low"] = coeffs
    sw_list: list[float] = []
    diff_list: list[float] = []

    for day in (0, 1, 2):
        ema: dict[str, float] = {}
        wf = nb.load_day_wide(day).sort_index()
        mp = nb.index_map_timestamp_to_row_idx(wf)
        wsub = nb.subsample_wide(wf, step=STEP)
        d0 = int(nb.dte_from_csv_day(day))
        df = pd.read_csv(_DATA / f"prices_round_3_day_{day}.csv", sep=";")
        for ts, _row in wsub.iterrows():
            ts_i = int(ts)
            g = df[df["timestamp"] == ts_i]
            if OPT not in g["product"].values or U not in g["product"].values:
                continue
            ro = g[g["product"] == OPT].iloc[0].to_dict()
            ru = g[g["product"] == U].iloc[0].to_dict()
            _, _, bid_w, ask_w, bb, ba, wm = book_from_row(ro)
            _, _, _, _, ubb, uba, _ = book_from_row(ru)
            if ubb is None or uba is None:
                continue
            u_mid = 0.5 * float(ubb) + 0.5 * float(uba)
            bid_w, ask_w, wm, bb, ba = synthetic_walls_if_missing(bid_w, ask_w, bb, ba)
            if wm is None or bb is None or ba is None:
                continue
            t_idx = mp[ts_i]
            T = float(nb.expiration_time_years(d0, t_idx))
            ind = compute_option_indicators(cal, ema, u_mid, K, T, float(wm), float(bb), float(ba), OPT)
            if ind.get("switch_mean") is not None:
                sw_list.append(float(ind["switch_mean"]))
            if ind.get("current_theo_diff") is not None:
                diff_list.append(float(ind["current_theo_diff"]))
    return sw_list, diff_list


def collect_fair39(
    coeffs: list[float], cal_base: dict[str, Any], winding: bool
) -> tuple[list[float], list[float]]:
    cal = dict(cal_base)
    cal["coeffs_high_to_low"] = coeffs
    tfn = _make_t_probe(winding)
    merged = _fair_day39_merged()
    ema: dict[str, float] = {}
    sw_list: list[float] = []
    diff_list: list[float] = []

    book_cols = (
        "day",
        "timestamp",
        "product",
        "bid_price_1",
        "bid_volume_1",
        "bid_price_2",
        "bid_volume_2",
        "bid_price_3",
        "bid_volume_3",
        "ask_price_1",
        "ask_volume_1",
        "ask_price_2",
        "ask_volume_2",
        "ask_price_3",
        "ask_volume_3",
        "mid_price",
        "profit_and_loss",
    )

    for _, row in merged.iterrows():
        ts_i = int(row["timestamp"])
        ro_opt = {c: row.get(c) for c in book_cols if c in row.index}
        ro_opt["product"] = OPT
        _, _, bid_w, ask_w, bb, ba, wm = book_from_row(ro_opt)
        u_mid = float(row["S_mid"])
        bid_w, ask_w, wm, bb, ba = synthetic_walls_if_missing(bid_w, ask_w, bb, ba)
        if wm is None or bb is None or ba is None:
            continue
        T = float(tfn(ts_i))
        ind = compute_option_indicators(cal, ema, u_mid, K, T, float(wm), float(bb), float(ba), OPT)
        if ind.get("switch_mean") is not None:
            sw_list.append(float(ind["switch_mean"]))
        if ind.get("current_theo_diff") is not None:
            diff_list.append(float(ind["current_theo_diff"]))
    return sw_list, diff_list


def _write_suggestions(out_dir: Path, model_id: str, sw: np.ndarray, dd: np.ndarray, meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"VEV_5200 — {model_id}",
        f"n_switch={len(sw)} n_diff={len(dd)}",
        json.dumps(meta, indent=2),
        "",
        "switch_mean quantiles (IV_SCALPING_THR is compared to this):",
    ]
    if len(sw) > 0:
        for q in (0.5, 0.7, 0.8, 0.9, 0.95):
            lines.append(f"  p{int(q * 100)}: {float(np.quantile(sw, q)):.6f}")
        lines.append("")
        lines.append("Suggested IV_SCALPING_THR (align gate with data):")
        lines.append(f"  conservative ~ p80: {float(np.quantile(sw, 0.8)):.4f}")
        lines.append(f"  moderate ~ p70: {float(np.quantile(sw, 0.7)):.4f}")
        lines.append(f"  loose ~ p50: {float(np.quantile(sw, 0.5)):.4f}")
    else:
        lines.append("  (no valid switch_mean samples)")

    lines.extend(["", "|current_theo_diff| quantiles (wall_mid - BS(smile)):", ""])
    if len(dd) > 0:
        ad = np.abs(dd)
        for q in (0.5, 0.9, 0.95):
            lines.append(f"  p{int(q * 100)}: {float(np.quantile(ad, q)):.6f}")
    else:
        lines.append("  (no valid theo_diff samples)")

    lines.extend(
        [
            "",
            "Reference: polished defaults IV_SCALPING_THR=0.7, THR_OPEN=0.5, THR_CLOSE=0.",
        ]
    )
    (out_dir / "threshold_suggestions.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cal_base = load_calibration(_CAL)
    para = json.loads(_PARABOLA_JSON.read_text(encoding="utf-8"))

    piv_wind = _load_plot_iv((_REPO / "round3work/plotting/original_method/wind_down/combined_analysis").resolve())
    piv_nowind = _load_plot_iv(_REPO / "round3work/plotting/original_method/no_wind_down/combined_analysis")
    nb_wind = _load_nb(_REPO / "round3work/plotting/test_implementation/wind_down/nb_method_core.py")
    nb_nowind = _load_nb(_REPO / "round3work/plotting/test_implementation/no_wind_down/nb_method_core.py")

    summary: list[dict[str, Any]] = []

    for branch, subsets in para["branches"].items():
        for subset_name, payload in subsets.items():
            model_id = f"{branch}__{subset_name}"
            coeffs = payload["coeffs_high_to_low"]
            meta = {
                "branch": branch,
                "subset": subset_name,
                "parabola_rmse": payload.get("rmse"),
                "parabola_n": payload.get("n_points"),
                "iv_engine": payload.get("iv_engine"),
            }

            if branch.startswith("original_method"):
                piv = piv_wind if "wind_down" in branch else piv_nowind
                sw, diff = collect_historical_plot_iv(piv, coeffs, cal_base)
                meta["data"] = "historical ROUND_3 days 0-2"
                meta["t_source"] = "plot_iv_smile_round3.t_years_effective"
            elif branch.startswith("test_implementation"):
                nb = nb_wind if "wind_down" in branch else nb_nowind
                sw, diff = collect_historical_nb(nb, coeffs, cal_base)
                meta["data"] = "historical ROUND_3 days 0-2"
                meta["t_source"] = "nb_method_core.expiration_time_years"
            elif branch.startswith("truemethod"):
                winding = "wind_down" in branch
                sw, diff = collect_fair39(coeffs, cal_base, winding)
                meta["data"] = "fairs day 39 VEV_5200 book + extract mid"
                meta["t_source"] = "probe DTE=5 " + ("wind" if winding else "no_wind")
            else:
                raise RuntimeError(branch)

            swa = np.asarray(sw, dtype=float)
            dda = np.asarray(diff, dtype=float)
            _write_suggestions(_ROOT / model_id.replace("/", "_"), model_id, swa, dda, meta)
            summary.append(
                {
                    "model_id": model_id,
                    "n_switch": int(len(swa)),
                    "n_diff": int(len(dda)),
                    "switch_p70": float(np.quantile(swa, 0.7)) if len(swa) else None,
                    "switch_p80": float(np.quantile(swa, 0.8)) if len(swa) else None,
                    "switch_p95": float(np.quantile(swa, 0.95)) if len(swa) else None,
                    "abs_theo_diff_p90": float(np.quantile(np.abs(dda), 0.9)) if len(dda) else None,
                }
            )
            print("OK", model_id, "n=", len(swa))

    (_ROOT / "summary_all_models.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", _ROOT / "summary_all_models.json")


if __name__ == "__main__":
    main()
