"""
Global parabola fit only (no full figure regeneration) for six analysis branches × two strike sets.

For each branch we pool points and fit  IV ≈ poly₂(m_t)  with
  m_t = log(K/S) / sqrt(T)
using that branch's own T and IV conventions:
  - original_method wind / no_wind: Brent implied_vol_call from the respective plot_iv_smile_round3.
  - test_implementation wind / no_wind: notebook bisection IV + that branch's expiration_time_years.
  - truemethod wind / no_wind: Brent on true_fv + probe DTE=5 with / without intraday wind.

Strike sets:
  - all: 4000 … 6500 (10 vouchers)
  - near_5000_5500: 5000, 5100, …, 5500 only

Output: parabola_fits_six_branches.json next to this script.

Run from repo:
  python3 round3work/plotting/fit_parabola_six_branches.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PLOT = Path(__file__).resolve().parent
REPO = PLOT.parent.parent

STRIKES_ALL = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
STRIKES_NEAR = [5000, 5100, 5200, 5300, 5400, 5500]
STEP = 20


def _vouchers(strikes: list[int]) -> list[str]:
    return [f"VEV_{k}" for k in strikes]


def _load_plot_iv(combined_dir: Path) -> Any:
    path = combined_dir / "plot_iv_smile_round3.py"
    name = f"piv_{abs(hash(str(path))) % 10_000_000}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_nb(nb_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"nb_{nb_path.parent.name}", nb_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(nb_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_mt_historical(piv: Any, strikes: list[int], step: int = STEP) -> dict[str, Any]:
    """Frankfurt-style m_t with Brent IV (plot_iv_smile_round3)."""
    xs: list[float] = []
    ys: list[float] = []
    vouchers = _vouchers(strikes)
    for day in (0, 1, 2):
        wide = piv.load_day_wide(day).sort_index()
        wsub = piv.subsample_wide(wide, step=step)
        for ts, row in wsub.iterrows():
            day_i = int(row["day"])
            ts_i = int(ts)
            T = float(piv.t_years_effective(day_i, ts_i))
            if T <= 0:
                continue
            sqrt_t = math.sqrt(T)
            S = float(row["S"])
            if S <= 0:
                continue
            for v in vouchers:
                if v not in row.index:
                    continue
                K = float(v.split("_")[1])
                mid = float(row[v])
                iv = float(piv.implied_vol_call(mid, S, K, T, 0.0))
                if not np.isfinite(iv):
                    continue
                m_t = math.log(K / S) / sqrt_t
                if not np.isfinite(m_t):
                    continue
                xs.append(m_t)
                ys.append(iv)
    xf = np.asarray(xs, dtype=float)
    yf = np.asarray(ys, dtype=float)
    m = np.isfinite(xf) & np.isfinite(yf)
    xf, yf = xf[m], yf[m]
    if len(xf) < 30:
        return {"coeffs_high_to_low": [], "rmse": None, "n_points": int(len(xf))}
    coeff = np.polyfit(xf, yf, 2)
    resid = yf - np.polyval(coeff, xf)
    rmse = float(np.sqrt(np.mean(resid**2)))
    return {
        "coeffs_high_to_low": [float(c) for c in coeff],
        "rmse": rmse,
        "n_points": int(len(xf)),
        "x_axis": "log(K/S)/sqrt(T)",
        "iv_engine": "implied_vol_call_brent_plot_iv",
    }


def fit_mt_notebook(nb: Any, strikes: list[int], step: int = STEP) -> dict[str, Any]:
    """Same m_t axis; IV from round3.ipynb bisection in nb_method_core."""
    xs: list[float] = []
    ys: list[float] = []
    vouchers = _vouchers(strikes)
    for day in (0, 1, 2):
        wf = nb.load_day_wide(day).sort_index()
        mp = nb.index_map_timestamp_to_row_idx(wf)
        wsub = nb.subsample_wide(wf, step=step)
        d0 = int(nb.dte_from_csv_day(day))
        for ts, row in wsub.iterrows():
            ts_i = int(ts)
            t_idx = mp[ts_i]
            T = float(nb.expiration_time_years(d0, t_idx))
            if T <= 0:
                continue
            sqrt_t = math.sqrt(T)
            S = float(row["S"])
            if S <= 0:
                continue
            for v in vouchers:
                if v not in row.index:
                    continue
                K = float(v.split("_")[1])
                mid = float(row[v])
                iv = float(nb.implied_volatility_nb(S, K, mid, T, 0.0))
                if not np.isfinite(iv):
                    continue
                m_t = math.log(K / S) / sqrt_t
                if not np.isfinite(m_t):
                    continue
                xs.append(m_t)
                ys.append(iv)
    xf = np.asarray(xs, dtype=float)
    yf = np.asarray(ys, dtype=float)
    m = np.isfinite(xf) & np.isfinite(yf)
    xf, yf = xf[m], yf[m]
    if len(xf) < 30:
        return {"coeffs_high_to_low": [], "rmse": None, "n_points": int(len(xf))}
    coeff = np.polyfit(xf, yf, 2)
    resid = yf - np.polyval(coeff, xf)
    rmse = float(np.sqrt(np.mean(resid**2)))
    return {
        "coeffs_high_to_low": [float(c) for c in coeff],
        "rmse": rmse,
        "n_points": int(len(xf)),
        "x_axis": "log(K/S)/sqrt(T)",
        "iv_engine": "implied_volatility_nb_bisection",
    }


def fit_mt_true_fv(winding: bool, strikes: list[int]) -> dict[str, Any]:
    """True_fv day-39 bundle; Brent IV; T from truemethod probe DTE=5."""
    tmc = PLOT / "truemethod" / "common"
    sys.path.insert(0, str(tmc))
    _orig = sys.path[:]
    try:
        from iv_smile_true_fv import (  # noqa: WPS433
            compute_iv_panel_fv,
            global_quadratic_fit,
            load_true_fv_wide,
            make_t_years_fn,
        )
    finally:
        sys.path[:] = _orig

    wide = load_true_fv_wide(REPO)
    tfn = make_t_years_fn(winding)
    ivdf = compute_iv_panel_fv(wide, tfn)
    ks = set(strikes)
    sub = ivdf[ivdf["K"].isin(ks)].copy()
    coeff, rmse, n_g = global_quadratic_fit(sub)
    return {
        "coeffs_high_to_low": [float(c) for c in coeff] if len(coeff) else [],
        "rmse": float(rmse) if np.isfinite(rmse) else None,
        "n_points": int(n_g),
        "x_axis": "log(K/S)/sqrt(T)",
        "iv_engine": "implied_vol_call_brent_on_true_fv",
        "data": "fairs merged true_fv day_39",
    }


def _dual(
    fn: Any, *fn_args: Any, strikes_sets: tuple[list[int], list[int]] = (STRIKES_ALL, STRIKES_NEAR)
) -> dict[str, Any]:
    a, b = strikes_sets
    return {"all_strikes": fn(*fn_args, a), "near_5000_5500": fn(*fn_args, b)}


def main() -> None:
    om_wind = (PLOT / "original_method" / "wind_down" / "combined_analysis").resolve()
    om_nowind = PLOT / "original_method" / "no_wind_down" / "combined_analysis"
    piv_w = _load_plot_iv(om_wind)
    piv_nw = _load_plot_iv(om_nowind)

    nb_w = _load_nb(PLOT / "test_implementation" / "wind_down" / "nb_method_core.py")
    nb_nw = _load_nb(PLOT / "test_implementation" / "no_wind_down" / "nb_method_core.py")

    out: dict[str, Any] = {
        "description": "Global quadratic IV fit in m_t=log(K/S)/sqrt(T); coeffs numpy polyfit high→low (m², m, const).",
        "subsample_step": STEP,
        "branches": {
            "original_method_wind_down": _dual(fit_mt_historical, piv_w),
            "original_method_no_wind_down": _dual(fit_mt_historical, piv_nw),
            "test_implementation_wind_down": _dual(fit_mt_notebook, nb_w),
            "test_implementation_no_wind_down": _dual(fit_mt_notebook, nb_nw),
            "truemethod_wind_down": _dual(fit_mt_true_fv, True),
            "truemethod_no_wind_down": _dual(fit_mt_true_fv, False),
        },
    }

    dest = PLOT / "parabola_fits_six_branches.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", dest)


if __name__ == "__main__":
    main()
