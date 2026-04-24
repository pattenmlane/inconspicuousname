"""
480-cell IV grid: 10 VEV strikes × 12 parabola branches × {p70,p80,p90,p95} on switch_mean.

Parallel (default):
  - Phase 1: one process per model (up to --workers-phase1); tqdm on completed models.
  - Phase 2: one process per cell (up to --workers-phase2); tqdm on completed cells.
    Each worker uses its own IV_GRID_CONFIG temp file. Intraday tqdm is OFF in workers.

--serial: single-process mode with per-timestamp tqdm and verbose cell prints.

Run (from repo root):
  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
    python3 round3work/voucher_work/5200_work/threshold_analysis/run_iv_grid_pnl.py --data "$PWD/Prosperity4Data"
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

_THRESH = Path(__file__).resolve().parent
_WORK = _THRESH.parent
_REPO = _WORK.parent.parent.parent
_DATA = _REPO / "Prosperity4Data" / "ROUND_3"
_PARABOLA_JSON = _REPO / "round3work" / "plotting" / "parabola_fits_six_branches.json"
_CAL = _WORK / "calibration.json"
_ALG = _WORK / "trader_iv_grid.py"
_CFG_PATH = _WORK / ".iv_grid_active_config.json"
_STEP = 50
_VOUCHERS: list[tuple[str, int]] = [(f"VEV_{k}", k) for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
_PCTS: list[tuple[str, float]] = [("p70", 0.7), ("p80", 0.8), ("p90", 0.9), ("p95", 0.95)]

sys.path.insert(0, str(_WORK))
from frankfurt_iv_scalp_core import (  # noqa: E402
    book_from_row,
    compute_option_indicators,
    load_calibration,
    synthetic_walls_if_missing,
)


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


def _make_t_probe(winding: bool):
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


def collect_switch_plot_iv(
    piv: Any, coeffs: list[float], cal_base: dict[str, Any], opt: str, strike: int, step: int = _STEP
) -> list[float]:
    cal = dict(cal_base)
    cal["coeffs_high_to_low"] = coeffs
    sw_list: list[float] = []
    U = "VELVETFRUIT_EXTRACT"
    for day in (0, 1, 2):
        ema: dict[str, float] = {}
        df = pd.read_csv(_DATA / f"prices_round_3_day_{day}.csv", sep=";")
        ts_list = sorted(df["timestamp"].unique())[::step]
        for ts in ts_list:
            g = df[df["timestamp"] == ts]
            if opt not in g["product"].values or U not in g["product"].values:
                continue
            ro = g[g["product"] == opt].iloc[0].to_dict()
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
            ind = compute_option_indicators(cal, ema, u_mid, strike, T, float(wm), float(bb), float(ba), opt)
            if ind.get("switch_mean") is not None:
                sw_list.append(float(ind["switch_mean"]))
    return sw_list


def collect_switch_nb(
    nb: Any, coeffs: list[float], cal_base: dict[str, Any], opt: str, strike: int, step: int = _STEP
) -> list[float]:
    cal = dict(cal_base)
    cal["coeffs_high_to_low"] = coeffs
    sw_list: list[float] = []
    U = "VELVETFRUIT_EXTRACT"
    for day in (0, 1, 2):
        ema: dict[str, float] = {}
        wf = nb.load_day_wide(day).sort_index()
        mp = nb.index_map_timestamp_to_row_idx(wf)
        wsub = nb.subsample_wide(wf, step=step)
        d0 = int(nb.dte_from_csv_day(day))
        df = pd.read_csv(_DATA / f"prices_round_3_day_{day}.csv", sep=";")
        for ts, _row in wsub.iterrows():
            ts_i = int(ts)
            g = df[df["timestamp"] == ts_i]
            if opt not in g["product"].values or U not in g["product"].values:
                continue
            ro = g[g["product"] == opt].iloc[0].to_dict()
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
            ind = compute_option_indicators(cal, ema, u_mid, strike, T, float(wm), float(bb), float(ba), opt)
            if ind.get("switch_mean") is not None:
                sw_list.append(float(ind["switch_mean"]))
    return sw_list


def collect_switch_tm(
    coeffs: list[float], cal_base: dict[str, Any], opt: str, strike: int, winding: bool, step: int = _STEP
) -> list[float]:
    cal = dict(cal_base)
    cal["coeffs_high_to_low"] = coeffs
    tfn = _make_t_probe(winding)
    sw_list: list[float] = []
    U = "VELVETFRUIT_EXTRACT"
    for day in (0, 1, 2):
        ema: dict[str, float] = {}
        df = pd.read_csv(_DATA / f"prices_round_3_day_{day}.csv", sep=";")
        ts_list = sorted(df["timestamp"].unique())[::step]
        for ts in ts_list:
            g = df[df["timestamp"] == ts]
            if opt not in g["product"].values or U not in g["product"].values:
                continue
            ro = g[g["product"] == opt].iloc[0].to_dict()
            ru = g[g["product"] == U].iloc[0].to_dict()
            _, _, bid_w, ask_w, bb, ba, wm = book_from_row(ro)
            _, _, _, _, ubb, uba, _ = book_from_row(ru)
            if ubb is None or uba is None:
                continue
            u_mid = 0.5 * float(ubb) + 0.5 * float(uba)
            bid_w, ask_w, wm, bb, ba = synthetic_walls_if_missing(bid_w, ask_w, bb, ba)
            if wm is None or bb is None or ba is None:
                continue
            T = float(tfn(int(ts)))
            ind = compute_option_indicators(cal, ema, u_mid, strike, T, float(wm), float(bb), float(ba), opt)
            if ind.get("switch_mean") is not None:
                sw_list.append(float(ind["switch_mean"]))
    return sw_list


def model_id_to_t_kind(model_id: str) -> str:
    if model_id.startswith("original_method"):
        return "om_wind" if "wind_down" in model_id else "om_nowind"
    if model_id.startswith("test_implementation"):
        return "ti_wind" if "wind_down" in model_id else "ti_nowind"
    if model_id.startswith("truemethod"):
        return "tm_wind" if "wind_down" in model_id else "tm_nowind"
    raise ValueError(model_id)


def collect_switch_for_model(
    model_id: str,
    coeffs: list[float],
    cal_base: dict[str, Any],
    opt: str,
    strike: int,
    piv_w: Any,
    piv_nw: Any,
    nb_w: Any,
    nb_nw: Any,
) -> list[float]:
    branch = model_id.split("__")[0]
    if branch.startswith("original_method"):
        piv = piv_w if "wind_down" in branch else piv_nw
        return collect_switch_plot_iv(piv, coeffs, cal_base, opt, strike)
    if branch.startswith("test_implementation"):
        nb = nb_w if "wind_down" in branch else nb_nw
        return collect_switch_nb(nb, coeffs, cal_base, opt, strike)
    if branch.startswith("truemethod"):
        return collect_switch_tm(coeffs, cal_base, opt, strike, winding="wind_down" in branch)
    raise RuntimeError(model_id)


def pnl_option_and_total(result: Any, option_product: str) -> tuple[float, float]:
    tot = 0.0
    opt_pnl = 0.0
    for a in result.final_activities():
        p = float(a.profit_loss)
        tot += p
        if a.symbol == option_product:
            opt_pnl = p
    return opt_pnl, tot


def phase1_model_worker(
    payload: tuple[str, list[float], dict[str, Any]],
) -> tuple[str, list[tuple[str, int, dict[str, float]]]]:
    """One model × all vouchers → quantile dicts (picklable; runs in child process)."""
    model_id, coeffs, cal_base = payload
    piv_w = _load_plot_iv((_REPO / "round3work/plotting/original_method/wind_down/combined_analysis").resolve())
    piv_nw = _load_plot_iv((_REPO / "round3work/plotting/original_method/no_wind_down/combined_analysis").resolve())
    nb_w = _load_nb(_REPO / "round3work/plotting/test_implementation/wind_down/nb_method_core.py")
    nb_nw = _load_nb(_REPO / "round3work/plotting/test_implementation/no_wind_down/nb_method_core.py")
    out: list[tuple[str, int, dict[str, float]]] = []
    for opt, strike in _VOUCHERS:
        sw = collect_switch_for_model(model_id, coeffs, cal_base, opt, strike, piv_w, piv_nw, nb_w, nb_nw)
        arr = np.asarray(sw, dtype=float)
        if len(arr) == 0:
            qd = {lbl: float("nan") for lbl, _ in _PCTS}
        else:
            qd = {lbl: float(np.quantile(arr, q)) for lbl, q in _PCTS}
        out.append((opt, strike, qd))
    return model_id, out


def phase2_cell_worker(task: dict[str, Any]) -> dict[str, Any]:
    """One grid cell: 3× TestRunner in a subprocess; own config path."""
    cfg_path = Path(task["cfg_path"])
    data_root = Path(task["data_root"])
    repo = Path(task["repo_root"])
    bt = repo / "imc-prosperity-4-backtester"
    for p in (bt, bt / "prosperity4bt"):
        s = str(p.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)
    core_dir = Path(task["core_dir"])
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))

    from prosperity4bt.tools.data_reader import FileSystemReader
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.models.test_options import TradeMatchingMode

    os.environ["IV_GRID_CONFIG_PATH"] = str(cfg_path)
    spec = importlib.util.spec_from_file_location("trader_iv_grid", task["trader_path"])
    if spec is None or spec.loader is None:
        raise RuntimeError(task["trader_path"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Trader = mod.Trader

    reader = FileSystemReader(data_root)
    opt = str(task["opt"])
    show_bar = bool(task.get("show_progress_bar", False))

    base_cfg = {
        "model_id": task["model_id"],
        "coeffs_high_to_low": task["coeffs"],
        "IV_SCALPING_THR": float(task["thr"]),
        "TARGET_VOUCHER": opt,
        "K_STRIKE": int(task["strike"]),
        "UNDERLYING": "VELVETFRUIT_EXTRACT",
        "t_kind": str(task["t_kind"]),
    }
    pnl_opt_sum = 0.0
    pnl_tot_sum = 0.0
    for csv_day in (0, 1, 2):
        cfg = {**base_cfg, "csv_day": int(csv_day)}
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        r = TestRunner(
            Trader(),
            reader,
            3,
            int(csv_day),
            show_progress_bar=show_bar,
            print_output=False,
            trade_matching_mode=TradeMatchingMode.all,
        ).run()
        po, pt = pnl_option_and_total(r, opt)
        pnl_opt_sum += po
        pnl_tot_sum += pt

    return {
        "model_id": task["model_id"],
        "voucher": opt,
        "strike": int(task["strike"]),
        "percentile": str(task["pct_label"]),
        "IV_SCALPING_THR": float(task["thr"]),
        "pnl_option_sum_days_0_2": pnl_opt_sum,
        "pnl_total_sum_days_0_2": pnl_tot_sum,
        "skipped": False,
    }


def _load_trader_module():
    bt = _REPO / "imc-prosperity-4-backtester"
    for p in (bt, bt / "prosperity4bt"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    spec = importlib.util.spec_from_file_location("trader_iv_grid", _ALG)
    if spec is None or spec.loader is None:
        raise RuntimeError(str(_ALG))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    import argparse

    bt = _REPO / "imc-prosperity-4-backtester"
    for p in (bt, bt / "prosperity4bt"):
        s = str(p.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)

    from prosperity4bt.tools.data_reader import FileSystemReader
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.models.test_options import TradeMatchingMode

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=_REPO / "Prosperity4Data", help="Prosperity4Data root")
    ap.add_argument("--out", type=Path, default=_THRESH / "iv_grid_pnl_results.json", help="JSON results path")
    ap.add_argument(
        "--workers-phase1",
        type=int,
        default=None,
        help="Parallel models for quantiles (default: min(12, CPU count))",
    )
    ap.add_argument(
        "--workers-phase2",
        type=int,
        default=None,
        help="Parallel backtest cells (default: CPU count, capped at 64)",
    )
    ap.add_argument(
        "--serial",
        action="store_true",
        help="Single process; per-timestamp tqdm + verbose cell logs (slow)",
    )
    args = ap.parse_args()
    data_root: Path = args.data.resolve()
    cpu = os.cpu_count() or 1

    if not _DATA.is_dir():
        print(f"ERROR: missing {_DATA}", flush=True)
        sys.exit(1)

    cal_base = load_calibration(_CAL)
    para = json.loads(_PARABOLA_JSON.read_text(encoding="utf-8"))

    models: list[tuple[str, list[float]]] = []
    for branch, subsets in para["branches"].items():
        for subset_name, payload in subsets.items():
            models.append((f"{branch}__{subset_name}", list(payload["coeffs_high_to_low"])))

    n_models = len(models)
    n_v = len(_VOUCHERS)
    total_cells = n_v * n_models * len(_PCTS)

    w1 = 1 if args.serial else (args.workers_phase1 if args.workers_phase1 is not None else max(1, min(n_models, cpu)))
    w2 = 1 if args.serial else (args.workers_phase2 if args.workers_phase2 is not None else max(1, min(64, cpu)))

    print(
        f"IV grid: {n_v} vouchers × {n_models} models × {len(_PCTS)} percentiles = {total_cells} cells; "
        f"3 days/cell. Data: {data_root}\n"
        f"Parallel: phase1_workers={w1} phase2_workers={w2} serial={args.serial}",
        flush=True,
    )

    # --- phase 1: quantiles ---
    qkey: dict[tuple[str, str], dict[str, float]] = {}
    t0 = time.perf_counter()
    if args.serial or w1 == 1:
        piv_w = _load_plot_iv((_REPO / "round3work/plotting/original_method/wind_down/combined_analysis").resolve())
        piv_nw = _load_plot_iv((_REPO / "round3work/plotting/original_method/no_wind_down/combined_analysis").resolve())
        nb_w = _load_nb(_REPO / "round3work/plotting/test_implementation/wind_down/nb_method_core.py")
        nb_nw = _load_nb(_REPO / "round3work/plotting/test_implementation/no_wind_down/nb_method_core.py")
        total_q = n_models * n_v
        qbar = tqdm(total=total_q, desc="Phase1 switch_mean quantiles", unit="pair", dynamic_ncols=True)
        for model_id, coeffs in models:
            for opt, strike in _VOUCHERS:
                sw = collect_switch_for_model(model_id, coeffs, cal_base, opt, strike, piv_w, piv_nw, nb_w, nb_nw)
                arr = np.asarray(sw, dtype=float)
                qbar.set_postfix_str(f"{model_id[:28]} {opt}", refresh=False)
                if len(arr) == 0:
                    qkey[(model_id, opt)] = {lbl: float("nan") for lbl, _ in _PCTS}
                else:
                    qkey[(model_id, opt)] = {lbl: float(np.quantile(arr, q)) for lbl, q in _PCTS}
                qbar.update(1)
        qbar.close()
    else:
        w1c = max(1, min(w1, n_models))
        with ProcessPoolExecutor(max_workers=w1c) as ex:
            futures = {ex.submit(phase1_model_worker, (mid, c, cal_base)): mid for mid, c in models}
            pbar1 = tqdm(as_completed(futures), total=len(futures), desc="Phase1 models", dynamic_ncols=True)
            for fut in pbar1:
                model_id, pairs = fut.result()
                for opt, strike, qd in pairs:
                    qkey[(model_id, opt)] = qd
                pbar1.set_postfix_str(model_id[:36], refresh=False)

    print(f"Phase1 done in {time.perf_counter() - t0:.1f}s", flush=True)

    rows: list[dict[str, Any]] = []
    tmpd: str | None = None

    try:
        if args.serial or w2 == 1:
            piv_w = _load_plot_iv((_REPO / "round3work/plotting/original_method/wind_down/combined_analysis").resolve())
            piv_nw = _load_plot_iv((_REPO / "round3work/plotting/original_method/no_wind_down/combined_analysis").resolve())
            nb_w = _load_nb(_REPO / "round3work/plotting/test_implementation/wind_down/nb_method_core.py")
            nb_nw = _load_nb(_REPO / "round3work/plotting/test_implementation/no_wind_down/nb_method_core.py")
            reader = FileSystemReader(data_root)
            os.environ["IV_GRID_CONFIG_PATH"] = str(_CFG_PATH)
            trader_mod = _load_trader_module()
            Trader = trader_mod.Trader
            completed = 0
            t_run = time.perf_counter()
            for model_id, coeffs in models:
                t_kind = model_id_to_t_kind(model_id)
                for opt, strike in _VOUCHERS:
                    qd = qkey.get((model_id, opt), {})
                    for pct_label, _q in _PCTS:
                        cell_n = completed + 1
                        thr = float(qd.get(pct_label, float("nan")))
                        if not np.isfinite(thr):
                            completed += 1
                            print(
                                f"progress: {completed}/{total_cells} (cell {cell_n}) SKIP {model_id} {opt} {pct_label}",
                                flush=True,
                            )
                            rows.append(
                                {
                                    "model_id": model_id,
                                    "voucher": opt,
                                    "strike": strike,
                                    "percentile": pct_label,
                                    "IV_SCALPING_THR": None,
                                    "pnl_option_sum_days_0_2": None,
                                    "pnl_total_sum_days_0_2": None,
                                    "skipped": True,
                                }
                            )
                            continue

                        base_cfg = {
                            "model_id": model_id,
                            "coeffs_high_to_low": coeffs,
                            "IV_SCALPING_THR": thr,
                            "TARGET_VOUCHER": opt,
                            "K_STRIKE": strike,
                            "UNDERLYING": "VELVETFRUIT_EXTRACT",
                            "t_kind": t_kind,
                        }
                        print(
                            f"\n--- cell {cell_n}/{total_cells} {model_id} {opt} {pct_label} thr={thr:.6f} ---",
                            flush=True,
                        )
                        pnl_opt_sum = 0.0
                        pnl_tot_sum = 0.0
                        for csv_day in (0, 1, 2):
                            print(f"  intraday: round 3 day {csv_day}", flush=True)
                            cfg = {**base_cfg, "csv_day": csv_day}
                            _CFG_PATH.write_text(json.dumps(cfg), encoding="utf-8")
                            r = TestRunner(
                                Trader(),
                                reader,
                                3,
                                csv_day,
                                show_progress_bar=True,
                                print_output=False,
                                trade_matching_mode=TradeMatchingMode.all,
                            ).run()
                            po, pt = pnl_option_and_total(r, opt)
                            pnl_opt_sum += po
                            pnl_tot_sum += pt
                            print(
                                f"  day {csv_day} done: pnl_opt={po:,.0f} pnl_tot={pt:,.0f} (cum opt={pnl_opt_sum:,.0f})",
                                flush=True,
                            )

                        completed += 1
                        elapsed = time.perf_counter() - t_run
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total_cells - completed) / rate if rate > 0 else float("nan")
                        print(
                            f"progress: {completed}/{total_cells} pnl_opt={pnl_opt_sum:,.0f} "
                            f"| {elapsed:.0f}s elapsed ETA~{eta:.0f}s",
                            flush=True,
                        )
                        rows.append(
                            {
                                "model_id": model_id,
                                "voucher": opt,
                                "strike": strike,
                                "percentile": pct_label,
                                "IV_SCALPING_THR": thr,
                                "pnl_option_sum_days_0_2": pnl_opt_sum,
                                "pnl_total_sum_days_0_2": pnl_tot_sum,
                                "skipped": False,
                            }
                        )
        else:
            tmpd = tempfile.mkdtemp(prefix="iv_grid_cfg_")
            tasks: list[dict[str, Any]] = []
            cell_id = 0
            for model_id, coeffs in models:
                t_kind = model_id_to_t_kind(model_id)
                for opt, strike in _VOUCHERS:
                    qd = qkey.get((model_id, opt), {})
                    for pct_label, _q in _PCTS:
                        thr = float(qd.get(pct_label, float("nan")))
                        if not np.isfinite(thr):
                            rows.append(
                                {
                                    "model_id": model_id,
                                    "voucher": opt,
                                    "strike": strike,
                                    "percentile": pct_label,
                                    "IV_SCALPING_THR": None,
                                    "pnl_option_sum_days_0_2": None,
                                    "pnl_total_sum_days_0_2": None,
                                    "skipped": True,
                                }
                            )
                            continue
                        tasks.append(
                            {
                                "cell_id": cell_id,
                                "cfg_path": str(Path(tmpd) / f"cell_{cell_id}.json"),
                                "data_root": str(data_root),
                                "repo_root": str(_REPO),
                                "core_dir": str(_WORK),
                                "trader_path": str(_ALG),
                                "model_id": model_id,
                                "coeffs": coeffs,
                                "opt": opt,
                                "strike": strike,
                                "pct_label": pct_label,
                                "thr": thr,
                                "t_kind": t_kind,
                                "show_progress_bar": False,
                            }
                        )
                        cell_id += 1

            n_tasks = len(tasks)
            t_run = time.perf_counter()
            if n_tasks == 0:
                print("Phase2: no cells to run (all skipped)", flush=True)
            else:
                w2c = max(1, min(w2, n_tasks))
                with ProcessPoolExecutor(max_workers=w2c) as ex:
                    futures = {ex.submit(phase2_cell_worker, t): t for t in tasks}
                    pbar2 = tqdm(
                        as_completed(futures),
                        total=n_tasks,
                        desc="Phase2 backtest cells",
                        dynamic_ncols=True,
                    )
                    for fut in pbar2:
                        row = fut.result()
                        rows.append(row)
                        pbar2.set_postfix_str(
                            f"{row['voucher']} {row['percentile']} pnl={row['pnl_option_sum_days_0_2']:,.0f}",
                            refresh=False,
                        )

                elapsed = time.perf_counter() - t_run
                print(
                    f"Phase2 done: {n_tasks} cells in {elapsed:.1f}s ({n_tasks / elapsed:.2f} cells/s)",
                    flush=True,
                )

    finally:
        if tmpd and os.path.isdir(tmpd):
            shutil.rmtree(tmpd, ignore_errors=True)

    def _row_sort_key(r: dict[str, Any]) -> tuple[str, str, str]:
        return (str(r.get("model_id", "")), str(r.get("voucher", "")), str(r.get("percentile", "")))

    rows.sort(key=_row_sort_key)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
