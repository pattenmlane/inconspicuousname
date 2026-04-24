Round 3 plotting — two DTE conventions
========================================

**Does the legacy path `original_method/combined_analysis/` wind down?**  
Yes. It uses `t_years_effective` = (calendar DTE for CSV day 0/1/2 minus intraday progress) / 365
(Frankfurt-style ~one day decay over the session). See `plot_iv_smile_round3.py`.

**Layout**

* **`wind_down/`** — `combined_analysis` and `pervoucher_analysis` are **symlinks** to the
  canonical folders above (same code and outputs as today; nothing duplicated).

* **`no_wind_down/`** — **full copy** of `combined_analysis` with `plot_iv_smile_round3.py`
  patched so `t_years_effective(day, ts) = dte_from_csv_day(day) / 365` (timestamp ignored for T).
  **`no_wind_down/pervoucher_analysis/`** holds copies of `build_pervoucher.py` and
  `interactive_visualizer.py` that resolve `../combined_analysis` to the no-wind copy.
  Re-run `build_pervoucher.py` from that directory to emit figures under `no_wind_down/pervoucher_analysis/`.

Historical CSV days 0–2 still map to **DTE 8, 7, 6** at session open (`dte_from_csv_day`).

Notebook test bundle: `../test_implementation/wind_down/` vs `../test_implementation/no_wind_down/`
(see `../test_implementation/README.txt`).
