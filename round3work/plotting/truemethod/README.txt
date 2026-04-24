True-method plotting (hold-1 **true_fv** from `round3work/fairs/`, not historical mids).

Branches
--------
* `wind_down/` — DTE 5d with same intraday winding fraction as historical analysis.
* `no_wind_down/` — T = 5/365 flat for the session.

Run
---
  python3 round3work/plotting/truemethod/wind_down/combined_analysis/run_truemethod_plots.py
  python3 round3work/plotting/truemethod/no_wind_down/combined_analysis/run_truemethod_plots.py

Shared code: `common/true_fv_loader.py`, `common/iv_smile_true_fv.py`.

Read: `FINDINGS_AND_USEFULNESS.md`.
