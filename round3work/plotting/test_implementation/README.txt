Notebook-method test plots (`round3.ipynb` style bisection IV + smile in m = log(K/S)/sqrt(T)).

Two branches (mirrors `original_method/wind_down` vs `no_wind_down`):

* **wind_down/** — session **T decays** with row index: (DTE_open − t/10_000)/365 (original notebook convention).
  Uses `original_method/wind_down/combined_analysis` on PYTHONPATH.

* **no_wind_down/** — **flat T** = DTE_open/365 for the whole CSV day (no intraday wind).
  Uses `original_method/no_wind_down/combined_analysis`.

Regenerate outputs (each writes PNG/CSV **into its own folder**):

  python3 round3work/plotting/test_implementation/wind_down/run_nb_method_plots.py
  python3 round3work/plotting/test_implementation/no_wind_down/run_nb_method_plots.py

Legacy one-liner (same as wind_down):

  python3 round3work/plotting/test_implementation/run_nb_method_plots.py
