"""
Backward-compatible entrypoint: runs the **wind-down** notebook-method bundle.

For explicit branches:
  python3 round3work/plotting/test_implementation/wind_down/run_nb_method_plots.py
  python3 round3work/plotting/test_implementation/no_wind_down/run_nb_method_plots.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "wind_down" / "run_nb_method_plots.py"
sys.argv[0] = str(_TARGET)
runpy.run_path(str(_TARGET), run_name="__main__")
