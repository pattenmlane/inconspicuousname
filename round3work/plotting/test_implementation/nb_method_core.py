"""
Shim: re-exports the **wind-down** `nb_method_core` from `test_implementation/wind_down/`.

Import the no-wind version explicitly:
  import importlib.util
  spec = importlib.util.spec_from_file_location(
      "nb_method_core_nowind",
      Path(...)/ "no_wind_down" / "nb_method_core.py",
  )
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_path = Path(__file__).resolve().parent / "wind_down" / "nb_method_core.py"
_spec = importlib.util.spec_from_file_location(__name__, _path)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load wind_down nb_method_core")
_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
