"""Same as trader_v0 with a small parameter sweep (tighter RV window, stronger VEV widen)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_r3v_rv_mod", Path(__file__).resolve().parent / "trader_v0.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


class Trader(_mod.Trader):
    K_WIDEN = 14.0
    K_TIGHTEN = 4.0
    RV_WIN = 30
    BASE_VEV_HALF = 2
    ORDER_SIZE_VEV = 14
