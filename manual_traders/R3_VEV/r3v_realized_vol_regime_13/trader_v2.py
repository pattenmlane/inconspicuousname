"""v0 base with smaller hydrogel footprint (day 2 hydrogel was large negative in v0/v1)."""
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
    BASE_H_HALF = 3
    REG_H_SCALE = 0.08
    ORDER_SIZE_H = 5
