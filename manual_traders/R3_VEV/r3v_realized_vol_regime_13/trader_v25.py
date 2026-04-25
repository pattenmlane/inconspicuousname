"""v14 + slightly wider extract/hydrogel MM widths."""
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
    REG_H_SCALE = 0.1
    ORDER_SIZE_H = 5
    BASE_H_HALF = 4
    TAKE_EDGE_MULT = 0.63
    MAX_TAKE_PER_SIDE = 18
    THETA_REGIME_WEIGHT = 0.16
    THETA_REGIME_NORM = 0.04
    REG_EX_SCALE = 0.5
