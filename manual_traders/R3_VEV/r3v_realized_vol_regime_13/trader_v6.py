"""v4 + moderate delta hedge on extract (BS call delta × VEV pos vs same smile)."""
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
    REG_H_SCALE = 0.08
    ORDER_SIZE_H = 5
    BASE_H_HALF = 4
    TAKE_EDGE_MULT = 0.70
    MAX_TAKE_PER_SIDE = 16
    DELTA_HEDGE_STRENGTH = 0.4
    MAX_D_HEDGE_QTY = 50
