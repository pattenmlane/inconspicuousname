"""v33 + wing-only vega–spread offsets: same tape cross-section as v44 but only widen strikes above pool mean spread (no negative local adds that tighten OTM)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_r3v_rv_mod", Path(__file__).resolve().parent / "trader_v0.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

_RAW_OFF = {
    "VEV_4000": 15.071833333333334,
    "VEV_4500": 10.136833333333334,
    "VEV_5000": 0.38349999999999973,
    "VEV_5100": -1.3648333333333333,
    "VEV_5200": -2.758166666666667,
    "VEV_5300": -3.5181666666666667,
    "VEV_5400": -4.269833333333334,
    "VEV_5500": -4.4515,
    "VEV_6000": -4.614833333333333,
    "VEV_6500": -4.614833333333333,
}
_POS_ONLY = {k: max(0.0, float(v)) for k, v in _RAW_OFF.items()}


class Trader(_mod.Trader):
    REG_H_SCALE = 0.08
    ORDER_SIZE_H = 5
    BASE_H_HALF = 4
    TAKE_EDGE_MULT = 0.635
    MAX_TAKE_PER_SIDE = 18
    THETA_REGIME_WEIGHT = 0.16
    THETA_REGIME_NORM = 0.04
    REG_EX_SCALE = 0.55
    GAMMA_REGIME_WEIGHT = 0.12
    GAMMA_REGIME_NORM = 0.0008
    ORDER_SIZE_VEV_MAP = {
        "VEV_4000": 18,
        "VEV_4500": 18,
        "VEV_5000": 18,
        "VEV_5100": 16,
        "VEV_5200": 14,
        "VEV_5300": 12,
        "VEV_5400": 10,
        "VEV_5500": 8,
        "VEV_6000": 4,
        "VEV_6500": 3,
    }
    MAX_TAKE_PER_SIDE_MAP = {
        "VEV_4000": 30,
        "VEV_4500": 30,
        "VEV_5000": 28,
        "VEV_5100": 24,
        "VEV_5200": 20,
        "VEV_5300": 18,
        "VEV_5400": 14,
        "VEV_5500": 10,
        "VEV_6000": 6,
        "VEV_6500": 6,
    }

    VEV_HALF_LOCAL_ADD_MAP = _POS_ONLY
    VEV_HALF_LOCAL_ADD_SCALE = 0.09
