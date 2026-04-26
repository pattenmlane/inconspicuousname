"""v46 retune: joint 5200/5300 tight gate with gentler half-spread adjustment (tight regime does not tighten quotes; loose widens slightly). No hydrogel."""
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
    TRADE_HYDROGEL = False

    USE_TIGHT_GATE_5200_5300 = True
    TIGHT_SPREAD_TH = 2.0
    TIGHT_GATE_TIGHT_VEV_SIZE_MULT = 1.1
    TIGHT_GATE_TIGHT_EX_SIZE_MULT = 1.1
    TIGHT_GATE_TIGHT_VEV_HALF_MULT = 1.0
    TIGHT_GATE_TIGHT_EX_HALF_MULT = 1.0
    TIGHT_GATE_LOOSE_VEV_SIZE_MULT = 0.88
    TIGHT_GATE_LOOSE_EX_SIZE_MULT = 0.88
    TIGHT_GATE_LOOSE_VEV_HALF_MULT = 1.05
    TIGHT_GATE_LOOSE_EX_HALF_MULT = 1.05

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
