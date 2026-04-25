"""
Grid search over SKEW_FACTOR for trader_vev_model_aware.py.

Run from repo root:
  python3 round3work/velvetfruit_work/grid_skew_vev.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path("imc-prosperity-4-backtester")))
sys.path.insert(0, str(Path("imc-prosperity-4-backtester/prosperity4bt")))

from prosperity4bt.tools.data_reader import PackageResourcesReader
from prosperity4bt.test_runner import TestRunner
from prosperity4bt.models.test_options import TradeMatchingMode

TRADER_PATH = Path("round3work/velvetfruit_work/trader_vev_model_aware.py")
SKEW_VALUES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]
FLATTEN = 150

reader = PackageResourcesReader()

results = []
for skew in SKEW_VALUES:
    spec = importlib.util.spec_from_file_location("trader_vev", TRADER_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.SKEW_FACTOR       = skew
    mod.FLATTEN_THRESHOLD = FLATTEN

    total = 0
    for day in [0, 1, 2]:
        runner = TestRunner(
            mod.Trader(), reader, round=3, day=day,
            trade_matching_mode=TradeMatchingMode.worse
        )
        result = runner.run()
        total += sum(row.profit_loss for row in result.final_activities())

    results.append((skew, total))
    print(f"  skew={skew:.2f}  total={total:,.0f}")

best = max(results, key=lambda x: x[1])
print(f"\nBest: skew={best[0]:.2f}  total={best[1]:,.0f}")
