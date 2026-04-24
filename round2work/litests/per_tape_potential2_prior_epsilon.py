#!/usr/bin/env python3
"""Per-tape (per Prosperity4Data round/day) PnL for patched ``potential2_osmium_only``.

Uses ``TestRunner`` in-process (same engine as ``python3 -m prosperity4bt``).

Example::

  cd /path/to/ProsperityRepo
  PYTHONPATH=\"$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt\" \\
  python3 round2work/litests/per_tape_potential2_prior_epsilon.py
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIT = REPO / "round2work" / "litests"
SRC = LIT / "potential2_osmium_only.py"
DATA_DEFAULT = REPO / "Prosperity4Data"
OSMIUM = "ASH_COATED_OSMIUM"

# Prosperity4Data tapes we have been using (prices_round_* present)
TAPES: list[tuple[int, int]] = [
    (1, -2),
    (1, -1),
    (1, 0),
    (1, 19),
    (1, 119),
    (2, -1),
    (2, 0),
    (2, 1),
]


def _patch_source(text: str, prior: int, epsilon: float) -> str:
    t = re.sub(
        r"^(\s*)eplison = 0\.65\s*$",
        rf"\1eplison = {epsilon}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    t = re.sub(
        r"^(\s*)PRIOR_STRENGTH = 2000\s*$",
        rf"\1PRIOR_STRENGTH = {prior}",
        t,
        count=1,
        flags=re.MULTILINE,
    )
    if t == text:
        raise SystemExit(f"Patch failed for {SRC}")
    return t


def _bootstrap_datamodel() -> None:
    for p in (
        REPO / "imc-prosperity-4-backtester",
        REPO / "imc-prosperity-4-backtester" / "prosperity4bt",
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    from prosperity4bt import datamodel as _dm

    dm = types.ModuleType("datamodel")
    for n in ("Order", "OrderDepth", "TradingState"):
        setattr(dm, n, getattr(_dm, n))
    sys.modules["datamodel"] = dm


def _load_trader_class(py: Path) -> type:
    _bootstrap_datamodel()
    name = "p2tape_" + py.stem.replace(".", "_")
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, py)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.Trader


def _run_tape(trader_cls: type, data_root: Path, rnd: int, day: int, match: str) -> int:
    from prosperity4bt.models.test_options import TradeMatchingMode
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import FileSystemReader

    mode = TradeMatchingMode.all if match == "all" else TradeMatchingMode.worse
    reader = FileSystemReader(data_root)
    runner = TestRunner(
        trader_cls(),
        reader,
        rnd,
        day,
        show_progress_bar=False,
        print_output=False,
        trade_matching_mode=mode,
    )
    result = runner.run()
    for row in result.final_activities():
        if row.symbol == OSMIUM:
            return int(round(float(row.profit_loss)))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_DEFAULT)
    ap.add_argument("--match", choices=("all", "worse"), default="all")
    ap.add_argument(
        "--configs",
        type=str,
        default="2000:0.65,8000:0.35,8000:0.5,4000:0.35,500:0.65",
        help="Comma list prior:epsilon (epsilon uses eplison patch)",
    )
    args = ap.parse_args()

    configs: list[tuple[str, int, float]] = []
    for chunk in args.configs.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(":")
        configs.append((chunk, int(a), float(b)))

    base_text = SRC.read_text(encoding="utf-8")
    td = Path(tempfile.mkdtemp(prefix="p2tape_"))
    try:
        paths: list[tuple[str, Path]] = []
        for label, prior, eps in configs:
            body = _patch_source(base_text, prior, eps)
            p = td / f"p{prior}_e{str(eps).replace('.', 'p')}.py"
            p.write_text(body, encoding="utf-8")
            paths.append((label, p))

        classes = {label: _load_trader_class(p) for label, p in paths}

        # header
        colw = 10
        tape_labels = [f"R{r}d{d}" for r, d in TAPES]
        print(f"{'tape':<8}", end="")
        for label, _, _ in configs:
            print(f"  {label:>{colw}}", end="")
        print()
        print("-" * (8 + (colw + 2) * len(configs)))

        totals = [0] * len(configs)
        for ti, (rnd, day) in enumerate(TAPES):
            tape = tape_labels[ti]
            print(f"{tape:<8}", end="")
            for ci in range(len(configs)):
                label = configs[ci][0]
                pnl = _run_tape(classes[label], args.data, rnd, day, args.match)
                totals[ci] += pnl
                print(f"  {pnl:>{colw},}", end="")
            print()

        print("-" * (8 + (colw + 2) * len(configs)))
        print(f"{'SUM':<8}", end="")
        for t in totals:
            print(f"  {t:>{colw},}", end="")
        print()
        print()
        print("configs:", args.configs)
        print("data:", args.data)
        print("match:", args.match)
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
