#!/usr/bin/env python3
"""Run analysis + validation + exact-rule scripts on all known fair sessions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(args: list[str]) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=str(HERE))


def main() -> None:
    scripts = [
        [PY, str(HERE / "analyze_osmium_quote_rules.py"), "--all-sessions"],
        [PY, str(HERE / "analyze_osmium_near_fv.py"), "--all-sessions"],
        [PY, str(HERE / "validate_osmium_inner.py"), "--all-sessions"],
        [PY, str(HERE / "validate_osmium_wall.py"), "--all-sessions"],
        [PY, str(HERE / "validate_osmium_near_fv.py"), "--all-sessions"],
        [PY, str(HERE / "osmium_inner_exact_rule.py"), "--all-sessions"],
        [PY, str(HERE / "osmium_wall_exact_rule.py"), "--all-sessions"],
        [PY, str(HERE / "osmium_near_fv_exact_rule.py"), "--all-sessions"],
    ]
    for cmd in scripts:
        run(cmd)
    print("\nDone. See MULTISESSION txt files and per-session outputs in each data_dir.")


if __name__ == "__main__":
    main()
