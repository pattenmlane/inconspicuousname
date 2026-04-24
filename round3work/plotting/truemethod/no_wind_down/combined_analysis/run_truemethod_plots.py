"""Generate true_fv IV plots (DTE=5 flat over the session — no intraday winding)."""
from __future__ import annotations

import sys
from pathlib import Path

_COMMON = Path(__file__).resolve().parent.parent.parent / "common"
sys.path.insert(0, str(_COMMON))

from iv_smile_true_fv import run_all  # noqa: E402

if __name__ == "__main__":
    run_all(winding=False, out_dir=Path(__file__).resolve().parent)
