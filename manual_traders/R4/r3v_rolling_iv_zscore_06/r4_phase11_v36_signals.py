#!/usr/bin/env python3
"""Build `r4_v36_signals.json` from `r4_v26_signals.json` by dropping merged T >= day3 offset."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V26 = ROOT / "outputs" / "r4_v26_signals.json"
OUT = ROOT / "outputs" / "r4_v36_signals.json"


def main() -> None:
    obj = json.loads(V26.read_text())
    cum = int(obj["day_cum_offset"]["3"])
    tr = sorted(int(x) for x in obj["mark67_extract_buy_aggr_filtered_merged_ts"])
    kept = [t for t in tr if t < cum]
    out = dict(obj)
    out["mark67_extract_buy_aggr_filtered_merged_ts"] = kept
    out["rule"] = str(obj.get("rule", "")) + "_drop_merged_day3_triggers"
    out["n_triggers_v26"] = len(tr)
    out["n_triggers_v36"] = len(kept)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT, "n", len(kept))


if __name__ == "__main__":
    main()
