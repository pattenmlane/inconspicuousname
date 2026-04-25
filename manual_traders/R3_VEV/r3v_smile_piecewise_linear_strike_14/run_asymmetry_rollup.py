#!/usr/bin/env python3
"""
Roll up up/down betas and asymmetry (beta_up - beta_down) from
analysis_outputs/underlying_to_voucher_response.json for inner strikes 5000-5500.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "analysis_outputs" / "underlying_to_voucher_response.json"


def main() -> None:
    data = json.loads(SRC.read_text())
    inner = (5000, 5100, 5200, 5300, 5400, 5500)
    by_sym: dict[str, list[dict]] = {f"VEV_{k}": [] for k in inner}
    for day, syms in data.get("by_day", {}).items():
        for s in inner:
            key = f"VEV_{s}"
            row = syms.get(key)
            if row:
                by_sym[key].append(
                    {
                        "day": day,
                        "beta_up": row["beta_up"],
                        "beta_down": row["beta_down"],
                        "asym": float(row["beta_up"]) - float(row["beta_down"]),
                    }
                )
    out: dict = {
        "source": str(SRC),
        "inner_strikes_5000_5500": {},
        "interpretation": (
            "Across days 0-2, |beta_up - beta_down| is tiny (mostly <0.02) for inner strikes, "
            "so aggressive asymmetric quoting from up/down betas is not data-supported. "
            "Short-horizon extract momentum is used instead, consistent with prior IV–extract comovement."
        ),
    }
    for k in inner:
        key = f"VEV_{k}"
        rows = by_sym[key]
        if not rows:
            out["inner_strikes_5000_5500"][key] = {}
            continue
        asyms = [r["asym"] for r in rows]
        out["inner_strikes_5000_5500"][key] = {
            "asym_median": sorted(asyms)[len(asyms) // 2],
            "asym_max_abs": max(abs(x) for x in asyms),
            "per_day": rows,
        }
    out_path = ROOT / "analysis_outputs" / "underlying_up_down_asymmetry_rollup.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
