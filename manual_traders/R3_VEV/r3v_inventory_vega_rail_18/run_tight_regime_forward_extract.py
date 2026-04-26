"""Aligned series: extract mid; joint-tight = s5200<=TH and s5300<=TH. K-step forward extract mid change (STRATEGY K=20)."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = (
    REPO
    / "manual_traders/R3_VEV/r3v_inventory_vega_rail_18"
    / "analysis_outputs"
    / "tight_regime_forward_extract_k20_by_day.json"
)
TH, K = 2, 20
GATE = ("VEV_5200", "VEV_5300")


def bbo_spread(row: dict) -> tuple[float, float, float, int] | None:
    if not row.get("bid_price_1") or not row.get("ask_price_1"):
        return None
    bb = int(row["bid_price_1"])
    ba = int(row["ask_price_1"])
    if ba < bb:
        return None
    return float(bb), float(ba), 0.5 * (bb + ba), int(ba - bb)


def main() -> None:
    out: dict = {}
    for day in (0, 1, 2, 3):
        path = DATA / f"prices_round_3_day_{day}.csv"
        if not path.exists():
            continue
        by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
        with path.open() as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                by_ts[int(row["timestamp"])][row["product"]] = row
        tss = sorted(by_ts)
        mids: list[float] = []
        is_tight: list[bool] = []
        for ts in tss:
            d = by_ts[ts]
            u = d.get("VELVETFRUIT_EXTRACT")
            if u is None:
                mids.append(float("nan"))
                is_tight.append(False)
                continue
            bu = bbo_spread(u)
            if bu is None:
                mids.append(float("nan"))
                is_tight.append(False)
                continue
            _, _, mu, _ = bu
            sp52 = bbo_spread(d.get("VEV_5200") or {})
            sp53 = bbo_spread(d.get("VEV_5300") or {})
            if sp52 is None or sp53 is None:
                mids.append(float(mu))
                is_tight.append(False)
                continue
            s52, s53 = sp52[3], sp53[3]
            mids.append(float(mu))
            is_tight.append(s52 <= TH and s53 <= TH)
        # valid indices: have mid and K forward
        fw_t: list[float] = []
        fw_n: list[float] = []
        for j in range(len(mids) - K):
            if math.isnan(mids[j]) or math.isnan(mids[j + K]):
                continue
            f = mids[j + K] - mids[j]
            if is_tight[j]:
                fw_t.append(f)
            else:
                fw_n.append(f)
        def share_pos(v: list[float]) -> float | None:
            if not v:
                return None
            return float(sum(1 for x in v if x > 0) / len(v))

        out[str(day)] = {
            "n_rows": len(mids),
            "k": K,
            "n_forward_tight": len(fw_t),
            "n_forward_not_tight": len(fw_n),
            "mean_fw_tight": float(sum(fw_t) / len(fw_t)) if fw_t else None,
            "mean_fw_not_tight": float(sum(fw_n) / len(fw_n)) if fw_n else None,
            "share_pos_fw_tight": share_pos(fw_t),
            "share_pos_fw_not_tight": share_pos(fw_n),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
