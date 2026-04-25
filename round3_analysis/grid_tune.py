"""Grid-tune hydrogel & voucher knobs against the official IMC backtester."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

STRAT = "round3_analysis/strats/strat_combined_v3.py"


def replace(path, old, new):
    p = Path(path)
    txt = p.read_text()
    if old not in txt:
        raise SystemExit(f"old not found: {old}")
    p.write_text(txt.replace(old, new))


def restore(path):
    return Path(path).read_text()


def total_profit(out):
    m = re.search(r"Total profit: ([-\d,]+)", out.split("Profit summary:")[-1])
    if m:
        return int(m.group(1).replace(",", ""))
    return None


def run_backtest():
    cmd = ["env", "PYTHONPATH=imc-prosperity-4-backtester", "python3", "-m",
           "prosperity4bt", STRAT, "3", "--data", "/tmp/btdata", "--no-out",
           "--no-vis", "--no-progress"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    return total_profit(out), out


def grid_hydro_band_skew():
    bands = [10, 15, 20, 25, 30]
    skews = [0.02, 0.04, 0.06, 0.08]
    orig = restore(STRAT)
    results = []
    for b in bands:
        for s in skews:
            txt = orig.replace("HYDRO_BAND = 20.0", f"HYDRO_BAND = {b:.1f}")
            txt = txt.replace("HYDRO_SKEW = 0.04", f"HYDRO_SKEW = {s:.3f}")
            Path(STRAT).write_text(txt)
            tp, _ = run_backtest()
            results.append((b, s, tp))
            print(f"  band={b} skew={s} pnl={tp}")
    Path(STRAT).write_text(orig)
    return results


def main():
    print("Grid-tune HYDRO_BAND, HYDRO_SKEW")
    print("-" * 40)
    res = grid_hydro_band_skew()
    res.sort(key=lambda r: r[2] or -1e9, reverse=True)
    print("\nTop 5:")
    for b, s, p in res[:5]:
        print(f"  band={b} skew={s} pnl={p}")


if __name__ == "__main__":
    main()
