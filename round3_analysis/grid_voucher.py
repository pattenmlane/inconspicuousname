"""Grid-tune voucher params."""
import re, subprocess
from pathlib import Path
STRAT = "round3_analysis/strats/strat_combined_v3.py"

def total_profit(out):
    m = re.search(r"Total profit: ([-\d,]+)", out.split("Profit summary:")[-1])
    return int(m.group(1).replace(",", "")) if m else None

def run():
    out = subprocess.run(["env", "PYTHONPATH=imc-prosperity-4-backtester", "python3",
                          "-m", "prosperity4bt", STRAT, "3", "--data", "/tmp/btdata",
                          "--no-out", "--no-vis", "--no-progress"],
                         capture_output=True, text=True).stdout
    return total_profit(out)

orig = Path(STRAT).read_text()
results = []
for size in [50, 100, 150, 200, 300]:
    for cap in [100, 150, 250, 300]:
        for skew in [0, 1, 2]:
            txt = orig.replace("VOUCHER_SIZE = 100", f"VOUCHER_SIZE = {size}")
            txt = txt.replace("VOUCHER_SOFT_CAP = 250", f"VOUCHER_SOFT_CAP = {cap}")
            txt = txt.replace("VOUCHER_SKEW_PER100 = 1", f"VOUCHER_SKEW_PER100 = {skew}")
            Path(STRAT).write_text(txt)
            p = run()
            results.append((size, cap, skew, p))
            print(f"size={size} cap={cap} skew={skew} pnl={p}")
Path(STRAT).write_text(orig)
results.sort(key=lambda r: r[3] or -1e9, reverse=True)
print("\nTop 5:")
for r in results[:5]:
    print(r)
