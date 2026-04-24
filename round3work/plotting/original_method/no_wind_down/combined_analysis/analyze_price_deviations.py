"""
Map smile-fitted IV to theoretical option price vs mid; lag-1 autocorr of price deviation.
Supports tips: deviation meaning + mean-reversion evidence (cf. Frankfurt writeup).
Outputs in round3work/plotting/original_method/combined_analysis/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_iv_smile_round3 import STRIKES, bs_call_price, build_iv_and_residual_dataframe, dte_from_csv_day

OUT_DIR = Path(__file__).resolve().parent


def lag1_autocorr(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 3:
        return float("nan")
    a = s.to_numpy(dtype=float)
    x, y = a[:-1], a[1:]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    _, resdf = build_iv_and_residual_dataframe(step=20)
    if "t_years" in resdf.columns:
        T_arr = resdf["t_years"].to_numpy(dtype=float)
    else:
        T_arr = resdf["dte"].to_numpy(dtype=float) / 365.0
    iv_fit = resdf["iv_fit"].to_numpy(dtype=float)
    S = resdf["S"].to_numpy(dtype=float)
    K = resdf["K"].to_numpy(dtype=float)
    mid = resdf["mid"].to_numpy(dtype=float)
    n = len(resdf)
    theo = np.full(n, np.nan, dtype=float)
    for i in range(n):
        sig = iv_fit[i]
        if not np.isfinite(sig) or S[i] <= 0 or K[i] <= 0 or T_arr[i] <= 0:
            continue
        theo[i] = bs_call_price(float(S[i]), float(K[i]), float(T_arr[i]), float(sig))
    resdf = resdf.assign(theoretical_mid=theo, price_deviation=mid - theo)

    rows = []
    for (day, v), g in resdf.groupby(["day", "voucher"]):
        g = g.sort_values("timestamp")
        ac = lag1_autocorr(g["price_deviation"])
        rows.append(
            {
                "day": day,
                "voucher": v,
                "lag1_autocorr_price_dev": ac,
                "mean_abs_price_dev": float(np.nanmean(np.abs(g["price_deviation"]))),
            }
        )
    ac_tbl = pd.DataFrame(rows)
    ac_tbl.to_csv(OUT_DIR / "price_deviation_lag1_autocorr.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, day in zip(axes, (0, 1, 2)):
        sub = ac_tbl[ac_tbl["day"] == day].set_index("voucher").reindex([f"VEV_{k}" for k in STRIKES])
        ys = sub["lag1_autocorr_price_dev"].to_numpy(dtype=float)
        ax.bar(range(len(STRIKES)), ys, tick_label=[str(k) for k in STRIKES])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"day {day} (DTE {dte_from_csv_day(day)} open, winds intraday)")
        ax.set_xlabel("strike")
        ax.set_ylabel("lag-1 autocorr(dev)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Price deviation (mid − BS(iv_fit)) lag-1 autocorrelation", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "price_deviation_lag1_autocorr_bars.png", dpi=160)
    plt.close()

    # Example time series: near-ATM
    for target in ("VEV_5000", "VEV_5100"):
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for ax, day in zip(axes, (0, 1, 2)):
            g = resdf[(resdf["day"] == day) & (resdf["voucher"] == target)].sort_values("timestamp")
            ax.plot(g["timestamp"], g["price_deviation"], lw=0.8, alpha=0.9)
            ax.set_ylabel("dev")
            ax.set_title(f"{target} day={day}")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("timestamp")
        fig.suptitle(f"Price deviation vs time — {target}", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"price_deviation_timeseries_{target}.png", dpi=160)
        plt.close()

    print("Wrote", OUT_DIR / "price_deviation_lag1_autocorr.csv", "and related PNGs")


if __name__ == "__main__":
    main()
