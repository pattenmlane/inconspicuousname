VEV_5200 only — Frankfurt-style IV scalping (logic cloned from Prosperity3Winner/FrankfurtHedgehogs_polished.py OptionTrader.get_iv_scalping_orders + calculate_indicators slice).

DTE (historical Round 3): CSV day 0 → 8, day 1 → 7, day 2 → 6 at open; intraday winding via plot_iv_smile_round3.t_years_effective (same as combined_analysis research).

Files:
  calibration.json          — smile coeffs + thresholds (update coeffs after re-running overall fit).
  frankfurt_iv_scalp_core.py — BS, smile IV, EMAs, iv-scalp decision (fixes new_switch_mean → switch_means).
  analyze_vev5200_thresholds.py — optional: quantiles on switch_mean / theo_diff for threshold tuning.
  backtest_vev5200_iv_scalp.py — walk Prosperity4Data ROUND_3 CSVs; simplified full fills at limit prices.
  trader_vev5200_iv_scalp_frankfurt.py — competition Trader (PYTHONPATH: imc-prosperity-4-backtester, 5200_work, combined_analysis).

Run order:
  python3 round3work/voucher_work/overall_work/fit_global_smile.py
  python3 round3work/voucher_work/5200_work/analyze_vev5200_thresholds.py   # optional
  python3 round3work/voucher_work/5200_work/backtest_vev5200_iv_scalp.py

Note: Frankfurt default IV_SCALPING_THR=0.7 is often above historical switch_mean on Round-3
VEV_5200 tape (see threshold_suggestions.txt after analyze). If backtest prints zero trades,
lower IV_SCALPING_THR in calibration.json toward ~0.35–0.40 for research backtests only.
