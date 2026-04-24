Overall voucher analysis (all 10 VEV_* on VELVETFRUIT_EXTRACT).

Run (from repo root):
  python3 round3work/voucher_work/overall_work/fit_global_smile.py

Outputs:
  fitted_smile_coeffs.json — quadratic IV smile in m_t = log(K/S)/sqrt(T) (Frankfurt-style),
                            pooled over historical days 0–2 (subsampled for speed).

DTE convention (hardcoded, Round 3 historical):
  CSV day 0 → DTE 8 at open, day 1 → 7, day 2 → 6; intraday winding via
  round3work/plotting/original_method/combined_analysis/plot_iv_smile_round3.py
