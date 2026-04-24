Six-branch global IV parabola (only the fit — no figure regeneration)

  python3 round3work/plotting/fit_parabola_six_branches.py

Writes: parabola_fits_six_branches.json

Each branch has two strike pools:
  - all_strikes: VEV_4000 … VEV_6500
  - near_5000_5500: VEV_5000 … VEV_5500

Fit: IV ≈ poly₂(m) with m = log(K/S)/sqrt(T), coeffs numpy polyfit order high→low (m², m, const).

Branches:
  - original_method_wind_down / no_wind_down — Brent IV from that folder’s plot_iv_smile_round3; historical days 0–2.
  - test_implementation_wind_down / no_wind_down — notebook bisection IV + that nb_method_core T rule; same historical days.
  - truemethod_* — Brent IV on merged true_fv (fairs day 39), DTE=5 with or without intraday wind.
