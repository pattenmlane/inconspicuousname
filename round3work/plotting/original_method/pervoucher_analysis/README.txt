Per-voucher diagnostics (original pipeline: winding DTE, Brent IV, quadratic smile in log(S/K)).

Layout
  _common/           — underlying Fig 8 analogs (same for every VEV_*)
  <VEV_K>/day0|1|2/ — session plots + resdf_slice.csv + stats_summary.csv
  <VEV_K>/combined/ — concatenated timelines + pooled stats + full resdf CSV
  <VEV_K>/FINDINGS.txt — metrics + strategy notes (IV scalping vs gamma/hybrid)

Regenerate (slow: full-session IV solve × 3 days once, then all vouchers):
  python3 round3work/plotting/original_method/pervoucher_analysis/build_pervoucher.py

Interactive dashboard (multi-voucher / multi-day, Plotly):
  pip install -r round3work/plotting/original_method/pervoucher_analysis/requirements-viz.txt
  streamlit run round3work/plotting/original_method/pervoucher_analysis/interactive_visualizer.py

Cross-strike plots and methodology text live in ../combined_analysis/.
