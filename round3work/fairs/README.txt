Round 3 hold-1 fair probes (website .log → prices / trades / true_fv).

Layout
  <Product>fair/<submissionId>/
    *.log                          — Prosperity JSON export (source of truth)
    *.py / *.json                  — submission copy for that run
    prices_round_3_day_39.csv      — full book rows for the probed product (day column = 39)
    trades_round_3_day_39.csv      — tradeHistory rows for that symbol
    <PRODUCT>_true_fv_day39.csv    — true_fv = profit_and_loss + buy_price
    export_meta_day39.txt          — buy_price, row counts, paths
    *.zip                          — optional archive of the run

Products covered (Round 3)
  VEV_4000 … VEV_6500 (10 *fair folders), VELVETFRUIT_EXTRACTfair, hydrogel/ (HYDROGEL_PACK).

Regenerate CSVs from logs (after editing exporter or moving folders):
  cd round3work
  find fairs -mindepth 3 -maxdepth 3 -name '*.log' | while read -r log; do
    python3 fairs/export_hold1_log_round3.py --log "$log" --out-dir "$(dirname "$log")"
  done

New probe: copy ../trader_hold1_fair_probe.py, set TARGET_PRODUCT, upload, export .log into a new
<Product>fair/<id>/ or hydrogel/<id>/ folder, then run the command above.
