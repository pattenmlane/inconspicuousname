Round 4 Phase 1 — automated summary (see CSVs in same folder)
Global mean same-symbol fwd_mid_20: 0.0918439

Top directed pairs by count:
  Mark 01 -> Mark 22: n=1339
  Mark 14 -> Mark 38: n=728
  Mark 38 -> Mark 14: n=714
  Mark 55 -> Mark 14: n=331
  Mark 14 -> Mark 55: n=316
  Mark 01 -> Mark 55: n=260
  Mark 55 -> Mark 01: n=244
  Mark 67 -> Mark 49: n=89
  Mark 14 -> Mark 22: n=83
  Mark 67 -> Mark 22: n=75
  Mark 38 -> Mark 22: n=19
  Mark 22 -> Mark 55: n=18

Aggressor bucket same-symbol fwd20:
aggressor,n,mean_fwd20,median_fwd20,t_fwd20_mean0,ci_fwd20_lo,ci_fwd20_hi,frac_pos
buy_aggr,1500,0.31033333333333335,0.5,1.9982595967788521,-0.009791666666666662,0.59075,0.5186666666666667
sell_aggr,2776,-0.026476945244956772,0.0,-0.3127696466857825,-0.1982303674351585,0.11942543227665708,0.3022334293948127

Burst event study:
Burst vs control (VELVETFRUIT_EXTRACT forward, same row index K)
K=5  burst n=1608 mean=0.028918  control n=1608 mean=0.17662  welch_t=-1.948
K=20  burst n=1608 mean=0.33427  control n=1607 mean=0.16895  welch_t=1.059
K=100  burst n=1592 mean=0.4777  control n=1596 mean=-0.067669  welch_t=1.649

Burst vs control (VEV_5300 forward at trade timestamps, same row index K)
K=5  burst n=1608 mean=0.18968  control n=1608 mean=0.016791  welch_t=5.622
K=20  burst n=1608 mean=0.3041  control n=1607 mean=0.018357  welch_t=4.818
K=100  burst n=1592 mean=0.30402  control n=1596 mean=-0.12813  welch_t=3.500

Signed-flow lead–lag (head):
 day         flow_symbol  lag_trade_events   n  corr_net_flow_lagged_vs_extract_fwd20
   1 VELVETFRUIT_EXTRACT                 0 446                               0.105661
   1 VELVETFRUIT_EXTRACT                 1 445                               0.011832
   1 VELVETFRUIT_EXTRACT                 2 444                               0.129793
   1 VELVETFRUIT_EXTRACT                 3 443                               0.077554
   1 VELVETFRUIT_EXTRACT                 5 441                               0.013589
   1       HYDROGEL_PACK                 0 375                              -0.084644
   1       HYDROGEL_PACK                 1 374                               0.033309
   1       HYDROGEL_PACK                 2 373                              -0.020489
   1       HYDROGEL_PACK                 3 372                               0.008904
   1       HYDROGEL_PACK                 5 370                              -0.059676
   1            VEV_5300                 0  39                               0.120559
   1            VEV_5300                 1  38                              -0.025322
