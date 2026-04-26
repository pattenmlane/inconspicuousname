Pooled days [1, 2, 3]: n_tight=7706 n_loose=22234
mean_fwd_extract_K20 tight=0.323514 loose=-0.121031 welch_t=7.9351 p=2.268e-15
P(tight)=0.2574
corr(s5200,s5300)=0.3071

--- By day tight vs loose (extract fwd K=20) ---
 day  n_tight  n_loose  mean_fwd_extract_tight  mean_fwd_extract_loose  welch_t      p_value  P_tight
   1     1381     8599                0.599203               -0.048320 5.324774 1.132599e-07 0.138377
   2     1727     8253                0.464100               -0.018115 4.238249 2.336409e-05 0.173046
   3     4598     5382                0.187908               -0.395020 6.781484 1.257856e-11 0.460721

--- Spread correlations by day ---
 day  corr_s5200_s5300  corr_s5200_m_ext  corr_s5300_m_ext  corr_s5200_fwd_k  corr_s5300_fwd_k  corr_s_ext_fwd_k
   1          0.235549          0.388153          0.324251         -0.066002         -0.040869         -0.026243
   2          0.264989          0.465786          0.406942         -0.051655         -0.048609         -0.008297
   3          0.184113          0.425123          0.351445         -0.072799         -0.053694         -0.043448

--- Mark01->22 gate summary ---
Mark 01 -> Mark 22 all symbols pooled:
  n_all=1339 mean_fwd20=0.0085885
  n_tight=1336 mean=0.00860778
  n_loose=3 mean=0
Welch tight vs loose: (0.008607784431137725, 0.0, 0.43663778356201505, 0.6624446652078051)
