Round 4 Phase 2 summary (see CSV/txt files)

EXTRACT: corr(spread, |Δmid next row|) = -0.2160
EXTRACT: corr(microprice-mid, |Δmid next row|) = 0.0104
(next row = consecutive timestamp row per day)

Mark 67 fwd20 by Sonic gate:
sonic_tight,n,mean_fwd20_same_sym,frac_pos
True,27,2.0555555555555554,0.5925925925925926
False,138,1.8043478260869565,0.6594202898550725

Mark55→14 extract by Sonic:
sonic_tight,n,mean_fwd20
True,86,0.4418604651162791
False,245,0.6224489795918368

Basket burst vs control (extract):
horizon_K,n_burst_ts,n_control_ts,mean_burst,mean_control
5,314,314,-0.0015923566878980893,0.012738853503184714
20,314,314,0.2627388535031847,0.03662420382165605
100,311,311,0.13183279742765272,-0.6688102893890675
