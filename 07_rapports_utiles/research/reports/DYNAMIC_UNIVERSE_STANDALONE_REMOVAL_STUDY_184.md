# Dynamic Universe Standalone Removal Study

## Demotion shortlist

ticker  retain_score  dead_score  latest_rank  latest_score  days_top15_trend  realized_pnl_eur  buy_count
   UEC      0.679235    0.457539         33.0      0.557057               622     -10212.292625        2.0
  AXON      0.658743    0.784143        178.0     -0.440109               637      72791.772068        6.0
   NET      0.701093    0.652363        110.0      0.085113               422      11521.664590        2.0
   WPM      0.759836    0.486496         46.0      0.442527               331        493.970090        2.0
    ZS      0.588525    0.824912        181.0     -0.497915               444        886.914249        4.0
  RACE      0.430464    0.890132        145.0     -0.127221               267        -66.092028        2.0
  AVAV      0.449727    0.945224        161.0     -0.217376               375      -1348.647886        3.0
  MARA      0.485519    0.931739        175.0     -0.367431               628     -31076.704874        8.0

## Phase 2 walk-forward shortlist

ticker  phase1_score  full_delta_roi_pct  oos_delta_roi_pct  full_delta_sharpe  oos_delta_sharpe  dead_score
    ZS    319.198600        15741.784209          -0.161731           0.025149         -0.000008    0.824912
  AVAV    290.366136        13614.024593           0.000000           0.025418          0.000000    0.945224
  RACE    181.730194         8880.172842          -0.694492           0.015289         -0.000062    0.890132
   WPM    115.398213         5645.781592          -0.694492           0.004046         -0.000062    0.486496

## Standalone removals

ticker selection_status  selection_score  full_delta_roi_pct  oos_delta_roi_pct  full_delta_sharpe  oos_delta_sharpe  full_delta_maxdd_pct  oos_delta_maxdd_pct  mean_delta_roi_2017_2025  mean_delta_sharpe_2017_2025
    ZS     watch_remove        62.853544        15741.784209          -0.161731           0.025149         -0.000008              0.000000             0.000000                  1.044335                     0.046060
  MARA     watch_remove        43.492126       -49300.852104         654.785186          -0.027994          0.071774             -0.118499            -0.028936                  0.000000                     0.000000
  AVAV     watch_remove        31.164564        13614.024593           0.000000           0.025418          0.000000              3.276338             0.000000                  0.361534                     0.017273
  AXON     watch_remove        28.144251         -511.949985         385.601857           0.006839          0.051137             -2.301789             6.705828                  0.000000                     0.000000
  RACE     watch_remove        25.608478         8880.172842          -0.694492           0.015289         -0.000062             -0.104944             0.000035                  0.664118                     0.010318
   UEC     watch_remove         8.915225         1926.461055          85.055737           0.004314          0.004352              0.000000            -0.004225                  0.000000                     0.000000
   WPM     watch_remove       -30.158839         5645.781592          -0.694492           0.004046         -0.000062              0.016670             0.000035                 -5.434602                     0.019358
   NET    reject_remove         6.515392        -1793.135897          -0.161731          -0.004397         -0.000008             -0.725712             0.000000                  0.000000                     0.000000