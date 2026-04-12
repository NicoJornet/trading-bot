# Dynamic Universe Swap Study (184)

## Baseline

- full ROI: `131953.95%`
- full Sharpe: `1.89360`
- full MaxDD: `-44.65%`
- OOS ROI: `8848.88%`
- OOS Sharpe: `2.78418`
- OOS MaxDD: `-34.89%`

## Candidate shortlist

   ticker dynamic_status recommendation scan_algo_fit  scan_algo_compat_score  recent_score  full_delta_roi_pct  oos_delta_roi_pct
     AXTI       approved            add          high               11.000000     15.121425       396860.138933        6504.679493
  0568.HK       approved            add          high                9.500000      3.706572        82296.690681        2105.670804
042660.KS         review            add           low                4.725000      0.382396       119040.435935        1996.838512
272210.KS       approved            add          high                8.413333      1.974996       122584.506751        1975.767486

## Demotion shortlist

ticker  retain_score  latest_rank  latest_score  days_top15_trend  realized_pnl_eur  buy_count
   UEC      0.662634         36.0      0.557057               576      -8618.389380        2.0
  AXON      0.652419        181.0     -0.440109               616      46835.236618        6.0
   NET      0.678763        113.0      0.085113               390       1264.659157        1.0
   WPM      0.753226         49.0      0.442527               329        485.924978        2.0
    ZS      0.589113        184.0     -0.497915               431        872.570137        4.0
  RACE      0.429167        148.0     -0.127221               257        -65.149823        2.0

## Top single swaps

candidate removed recommendation  full_delta_roi_pct  oos_delta_roi_pct  full_delta_sharpe  oos_delta_sharpe  full_delta_maxdd_pct  oos_delta_maxdd_pct
  0568.HK     UEC        promote        20184.671078       2.269995e+02           0.023067     -1.891750e-03              0.000000            -0.005499
     AXTI     UEC        promote        20184.671078       2.269995e+02           0.023067     -1.891750e-03              0.000000            -0.005499
  0568.HK      ZS        promote        19042.356731       2.364686e-11           0.024349      4.440892e-16              0.000000             0.000000
  0568.HK    RACE        promote        10719.533655       2.364686e-11           0.014766      4.440892e-16             -0.106626             0.000000
  0568.HK    AXON          watch        -2397.081620       6.773504e+02           0.004639      4.813091e-02             -2.304796             6.708974
272210.KS    AXON          watch        15703.438314       6.773504e+02           0.023300      4.813091e-02             -2.304796             6.708974
     AXTI    AXON          watch        -2397.081620       6.773504e+02           0.004639      4.813091e-02             -2.304796             6.708974
  0568.HK     NET          watch        -1075.186875       2.364686e-11          -0.002980      4.440892e-16              0.000000             0.000000
272210.KS      ZS         reject        40286.678686       0.000000e+00           0.043038      0.000000e+00              0.000000             0.000000
272210.KS    RACE         reject        30767.331283       0.000000e+00           0.033462      0.000000e+00             -0.106626             0.000000
272210.KS     NET         reject        17657.814998       0.000000e+00           0.014889      0.000000e+00              0.000000             0.000000
     AXTI      ZS         reject        19042.356731      -1.818989e-12           0.024349      0.000000e+00              0.000000             0.000000
     AXTI    RACE         reject        10719.533655      -1.818989e-12           0.014766      0.000000e+00             -0.106626             0.000000
     AXTI     NET         reject        -1075.186875      -1.818989e-12          -0.002980      0.000000e+00              0.000000             0.000000
  0568.HK     WPM         reject          -17.946444      -5.583805e+02          -0.010731     -4.816369e-02              0.007432             0.014806
     AXTI     WPM         reject          -17.946444      -5.583805e+02          -0.010731     -4.816369e-02              0.007432             0.014806

## Phase 2 walk-forward shortlist

candidate removed recommendation  phase1_score  full_delta_roi_pct  oos_delta_roi_pct  full_delta_sharpe  oos_delta_sharpe
  0568.HK     UEC        promote    422.025761        20184.671078       2.269995e+02           0.023067     -1.891750e-03
     AXTI     UEC        promote    422.025761        20184.671078       2.269995e+02           0.023067     -1.891750e-03
  0568.HK      ZS        promote    381.090629        19042.356731       2.364686e-11           0.024349      4.440892e-16
272210.KS    AXON          watch    373.891835        15703.438314       6.773504e+02           0.023300      4.813091e-02

## Walk-forward summary

              swap  mean_delta_roi_2017_2025  mean_delta_sharpe_2017_2025  mean_delta_maxdd_2017_2025  roi_wins_2017_2025  sharpe_wins_2017_2025  delta_roi_2026_ytd  delta_sharpe_2026_ytd  delta_maxdd_2026_ytd
    0568.HK_for_ZS                  2.469821                     0.048259                   -0.378494                   4                      4                 0.0           0.000000e+00                   0.0
272210.KS_for_AXON                 -3.192093                     0.008973                    0.258216                   2                      3                 0.0           0.000000e+00                   0.0
   0568.HK_for_UEC                  0.288163                    -0.005033                   -0.444831                   3                      2                 0.0           0.000000e+00                   0.0
      AXTI_for_UEC                  0.288163                    -0.005033                   -0.444831                   2                      1                 0.0          -1.776357e-15                   0.0