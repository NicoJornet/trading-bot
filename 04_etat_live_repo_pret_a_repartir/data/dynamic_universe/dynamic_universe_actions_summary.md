# Dynamic Universe Actions Summary

- as_of: `2026-04-11`
- approved single swaps: `1`
- approved combo swaps: `0`
- approved standalone adds: `8`
- approved standalone removes: `0`
- selected additions: `8`
- selected demotions: `1`

## Selected moves

 action_group    action_type    ticker paired_ticker   side selection_status  selection_score          reason
  0568.HK->ZS         single   0568.HK            ZS    ADD         approved       144.076323 approved_single
  0568.HK->ZS         single        ZS       0568.HK REMOVE         approved       144.076323 approved_single
     ADD:AXTI standalone_add      AXTI                  ADD     approved_add     16838.900761    approved_add
ADD:006800.KS standalone_add 006800.KS                  ADD     approved_add      5420.413830    approved_add
ADD:272210.KS standalone_add 272210.KS                  ADD     approved_add      5414.633350    approved_add
ADD:006400.KS standalone_add 006400.KS                  ADD     approved_add      5386.092520    approved_add
     ADD:ONDS standalone_add      ONDS                  ADD     approved_add      2138.149750    approved_add
  ADD:2383.TW standalone_add   2383.TW                  ADD     approved_add      1321.093720    approved_add
   ADD:285A.T standalone_add    285A.T                  ADD     approved_add       492.451290    approved_add

## Approved standalone adds

   ticker  selection_score promotion_stage dynamic_status entry_timing_bucket  entry_timing_score  shadow_probation_score  portfolio_diversification_score  portfolio_crowding_score  profile_count scan_algo_fit  scan_algo_compat_score_v2  recent_score
     AXTI     16838.900761   approved_live       approved                 hot              3.3380                  1.2842                           0.3125                    0.4548              3          high                  11.164447     15.121425
006800.KS      5420.413830   approved_live       approved               clean              5.0313                  2.0110                           0.7079                    0.0000              3          high                  10.055753      3.587875
272210.KS      5414.633350   approved_live       approved               clean              3.1104                  2.1220                           0.7754                    0.0000              3          high                   9.712875      1.974996
006400.KS      5386.092520   approved_live       approved        constructive              2.6815                  1.4763                           0.7401                    0.0000              3          high                  10.670628      1.207930
  0568.HK      3823.292830   approved_live       approved        constructive              2.9846                  1.6624                           0.6886                    0.0938              3          high                  11.572404      3.706572
     ONDS      2138.149750   approved_live          watch        constructive              2.0235                  1.8132                           0.0000                    0.0000              1          high                  11.425867      3.212938
  2383.TW      1321.093720   approved_live       approved               clean              4.4342                  1.6886                           0.6111                    0.1200              3          high                  12.305714      2.703681
   285A.T       492.451290   approved_live       approved               clean              6.2243                  2.4830                           0.5242                    0.2000              3          high                  13.749797      6.752062

## Approved standalone removes

(none)

## Approved single swaps

candidate removed  selection_score pair_memory_label  pair_memory_score entry_timing_bucket  entry_timing_score  portfolio_diversification_score  portfolio_crowding_score  full_delta_roi_pct  oos_delta_roi_pct  mean_delta_roi_2017_2025  mean_delta_sharpe_2017_2025
  0568.HK      ZS       144.076323         preferred           339.1752        constructive              2.9846                           0.6886                    0.0938        19042.356731       2.364686e-11                  2.469821                     0.048259

## Approved combo swaps

(none)

## Watch single swaps

candidate removed  selection_score  full_delta_roi_pct  oos_delta_roi_pct  mean_delta_roi_2017_2025  mean_delta_sharpe_2017_2025
  0568.HK     UEC        73.771861        20184.671078       2.269995e+02                  0.288163                    -0.005033
     AXTI     UEC        73.363291        20184.671078       2.269995e+02                  0.288163                    -0.005033
272210.KS    AXON        66.448493        15703.438314       6.773504e+02                 -3.192093                     0.008973
272210.KS      ZS              NaN        40286.678686       0.000000e+00                       NaN                          NaN
272210.KS    RACE              NaN        30767.331283       0.000000e+00                       NaN                          NaN
272210.KS     NET              NaN        17657.814998       0.000000e+00                       NaN                          NaN
  0568.HK    RACE              NaN        10719.533655       2.364686e-11                       NaN                          NaN
  0568.HK    AXON              NaN        -2397.081620       6.773504e+02                       NaN                          NaN
  0568.HK     NET              NaN        -1075.186875       2.364686e-11                       NaN                          NaN
     AXTI      ZS              NaN        19042.356731      -1.818989e-12                       NaN                          NaN
     AXTI    RACE              NaN        10719.533655      -1.818989e-12                       NaN                          NaN
     AXTI    AXON              NaN        -2397.081620       6.773504e+02                       NaN                          NaN

## Watch combo swaps

                            combo  selection_score  full_delta_roi_pct  oos_delta_roi_pct  mean_delta_roi_2017_2025  mean_delta_sharpe_2017_2025
             FIEE->UEC + UI->MELI       232.451333        71810.992501        3269.204762                  6.930276                    -0.001481
            FIEE->PAAS + UI->MELI       224.648692       244115.548613        3610.427756                  3.799245                     0.001325
             FIEE->PAAS + UI->UEC       203.326375       314231.180926        3269.204762                  5.895803                    -0.015083
             FIEE->PAAS + UI->WPM       126.816234       208352.445865        2697.314416                  1.467639                    -0.020797
FIEE->PAAS + UI->WPM + CIEN->MELI              NaN        84940.864338        2697.314416                       NaN                          NaN
FIEE->PAAS + UI->UEC + CIEN->MELI              NaN       128936.441150        2525.838198                       NaN                          NaN
              FIEE->UEC + UI->WPM              NaN       124971.964268        2381.464056                       NaN                          NaN
          FIEE->PAAS + CIEN->MELI              NaN       177057.712819        1786.297460                       NaN                          NaN
 FIEE->UEC + UI->WPM + CIEN->MELI              NaN       -20271.366160        1693.053341                       NaN                          NaN
             UI->UEC + CIEN->MELI              NaN        89414.876516        1552.306813                       NaN                          NaN
             UI->WPM + CIEN->MELI              NaN         5250.030195        1097.455075                       NaN                          NaN
           FIEE->UEC + CIEN->MELI              NaN        48775.121994         863.090222                       NaN                          NaN

## Watch standalone adds

       ticker  selection_score      promotion_stage dynamic_status  portfolio_diversification_score  portfolio_crowding_score  profile_count scan_algo_fit  scan_algo_compat_score_v2  recent_score
       PLS.AX     15504.868010 targeted_integration          watch                           0.6063                    0.1200              3          high                  11.228013      1.718995
       9501.T      5356.977120          watch_queue         review                           0.7336                    0.0000              3           low                   5.926773      0.124049
         INTC      5294.808950          watch_queue       approved                           0.2923                    0.2286              3        medium                   7.573789      1.215276
    042660.KS      5135.465880          watch_queue         review                           0.7194                    0.0000              2           low                   5.755471      0.382396
       1605.T      5109.276570          watch_queue         review                           0.8076                    0.0000              3           low                   6.829119      0.703179
      6443.TW      2789.126590          watch_queue       approved                           0.6432                    0.1200              3        medium                   9.334782      1.700872
           AR      1898.558720          watch_queue         review                           0.7636                    0.0000              2           low                   6.797791      0.206072
         KEYS      1830.457100          watch_queue          watch                           0.0000                    0.0000              3        medium                   9.345636      1.058250
       6857.T      1433.217990 targeted_integration          watch                           0.4722                    0.2000              3          high                  10.524191      1.451469
         UCTT       971.351205 targeted_integration    prime_watch                           0.2420                    0.3897              3          high                   9.947366      1.309426
           HL       957.249150          watch_queue          watch                           0.0000                    0.0000              1        medium                   7.615295      1.225671
           PR       938.968260 targeted_integration          watch                           0.7423                    0.0000              3        medium                   7.468573      0.698515
       GFI.JO       790.738800          watch_queue       approved                           0.0000                    0.0000              3        medium                   7.095092      0.428291
ADANIPOWER.NS       770.469550          watch_queue       approved                           0.0000                    0.0000              2        medium                   7.004169      0.288882
      3037.TW       668.689350 targeted_integration          watch                           0.0000                    0.0000              1          high                  11.397378      2.855331
      2345.TW       616.367040 targeted_integration          watch                           0.6052                    0.1200              2          high                  11.284288      1.146215
          OVV       520.204450          watch_queue          watch                           0.7470                    0.0000              2        medium                   7.685295      0.454361
          TER       507.323760 targeted_integration          watch                           0.2240                    0.4954              3          high                  11.208605      2.366201
         LUNR       429.118300          watch_queue          watch                           0.0000                    0.0000              1        medium                   7.565495      1.484660
          FTI       426.743700 targeted_integration       approved                           0.0000                    0.0000              3          high                  10.354666      0.916307

## Watch standalone removes

ticker  selection_score  dead_score  full_delta_roi_pct  oos_delta_roi_pct  mean_delta_roi_2017_2025  mean_delta_sharpe_2017_2025
    ZS        62.853544    0.824912        15741.784209          -0.161731                  1.044335                     0.046060
  MARA        43.492126    0.931739       -49300.852104         654.785186                  0.000000                     0.000000
  AVAV        31.164564    0.945224        13614.024593           0.000000                  0.361534                     0.017273
  AXON        28.144251    0.784143         -511.949985         385.601857                  0.000000                     0.000000
  RACE        25.608478    0.890132         8880.172842          -0.694492                  0.664118                     0.010318
   UEC         8.915225    0.457539         1926.461055          85.055737                  0.000000                     0.000000
   WPM       -30.158839    0.486496         5645.781592          -0.694492                 -5.434602                     0.019358