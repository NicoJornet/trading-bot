# Dynamic Universe Discovery (broad)

## Purpose

- build a dynamic candidate pool without relying only on a fixed universe
- combine broad discovery with local structural leader sentinels and, when enabled, Yahoo expansion layers
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Source Profile

- `local_structural`

## Current raw leaders

- AXTI, SNDK, LITE, WDC, BE, 006800.KS, 0568.HK, MU, LRCX, VRT

## Seed tickers

- AXTI, SNDK, LITE, WDC, BE, 006800.KS

## Keywords

- (none)

## Discovery summary

- total discovered symbols: `143`
- new symbols outside active/reserve/exclusions: `98`

## Top new candidates by discovery score

   ticker  priority_score  source_count                                                                                                   source_types    marketCap  averageDailyVolume3Month                 sector                            industry
   285A.T         56.2640             7 historical_winner_cousin|horizon_emerging|industry_relative_leader|local_cluster_leader|universe_base_largecap 1.642841e+13                34125386.0             Technology                      Semiconductors
       PL         48.7654             6     horizon_emerging|industry_relative_leader|local_cluster_leader|sector_relative_leader|universe_base_midcap 1.200099e+10                14187822.0            Industrials                 Aerospace & Defense
   5803.T         42.5969             5                          horizon_emerging|industry_relative_leader|local_cluster_leader|universe_base_largecap 9.320922e+12                62997573.0            Industrials                       Conglomerates
006800.KS         39.8871             5                                           historical_winner_cousin|local_cluster_leader|sector_relative_leader 3.860235e+13                 6251020.0     Financial Services                     Capital Markets
272210.KS         37.3584             5                                                                  historical_winner_cousin|local_cluster_leader 2.492620e+13                 3234302.0            Industrials                 Aerospace & Defense
   8802.T         35.3444             4                            horizon_pullback|local_cluster_leader|sector_relative_leader|universe_base_largecap 5.340088e+12                 4216232.0            Real Estate           Real Estate - Diversified
     WULF         34.7262             4                          horizon_emerging|industry_relative_leader|sector_relative_leader|universe_base_midcap 8.002166e+09                32261596.0     Financial Services                     Capital Markets
     AXTI         34.2017             4                          horizon_emerging|industry_relative_leader|sector_relative_leader|universe_base_midcap 3.567035e+09                 9816487.0             Technology Semiconductor Equipment & Materials
  0883.HK         33.8290             4                                               historical_winner_cousin|horizon_pullback|universe_base_largecap 1.431382e+12               143530238.0                 Energy                       Oil & Gas E&P
   PLS.AX         32.8591             4                                           industry_relative_leader|sector_relative_leader|universe_base_midcap 1.726456e+10                27722127.0        Basic Materials    Other Industrial Metals & Mining
      CDE         32.8556             4                                           industry_relative_leader|sector_relative_leader|universe_base_midcap 1.668722e+10                28205306.0        Basic Materials                                Gold
042660.KS         29.1129             4                                                                                       historical_winner_cousin 3.780941e+13                 2241938.0            Industrials                 Aerospace & Defense
   1605.T         28.3185             3                                             historical_winner_cousin|horizon_pullback|industry_relative_leader 4.819549e+12                10179613.0                 Energy                       Oil & Gas E&P
  2383.TW         27.6402             3                                                 horizon_emerging|industry_relative_leader|local_cluster_leader 1.225458e+12                 3773877.0             Technology               Electronic Components
000270.KS         27.5374             3                                           historical_winner_cousin|local_cluster_leader|sector_relative_leader 6.023594e+13                 1831216.0      Consumer Cyclical                  Auto Manufacturers
  0016.HK         27.1346             3                                             local_cluster_leader|sector_relative_leader|universe_base_largecap 3.732341e+11                 7003715.0            Real Estate           Real Estate - Development
  0857.HK         26.2955             3                                                                    local_cluster_leader|universe_base_largecap 2.693680e+12               154095557.0                 Energy                Oil & Gas Integrated
   7203.T         25.6675             3                                                                historical_winner_cousin|universe_base_largecap 4.441778e+13                24019943.0      Consumer Cyclical                  Auto Manufacturers
006400.KS         25.0115             3                                                                      historical_winner_cousin|horizon_emerging 3.785164e+13                 1069777.0            Industrials        Electrical Equipment & Parts
   CCO.TO         22.2682             3                                                                                       historical_winner_cousin 7.000321e+10                 1042085.0                 Energy                             Uranium
     INTC         20.4829             2                                                                          horizon_emerging|local_cluster_leader 3.132106e+11               108771117.0             Technology                      Semiconductors
   5713.T         19.7864             2                                                                  historical_winner_cousin|local_cluster_leader 2.701737e+12                 6343450.0        Basic Materials    Other Industrial Metals & Mining
   8316.T         19.6192             2                                                                        horizon_pullback|universe_base_largecap 2.002925e+13                15047175.0     Financial Services                 Banks - Diversified
  AMXB.MX         19.5066             2                                                                    local_cluster_leader|universe_base_largecap 1.352137e+12                54209017.0 Communication Services                    Telecom Services
   6857.T         19.4269             2                                                                        horizon_pullback|universe_base_largecap 1.814310e+13                10429808.0             Technology Semiconductor Equipment & Materials

## Top algo-compatible candidates before backtest

   ticker scan_algo_fit  scan_algo_compat_score scan_quality_compounder_fit  scan_quality_compounder_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                                                                                   source_types
  6869.HK          high                 11.0000                        high                         9.8459                        3.0                     665.0                    308.0        7.2767         11.8031                                                                                       historical_winner_cousin
       PL          high                 10.9333                        high                         9.3023                        4.0                     309.0                    128.0        4.7598         48.7654     horizon_emerging|industry_relative_leader|local_cluster_leader|sector_relative_leader|universe_base_midcap
     CIEN          high                 10.9000                        high                         9.8548                        5.0                     346.0                    124.0        4.3554         18.9024                                                                          horizon_emerging|local_cluster_leader
   285A.T          high                  9.7517                        high                         9.5655                        4.0                      92.0                     91.0        6.7521         56.2640 historical_winner_cousin|horizon_emerging|industry_relative_leader|local_cluster_leader|universe_base_largecap
  2383.TW          high                  9.3667                        high                         9.2453                        7.0                     651.0                    133.0        2.7037         27.6402                                                 horizon_emerging|industry_relative_leader|local_cluster_leader
   5803.T          high                  9.3333                        high                         9.2491                        7.0                     698.0                    329.0        2.8280         42.5969                          horizon_emerging|industry_relative_leader|local_cluster_leader|universe_base_largecap
     WULF          high                  9.3000                        high                         8.7416                        8.0                     750.0                    350.0        3.2474         34.7262                          horizon_emerging|industry_relative_leader|sector_relative_leader|universe_base_midcap
     SATS          high                  9.1037                        high                         8.8187                        9.0                     233.0                     93.0        2.3532         19.3759                                                                industry_relative_leader|sector_relative_leader
     CIFR          high                  9.1000                      medium                         7.7639                        9.0                     426.0                    230.0        2.7091         16.0528                                                                                           universe_base_midcap
   6857.T          high                  9.0333                      medium                         8.0656                        9.0                     726.0                    211.0        1.4515         19.4269                                                                        horizon_pullback|universe_base_largecap
000660.KS          high                  8.7987                      medium                         7.9397                        8.0                     354.0                     48.0        2.4435         17.4276                                                                                       historical_winner_cousin
272210.KS          high                  8.3467                      medium                         7.1464                        9.0                     248.0                     20.0        1.9750         37.3584                                                                  historical_winner_cousin|local_cluster_leader
   PLS.AX          high                  7.7500                      medium                         7.8026                       11.0                    1070.0                    765.0        1.7190         32.8591                                           industry_relative_leader|sector_relative_leader|universe_base_midcap
  2345.TW          high                  7.7167                      medium                         8.1736                       11.0                     786.0                    309.0        1.1462         18.9031                                                                      historical_winner_cousin|horizon_emerging
  1428.HK        medium                  7.6833                      medium                         7.2960                       11.0                     304.0                    128.0        1.6800         16.6402                                                                                           universe_base_midcap
      RIG        medium                  7.6500                      medium                         7.2710                       11.0                     359.0                    145.0        1.2633         18.4171                                                                      local_cluster_leader|universe_base_midcap
     NBIS          high                  7.6355                      medium                         7.2973                        9.0                     105.0                     37.0        2.6640         19.3815                                                                        horizon_emerging|sector_relative_leader
034020.KS        medium                  7.5833                      medium                         7.4718                       11.0                     479.0                    196.0        1.4778         12.1815                                                                                       industry_relative_leader
  2489.TW        medium                  7.5598                      medium                         7.0938                        9.0                     129.0                      8.0        1.7818         15.9675                                                                                           universe_base_midcap
     VIAV          high                  7.5563                      medium                         7.0183                        8.0                     126.0                      0.0        2.5318         17.4654                                                                          horizon_emerging|universe_base_midcap
  6443.TW        medium                  7.4833                      medium                         6.7420                       12.0                     299.0                    180.0        0.9083         16.1913                                                                                           universe_base_midcap
      CDE        medium                  7.4500                      medium                         6.6437                       15.0                     631.0                    316.0        1.2748         32.8556                                           industry_relative_leader|sector_relative_leader|universe_base_midcap
006400.KS        medium                  6.8687                      medium                         7.2887                       12.0                     266.0                     28.0        1.2490         25.0115                                                                      historical_winner_cousin|horizon_emerging
     EXAS        medium                  5.9167                      medium                         7.0958                       19.0                     904.0                    483.0        0.9417         17.6129                                                                    sector_relative_leader|universe_base_midcap
   5713.T        medium                  5.9022                         low                         5.5881                       11.0                     113.0                     11.0        1.3202         19.7864                                                                  historical_winner_cousin|local_cluster_leader

## Recommended additions

   name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct  days_top15_trend_if_added  full_buy_count_if_added                                                                                                   source_types
2383.TW          24413.1393             0.0307                   0.0           851.2915            0.0263              -0.0193                      651.0                      6.0                                                 horizon_emerging|industry_relative_leader|local_cluster_leader
 285A.T           4301.3755             0.0167                   0.0           292.8280            0.0551               0.0000                       92.0                      4.0 historical_winner_cousin|horizon_emerging|industry_relative_leader|local_cluster_leader|universe_base_largecap

## Watchlist additions

(none)

## Combo tests

          name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
2383.TW+285A.T          29509.6022             0.0477                   0.0          1171.8443            0.0812              -0.0193