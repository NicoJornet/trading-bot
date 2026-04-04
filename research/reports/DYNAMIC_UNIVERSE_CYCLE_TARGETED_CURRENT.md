# Dynamic Universe Discovery (targeted)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, VRT, ALB, LRCX, RKLB, SQM

## Seed tickers

- SNDK, LITE, WDC, BE

## Keywords

- (none)

## Discovery summary

- total discovered symbols: `118`
- new symbols outside active/reserve/exclusions: `59`

## Top new candidates by discovery score

ticker  priority_score  source_count                                          source_types  marketCap  averageDailyVolume3Month  sector  industry
  DELL            26.5             3                     industry_top_performing|seed_peer        NaN                       NaN     NaN       NaN
   STX            26.5             3 industry_top_growth|industry_top_performing|seed_peer        NaN                       NaN     NaN       NaN
  CSCO            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  INTC            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   IBM            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   GLW            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   ACN            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   ADP            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  SNPS            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   MSI            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  CIEN            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   TEL            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  MPWR            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  SNOW            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  ADSK            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  KEYS            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   TER            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   GEV            20.0             2                                 etf_holding|seed_peer        NaN                       NaN     NaN       NaN
  UMAC            17.5             2                               industry_top_performing        NaN                       NaN     NaN       NaN
   OSS            17.5             2                                   industry_top_growth        NaN                       NaN     NaN       NaN
  PSTG            17.5             2                                   industry_top_growth        NaN                       NaN     NaN       NaN
   UNP            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   FDX            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   UPS            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   JCI            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                          source_types
  CIEN          high                 11.0000                        4.0                     361.0                    168.0        3.4738            24.0                                             seed_peer
   STX          high                 10.5000                        5.0                     242.0                     50.0        2.0120            26.5 industry_top_growth|industry_top_performing|seed_peer
   FIX          high                  9.3500                        6.0                     814.0                    115.0        1.7009            12.0                                             seed_peer
   TER          high                  8.9320                        6.0                     330.0                     48.0        1.6966            24.0                                             seed_peer
   OSS        medium                  8.7280                        8.0                     344.0                     42.0        1.1065            17.5                                   industry_top_growth
   GEV        medium                  7.8695                        9.0                     153.0                      7.0        1.0067            20.0                                 etf_holding|seed_peer
   GLW        medium                  6.7830                        7.0                      46.0                      0.0        1.3490            24.0                                             seed_peer
  KEYS        medium                  5.9230                       16.0                     213.0                     47.0        0.7361            24.0                                             seed_peer
  DELL           low                  5.7060                       25.0                     270.0                     34.0        0.5015            26.5                     industry_top_performing|seed_peer
  UMAC           low                  5.2005                       44.0                     181.0                    160.0        0.4641            17.5                               industry_top_performing
  MPWR           low                  4.5580                       27.0                     196.0                      0.0        0.5047            24.0                                             seed_peer
  PSTG          weak                  4.1460                      115.0                     251.0                     44.0        0.0287            17.5                                   industry_top_growth
   GWW           low                  3.8995                       94.0                     151.0                     46.0        0.1446            12.0                                             seed_peer
  INTC           low                  3.8225                       16.0                      45.0                      0.0        0.7477            24.0                                             seed_peer
   URI          weak                  3.7210                      131.0                     241.0                     19.0       -0.0516            12.0                                             seed_peer
   CSX          weak                  3.6805                       71.0                     141.0                      0.0        0.2567            12.0                                             seed_peer
   FDX          weak                  3.1115                       41.0                      63.0                      0.0        0.4520            12.0                                             seed_peer
   WCN          weak                  2.6640                      141.0                     120.0                      6.0       -0.0932            12.0                                             seed_peer
   DAL          weak                  2.6175                       65.0                      35.0                      0.0        0.2978            12.0                                             seed_peer
   WAB          weak                  2.5835                       63.0                      27.0                      0.0        0.3024            12.0                                             seed_peer
   MSI          weak                  2.5030                      117.0                      86.0                      0.0        0.0160            24.0                                             seed_peer
  SNPS          weak                  2.4260                      146.0                     112.0                      0.0       -0.1441            24.0                                             seed_peer
   JCI          weak                  2.3605                       53.0                       1.0                      0.0        0.3640            12.0                                             seed_peer
   AME          weak                  2.1500                       80.0                       0.0                      0.0        0.1865            12.0                                             seed_peer
  CSCO          weak                  2.1000                       82.0                       0.0                      0.0        0.1764            24.0                                             seed_peer

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                          source_types
 STX          37415.6264                NaN               55.1534          1587.2722               NaN              71.1891 industry_top_growth|industry_top_performing|seed_peer
 FIX          43235.9798                NaN               55.0777          1496.9037               NaN              71.1891                                             seed_peer
 TER          42730.9677                NaN               55.1237          1496.9037               NaN              71.1891                                             seed_peer
 OSS          37374.7408                NaN               55.1237          1496.9037               NaN              71.1891                                   industry_top_growth
 GEV          36510.6523                NaN               55.1237          1496.9037               NaN              71.1891                                 etf_holding|seed_peer
INTC          36510.6523                NaN               55.1237          1496.9037               NaN              71.1891                                             seed_peer
CIEN          35247.8184                NaN               55.5490          1496.9037               NaN              71.1891                                             seed_peer

## Combo tests

   name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
FIX+STX          40567.4173                NaN               55.0777          1587.2722               NaN              71.1891