# Dynamic Universe Discovery (targeted)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, VRT, RKLB, LRCX, ALB, AMAT

## Seed tickers

- SNDK, WDC, BE, MU

## Keywords

- (none)

## Discovery summary

- total discovered symbols: `119`
- new symbols outside active/reserve/exclusions: `60`

## Top new candidates by discovery score

ticker  priority_score  source_count                                          source_types  marketCap  averageDailyVolume3Month  sector  industry
  DELL            26.5             3                     industry_top_performing|seed_peer        NaN                       NaN     NaN       NaN
   STX            26.5             3 industry_top_growth|industry_top_performing|seed_peer        NaN                       NaN     NaN       NaN
  CSCO            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   IBM            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  INTC            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   ACN            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   GLW            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   ADP            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  SNPS            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   MSI            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   TEL            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  CIEN            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  SNOW            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  MPWR            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  ADSK            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  KEYS            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   TER            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   GEV            20.0             2                                 etf_holding|seed_peer        NaN                       NaN     NaN       NaN
   OSS            17.5             2           industry_top_growth|industry_top_performing        NaN                       NaN     NaN       NaN
  PSTG            17.5             2                                   industry_top_growth        NaN                       NaN     NaN       NaN
   UNP            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   UPS            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   FDX            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   JCI            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   MMM            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                          source_types
  CIEN          high                 11.0000                        4.0                     356.0                    163.0        2.7513            24.0                                             seed_peer
   STX          high                 10.4640                        5.0                     237.0                     46.0        1.6988            26.5 industry_top_growth|industry_top_performing|seed_peer
   FIX          high                  9.3500                        6.0                     809.0                    115.0        1.4486            12.0                                             seed_peer
   TER          high                  8.9320                        6.0                     325.0                     48.0        1.5506            24.0                                             seed_peer
   OSS        medium                  8.7780                        7.0                     339.0                     42.0        1.0747            17.5           industry_top_growth|industry_top_performing
   GEV        medium                  7.8170                       10.0                     148.0                      7.0        0.8798            20.0                                 etf_holding|seed_peer
  KEYS        medium                  7.4230                       13.0                     213.0                     47.0        0.6414            24.0                                             seed_peer
   GLW        medium                  6.6805                        7.0                      41.0                      0.0        1.1017            24.0                                             seed_peer
  DELL           low                  5.7560                       19.0                     270.0                     34.0        0.5311            26.5                     industry_top_performing|seed_peer
  MPWR           low                  4.6080                       29.0                     196.0                      0.0        0.3799            24.0                                             seed_peer
  PSTG          weak                  4.3960                       97.0                     251.0                     44.0       -0.0450            17.5                                   industry_top_growth
   GWW          weak                  3.8495                       97.0                     151.0                     46.0        0.0849            12.0                                             seed_peer
   URI          weak                  3.8210                      125.0                     241.0                     19.0       -0.0469            12.0                                             seed_peer
   CSX           low                  3.6805                       75.0                     141.0                      0.0        0.2235            12.0                                             seed_peer
   FDX          weak                  3.1115                       38.0                      63.0                      0.0        0.4185            12.0                                             seed_peer
  INTC          weak                  2.9725                       32.0                      45.0                      0.0        0.5362            24.0                                             seed_peer
   DAL          weak                  2.7175                       60.0                      35.0                      0.0        0.2034            12.0                                             seed_peer
   MSI          weak                  2.6530                      116.0                      86.0                      0.0        0.0528            24.0                                             seed_peer
   WAB          weak                  2.5835                       65.0                      27.0                      0.0        0.2555            12.0                                             seed_peer
   WCN          weak                  2.5640                      149.0                     120.0                      6.0       -0.1326            12.0                                             seed_peer
  SNPS          weak                  2.4760                      142.0                     112.0                      0.0       -0.1315            24.0                                             seed_peer
   JCI          weak                  2.4105                       52.0                       1.0                      0.0        0.3408            12.0                                             seed_peer
  CSCO          weak                  2.2500                       66.0                       0.0                      0.0        0.2434            24.0                                             seed_peer
   ROK          weak                  2.1500                       84.0                       0.0                      0.0        0.1202            12.0                                             seed_peer
   AME          weak                  2.1000                       85.0                       0.0                      0.0        0.1316            12.0                                             seed_peer

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                          source_types
 STX          37415.6264                NaN               55.1534          1587.2722               NaN              71.1891 industry_top_growth|industry_top_performing|seed_peer
 FIX          43235.9798                NaN               55.0777          1496.9037               NaN              71.1891                                             seed_peer
 TER          42730.9677                NaN               55.1237          1496.9037               NaN              71.1891                                             seed_peer
CIEN          39586.8970                NaN               55.5490          1496.9037               NaN              71.1891                                             seed_peer
 OSS          37374.7408                NaN               55.1237          1496.9037               NaN              71.1891           industry_top_growth|industry_top_performing
 GEV          36510.6523                NaN               55.1237          1496.9037               NaN              71.1891                                 etf_holding|seed_peer

## Combo tests

   name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
FIX+STX          40567.4173                NaN               55.0777          1587.2722               NaN              71.1891