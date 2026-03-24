# Dynamic Universe Discovery (targeted)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, VRT, LRCX, RKLB, AMAT, ALB

## Seed tickers

- SNDK, LITE, WDC, BE

## Keywords

- (none)

## Discovery summary

- total discovered symbols: `117`
- new symbols outside active/reserve/exclusions: `58`

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
  SNOW            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   TEL            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  CIEN            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  MPWR            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  ADSK            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  KEYS            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  COHR            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   GEV            20.0             2                                 etf_holding|seed_peer        NaN                       NaN     NaN       NaN
   OSS            17.5             2           industry_top_growth|industry_top_performing        NaN                       NaN     NaN       NaN
  PSTG            17.5             2                                   industry_top_growth        NaN                       NaN     NaN       NaN
   UNP            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   FDX            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   UPS            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   JCI            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   MMM            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                          source_types
  CIEN          high                 11.0000                        4.0                     355.0                    165.0        3.0524            24.0                                             seed_peer
   FIX          high                  9.3500                        6.0                     807.0                    115.0        1.5916            12.0                                             seed_peer
  COHR          high                  8.9500                        6.0                     529.0                     50.0        1.7101            24.0                                             seed_peer
   STX          high                  8.9370                        6.0                     234.0                     43.0        1.8358            26.5 industry_top_growth|industry_top_performing|seed_peer
   OSS        medium                  8.7780                        6.0                     339.0                     42.0        1.1483            17.5           industry_top_growth|industry_top_performing
   GEV        medium                  7.7855                       10.0                     145.0                      7.0        0.8899            20.0                                 etf_holding|seed_peer
   GLW        medium                  6.6490                        9.0                      38.0                      0.0        1.0364            24.0                                             seed_peer
  KEYS        medium                  6.0040                       17.0                     216.0                     56.0        0.6835            24.0                                             seed_peer
  DELL           low                  4.9560                       33.0                     274.0                     34.0        0.4323            26.5                     industry_top_performing|seed_peer
  MPWR           low                  4.6000                       39.0                     220.0                      0.0        0.4084            24.0                                             seed_peer
  PSTG          weak                  4.3960                      106.0                     251.0                     44.0       -0.0067            17.5                                   industry_top_growth
   GWW          weak                  3.8495                      106.0                     151.0                     46.0        0.0671            12.0                                             seed_peer
   URI          weak                  3.7710                      131.0                     245.0                     19.0       -0.0418            12.0                                             seed_peer
   CSX          weak                  3.7305                       82.0                     141.0                      0.0        0.2046            12.0                                             seed_peer
   FDX          weak                  3.1850                       39.0                      70.0                      0.0        0.4399            12.0                                             seed_peer
  INTC          weak                  3.0830                       26.0                      46.0                      0.0        0.5725            24.0                                             seed_peer
  SNPS          weak                  2.7520                      133.0                     124.0                      0.0       -0.0776            24.0                                             seed_peer
   MSI          weak                  2.7160                      107.0                      92.0                      0.0        0.0605            24.0                                             seed_peer
   DAL          weak                  2.6675                       72.0                      35.0                      0.0        0.1765            12.0                                             seed_peer
   WAB          weak                  2.6335                       68.0                      27.0                      0.0        0.2600            12.0                                             seed_peer
   WCN          weak                  2.5140                      143.0                     120.0                      6.0       -0.1001            12.0                                             seed_peer
   JCI          weak                  2.4105                       43.0                       1.0                      0.0        0.3579            12.0                                             seed_peer
  CSCO          weak                  2.2000                       87.0                       0.0                      0.0        0.1872            24.0                                             seed_peer
   AME          weak                  2.1500                       91.0                       0.0                      0.0        0.1477            12.0                                             seed_peer
   ROK          weak                  2.1000                       92.0                       0.0                      0.0        0.1488            12.0                                             seed_peer

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                          source_types
 STX          42114.7034                NaN               55.1534          1587.2722               NaN              71.1891 industry_top_growth|industry_top_performing|seed_peer
 FIX          43235.9798                NaN               55.0777          1496.9037               NaN              71.1891                                             seed_peer
CIEN          39586.8970                NaN               55.5490          1496.9037               NaN              71.1891                                             seed_peer
COHR          40883.9018                NaN               55.1237          1485.7024               NaN              68.3561                                             seed_peer

## Combo tests

   name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
FIX+STX          45691.2422                NaN               55.0777          1587.2722               NaN              71.1891