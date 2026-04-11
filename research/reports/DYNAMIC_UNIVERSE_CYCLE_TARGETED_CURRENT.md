# Dynamic Universe Discovery (targeted)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, LRCX, VRT, ALB, AMAT, UI

## Seed tickers

- SNDK, LITE, WDC, BE

## Keywords

- (none)

## Discovery summary

- total discovered symbols: `117`
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
  COHR            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   TER            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  KEYS            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  CRWV            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   GEV            20.0             2                                 etf_holding|seed_peer        NaN                       NaN     NaN       NaN
  ALOT            17.5             2                               industry_top_performing        NaN                       NaN     NaN       NaN
   OSS            17.5             2                                   industry_top_growth        NaN                       NaN     NaN       NaN
  PSTG            17.5             2                                   industry_top_growth        NaN                       NaN     NaN       NaN
   UNP            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   FDX            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   JCI            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   UPS            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                          source_types
  CIEN          high                 11.0000                        4.0                     366.0                    173.0        4.3554            24.0                                             seed_peer
   STX          high                 10.5450                        5.0                     247.0                     55.0        3.2753            26.5 industry_top_growth|industry_top_performing|seed_peer
  COHR          high                 10.4590                        5.0                     530.0                     51.0        2.8029            24.0                                             seed_peer
   FIX        medium                  9.3000                        6.0                     819.0                    115.0        2.1230            12.0                                             seed_peer
   TER        medium                  8.8820                        6.0                     335.0                     48.0        2.5241            24.0                                             seed_peer
   OSS        medium                  8.6280                       10.0                     349.0                     42.0        1.3329            17.5                                   industry_top_growth
   GEV        medium                  7.8220                       10.0                     158.0                      7.0        1.3317            20.0                                 etf_holding|seed_peer
  KEYS        medium                  7.3230                       11.0                     218.0                     47.0        1.0802            24.0                                             seed_peer
   GLW        medium                  6.7855                        6.0                      51.0                      0.0        1.9363            24.0                                             seed_peer
  INTC           low                  6.7250                        8.0                      50.0                      0.0        1.3464            24.0                                             seed_peer
  MPWR           low                  5.3185                       16.0                     197.0                      0.0        1.0241            24.0                                             seed_peer
  DELL           low                  4.8060                       32.0                     270.0                     34.0        0.7121            26.5                     industry_top_performing|seed_peer
  ALOT          weak                  4.0040                       69.0                     207.0                      6.0        0.2181            17.5                               industry_top_performing
  PSTG          weak                  3.8960                      124.0                     251.0                     44.0        0.0773            17.5                                   industry_top_growth
   GWW          weak                  3.7495                       90.0                     151.0                     46.0        0.2212            12.0                                             seed_peer
   URI          weak                  3.6210                      127.0                     241.0                     19.0        0.0382            12.0                                             seed_peer
   CSX          weak                  3.4805                       67.0                     141.0                      0.0        0.3422            12.0                                             seed_peer
   FDX          weak                  3.1115                       31.0                      63.0                      0.0        0.6212            12.0                                             seed_peer
   WAB          weak                  2.5335                       57.0                      27.0                      0.0        0.4436            12.0                                             seed_peer
  SNPS          weak                  2.3760                      144.0                     112.0                      0.0       -0.1157            24.0                                             seed_peer
   MSI          weak                  2.3530                      124.0                      86.0                      0.0        0.0419            24.0                                             seed_peer
  CRWV          weak                  2.3500                       38.0                       0.0                      0.0        0.4917            24.0                                             seed_peer
   DAL          weak                  2.3175                       67.0                      35.0                      0.0        0.4213            12.0                                             seed_peer
   JCI          weak                  2.3105                       40.0                       1.0                      0.0        0.5833            12.0                                             seed_peer
   ROK          weak                  2.2000                       63.0                       0.0                      0.0        0.3746            12.0                                             seed_peer

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                          source_types
 STX          37415.6264                NaN               55.1534          1587.2722               NaN              71.1891 industry_top_growth|industry_top_performing|seed_peer
 FIX          43235.9798                NaN               55.0777          1496.9037               NaN              71.1891                                             seed_peer
 TER          42730.9677                NaN               55.1237          1496.9037               NaN              71.1891                                             seed_peer
CIEN          35247.8184                NaN               55.5490          1496.9037               NaN              71.1891                                             seed_peer
COHR          36312.3355                NaN               55.1237          1485.7024               NaN              68.3561                                             seed_peer

## Combo tests

   name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
FIX+STX          40567.4173                NaN               55.0777          1587.2722               NaN              71.1891