# Dynamic Universe Discovery (targeted)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, VRT, ALB, LRCX, RKLB, AMAT

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
   IBM            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  INTC            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   ACN            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   GLW            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   ADP            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   MSI            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  SNPS            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   TEL            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  CIEN            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  SNOW            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  MPWR            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  ADSK            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
  KEYS            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   TER            24.0             3                                             seed_peer        NaN                       NaN     NaN       NaN
   GEV            20.0             2                                 etf_holding|seed_peer        NaN                       NaN     NaN       NaN
  ALOT            17.5             2                               industry_top_performing        NaN                       NaN     NaN       NaN
   OSS            17.5             2                                   industry_top_growth        NaN                       NaN     NaN       NaN
  PSTG            17.5             2                                   industry_top_growth        NaN                       NaN     NaN       NaN
   UNP            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   FDX            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   UPS            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN
   JCI            12.0             1                                             seed_peer        NaN                       NaN     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                          source_types
  CIEN          high                 11.0000                        4.0                     357.0                    164.0        2.9483            24.0                                             seed_peer
   FIX          high                  9.3500                        6.0                     810.0                    115.0        1.5664            12.0                                             seed_peer
   STX          high                  8.9640                        6.0                     238.0                     46.0        1.6929            26.5 industry_top_growth|industry_top_performing|seed_peer
   TER          high                  8.9320                        6.0                     326.0                     48.0        1.5313            24.0                                             seed_peer
   OSS        medium                  8.7280                        9.0                     340.0                     42.0        0.9460            17.5                                   industry_top_growth
   GEV        medium                  7.8275                       10.0                     149.0                      7.0        0.8916            20.0                                 etf_holding|seed_peer
   GLW        medium                  6.7410                        7.0                      42.0                      0.0        1.1586            24.0                                             seed_peer
  KEYS        medium                  5.9230                       19.0                     213.0                     47.0        0.6333            24.0                                             seed_peer
  DELL           low                  5.7560                       25.0                     270.0                     34.0        0.5064            26.5                     industry_top_performing|seed_peer
  MPWR           low                  4.5580                       36.0                     196.0                      0.0        0.3949            24.0                                             seed_peer
  PSTG          weak                  4.3460                      105.0                     251.0                     44.0       -0.0754            17.5                                   industry_top_growth
   GWW           low                  3.8995                      103.0                     151.0                     46.0        0.0773            12.0                                             seed_peer
   URI          weak                  3.8210                      123.0                     241.0                     19.0       -0.0487            12.0                                             seed_peer
  ALOT          weak                  3.8040                      118.0                     207.0                      6.0        0.0475            17.5                               industry_top_performing
   CSX           low                  3.6805                       78.0                     141.0                      0.0        0.2162            12.0                                             seed_peer
   FDX          weak                  3.1115                       48.0                      63.0                      0.0        0.3791            12.0                                             seed_peer
  INTC          weak                  3.0225                       31.0                      45.0                      0.0        0.4827            24.0                                             seed_peer
   DAL          weak                  2.7175                       65.0                      35.0                      0.0        0.1792            12.0                                             seed_peer
   WCN          weak                  2.6140                      146.0                     120.0                      6.0       -0.1348            12.0                                             seed_peer
   MSI          weak                  2.6030                      114.0                      86.0                      0.0        0.0241            24.0                                             seed_peer
   WAB          weak                  2.5835                       68.0                      27.0                      0.0        0.2291            12.0                                             seed_peer
  SNPS          weak                  2.4260                      148.0                     112.0                      0.0       -0.1894            24.0                                             seed_peer
   JCI          weak                  2.4105                       52.0                       1.0                      0.0        0.3474            12.0                                             seed_peer
  CSCO          weak                  2.2500                       75.0                       0.0                      0.0        0.1933            24.0                                             seed_peer
   TEL          weak                  2.1500                       86.0                       0.0                      0.0        0.0913            24.0                                             seed_peer

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                          source_types
 STX          37415.6264                NaN               55.1534          1587.2722               NaN              71.1891 industry_top_growth|industry_top_performing|seed_peer
 FIX          43235.9798                NaN               55.0777          1496.9037               NaN              71.1891                                             seed_peer
 TER          42730.9677                NaN               55.1237          1496.9037               NaN              71.1891                                             seed_peer
CIEN          39586.8970                NaN               55.5490          1496.9037               NaN              71.1891                                             seed_peer
DELL          36510.6523                NaN               55.1237          1496.9037               NaN              71.1891                     industry_top_performing|seed_peer
CSCO          36510.6523                NaN               55.1237          1496.9037               NaN              71.1891                                             seed_peer
KEYS          36408.7768                NaN               55.1534          1496.9037               NaN              71.1891                                             seed_peer
 GLW          35404.0002                NaN               55.1534          1496.9037               NaN              71.1891                                             seed_peer

## Combo tests

   name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
FIX+STX          40567.4173                NaN               55.0777          1587.2722               NaN              71.1891