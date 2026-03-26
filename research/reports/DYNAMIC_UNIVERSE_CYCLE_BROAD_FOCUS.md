# Dynamic Universe Discovery (broad)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, VRT, RKLB, LRCX, ALB, AMAT

## Seed tickers

- SNDK, LITE, WDC, BE, MU, VRT

## Keywords

- (none)

## Discovery summary

- total discovered symbols: `127`
- new symbols outside active/reserve/exclusions: `68`

## Top new candidates by discovery score

ticker  priority_score  source_count                                                             source_types  marketCap  averageDailyVolume3Month  sector  industry
  DELL            40.0             5                     industry_top_performing|sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   STX            40.0             5 industry_top_growth|industry_top_performing|sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  CSCO            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   IBM            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  INTC            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   ACN            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   GLW            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   ADP            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  SNPS            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   MSI            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   TEL            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  CIEN            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  SNOW            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  MPWR            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  ADSK            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  KEYS            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   TER            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   GEV            33.5             4                                 etf_holding|sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   UNP            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   UPS            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   FDX            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   JCI            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   MMM            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   ITW            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   CSX            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                                             source_types
  AAOI          high                 10.9667                        3.0                     761.0                    466.0        2.7746            11.5                              industry_top_growth|industry_top_performing
  FIEE          high                 10.9333                        3.0                     921.0                    626.0        6.2221            11.5                                                  industry_top_performing
  CIEN          high                 10.8667                        4.0                     356.0                    163.0        2.7445            37.5                                             sector_top_company|seed_peer
  LASR          high                 10.8550                        3.0                     213.0                     95.0        3.2089            11.5                                                  industry_top_performing
  AMPX          high                 10.8333                        5.0                     272.0                    139.0        2.0745            17.5                                                  industry_top_performing
  NDBI          high                 10.7060                        1.0                     172.0                    168.0      110.3346            17.5                                                  industry_top_performing
   STX          high                 10.3140                        5.0                     237.0                     46.0        1.6885            40.0 industry_top_growth|industry_top_performing|sector_top_company|seed_peer
   FIX          high                  9.2000                        6.0                     809.0                    115.0        1.4440            25.5                                             sector_top_company|seed_peer
   OCC          high                  9.1333                        7.0                     344.0                    162.0        1.1783            11.5                                                  industry_top_performing
  WATT          high                  9.1000                        7.0                     392.0                    110.0        1.3204            11.5                                                  industry_top_performing
  POWL          high                  9.0333                        7.0                     603.0                    282.0        1.1160            17.5                              industry_top_growth|industry_top_performing
   TER          high                  8.7653                        6.0                     325.0                     48.0        1.5425            37.5                                             sector_top_company|seed_peer
   OSS        medium                  8.6447                        7.0                     339.0                     42.0        1.0735            17.5                              industry_top_growth|industry_top_performing
  RFIL        medium                  7.6833                       15.0                     349.0                    255.0        0.7095            17.5                                                  industry_top_performing
  VIAV        medium                  7.6212                        6.0                     129.0                      0.0        1.7057            11.5                                                  industry_top_performing
   GEV        medium                  7.6170                       10.0                     148.0                      7.0        0.8756            33.5                                 etf_holding|sector_top_company|seed_peer
  KEYS        medium                  7.2397                       13.0                     213.0                     47.0        0.6354            37.5                                             sector_top_company|seed_peer
   GLW        medium                  6.4972                        7.0                      41.0                      0.0        1.0983            37.5                                             sector_top_company|seed_peer
  DELL           low                  5.5560                       19.0                     270.0                     34.0        0.5292            40.0                     industry_top_performing|sector_top_company|seed_peer
  ULBI           low                  4.9667                       84.0                     354.0                    189.0        0.0724            17.5                                                      industry_top_growth
  EXTR          weak                  4.5333                      121.0                     741.0                    311.0       -0.1256            11.5                                                      industry_top_growth
  MPWR           low                  4.3913                       30.0                     196.0                      0.0        0.3754            37.5                                             sector_top_company|seed_peer
   EME           low                  4.3667                       34.0                     271.0                      0.0        0.4095            25.5                                             sector_top_company|seed_peer
  HLIT          weak                  4.3227                      125.0                     324.0                     84.0       -0.0579            11.5                                                      industry_top_growth
  PSTG          weak                  4.2960                       97.0                     251.0                     44.0       -0.0459            17.5                                                      industry_top_growth

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                                             source_types
NDBI          49495.4848                NaN               55.1237          2066.5084               NaN              71.1891                                                  industry_top_performing
FIEE          30539.0118                NaN               55.4783          1850.7399               NaN              69.8920                                                  industry_top_performing
AAOI          47147.4114                NaN               55.1468          1675.6433               NaN              64.2289                              industry_top_growth|industry_top_performing
 STX          37415.6264                NaN               55.1534          1587.2722               NaN              71.1891 industry_top_growth|industry_top_performing|sector_top_company|seed_peer
 OCC          35493.9941                NaN               54.6808          1576.6466               NaN              71.1891                                                  industry_top_performing
WATT          52554.6743                NaN               54.8462          1496.9037               NaN              71.1891                                                  industry_top_performing
 TER          42730.9677                NaN               55.1237          1496.9037               NaN              71.1891                                             sector_top_company|seed_peer
CIEN          39586.8970                NaN               55.5490          1496.9037               NaN              71.1891                                             sector_top_company|seed_peer
 FIX          38386.8310                NaN               55.0777          1496.9037               NaN              71.1891                                             sector_top_company|seed_peer
LASR          39943.0153                NaN               55.1237          1451.8316               NaN              71.1891                                                  industry_top_performing
AMPX          33551.7254                NaN               55.1237          1207.1984               NaN              71.1891                                                  industry_top_performing

## Combo tests

     name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
FIEE+NDBI          35599.1475                NaN               55.4783          2472.8802               NaN              69.8920
AAOI+NDBI          56403.0193                NaN               55.1468          2293.0034               NaN              64.2289
AAOI+FIEE          46645.1307                NaN               55.1060          1961.1300               NaN              64.2569