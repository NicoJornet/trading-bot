# Dynamic Universe Discovery (broad)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, VRT, ALB, LRCX, RKLB, AMAT

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
   MSI            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  SNPS            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   TEL            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  CIEN            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  SNOW            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  MPWR            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  ADSK            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  KEYS            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   TER            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   GEV            33.5             4                                 etf_holding|sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   UNP            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   FDX            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   UPS            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   JCI            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   MMM            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   ITW            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   CSX            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                                             source_types
  AAOI          high                 11.0000                        3.0                     762.0                    467.0        2.9861            11.5                              industry_top_growth|industry_top_performing
  FIEE          high                 10.9667                        3.0                     922.0                    627.0        5.9387            11.5                                                  industry_top_performing
  CIEN          high                 10.9000                        4.0                     357.0                    164.0        2.9483            37.5                                             sector_top_company|seed_peer
  LASR          high                 10.8973                        3.0                     214.0                     96.0        3.2894            11.5                                                  industry_top_performing
  AMPX          high                 10.8667                        5.0                     273.0                    140.0        2.0676            17.5                                                  industry_top_performing
   FIX          high                  9.2333                        6.0                     810.0                    115.0        1.5664            25.5                                             sector_top_company|seed_peer
  POWL          high                  9.1667                        8.0                     604.0                    282.0        1.2398            17.5                              industry_top_growth|industry_top_performing
   OCC          high                  9.1333                        9.0                     345.0                    162.0        1.0270            11.5                                                  industry_top_performing
  WATT        medium                  9.0667                        9.0                     393.0                    110.0        1.1107            11.5                                                  industry_top_performing
  RFIL          high                  9.0000                       10.0                     350.0                    255.0        0.8836            17.5                                                  industry_top_performing
   STX          high                  8.8473                        6.0                     238.0                     46.0        1.6929            40.0 industry_top_growth|industry_top_performing|sector_top_company|seed_peer
   TER          high                  8.7987                        6.0                     326.0                     48.0        1.5313            37.5                                             sector_top_company|seed_peer
   OSS        medium                  8.5780                        9.0                     340.0                     42.0        0.9460            17.5                                                      industry_top_growth
  VIAV        medium                  7.6650                        6.0                     130.0                      0.0        1.7023            11.5                                                  industry_top_performing
   GEV        medium                  7.6608                       10.0                     149.0                      7.0        0.8916            33.5                                 etf_holding|sector_top_company|seed_peer
   GLW        medium                  6.6410                        7.0                      42.0                      0.0        1.1586            37.5                                             sector_top_company|seed_peer
  KEYS           low                  5.7397                       19.0                     213.0                     47.0        0.6333            37.5                                             sector_top_company|seed_peer
  DELL           low                  5.5560                       25.0                     270.0                     34.0        0.5064            40.0                     industry_top_performing|sector_top_company|seed_peer
  ULBI           low                  4.9000                       92.0                     354.0                    189.0        0.0613            17.5                                                      industry_top_growth
  EXTR          weak                  4.5667                      118.0                     741.0                    311.0       -0.1169            11.5                                                      industry_top_growth
   EME           low                  4.4000                       39.0                     271.0                      0.0        0.4505            25.5                                             sector_top_company|seed_peer
  MPWR           low                  4.3913                       36.0                     196.0                      0.0        0.3949            37.5                                             sector_top_company|seed_peer
  HLIT          weak                  4.3560                      123.0                     324.0                     84.0       -0.0554            11.5                                                      industry_top_growth
  PSTG          weak                  4.2627                      105.0                     251.0                     44.0       -0.0754            17.5                                                      industry_top_growth
   URI          weak                  3.8043                      123.0                     241.0                     19.0       -0.0487            25.5                                             sector_top_company|seed_peer

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                         source_types
FIEE          30539.0118                NaN               55.4783          1850.7399               NaN              69.8920                              industry_top_performing
AAOI          47147.4114                NaN               55.1468          1675.6433               NaN              64.2289          industry_top_growth|industry_top_performing
CIEN          39586.8970                NaN               55.5490          1496.9037               NaN              71.1891                         sector_top_company|seed_peer
 GEV          36510.6523                NaN               55.1237          1496.9037               NaN              71.1891             etf_holding|sector_top_company|seed_peer
DELL          36510.6523                NaN               55.1237          1496.9037               NaN              71.1891 industry_top_performing|sector_top_company|seed_peer
VIAV          35404.0002                NaN               55.1534          1496.9037               NaN              71.1891                              industry_top_performing
 GLW          35404.0002                NaN               55.1534          1496.9037               NaN              71.1891                         sector_top_company|seed_peer
RFIL          33100.5694                NaN               55.1754          1496.9037               NaN              71.1891                              industry_top_performing
LASR          39943.0153                NaN               55.1237          1451.8316               NaN              71.1891                              industry_top_performing
POWL          35602.9688                NaN               55.1534          1388.7678               NaN              70.6962          industry_top_growth|industry_top_performing
AMPX          33551.7254                NaN               55.1237          1207.1984               NaN              71.1891                              industry_top_performing

## Combo tests

     name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
AAOI+FIEE          46645.1307                NaN               55.1060          1961.1300               NaN              64.2569
CIEN+FIEE          26123.7667                NaN               55.9064          1850.7399               NaN              69.8920
AAOI+CIEN          40437.2777                NaN               55.5723          1675.6433               NaN              64.2289