# Dynamic Universe Discovery (broad)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, VRT, ALB, LRCX, RKLB, SQM

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
  INTC            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   IBM            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   GLW            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   ACN            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   ADP            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  SNPS            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   MSI            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  CIEN            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   TEL            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  MPWR            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  SNOW            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  ADSK            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  KEYS            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   TER            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   GEV            33.5             4                                 etf_holding|sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   UNP            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   FDX            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   UPS            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   JCI            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   CSX            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   MMM            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   ITW            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                                             source_types
  AAOI          high                 11.0000                        3.0                     766.0                    471.0        3.8131            11.5                              industry_top_growth|industry_top_performing
  CIEN          high                 10.9667                        4.0                     361.0                    168.0        3.4738            37.5                                             sector_top_company|seed_peer
  FIEE          high                 10.9333                        4.0                     926.0                    631.0        5.2010            11.5                                                  industry_top_performing
  AMPX          high                 10.9000                        5.0                     277.0                    144.0        2.4178            17.5                                                  industry_top_performing
   STX          high                 10.4167                        5.0                     242.0                     50.0        2.0120            40.0 industry_top_growth|industry_top_performing|sector_top_company|seed_peer
   FIX          high                  9.2667                        6.0                     814.0                    115.0        1.7009            25.5                                             sector_top_company|seed_peer
  POWL          high                  9.1667                        8.0                     608.0                    282.0        1.2803            17.5                              industry_top_growth|industry_top_performing
  WATT          high                  9.1333                        8.0                     397.0                    110.0        1.3249            11.5                                                  industry_top_performing
  RFIL          high                  9.0667                        9.0                     354.0                    255.0        0.8430            17.5                                                  industry_top_performing
   TER          high                  8.8320                        6.0                     330.0                     48.0        1.6966            37.5                                             sector_top_company|seed_peer
   OSS        medium                  8.5780                        8.0                     344.0                     42.0        1.1065            17.5                                                      industry_top_growth
  AEIS        medium                  8.3603                        6.0                     420.0                      3.0        1.4687            17.5                                                  industry_top_performing
  VIAV        medium                  7.7403                        6.0                     134.0                      0.0        1.8841            11.5                                                  industry_top_performing
   GEV        medium                  7.7028                        9.0                     153.0                      7.0        1.0067            33.5                                 etf_holding|sector_top_company|seed_peer
   GLW        medium                  6.6830                        7.0                      46.0                      0.0        1.3490            37.5                                             sector_top_company|seed_peer
  KEYS        medium                  5.7730                       16.0                     213.0                     47.0        0.7361            37.5                                             sector_top_company|seed_peer
  QUIK           low                  5.5983                       21.0                     283.0                     35.0        0.7690            11.5                                                  industry_top_performing
  DELL           low                  5.5227                       25.0                     270.0                     34.0        0.5015            40.0                     industry_top_performing|sector_top_company|seed_peer
  UMAC           low                  5.0672                       44.0                     181.0                    160.0        0.4641            17.5                                                  industry_top_performing
  ULBI           low                  4.8333                       90.0                     354.0                    189.0        0.1533            17.5                                                      industry_top_growth
  EXTR          weak                  4.5667                      121.0                     741.0                    311.0       -0.0533            11.5                                                      industry_top_growth
  MPWR           low                  4.3913                       27.0                     196.0                      0.0        0.5047            37.5                                             sector_top_company|seed_peer
   EME           low                  4.3333                       44.0                     271.0                      0.0        0.5148            25.5                                             sector_top_company|seed_peer
  HLIT          weak                  4.2560                      135.0                     324.0                     84.0       -0.0810            11.5                                                      industry_top_growth
  PSTG          weak                  4.1293                      115.0                     251.0                     44.0        0.0287            17.5                                                      industry_top_growth

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                                             source_types
FIEE          30539.0118                NaN               55.4783          1850.7399               NaN              69.8920                                                  industry_top_performing
AAOI          41882.4424                NaN               55.1468          1675.6433               NaN              64.2289                              industry_top_growth|industry_top_performing
 STX          37415.6264                NaN               55.1534          1587.2722               NaN              71.1891 industry_top_growth|industry_top_performing|sector_top_company|seed_peer
WATT          52554.6743                NaN               54.8462          1496.9037               NaN              71.1891                                                  industry_top_performing
 TER          42730.9677                NaN               55.1237          1496.9037               NaN              71.1891                                             sector_top_company|seed_peer
 FIX          38386.8310                NaN               55.0777          1496.9037               NaN              71.1891                                             sector_top_company|seed_peer
CIEN          35247.8184                NaN               55.5490          1496.9037               NaN              71.1891                                             sector_top_company|seed_peer
POWL          35602.9688                NaN               55.1534          1388.7678               NaN              70.6962                              industry_top_growth|industry_top_performing
AMPX          33551.7254                NaN               55.1237          1207.1984               NaN              71.1891                                                  industry_top_performing

## Combo tests

     name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
AAOI+FIEE          41129.0446                NaN               55.1060          1961.1300               NaN              64.2569
 FIEE+STX          27487.7243                NaN               55.3728          1858.7371               NaN              69.8920
 AAOI+STX          43973.3940                NaN               55.1468          1763.9564               NaN              64.2289