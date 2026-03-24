# Dynamic Universe Discovery (broad)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, VRT, LRCX, RKLB, AMAT, ALB

## Seed tickers

- SNDK, LITE, WDC, BE, MU, VRT

## Keywords

- (none)

## Discovery summary

- total discovered symbols: `125`
- new symbols outside active/reserve/exclusions: `66`

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
  SNOW            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   TEL            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  CIEN            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  MPWR            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  ADSK            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  KEYS            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  COHR            37.5             5                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   GEV            33.5             4                                 etf_holding|sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   UNP            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   FDX            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   UPS            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   JCI            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   MMM            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   ITW            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  CTAS            25.5             3                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                                             source_types
  FIEE          high                 11.0000                        3.0                     930.0                    626.0        6.5677            11.5                                                  industry_top_performing
  AAOI          high                 10.9333                        3.0                     759.0                    463.0        2.7931            11.5                              industry_top_growth|industry_top_performing
  CIEN          high                 10.9000                        4.0                     355.0                    165.0        3.0524            37.5                                             sector_top_company|seed_peer
  LASR          high                 10.8947                        3.0                     218.0                     92.0        3.4553            11.5                                                  industry_top_performing
  AMPX          high                 10.8667                        5.0                     269.0                    136.0        3.2778            17.5                                                  industry_top_performing
   FIX          high                  9.2667                        6.0                     807.0                    115.0        1.5916            25.5                                             sector_top_company|seed_peer
  WATT        medium                  9.2000                        6.0                     393.0                    110.0        1.7303            11.5                                                  industry_top_performing
  POWL          high                  9.1000                        7.0                     603.0                    282.0        1.2568            17.5                              industry_top_growth|industry_top_performing
  COHR          high                  8.8500                        6.0                     529.0                     50.0        1.7101            37.5                                             sector_top_company|seed_peer
   STX          high                  8.8203                        6.0                     234.0                     43.0        1.8358            40.0 industry_top_growth|industry_top_performing|sector_top_company|seed_peer
   OSS        medium                  8.6113                        6.0                     339.0                     42.0        1.1483            17.5                              industry_top_growth|industry_top_performing
  SKYT        medium                  7.9253                        6.0                     138.0                     27.0        1.7089            11.5                                                  industry_top_performing
   GEV        medium                  7.6188                       10.0                     145.0                      7.0        0.8899            33.5                                 etf_holding|sector_top_company|seed_peer
  VIAV        medium                  7.5212                        6.0                     129.0                      0.0        1.6681            11.5                                                  industry_top_performing
   GLW        medium                  6.4657                        9.0                      38.0                      0.0        1.0364            37.5                                             sector_top_company|seed_peer
  RFIL        medium                  6.2500                       16.0                     347.0                    255.0        0.7994            17.5                                                  industry_top_performing
  KEYS           low                  5.8207                       17.0                     216.0                     56.0        0.6835            37.5                                             sector_top_company|seed_peer
  ULBI           low                  4.8333                      103.0                     356.0                    189.0        0.0725            17.5                                                      industry_top_growth
  DELL           low                  4.7393                       33.0                     274.0                     34.0        0.4323            40.0                     industry_top_performing|sector_top_company|seed_peer
   EME           low                  4.4667                       31.0                     271.0                      0.0        0.4599            25.5                                             sector_top_company|seed_peer
  MPWR           low                  4.4000                       39.0                     220.0                      0.0        0.4084            37.5                                             sector_top_company|seed_peer
  HLIT          weak                  4.3433                      133.0                     327.0                     90.0       -0.0965            11.5                                                      industry_top_growth
  EXTR          weak                  4.3000                      138.0                     745.0                    311.0       -0.1428            11.5                                                      industry_top_growth
  PSTG          weak                  4.2960                      106.0                     251.0                     44.0       -0.0067            17.5                                                      industry_top_growth
  EOSE          weak                  4.1333                      153.0                     463.0                    298.0       -0.1930            17.5                                                      industry_top_growth

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                source_types
FIEE          30539.0118                NaN               55.4783          1850.7399               NaN              69.8920                     industry_top_performing
AAOI          47147.4114                NaN               55.1468          1675.6433               NaN              64.2289 industry_top_growth|industry_top_performing
CIEN          39586.8970                NaN               55.5490          1496.9037               NaN              71.1891                sector_top_company|seed_peer
LASR          39943.0153                NaN               55.1237          1451.8316               NaN              71.1891                     industry_top_performing
POWL          40076.9056                NaN               55.1534          1388.7678               NaN              70.6962 industry_top_growth|industry_top_performing
AMPX          33551.7254                NaN               55.1237          1207.1984               NaN              71.1891                     industry_top_performing

## Combo tests

     name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
AAOI+FIEE          46645.1307                NaN               55.1060          1961.1300               NaN              64.2569
CIEN+FIEE          29455.1757                NaN               55.9064          1850.7399               NaN              69.8920
AAOI+CIEN          45407.6369                NaN               55.5723          1675.6433               NaN              64.2289