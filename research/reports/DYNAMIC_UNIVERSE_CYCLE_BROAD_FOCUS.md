# Dynamic Universe Discovery (broad)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, LRCX, VRT, ALB, AMAT, UI

## Seed tickers

- SNDK, LITE, WDC, BE, MU, LRCX

## Keywords

- (none)

## Discovery summary

- total discovered symbols: `134`
- new symbols outside active/reserve/exclusions: `76`

## Top new candidates by discovery score

ticker  priority_score  source_count                                                             source_types  marketCap  averageDailyVolume3Month  sector  industry
  INTC            46.0             6                     industry_top_performing|sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  DELL            46.0             6                     industry_top_performing|sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   STX            46.0             6 industry_top_growth|industry_top_performing|sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  CSCO            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   IBM            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   GLW            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   ACN            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   ADP            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  SNPS            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   MSI            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  CIEN            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   TEL            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  MPWR            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  COHR            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   TER            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  KEYS            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
  CRWV            43.5             6                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   GEV            27.5             3                                 etf_holding|sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   UNP            19.5             2                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   FDX            19.5             2                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   JCI            19.5             2                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   UPS            19.5             2                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   MMM            19.5             2                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   CSX            19.5             2                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN
   ITW            19.5             2                                             sector_top_company|seed_peer        NaN                       NaN     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                                             source_types
  AAOI          high                 11.0000                        3.0                     771.0                    476.0        7.4120            11.5                              industry_top_growth|industry_top_performing
  CIEN          high                 10.9667                        4.0                     366.0                    173.0        4.3554            43.5                                             sector_top_company|seed_peer
  FIEE          high                 10.9333                        4.0                     931.0                    636.0        3.8519            11.5                                                  industry_top_performing
  AMPX          high                 10.9000                        5.0                     282.0                    149.0        3.1764            11.5                                                  industry_top_performing
   STX          high                 10.4617                        5.0                     247.0                     55.0        3.2753            46.0 industry_top_growth|industry_top_performing|sector_top_company|seed_peer
  COHR          high                 10.3923                        5.0                     530.0                     51.0        2.8029            43.5                                             sector_top_company|seed_peer
  POWL          high                  9.2667                        6.0                     613.0                    282.0        1.9559            11.5                              industry_top_growth|industry_top_performing
   FIX        medium                  9.2333                        6.0                     819.0                    115.0        2.1230            19.5                                             sector_top_company|seed_peer
   OCC        medium                  9.1667                        6.0                     354.0                    162.0        1.6927            11.5                                                  industry_top_performing
  WATT          high                  9.1333                        8.0                     402.0                    110.0        1.5894            11.5                                                  industry_top_performing
  RFIL          high                  9.0667                        9.0                     359.0                    255.0        1.2789            11.5                                                  industry_top_performing
   TER        medium                  8.8320                        6.0                     335.0                     48.0        2.5241            43.5                                             sector_top_company|seed_peer
   OSS        medium                  8.5113                       10.0                     349.0                     42.0        1.3329            17.5                                                      industry_top_growth
   GEV        medium                  7.7220                       10.0                     158.0                      7.0        1.3317            27.5                                 etf_holding|sector_top_company|seed_peer
  KEYS        medium                  7.2397                       11.0                     218.0                     47.0        1.0802            43.5                                             sector_top_company|seed_peer
  QUIK        medium                  7.0983                       14.0                     285.0                     35.0        0.9809            11.5                                                  industry_top_performing
   GLW        medium                  6.7355                        6.0                      51.0                      0.0        1.9363            43.5                                             sector_top_company|seed_peer
  INTC           low                  6.6250                        8.0                      50.0                      0.0        1.3464            46.0                     industry_top_performing|sector_top_company|seed_peer
  MPWR           low                  5.2185                       16.0                     197.0                      0.0        1.0241            43.5                                             sector_top_company|seed_peer
   SYM           low                  5.1667                       42.0                     440.0                    249.0        0.6696            19.5                                             sector_top_company|seed_peer
  ULBI           low                  4.7667                       81.0                     354.0                    189.0        0.2459            11.5                                                      industry_top_growth
  DELL           low                  4.7060                       32.0                     270.0                     34.0        0.7121            46.0                     industry_top_performing|sector_top_company|seed_peer
  EXTR          weak                  4.7000                       91.0                     741.0                    311.0        0.1981            11.5                                                      industry_top_growth
   EME           low                  4.3667                       38.0                     271.0                      0.0        0.6233            19.5                                             sector_top_company|seed_peer
  HLIT          weak                  4.3227                      124.0                     324.0                     84.0        0.0527            11.5                                                      industry_top_growth

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                                             source_types
FIEE          30539.0118                NaN               55.4783          1850.7399               NaN              69.8920                                                  industry_top_performing
AAOI          41882.4424                NaN               55.1468          1675.6433               NaN              64.2289                              industry_top_growth|industry_top_performing
 STX          37415.6264                NaN               55.1534          1587.2722               NaN              71.1891 industry_top_growth|industry_top_performing|sector_top_company|seed_peer
CIEN          35247.8184                NaN               55.5490          1496.9037               NaN              71.1891                                             sector_top_company|seed_peer
RFIL          33100.5694                NaN               55.1754          1496.9037               NaN              71.1891                                                  industry_top_performing
COHR          36312.3355                NaN               55.1237          1485.7024               NaN              68.3561                                             sector_top_company|seed_peer
AMPX          29798.1196                NaN               55.1237          1207.1984               NaN              71.1891                                                  industry_top_performing

## Combo tests

     name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
AAOI+FIEE          41129.0446                NaN               55.1060          1961.1300               NaN              64.2569
 FIEE+STX          27487.7243                NaN               55.3728          1858.7371               NaN              69.8920
 AAOI+STX          43973.3940                NaN               55.1468          1763.9564               NaN              64.2289