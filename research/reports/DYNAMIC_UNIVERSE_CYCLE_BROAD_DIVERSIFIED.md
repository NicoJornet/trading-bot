# Dynamic Universe Discovery (broad)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- LITE, WDC, BE, MU, VRT, RKLB, LRCX, ALB, UI, DNN

## Seed tickers

- SNDK, LITE, WDC, BE, MU, VRT, RKLB, LRCX

## Keywords

- optical, networking, datacenter, power equipment, electrification, aerospace defense, uranium, industrial automation

## Discovery summary

- total discovered symbols: `1035`
- new symbols outside active/reserve/exclusions: `882`

## Top new candidates by discovery score

ticker  priority_score  source_count                                                                         source_types    marketCap  averageDailyVolume3Month  sector  industry
  DELL         61.3242             8 custom_screen|industry_top_performing|predefined_screen|sector_top_company|seed_peer 1.172165e+11                 8490155.0     NaN       NaN
  INTC         58.5338             8                         custom_screen|predefined_screen|sector_top_company|seed_peer 2.205816e+11               106113873.0     NaN       NaN
   GLW         52.8903             7                                           custom_screen|sector_top_company|seed_peer 1.170494e+11                11006525.0     NaN       NaN
  CSCO         52.8005             7                                           custom_screen|sector_top_company|seed_peer 3.253726e+11                23332288.0     NaN       NaN
  CIEN         52.4005             7                                       predefined_screen|sector_top_company|seed_peer 5.589362e+10                 3310193.0     NaN       NaN
   GEV         48.4845             6                               custom_screen|etf_holding|sector_top_company|seed_peer 2.391554e+11                 2830350.0     NaN       NaN
   STX         44.7671             6             industry_top_growth|industry_top_performing|sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   IBM         42.2671             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   ACN         42.2671             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   ADP         42.2671             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  SNPS         42.2671             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   MSI         42.2671             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   TEL         42.2671             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  SNOW         42.2671             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  MPWR         42.2671             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  ADSK         42.2671             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  KEYS         42.2671             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   TER         42.2671             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   JCI         40.1242             5                                           custom_screen|sector_top_company|seed_peer 8.055403e+10                 5216041.0     NaN       NaN
   CSX         39.9865             5                                           custom_screen|sector_top_company|seed_peer 7.364774e+10                13615895.0     NaN       NaN
   DAL         39.7671             5                                           custom_screen|sector_top_company|seed_peer 4.357688e+10                10618566.0     NaN       NaN
   UPS         39.3599             5                                           custom_screen|sector_top_company|seed_peer 8.272415e+10                 6580793.0     NaN       NaN
  CTAS         38.1121             5                                           custom_screen|sector_top_company|seed_peer 6.758740e+10                 2026710.0     NaN       NaN
    VZ         33.8981             4                                         custom_screen|etf_holding|sector_top_company 2.141740e+11                31883550.0     NaN       NaN
     T         33.7498             4                                         custom_screen|etf_holding|sector_top_company 2.056649e+11                45308001.0     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                                                         source_types
  AXTI          high                 11.0000                        1.0                     674.0                    182.0       18.2420         19.9768                                                custom_screen|industry_top_performing
    PL          high                 10.9750                        3.0                     296.0                    150.0        3.6298         20.3671                                                custom_screen|industry_top_performing
  AAOI          high                 10.9500                        3.0                     761.0                    466.0        2.7765         20.1647                            custom_screen|industry_top_growth|industry_top_performing
  CIEN          high                 10.9250                        4.0                     356.0                    163.0        2.7458         52.4005                                       predefined_screen|sector_top_company|seed_peer
  AMPX          high                 10.9000                        5.0                     272.0                    139.0        2.0790         25.8111                                                custom_screen|industry_top_performing
   STX          high                 10.3890                        5.0                     237.0                     46.0        1.6903         44.7671             industry_top_growth|industry_top_performing|sector_top_company|seed_peer
   FIX          high                  9.3250                        6.0                     809.0                    115.0        1.4415         30.2671                                                         sector_top_company|seed_peer
   RIG          high                  9.2500                        7.0                     351.0                    148.0        0.9768         19.3686                                                     custom_screen|sector_top_company
   TER          high                  8.8820                        6.0                     325.0                     48.0        1.5389         42.2671                                                         sector_top_company|seed_peer
   APA        medium                  7.8510                       11.0                     281.0                     89.0        0.8615         19.3232                                                     custom_screen|sector_top_company
  NBIS        medium                  7.6845                        6.0                      95.0                     43.0        1.1637         19.2923                                                     custom_screen|sector_top_company
   WBD        medium                  7.6460                       11.0                     217.0                     69.0        0.7236         19.2570                                                     custom_screen|sector_top_company
  KEYS        medium                  7.4230                       13.0                     213.0                     47.0        0.6371         42.2671                                                         sector_top_company|seed_peer
   GLW        medium                  6.7055                        7.0                      41.0                      0.0        1.0993         52.8903                                           custom_screen|sector_top_company|seed_peer
   GEV        medium                  6.5920                       11.0                     148.0                      7.0        0.8758         48.4845                               custom_screen|etf_holding|sector_top_company|seed_peer
   CDE           low                  6.3500                       22.0                     625.0                    324.0        0.5651         23.5415                                   custom_screen|predefined_screen|sector_top_company
    PR        medium                  6.3000                       23.0                     461.0                    282.0        0.5335         19.4290                                                     custom_screen|sector_top_company
  DELL           low                  5.7810                       19.0                     270.0                     34.0        0.5286         61.3242 custom_screen|industry_top_performing|predefined_screen|sector_top_company|seed_peer
    CF           low                  5.3450                       28.0                     284.0                     80.0        0.6138         19.0643                                                     custom_screen|sector_top_company
   EQT           low                  5.1500                       69.0                     228.0                    133.0        0.2479         19.4082                                                     custom_screen|sector_top_company
  MPWR           low                  4.5580                       30.0                     196.0                      0.0        0.3709         42.2671                                                         sector_top_company|seed_peer
   EME           low                  4.5250                       34.0                     271.0                      0.0        0.4080         30.2671                                                         sector_top_company|seed_peer
   OKE          weak                  3.7990                       88.0                     180.0                      1.0        0.1283         19.2850                                                     custom_screen|sector_top_company
   GWW          weak                  3.7495                       97.0                     151.0                     46.0        0.0833         30.2671                                                         sector_top_company|seed_peer
  ODFL          weak                  3.7000                       70.0                     150.0                      0.0        0.2207         30.2671                                                         sector_top_company|seed_peer

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                                             source_types
AXTI          65910.6557                NaN               55.1763          2580.3124               NaN              71.1891                                    custom_screen|industry_top_performing
 RIG          59004.4407                NaN               50.5300          1723.2868               NaN              69.0504                                         custom_screen|sector_top_company
  PL          46237.5863                NaN               55.1237          1696.4184               NaN              71.1891                                    custom_screen|industry_top_performing
AAOI          47147.4114                NaN               55.1468          1675.6433               NaN              64.2289                custom_screen|industry_top_growth|industry_top_performing
 STX          37415.6264                NaN               55.1534          1587.2722               NaN              71.1891 industry_top_growth|industry_top_performing|sector_top_company|seed_peer
 APA          35903.5857                NaN               55.1534          1569.9934               NaN              71.1664                                         custom_screen|sector_top_company
 FIX          43235.9798                NaN               55.0777          1496.9037               NaN              71.1891                                             sector_top_company|seed_peer
 TER          42730.9677                NaN               55.1237          1496.9037               NaN              71.1891                                             sector_top_company|seed_peer
CIEN          39586.8970                NaN               55.5490          1496.9037               NaN              71.1891                           predefined_screen|sector_top_company|seed_peer
KEYS          36408.7768                NaN               55.1534          1496.9037               NaN              71.1891                                             sector_top_company|seed_peer
 WBD          35404.0002                NaN               55.1534          1496.9037               NaN              71.1891                                         custom_screen|sector_top_company
 GLW          35404.0002                NaN               55.1534          1496.9037               NaN              71.1891                               custom_screen|sector_top_company|seed_peer
NBIS          30480.4231                NaN               55.1237          1234.6998               NaN              70.5157                                         custom_screen|sector_top_company
AMPX          33551.7254                NaN               55.1237          1207.1984               NaN              71.1891                                    custom_screen|industry_top_performing

## Combo tests

     name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
 AXTI+RIG         112186.5162                NaN               50.5118          2960.5679               NaN              69.0504
AAOI+AXTI          77202.2166                NaN               55.1728          2861.1792               NaN              64.2289
  AXTI+PL          53364.7381                NaN               55.1763          2072.1559               NaN              71.1891
   PL+RIG          66362.7669                NaN               50.5300          1950.9794               NaN              69.0504