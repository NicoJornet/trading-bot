# Dynamic Universe Discovery (broad)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, VRT, LRCX, RKLB, AMAT, ALB

## Seed tickers

- SNDK, LITE, WDC, BE, MU, VRT, LRCX, RKLB

## Keywords

- optical, networking, datacenter, power equipment, electrification, aerospace defense, uranium, industrial automation

## Discovery summary

- total discovered symbols: `1033`
- new symbols outside active/reserve/exclusions: `881`

## Top new candidates by discovery score

ticker  priority_score  source_count                                                                         source_types    marketCap  averageDailyVolume3Month  sector  industry
  DELL         61.2493             8 custom_screen|industry_top_performing|predefined_screen|sector_top_company|seed_peer 1.090721e+11                 8219232.0     NaN       NaN
  INTC         58.5484             8                         custom_screen|predefined_screen|sector_top_company|seed_peer 2.198378e+11               105641771.0     NaN       NaN
   GLW         52.8712             7                                           custom_screen|sector_top_company|seed_peer 1.123655e+11                10647549.0     NaN       NaN
  CSCO         52.6713             7                                           custom_screen|sector_top_company|seed_peer 3.114253e+11                23518152.0     NaN       NaN
  CIEN         52.4259             7                                       predefined_screen|sector_top_company|seed_peer 5.770294e+10                 3356249.0     NaN       NaN
   GEV         48.5160             6                               custom_screen|etf_holding|sector_top_company|seed_peer 2.394783e+11                 2802350.0     NaN       NaN
   STX         44.7783             6             industry_top_growth|industry_top_performing|sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   IBM         42.2783             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   ACN         42.2783             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   ADP         42.2783             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  SNPS         42.2783             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   MSI         42.2783             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  SNOW         42.2783             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   TEL         42.2783             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  MPWR         42.2783             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  ADSK         42.2783             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  KEYS         42.2783             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  COHR         42.2783             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   CSX         39.9753             5                                           custom_screen|sector_top_company|seed_peer 7.241114e+10                13589393.0     NaN       NaN
   DAL         39.6626             5                                           custom_screen|sector_top_company|seed_peer 4.253840e+10                10374142.0     NaN       NaN
   UPS         39.3601             5                                           custom_screen|sector_top_company|seed_peer 8.298755e+10                 6612171.0     NaN       NaN
   MMM         38.6162             5                                           custom_screen|sector_top_company|seed_peer 7.719315e+10                 4172930.0     NaN       NaN
    VZ         33.9439             4                                         custom_screen|etf_holding|sector_top_company 2.133305e+11                31910359.0     NaN       NaN
     T         33.7590             4                                         custom_screen|etf_holding|sector_top_company 2.038926e+11                45032830.0     NaN       NaN
   NEE         33.5421             4                                         custom_screen|etf_holding|sector_top_company 1.879962e+11                 9708245.0     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                                                         source_types
  AXTI          high                 11.0000                        1.0                     677.0                    179.0       20.8451         25.9506                              custom_screen|industry_top_performing|predefined_screen
    PL          high                 10.9750                        3.0                     293.0                    147.0        3.7949         20.3940                                                custom_screen|industry_top_performing
  ONDS          high                 10.9500                        3.0                     248.0                    175.0        6.2471         23.5760                                                      custom_screen|predefined_screen
  AAOI          high                 10.9250                        3.0                     759.0                    463.0        2.7931         26.1191          custom_screen|industry_top_growth|industry_top_performing|predefined_screen
  CIEN          high                 10.9000                        4.0                     355.0                    165.0        3.0524         52.4259                                       predefined_screen|sector_top_company|seed_peer
  AMPX          high                 10.8750                        5.0                     269.0                    136.0        3.2778         25.8432                                                custom_screen|industry_top_performing
   FIX          high                  9.2750                        6.0                     807.0                    115.0        1.5916         30.2783                                                         sector_top_company|seed_peer
  COHR          high                  8.8500                        6.0                     529.0                     50.0        1.7101         42.2783                                                         sector_top_company|seed_peer
   STX          high                  8.8370                        6.0                     234.0                     43.0        1.8358         44.7783             industry_top_growth|industry_top_performing|sector_top_company|seed_peer
   FTI        medium                  7.9500                       11.0                     285.0                    119.0        0.9313         19.1496                                                     custom_screen|sector_top_company
   RIG        medium                  7.9250                       11.0                     348.0                    148.0        0.9054         25.3306                                   custom_screen|predefined_screen|sector_top_company
   CDE        medium                  7.9000                       12.0                     632.0                    324.0        0.7344         24.2594                                   custom_screen|predefined_screen|sector_top_company
   GEV        medium                  7.8105                       10.0                     145.0                      7.0        0.8899         48.5160                               custom_screen|etf_holding|sector_top_company|seed_peer
  NBIS        medium                  7.6780                        6.0                      92.0                     43.0        1.4965         19.4918                                                     custom_screen|sector_top_company
   WBD        medium                  7.5960                       14.0                     217.0                     69.0        0.7946         19.3441                                                     custom_screen|sector_top_company
   GLW        medium                  6.6490                        9.0                      38.0                      0.0        1.0364         52.8712                                           custom_screen|sector_top_company|seed_peer
   APA        medium                  6.2260                       17.0                     280.0                     89.0        0.7517         19.2546                                                     custom_screen|sector_top_company
  KEYS        medium                  5.9540                       17.0                     216.0                     56.0        0.6835         42.2783                                                         sector_top_company|seed_peer
    PR           low                  5.4250                       39.0                     461.0                    282.0        0.4610         19.3621                                                     custom_screen|sector_top_company
   OVV           low                  5.3250                       46.0                     499.0                    145.0        0.3998         19.0716                                                     custom_screen|sector_top_company
   EQT           low                  5.1250                       77.0                     236.0                    133.0        0.2587         19.3582                                                     custom_screen|sector_top_company
  DELL           low                  4.8810                       33.0                     274.0                     34.0        0.4323         61.2493 custom_screen|industry_top_performing|predefined_screen|sector_top_company|seed_peer
   EME           low                  4.6000                       31.0                     271.0                      0.0        0.4599         30.2783                                                         sector_top_company|seed_peer
  MPWR           low                  4.5500                       39.0                     220.0                      0.0        0.4084         42.2783                                                         sector_top_company|seed_peer
    NU          weak                  3.8155                       95.0                     181.0                     10.0        0.0587         24.1810                                   custom_screen|predefined_screen|sector_top_company

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                                                source_types
AXTI          65910.6557                NaN               55.1763          2580.3124               NaN              71.1891                     custom_screen|industry_top_performing|predefined_screen
  PL          46237.5863                NaN               55.1237          1696.4184               NaN              71.1891                                       custom_screen|industry_top_performing
AAOI          47147.4114                NaN               55.1468          1675.6433               NaN              64.2289 custom_screen|industry_top_growth|industry_top_performing|predefined_screen
 STX          42114.7034                NaN               55.1534          1587.2722               NaN              71.1891    industry_top_growth|industry_top_performing|sector_top_company|seed_peer
ONDS          42624.8625                NaN               55.1237          1558.5751               NaN              71.1891                                             custom_screen|predefined_screen
 FIX          43235.9798                NaN               55.0777          1496.9037               NaN              71.1891                                                sector_top_company|seed_peer
CIEN          39586.8970                NaN               55.5490          1496.9037               NaN              71.1891                              predefined_screen|sector_top_company|seed_peer
AMPX          33551.7254                NaN               55.1237          1207.1984               NaN              71.1891                                       custom_screen|industry_top_performing

## Combo tests

     name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
AAOI+AXTI          77202.2166                NaN               55.1728          2861.1792               NaN              64.2289
 AXTI+STX          64370.2842                NaN               55.1763          2517.7576               NaN              71.1891
  AXTI+PL          53364.7381                NaN               55.1763          2072.1559               NaN              71.1891
  AAOI+PL          53462.4711                NaN               55.1468          1913.5618               NaN              66.8544