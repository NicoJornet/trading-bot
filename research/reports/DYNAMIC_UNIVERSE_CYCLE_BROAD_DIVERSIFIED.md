# Dynamic Universe Discovery (broad)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, LRCX, VRT, ALB, AMAT, UI

## Seed tickers

- SNDK, LITE, WDC, BE, MU, LRCX, VRT, ALB

## Keywords

- optical, networking, datacenter, power equipment, electrification, aerospace defense, uranium, industrial automation

## Discovery summary

- total discovered symbols: `1035`
- new symbols outside active/reserve/exclusions: `881`

## Top new candidates by discovery score

ticker  priority_score  source_count                                                                         source_types    marketCap  averageDailyVolume3Month  sector  industry
  CRWV         63.7140             9                         custom_screen|predefined_screen|sector_top_company|seed_peer 5.362019e+10                27291674.0     NaN       NaN
  INTC         61.8667             8 custom_screen|industry_top_performing|predefined_screen|sector_top_company|seed_peer 3.132106e+11               108771117.0     NaN       NaN
  DELL         55.2473             7               industry_top_performing|predefined_screen|sector_top_company|seed_peer 1.178262e+11                 8594582.0     NaN       NaN
  COHR         52.6686             7                                       predefined_screen|sector_top_company|seed_peer 5.765067e+10                 7314587.0     NaN       NaN
  CSCO         52.5290             7                                           custom_screen|sector_top_company|seed_peer 3.248590e+11                24001133.0     NaN       NaN
  CIEN         52.4686             7                                       predefined_screen|sector_top_company|seed_peer 7.016870e+10                 3430280.0     NaN       NaN
   STX         44.4865             6             industry_top_growth|industry_top_performing|sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   IBM         41.9865             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   GLW         41.9865             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   ACN         41.9865             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   ADP         41.9865             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  SNPS         41.9865             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   MSI         41.9865             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   TEL         41.9865             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  MPWR         41.9865             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   TER         41.9865             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  KEYS         41.9865             6                                                         sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
    AA         36.2913             4                               custom_screen|etf_holding|sector_top_company|seed_peer 1.926988e+10                 7376391.0     NaN       NaN
   CSX         34.3039             4                                           custom_screen|sector_top_company|seed_peer 7.856611e+10                13727240.0     NaN       NaN
   JCI         34.1594             4                                           custom_screen|sector_top_company|seed_peer 8.723779e+10                 4735238.0     NaN       NaN
   DAL         33.8604             4                                           custom_screen|sector_top_company|seed_peer 4.455734e+10                11731166.0     NaN       NaN
  PCAR         33.8386             4                                           custom_screen|sector_top_company|seed_peer 6.688852e+10                 3111267.0     NaN       NaN
  GOOG         33.6739             4                                         custom_screen|etf_holding|sector_top_company 3.819265e+12                21808966.0     NaN       NaN
   BAC         33.4367             4                                         custom_screen|etf_holding|sector_top_company 3.770629e+11                42204177.0     NaN       NaN
   NEE         33.3652             4                                         custom_screen|etf_holding|sector_top_company 1.961853e+11                 9824001.0     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                                                         source_types
  AAOI          high                 11.0000                        3.0                     771.0                    476.0        7.4120         26.3324          custom_screen|industry_top_growth|industry_top_performing|predefined_screen
  CIEN          high                 10.9750                        4.0                     366.0                    173.0        4.3554         52.4686                                       predefined_screen|sector_top_company|seed_peer
  AMPX          high                 10.9500                        5.0                     282.0                    149.0        3.1764         31.7536                              custom_screen|industry_top_performing|predefined_screen
   STX          high                 10.5200                        5.0                     247.0                     55.0        3.2753         44.4865             industry_top_growth|industry_top_performing|sector_top_company|seed_peer
  COHR          high                 10.4590                        5.0                     530.0                     51.0        2.8029         52.6686                                       predefined_screen|sector_top_company|seed_peer
  SATS          high                  9.3250                        6.0                     224.0                    109.0        2.3532         25.4150                                   custom_screen|predefined_screen|sector_top_company
  IREN        medium                  9.3000                        6.0                     440.0                    225.0        2.4014         22.0382                                                      custom_screen|predefined_screen
   FIX        medium                  9.2750                        6.0                     819.0                    115.0        2.1230         23.9865                                                         sector_top_company|seed_peer
    AA          high                  9.2250                        8.0                     397.0                    192.0        1.3216         36.2913                               custom_screen|etf_holding|sector_top_company|seed_peer
   TER        medium                  8.9070                        6.0                     335.0                     48.0        2.5241         41.9865                                                         sector_top_company|seed_peer
    HL        medium                  7.9000                       11.0                     609.0                    289.0        1.4214         27.4164                                           custom_screen|sector_top_company|seed_peer
   GEV        medium                  7.8970                       10.0                     158.0                      7.0        1.3317         31.9865                                             etf_holding|sector_top_company|seed_peer
   CDE        medium                  7.8500                       14.0                     634.0                    324.0        1.2748         27.4681                                           custom_screen|sector_top_company|seed_peer
  NBIS        medium                  7.8395                        6.0                     105.0                     43.0        2.6640         25.7014                                   custom_screen|predefined_screen|sector_top_company
  KEYS        medium                  7.3980                       11.0                     218.0                     47.0        1.0802         41.9865                                                         sector_top_company|seed_peer
   GLW        medium                  6.7855                        6.0                      51.0                      0.0        1.9363         41.9865                                                         sector_top_company|seed_peer
  INTC        medium                  6.7250                        8.0                      50.0                      0.0        1.3464         61.8667 custom_screen|industry_top_performing|predefined_screen|sector_top_company|seed_peer
    AU        medium                  6.3000                       17.0                     739.0                    371.0        1.1217         28.0493                                           custom_screen|sector_top_company|seed_peer
  CRDO           low                  6.2750                       18.0                     373.0                    162.0        0.8454         24.9498                                                industry_top_growth|predefined_screen
  MPWR           low                  5.3935                       16.0                     197.0                      0.0        1.0241         41.9865                                                         sector_top_company|seed_peer
   SYM           low                  5.3250                       42.0                     440.0                    249.0        0.6696         23.9865                                                         sector_top_company|seed_peer
    CF           low                  5.1200                       42.0                     284.0                     80.0        0.5422         27.8623                                           custom_screen|sector_top_company|seed_peer
  STLD           low                  4.9340                       59.0                     328.0                     76.0        0.4331         25.9865                                             etf_holding|sector_top_company|seed_peer
  DELL           low                  4.8810                       32.0                     270.0                     34.0        0.7121         55.2473               industry_top_performing|predefined_screen|sector_top_company|seed_peer
    MP           low                  4.8750                       83.0                     259.0                    124.0        0.5011         26.0681                                           custom_screen|sector_top_company|seed_peer

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                                                source_types
SATS          40280.5476                NaN               55.1534          1715.7774               NaN              71.1891                          custom_screen|predefined_screen|sector_top_company
AAOI          41882.4424                NaN               55.1468          1675.6433               NaN              64.2289 custom_screen|industry_top_growth|industry_top_performing|predefined_screen
 STX          37415.6264                NaN               55.1534          1587.2722               NaN              71.1891    industry_top_growth|industry_top_performing|sector_top_company|seed_peer
  AA          46013.8306                NaN               55.1534          1522.2026               NaN              71.1810                      custom_screen|etf_holding|sector_top_company|seed_peer
CIEN          35247.8184                NaN               55.5490          1496.9037               NaN              71.1891                              predefined_screen|sector_top_company|seed_peer
COHR          36312.3355                NaN               55.1237          1485.7024               NaN              68.3561                              predefined_screen|sector_top_company|seed_peer
NBIS          30480.4231                NaN               55.1237          1234.6998               NaN              70.5157                          custom_screen|predefined_screen|sector_top_company
AMPX          29798.1196                NaN               55.1237          1207.1984               NaN              71.1891                     custom_screen|industry_top_performing|predefined_screen
IREN          26582.6343                NaN               55.1237          1074.2290               NaN              63.9734                                             custom_screen|predefined_screen

## Combo tests

     name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
AAOI+SATS          47305.9439                NaN               55.1639          1939.3678               NaN              66.8544
 SATS+STX          42498.9186                NaN               55.1534          1815.7560               NaN              71.1891
 AAOI+STX          43973.3940                NaN               55.1468          1763.9564               NaN              64.2289
  AA+SATS          52347.3031                NaN               55.1534          1744.5503               NaN              71.1810