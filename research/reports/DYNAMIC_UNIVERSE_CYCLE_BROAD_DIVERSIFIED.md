# Dynamic Universe Discovery (broad)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, VRT, ALB, LRCX, RKLB, SQM

## Seed tickers

- SNDK, LITE, WDC, BE, MU, VRT, ALB, LRCX

## Keywords

- optical, networking, datacenter, power equipment, electrification, aerospace defense, uranium, industrial automation

## Discovery summary

- total discovered symbols: `1039`
- new symbols outside active/reserve/exclusions: `887`

## Top new candidates by discovery score

ticker  priority_score  source_count                                                             source_types    marketCap  averageDailyVolume3Month  sector  industry
  INTC         59.1843             8             custom_screen|predefined_screen|sector_top_company|seed_peer 2.529585e+11               108059024.0     NaN       NaN
  CIEN         58.4413             8                           predefined_screen|sector_top_company|seed_peer 6.334168e+10                 3485809.0     NaN       NaN
  DELL         55.2844             7   industry_top_performing|predefined_screen|sector_top_company|seed_peer 1.155532e+11                 8889191.0     NaN       NaN
   GLW         52.9687             7                               custom_screen|sector_top_company|seed_peer 1.270655e+11                11557730.0     NaN       NaN
  CSCO         52.1858             7                               custom_screen|sector_top_company|seed_peer 3.122155e+11                24087758.0     NaN       NaN
   STX         44.6655             6 industry_top_growth|industry_top_performing|sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   IBM         42.1655             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   ACN         42.1655             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   ADP         42.1655             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  SNPS         42.1655             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   MSI         42.1655             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   TEL         42.1655             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  MPWR         42.1655             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  SNOW         42.1655             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  ADSK         42.1655             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  KEYS         42.1655             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   TER         42.1655             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   CRH         40.5404             5                   custom_screen|etf_holding|sector_top_company|seed_peer 6.993237e+10                 5314496.0     NaN       NaN
   ECL         34.6655             4             etf_holding|industry_top_growth|sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   CSX         34.3383             4                               custom_screen|sector_top_company|seed_peer 7.666891e+10                14118201.0     NaN       NaN
   DAL         33.9644             4                               custom_screen|sector_top_company|seed_peer 4.360301e+10                11271527.0     NaN       NaN
    VZ         33.6858             4                             custom_screen|etf_holding|sector_top_company 2.083536e+11                32071458.0     NaN       NaN
     T         33.5525             4                             custom_screen|etf_holding|sector_top_company 1.983264e+11                45896650.0     NaN       NaN
   NEE         33.4913             4                             custom_screen|etf_holding|sector_top_company 1.940801e+11                 9976151.0     NaN       NaN
   UPS         33.2599             4                               custom_screen|sector_top_company|seed_peer 8.342089e+10                 6598820.0     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                                                source_types
  AXTI          high                 11.0000                        1.0                     679.0                    187.0       17.6757         25.9317                     custom_screen|industry_top_performing|predefined_screen
  AAOI          high                 10.9750                        3.0                     766.0                    471.0        3.8131         26.1963 custom_screen|industry_top_growth|industry_top_performing|predefined_screen
  CIEN          high                 10.9500                        4.0                     361.0                    168.0        3.4738         58.4413                              predefined_screen|sector_top_company|seed_peer
  AMPX          high                 10.9250                        5.0                     277.0                    144.0        2.4178         25.7276                                       custom_screen|industry_top_performing
   STX          high                 10.4500                        5.0                     242.0                     50.0        2.0120         44.6655    industry_top_growth|industry_top_performing|sector_top_company|seed_peer
   FIX          high                  9.3500                        6.0                     814.0                    115.0        1.7009         24.1655                                                sector_top_company|seed_peer
   RIG          high                  9.3000                        8.0                     356.0                    148.0        0.9292         25.2955                          custom_screen|predefined_screen|sector_top_company
    AA          high                  9.2500                        8.0                     392.0                    192.0        1.0638         28.3210                                  custom_screen|sector_top_company|seed_peer
    HL        medium                  9.2250                        8.0                     604.0                    289.0        1.2257         27.4249                                  custom_screen|sector_top_company|seed_peer
  LUNR        medium                  9.0230                        8.0                     176.0                    107.0        1.4847         23.5265                                             custom_screen|predefined_screen
   TER          high                  8.9070                        6.0                     330.0                     48.0        1.6966         42.1655                                                sector_top_company|seed_peer
   CDE        medium                  7.9250                       11.0                     630.0                    324.0        0.9870         27.3614                                  custom_screen|sector_top_company|seed_peer
   GEV        medium                  7.8695                        9.0                     153.0                      7.0        1.0067         32.1655                                    etf_holding|sector_top_company|seed_peer
   GLW        medium                  6.8080                        7.0                      46.0                      0.0        1.3490         52.9687                                  custom_screen|sector_top_company|seed_peer
  KEYS        medium                  5.9230                       16.0                     213.0                     47.0        0.7361         42.1655                                                sector_top_company|seed_peer
  DELL           low                  5.7060                       25.0                     270.0                     34.0        0.5015         55.2844      industry_top_performing|predefined_screen|sector_top_company|seed_peer
  IRDM           low                  5.4250                       32.0                     355.0                    205.0        0.6044         24.3547                          custom_screen|predefined_screen|sector_top_company
    CF           low                  5.3450                       26.0                     284.0                     80.0        0.5908         28.0428                                  custom_screen|sector_top_company|seed_peer
    MP           low                  5.1500                       69.0                     259.0                    124.0        0.2813         25.8922                                  custom_screen|sector_top_company|seed_peer
  MPWR           low                  4.5580                       27.0                     196.0                      0.0        0.5047         42.1655                                                sector_top_company|seed_peer
   EME           low                  4.4500                       44.0                     271.0                      0.0        0.5148         24.1655                                                sector_top_company|seed_peer
   CLF          weak                  4.1500                      150.0                     574.0                    321.0       -0.1886         25.8340                                  custom_screen|sector_top_company|seed_peer
   MOS          weak                  3.9740                      137.0                     217.0                     61.0       -0.0826         26.1540                                  custom_screen|sector_top_company|seed_peer
  INTC           low                  3.8475                       16.0                      45.0                      0.0        0.7477         59.1843                custom_screen|predefined_screen|sector_top_company|seed_peer
   GWW           low                  3.7995                       94.0                     151.0                     46.0        0.1446         24.1655                                                sector_top_company|seed_peer

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                                                source_types
AXTI          58193.3283                NaN               55.1763          2580.3124               NaN              71.1891                     custom_screen|industry_top_performing|predefined_screen
 RIG          59004.4407                NaN               50.5300          1723.2868               NaN              69.0504                          custom_screen|predefined_screen|sector_top_company
AAOI          41882.4424                NaN               55.1468          1675.6433               NaN              64.2289 custom_screen|industry_top_growth|industry_top_performing|predefined_screen
 STX          37415.6264                NaN               55.1534          1587.2722               NaN              71.1891    industry_top_growth|industry_top_performing|sector_top_company|seed_peer
  AA          46013.8306                NaN               55.1534          1522.2026               NaN              71.1810                                  custom_screen|sector_top_company|seed_peer
 FIX          43235.9798                NaN               55.0777          1496.9037               NaN              71.1891                                                sector_top_company|seed_peer
KEYS          36408.7768                NaN               55.1534          1496.9037               NaN              71.1891                                                sector_top_company|seed_peer
 GLW          35404.0002                NaN               55.1534          1496.9037               NaN              71.1891                                  custom_screen|sector_top_company|seed_peer
CIEN          35247.8184                NaN               55.5490          1496.9037               NaN              71.1891                              predefined_screen|sector_top_company|seed_peer
IRDM          29608.0747                NaN               49.7136          1442.7907               NaN              71.2074                          custom_screen|predefined_screen|sector_top_company
LUNR          30254.7405                NaN               55.1237          1238.1387               NaN              57.0924                                             custom_screen|predefined_screen
AMPX          33551.7254                NaN               55.1237          1207.1984               NaN              71.1891                                       custom_screen|industry_top_performing

## Combo tests

     name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
 AXTI+RIG         112186.5162                NaN               50.5118          2960.5679               NaN              69.0504
AAOI+AXTI          68156.9491                NaN               55.1728          2861.1792               NaN              64.2289
 AXTI+STX          56833.0136                NaN               55.1763          2517.7576               NaN              71.1891
 AAOI+RIG          76004.4377                NaN               50.5289          2125.9059               NaN              64.1802