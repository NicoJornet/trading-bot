# Dynamic Universe Discovery (broad)

## Purpose

- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe
- combine broad discovery with targeted peer expansion around current leaders
- score candidates with the current APEX baseline through single-addition backtests and small combo tests

## Current raw leaders

- SNDK, LITE, WDC, BE, MU, VRT, ALB, LRCX, RKLB, AMAT

## Seed tickers

- SNDK, LITE, WDC, BE, MU, VRT, ALB, LRCX

## Keywords

- optical, networking, datacenter, power equipment, electrification, aerospace defense, uranium, industrial automation

## Discovery summary

- total discovered symbols: `1015`
- new symbols outside active/reserve/exclusions: `861`

## Top new candidates by discovery score

ticker  priority_score  source_count                                                             source_types    marketCap  averageDailyVolume3Month  sector  industry
  INTC         58.5261             8             custom_screen|predefined_screen|sector_top_company|seed_peer 2.165562e+11               104905003.0     NaN       NaN
  DELL         55.2818             7   industry_top_performing|predefined_screen|sector_top_company|seed_peer 1.138567e+11                 8067052.0     NaN       NaN
  CSCO         52.7256             7                               custom_screen|sector_top_company|seed_peer 3.157715e+11                23332288.0     NaN       NaN
  CIEN         52.3749             7                           predefined_screen|sector_top_company|seed_peer 5.681314e+10                 3310193.0     NaN       NaN
   STX         44.8187             6 industry_top_growth|industry_top_performing|sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   GEV         42.4374             5                   custom_screen|etf_holding|sector_top_company|seed_peer 2.314797e+11                 2764322.0     NaN       NaN
   IBM         42.3187             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   ACN         42.3187             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   GLW         42.3187             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   ADP         42.3187             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   MSI         42.3187             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  SNPS         42.3187             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   TEL         42.3187             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  SNOW         42.3187             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  MPWR         42.3187             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  ADSK         42.3187             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
  KEYS         42.3187             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   TER         42.3187             6                                             sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   CRH         40.5759             5                   custom_screen|etf_holding|sector_top_company|seed_peer 6.803684e+10                 5152764.0     NaN       NaN
   ECL         34.8187             4             etf_holding|industry_top_growth|sector_top_company|seed_peer          NaN                       NaN     NaN       NaN
   CSX         34.3355             4                               custom_screen|sector_top_company|seed_peer 7.376862e+10                13446164.0     NaN       NaN
   JCI         34.1458             4                               custom_screen|sector_top_company|seed_peer 8.035816e+10                 5215154.0     NaN       NaN
    VZ         33.8995             4                             custom_screen|etf_holding|sector_top_company 2.121917e+11                31597718.0     NaN       NaN
     T         33.7867             4                             custom_screen|etf_holding|sector_top_company 2.037168e+11                44477611.0     NaN       NaN
   DAL         33.6453             4                               custom_screen|sector_top_company|seed_peer 4.234247e+10                10096813.0     NaN       NaN

## Top algo-compatible candidates before backtest

ticker scan_algo_fit  scan_algo_compat_score  scan_latest_rank_if_added  scan_days_top15_if_added  scan_days_top5_if_added  recent_score  priority_score                                                             source_types
  AAOI          high                 11.0000                        3.0                     762.0                    467.0        2.9861         20.1276                custom_screen|industry_top_growth|industry_top_performing
  CIEN          high                 10.9750                        4.0                     357.0                    164.0        2.9483         52.3749                           predefined_screen|sector_top_company|seed_peer
  AMPX          high                 10.9500                        5.0                     273.0                    140.0        2.0676         25.7601                                    custom_screen|industry_top_performing
   FIX          high                  9.3750                        6.0                     810.0                    115.0        1.5664         24.3187                                             sector_top_company|seed_peer
   RIG          high                  9.3500                        7.0                     352.0                    148.0        1.0837         25.3680                       custom_screen|predefined_screen|sector_top_company
    HL          high                  9.3000                        9.0                     600.0                    289.0        1.0480         27.3714                               custom_screen|sector_top_company|seed_peer
   STX          high                  8.9390                        6.0                     238.0                     46.0        1.6929         44.8187 industry_top_growth|industry_top_performing|sector_top_company|seed_peer
   TER          high                  8.9320                        6.0                     326.0                     48.0        1.5313         42.3187                                             sector_top_company|seed_peer
   CDE        medium                  8.0000                       11.0                     626.0                    324.0        0.6755         33.1241             custom_screen|predefined_screen|sector_top_company|seed_peer
    AA        medium                  7.9750                       15.0                     388.0                    192.0        0.6615         27.5966                               custom_screen|sector_top_company|seed_peer
   GEV        medium                  7.9025                       10.0                     149.0                      7.0        0.8916         42.4374                   custom_screen|etf_holding|sector_top_company|seed_peer
   GLW        medium                  6.7660                        7.0                      42.0                      0.0        1.1586         42.3187                                             sector_top_company|seed_peer
    AU        medium                  6.4250                       19.0                     737.0                    371.0        0.7435         27.1739                               custom_screen|sector_top_company|seed_peer
    CF        medium                  6.1950                       25.0                     284.0                     80.0        0.6724         28.0581                               custom_screen|sector_top_company|seed_peer
  KEYS        medium                  5.9730                       19.0                     213.0                     47.0        0.6333         42.3187                                             sector_top_company|seed_peer
  DELL           low                  5.8060                       25.0                     270.0                     34.0        0.5064         55.2818   industry_top_performing|predefined_screen|sector_top_company|seed_peer
    MP           low                  5.4000                       56.0                     259.0                    124.0        0.2537         26.0611                               custom_screen|sector_top_company|seed_peer
   EME           low                  4.6000                       39.0                     271.0                      0.0        0.4505         24.3187                                             sector_top_company|seed_peer
  MPWR           low                  4.5830                       36.0                     196.0                      0.0        0.3949         42.3187                                             sector_top_company|seed_peer
   CLF          weak                  4.1750                      148.0                     574.0                    321.0       -0.2583         25.7936                               custom_screen|sector_top_company|seed_peer
   MOS          weak                  3.9990                      138.0                     217.0                     61.0       -0.1376         26.1217                               custom_screen|sector_top_company|seed_peer
  ODFL          weak                  3.8250                       72.0                     150.0                      0.0        0.2195         24.3187                                             sector_top_company|seed_peer
   GWW          weak                  3.8245                      103.0                     151.0                     46.0        0.0773         24.3187                                             sector_top_company|seed_peer
   URI          weak                  3.7710                      123.0                     241.0                     19.0       -0.0487         24.3187                                             sector_top_company|seed_peer
    NU          weak                  3.7155                      111.0                     181.0                     10.0        0.0054         23.9611                       custom_screen|predefined_screen|sector_top_company

## Recommended additions

(none)

## Watchlist additions

name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct                                                             source_types
 RIG          59004.4407                NaN               50.5300          1723.2868               NaN              69.0504                       custom_screen|predefined_screen|sector_top_company
AAOI          47147.4114                NaN               55.1468          1675.6433               NaN              64.2289                custom_screen|industry_top_growth|industry_top_performing
  HL          28171.5134                NaN               55.0883          1605.6639               NaN              71.1891                               custom_screen|sector_top_company|seed_peer
 STX          37415.6264                NaN               55.1534          1587.2722               NaN              71.1891 industry_top_growth|industry_top_performing|sector_top_company|seed_peer
  AA          46013.8306                NaN               55.1534          1522.2026               NaN              71.1810                               custom_screen|sector_top_company|seed_peer
  CF          35382.2102                NaN               55.1534          1497.4248               NaN              71.1890                               custom_screen|sector_top_company|seed_peer
 FIX          43235.9798                NaN               55.0777          1496.9037               NaN              71.1891                                             sector_top_company|seed_peer
 TER          42730.9677                NaN               55.1237          1496.9037               NaN              71.1891                                             sector_top_company|seed_peer
CIEN          39586.8970                NaN               55.5490          1496.9037               NaN              71.1891                           predefined_screen|sector_top_company|seed_peer
 GEV          36510.6523                NaN               55.1237          1496.9037               NaN              71.1891                   custom_screen|etf_holding|sector_top_company|seed_peer
DELL          36510.6523                NaN               55.1237          1496.9037               NaN              71.1891   industry_top_performing|predefined_screen|sector_top_company|seed_peer
INTC          36510.6523                NaN               55.1237          1496.9037               NaN              71.1891             custom_screen|predefined_screen|sector_top_company|seed_peer
  AU          36339.7135                NaN               55.1247          1460.7348               NaN              69.6862                               custom_screen|sector_top_company|seed_peer
 CDE          38658.7536                NaN               55.1418          1364.6362               NaN              66.2254             custom_screen|predefined_screen|sector_top_company|seed_peer
AMPX          33551.7254                NaN               55.1237          1207.1984               NaN              71.1891                                    custom_screen|industry_top_performing

## Combo tests

    name  full_delta_roi_pct  full_delta_sharpe  full_delta_maxdd_pct  oos_delta_roi_pct  oos_delta_sharpe  oos_delta_maxdd_pct
AAOI+RIG          76004.4377                NaN               50.5289          2125.9059               NaN              64.1802
  HL+RIG          46884.7161                NaN               50.5841          1847.4877               NaN              69.0504
 RIG+STX          62352.8950                NaN               50.5300          1826.4786               NaN              69.0504
 AAOI+HL          32783.2354                NaN               55.0980          1798.5542               NaN              64.2289