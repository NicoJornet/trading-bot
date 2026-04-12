# Winner Recall Dashboard

As of `2026-04-11`.

Purpose:
- track whether historical missing winners are now being repaired by the current pipeline;
- separate remaining misses into universe / scan-governance / portfolio-like bottlenecks;
- quantify closure rates without opening a new backtest-heavy study.

## High-Level Summary

- `missing_universe`: `104` historical compatible misses;
  - now in current scan: `102` (`98.08%`)
  - now watch+: `31` (`29.81%`)
  - now targeted+: `16` (`15.38%`)
  - now approved live: `3` (`2.88%`)
- `missing_selection`: `76` historical compatible misses;
  - now in current scan: `59` (`77.63%`)
  - now watch+: `4` (`5.26%`)
  - now targeted+: `1` (`1.32%`)
  - now approved live: `1` (`1.32%`)

## Bucket Summary

| bucket | count | in_current_scan | in_current_scan_pct | watch_plus | watch_plus_pct | targeted_plus | targeted_plus_pct | approved_live | approved_live_pct | median_priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| missing_selection | 76 | 59 | 77.63 | 4 | 5.26 | 1 | 1.32 | 1 | 1.32 | 3.372 |
| missing_universe | 104 | 102 | 98.08 | 31 | 29.81 | 16 | 15.38 | 3 | 2.88 | 5.2126 |

## Stage Breakdown

| bucket | stage_bin | count | share_pct |
| --- | --- | --- | --- |
| missing_selection | review_or_discovered | 55 | 72.37 |
| missing_selection | not_in_current_scan | 17 | 22.37 |
| missing_selection | watch_queue | 3 | 3.95 |
| missing_selection | approved_live | 1 | 1.32 |
| missing_universe | review_or_discovered | 71 | 68.27 |
| missing_universe | watch_queue | 15 | 14.42 |
| missing_universe | targeted_integration | 13 | 12.5 |
| missing_universe | approved_live | 3 | 2.88 |
| missing_universe | not_in_current_scan | 2 | 1.92 |

## Families Repairing Best

| bucket | family | count | in_scan | watch_plus | targeted_plus | approved_live | avg_priority | avg_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| missing_selection | financial-services | 6 | 6 | 1 | 1 | 1 | 5.6259 | 2.1815 |
| missing_selection | semiconductors | 9 | 9 | 3 | 0 | 0 | 4.7674 | 1.3685 |
| missing_selection | uranium | 1 | 1 | 0 | 0 | 0 | 4.3005 | 1.25 |
| missing_selection | basic-materials | 1 | 1 | 0 | 0 | 0 | 4.2811 | 1.4 |
| missing_selection | technology | 2 | 2 | 0 | 0 | 0 | 3.92 | 1.4 |
| missing_selection | copper-miners | 2 | 2 | 0 | 0 | 0 | 3.8411 | 1.3028 |
| missing_selection | industrials | 6 | 6 | 0 | 0 | 0 | 3.6613 | 1.3676 |
| missing_selection | oil-gas | 9 | 9 | 0 | 0 | 0 | 3.5902 | 1.3815 |
| missing_selection | space-defense | 6 | 6 | 0 | 0 | 0 | 3.5312 | 1.3398 |
| missing_selection | cloud-software | 5 | 5 | 0 | 0 | 0 | 3.52 | 1.4 |
| missing_selection | communication-services | 3 | 3 | 0 | 0 | 0 | 3.3817 | 1.3074 |
| missing_selection | consumer-cyclical | 1 | 1 | 0 | 0 | 0 | 3.2045 | 0.6333 |
| missing_selection | healthcare | 4 | 4 | 0 | 0 | 0 | 3.1899 | 1.157 |
| missing_selection | industrial-compounders | 4 | 4 | 0 | 0 | 0 | 3.1589 | 1.3537 |
| missing_selection | unknown | 17 | 0 | 0 | 0 | 0 | 0.0 | 0.9333 |
| missing_universe | semiconductors | 11 | 11 | 7 | 5 | 0 | 8.3782 | 3.0578 |
| missing_universe | optical-networking | 7 | 7 | 5 | 3 | 0 | 8.2695 | 2.7102 |
| missing_universe | technology | 9 | 9 | 4 | 2 | 1 | 7.8978 | 2.3961 |
| missing_universe | industrials | 10 | 10 | 2 | 2 | 0 | 5.938 | 2.184 |
| missing_universe | space-defense | 2 | 2 | 2 | 1 | 1 | 12.9709 | 3.3875 |

## Top Historical Misses Still Unresolved

| ticker | recall_bucket | family | in_current_scan | promotion_stage | dynamic_status | snapshots_seen | persistence_score | priority_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INTC | missing_universe | semiconductors | 1 | watch_queue | approved | 14.0 | 3.4667 | 11.0808 |
| 6443.TW | missing_universe | technology | 1 | watch_queue | approved | 8.0 | 3.45 | 10.0011 |
| 5713.T | missing_universe | industrial-compounders | 1 | watch_queue | watch | 9.0 | 3.395 | 9.4506 |
| 1605.T | missing_universe | oil-gas | 1 | watch_queue | review | 9.0 | 3.4533 | 9.3539 |
| VIAV | missing_universe | optical-networking | 1 | watch_queue | watch | 13.0 | 1.7883 | 9.2366 |
| 042660.KS | missing_universe | space-defense | 1 | watch_queue | review | 8.0 | 3.325 | 8.2236 |
| OVV | missing_universe | oil-gas | 1 | watch_queue | watch | 11.0 | 3.4 | 8.2213 |
| 034020.KS | missing_universe | industrial-compounders | 1 | watch_queue | reject | 8.0 | 3.45 | 8.1599 |
| CCO.TO | missing_universe | energy | 1 | watch_queue | review | 11.0 | 1.2917 | 8.0142 |
| SATS | missing_universe | communication-services | 1 | watch_queue | reject | 11.0 | 5.2833 | 7.8837 |
| 9501.T | missing_universe | utilities | 1 | watch_queue | review | 9.0 | 3.3617 | 7.4749 |
| COHR | missing_universe | technology | 1 | watch_queue | reject | 12.0 | 3.45 | 7.4435 |
| AR | missing_universe | oil-gas | 1 | watch_queue | review | 11.0 | 2.7667 | 6.8415 |
| LRCX | missing_selection | semiconductors | 1 | watch_queue | watch | 14.0 | 1.4 | 6.62 |
| AMAT | missing_selection | semiconductors | 1 | watch_queue | watch | 14.0 | 1.4 | 6.62 |

## Historical Misses Already Progressing

| ticker | recall_bucket | family | promotion_stage | dynamic_status | snapshots_seen | persistence_score | priority_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 272210.KS | missing_universe | space-defense | approved_live | approved | 8.0 | 3.45 | 17.7183 |
| 006400.KS | missing_universe | industrial-compounders | approved_live | approved | 8.0 | 3.45 | 17.618 |
| 2383.TW | missing_universe | technology | approved_live | approved | 8.0 | 2.65 | 17.5445 |
| 006800.KS | missing_selection | financial-services | approved_live | approved | 8.0 | 6.7 | 16.927 |
| TER | missing_universe | semiconductors | targeted_integration | watch | 11.0 | 3.3583 | 12.6882 |
| 2345.TW | missing_universe | optical-networking | targeted_integration | watch | 8.0 | 3.45 | 12.4344 |
| 6857.T | missing_universe | semiconductors | targeted_integration | watch | 9.0 | 3.57 | 12.343 |
| UCTT | missing_universe | semiconductors | targeted_integration | prime_watch | 11.0 | 5.5667 | 12.1748 |
| PR | missing_universe | oil-gas | targeted_integration | watch | 11.0 | 5.2833 | 12.0698 |
| WBD | missing_universe | communication-services | targeted_integration | watch | 11.0 | 5.4889 | 11.3275 |
| 6869.HK | missing_universe | optical-networking | targeted_integration | reject | 11.0 | 3.3583 | 10.8081 |
| FIX | missing_universe | industrials | targeted_integration | watch | 14.0 | 3.5333 | 10.6741 |
| 000660.KS | missing_universe | semiconductors | targeted_integration | reject | 8.0 | 3.45 | 10.2379 |
| CIEN | missing_universe | optical-networking | targeted_integration | reject | 14.0 | 3.6833 | 10.1293 |
| 5803.T | missing_universe | industrials | targeted_integration | reject | 9.0 | 2.7667 | 10.0116 |

## Strategic Read

- If `missing_universe` closure keeps rising, the scan/universe repair is working.
- If names become `watch` or `targeted`, the gap is no longer a pure universe miss.
- If a whole family stays persistent but cannot reach `targeted+`, that starts to look like a scan/governance bottleneck.
- If a family reaches `targeted+` repeatedly but still never matters to the portfolio, that is the moment to reopen a portfolio/ranking study.
