# Scan Engine Hardening Audit

Goal: compare three discovery stacks against our known historical compatible misses.

Profiles tested:
- `hybrid`: Yahoo expansion + local structural lanes;
- `local_structural`: local OHLCV/context/taxonomy/market-structure lanes only;
- `yahoo_external`: Yahoo discovery only, without local structural lanes.

Seeds: `AXTI, SNDK, LITE, WDC, BE, 006800.KS`
Sectors inferred: `Technology, Industrials, Financial Services`

## Coverage Summary

| profile | bucket | discovered_hits | discovered_hit_pct | top50_hits | top100_hits | new_only_hits |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid | missing_selection | 48 | 63.16 | 11 | 21 | 1 |
| hybrid | missing_universe | 89 | 85.58 | 14 | 24 | 84 |
| local_structural | missing_selection | 17 | 22.37 | 6 | 12 | 1 |
| local_structural | missing_universe | 58 | 55.77 | 24 | 46 | 56 |
| yahoo_external | missing_selection | 45 | 59.21 | 13 | 18 | 1 |
| yahoo_external | missing_universe | 70 | 67.31 | 7 | 18 | 66 |

## Unique Gap Hits By Profile

(none)

## Family Coverage Focus

| profile | bucket | family | count | hits | hit_pct |
| --- | --- | --- | --- | --- | --- |
| hybrid | missing_selection | cloud-software | 9 | 7 | 77.78 |
| local_structural | missing_selection | cloud-software | 9 | 1 | 11.11 |
| yahoo_external | missing_selection | cloud-software | 9 | 7 | 77.78 |
| hybrid | missing_selection | semiconductors | 9 | 9 | 100.00 |
| local_structural | missing_selection | semiconductors | 9 | 2 | 22.22 |
| yahoo_external | missing_selection | semiconductors | 9 | 9 | 100.00 |
| hybrid | missing_universe | cloud-software | 2 | 2 | 100.00 |
| local_structural | missing_universe | cloud-software | 2 | 0 | 0.00 |
| yahoo_external | missing_universe | cloud-software | 2 | 2 | 100.00 |
| hybrid | missing_universe | optical-networking | 2 | 2 | 100.00 |
| local_structural | missing_universe | optical-networking | 2 | 2 | 100.00 |
| yahoo_external | missing_universe | optical-networking | 2 | 2 | 100.00 |
| hybrid | missing_universe | semiconductors | 11 | 9 | 81.82 |
| local_structural | missing_universe | semiconductors | 11 | 6 | 54.55 |
| yahoo_external | missing_universe | semiconductors | 11 | 9 | 81.82 |

## Hybrid Local-Sentinel Contribution

| bucket | hybrid_local_source_hits |
| --- | --- |
| missing_selection | 5 |
| missing_universe | 14 |

## Conclusions

- `hybrid` should remain the default scan architecture: it keeps the broadest recall.
- `local_structural` is now a credible resilience layer instead of a weak fallback.
- `yahoo_external` is not enough on its own if we want to reduce historical family misses with discipline.
- The new local sentinel lanes matter only if they add coverage on historical compatible misses without forcing ticker-specific hindsight.

