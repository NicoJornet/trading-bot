# Research Scoreboard Latest

- as_of: `2026-04-11`
- active baseline: `best_algo_184_dynamic_universe_staged_r8_uranium_exitguard`
- baseline frozen on: `2026-04-05`
- recommendation: `continue monitoring`

## Baseline Snapshot

- full replay: `ROI 920484.406288%`, `Sharpe 2.113031`, `MaxDD -44.80329%`
- OOS replay: `ROI 14687.404102%`, `Sharpe 2.879884`, `MaxDD -35.809446%`

## Data Freshness

- OHLCV max date: `2026-04-10`
- refreshed coverage: `379/380` = `99.74%`
- failed tickers: `MMC`

## Live Overlay

- approved standalone adds: `8`
- approved single swaps: `1`
- approved standalone removes: `0`
- selected demotions: `1`
- selected adds: `006400.KS, 006800.KS, 0568.HK, 2383.TW, 272210.KS, 285A.T, AXTI, ONDS`
- selected swaps: `ZS -> 0568.HK`

## Pipeline State

- tracked forward names: `195`
- names still in current scan: `176`
- approved / probation / targeted names: `24`
- stage or status progressions vs previous snapshot: `0`
- unresolved legacy selection gaps below targeted: `75`

## Top Forward Priorities

| ticker | monitor_bucket | promotion_stage | dynamic_status | pit_cluster_key | scan_candidate_track | history_emergence_persistence_score | pit_data_context_score | recent_score | monitor_priority_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 285A.T | quality_lane | approved_live | approved | semiconductors | quality_compounder | 6.1 | 0.766 | 6.7521 | 18.317 |
| 272210.KS | legacy_universe_gap | approved_live | approved | space-defense | persistent_leader | 6.5167 | 0.8768 | 1.975 | 17.7183 |
| 006400.KS | legacy_universe_gap | approved_live | approved | industrial-compounders | persistent_leader | 6.1 | 0.8875 | 1.2079 | 17.618 |
| 2383.TW | legacy_universe_gap | approved_live | approved | technology | quality_compounder | 5.5333 | 0.8875 | 2.7037 | 17.5445 |
| 006800.KS | legacy_selection_gap | approved_live | approved | financial-services | emerging_leader | 7.0 | 0.8875 | 3.5879 | 16.927 |
| ONDS | quality_lane | approved_live | watch | optical-networking | persistent_leader | 6.6167 | 0.4875 | 3.2129 | 16.2927 |
| FTI | quality_lane | targeted_integration | approved | oil-gas | fringe | 5.9 | 0.4875 | 0.9163 | 14.3991 |
| 3037.TW | quality_lane | targeted_integration | watch | technology | quality_compounder | 5.7333 | 0.4875 | 2.8553 | 12.7818 |
| TER | legacy_universe_gap | targeted_integration | watch | semiconductors | quality_compounder | 5.4917 | 0.8875 | 2.3662 | 12.6882 |
| PLS.AX | quality_lane | targeted_integration | watch | industrial-compounders | persistent_leader | 5.4917 | 0.6375 | 1.719 | 12.6154 |
| 2345.TW | legacy_universe_gap | targeted_integration | watch | optical-networking | persistent_leader | 5.2 | 0.8875 | 1.1462 | 12.4344 |
| 6857.T | legacy_universe_gap | targeted_integration | watch | semiconductors | persistent_leader | 5.2 | 0.8875 | 1.4515 | 12.343 |
| UCTT | legacy_universe_gap | targeted_integration | prime_watch | semiconductors | persistent_leader | 5.9 | 0.925 | 1.3094 | 12.1748 |
| PR | legacy_universe_gap | targeted_integration | watch | oil-gas | fringe | 5.4333 | 0.8875 | 0.6985 | 12.0698 |
| CIFR | quality_lane | targeted_integration | watch | technology | persistent_leader | 4.55 | 0.8564 | 2.5535 | 11.8525 |

## Family Summary

| cluster | names | approved_live | targeted | avg_priority | avg_persistence |
| --- | --- | --- | --- | --- | --- |
| semiconductors | 6 | 1 | 5 | 12.6212 | 5.5542 |
| technology | 4 | 1 | 3 | 12.9547 | 5.1166 |
| optical-networking | 4 | 1 | 3 | 12.4161 | 5.4584 |
| industrial-compounders | 2 | 1 | 1 | 15.1167 | 5.7958 |
| financial-services | 2 | 1 | 1 | 13.6699 | 6.275 |
| space-defense | 1 | 1 | 0 | 17.7183 | 6.5167 |
| oil-gas | 2 | 0 | 2 | 13.2344 | 5.6667 |
| industrials | 2 | 0 | 2 | 10.3428 | 4.8611 |
| communication-services | 1 | 0 | 1 | 11.3275 | 5.9 |

## Recent Progressions

(none)

## Unresolved Portfolio-Ranking Gaps

| ticker | promotion_stage | dynamic_status | pit_cluster_key | scan_candidate_track | scan_quality_compounder_fit | pit_data_context_score | monitor_priority_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LRCX | watch_queue | watch | semiconductors |  | weak | 0.925 | 6.62 |
| KLAC | watch_queue | watch | semiconductors |  | weak | 0.925 | 6.62 |
| AMAT | watch_queue | watch | semiconductors |  | weak | 0.8875 | 6.62 |
| CAT | review_queue | review | industrials |  | weak | 0.8875 | 4.32 |
| MRVL | review_queue | review | semiconductors |  | weak | 0.8875 | 4.32 |
| CVX | review_queue | review | oil-gas |  | weak | 0.925 | 4.32 |
| GE | review_queue | review | space-defense |  | weak | 0.8875 | 4.32 |
| AAPL | review_queue | review | technology |  | weak | 0.925 | 4.32 |
| ENI.MI | review_queue | review | oil-gas |  | weak | 0.8875 | 4.32 |
| AVGO | review_queue | review | semiconductors |  | weak | 0.8875 | 4.32 |
| CCJ | review_queue | review | uranium |  | weak | 0.8875 | 4.3005 |
| ALB | review_queue | review | basic-materials |  | weak | 0.925 | 4.2811 |
| FCX | review_queue | review | copper-miners |  | weak | 0.925 | 4.24 |
| TSM | review_queue | review | semiconductors |  | weak | 0.925 | 4.1817 |
| PWR | review_queue | review | industrials |  | weak | 0.8875 | 4.1622 |

## Local Portfolio Context

- local portfolio names: `LRCX, MU, PAAS`
- names already shared with live overlay: `none`
- local holdings outside current live overlay: `LRCX, MU, PAAS`
- note: this local book is useful for operational alignment, but it is not promotion evidence for the engine.

## Interpretation

- the scan repair is still doing useful work: multiple families now progress naturally through governance without forcing names into the engine.
- the remaining misses are concentrated in the ranking / portfolio layer, but they are still mostly in monitoring rather than in proof-forward mode.
- the right posture remains: keep `r8` stable, keep observing the repaired families, and only reopen engine work if a family persists across several snapshots and still fails structurally.

## References

- [SYSTEM_BASELINE_LATEST.md](C:/Users/nicol/Downloads/214/SYSTEM_BASELINE_LATEST.md)
- [dynamic_universe_actions_summary.md](C:/Users/nicol/Downloads/214/data/dynamic_universe/dynamic_universe_actions_summary.md)
- [dynamic_universe_quality_compounder_forward_monitor.md](C:/Users/nicol/Downloads/214/data/dynamic_universe/dynamic_universe_quality_compounder_forward_monitor.md)
- [WEEKLY_FORWARD_REVIEW_PROTOCOL.md](C:/Users/nicol/Downloads/214/WEEKLY_FORWARD_REVIEW_PROTOCOL.md)
