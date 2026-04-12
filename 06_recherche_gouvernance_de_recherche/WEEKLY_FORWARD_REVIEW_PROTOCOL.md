# Weekly Forward Review Protocol

Protocol frozen on `2026-04-08` for the stabilization phase around baseline `r8`.

## Objective

Run a disciplined weekly review of the repaired scan + governance pipeline without forcing names into the portfolio and without reopening free-form optimization.

## Active references

- active baseline: `r8`
- robust fallback: `r7`
- research posture: **monitor first, promote rarely**

## Weekly inputs to review

Open these files first:

- [RESEARCH_SCOREBOARD_LATEST.md](C:/Users/nicol/Downloads/214/RESEARCH_SCOREBOARD_LATEST.md)
- [WINNER_RECALL_DASHBOARD_LATEST.md](C:/Users/nicol/Downloads/214/WINNER_RECALL_DASHBOARD_LATEST.md)
- [ENGINE_RESEARCH_SCOREBOARD_LATEST.md](C:/Users/nicol/Downloads/214/ENGINE_RESEARCH_SCOREBOARD_LATEST.md)
- [dynamic_universe_quality_compounder_forward_monitor.md](C:/Users/nicol/Downloads/214/data/dynamic_universe/dynamic_universe_quality_compounder_forward_monitor.md)
- [dynamic_universe_actions_summary.md](C:/Users/nicol/Downloads/214/data/dynamic_universe/dynamic_universe_actions_summary.md)
- [dynamic_universe_summary.md](C:/Users/nicol/Downloads/214/data/dynamic_universe/dynamic_universe_summary.md)
- [dynamic_universe_promotion_queue.csv](C:/Users/nicol/Downloads/214/data/dynamic_universe/dynamic_universe_promotion_queue.csv)
- [ohlcv_refresh_summary.csv](C:/Users/nicol/Downloads/214/data/extracts/ohlcv_refresh_summary.csv)

## Review order

1. Confirm the data layer is fresh.
2. Check whether the live overlay changed.
3. Check whether historical missing winners are being repaired.
4. Check whether repaired scan families progressed in stage.
5. Separate scan wins from core-ranking misses.
6. Check blockage diagnosis and family escalation candidates.
7. Check whether a true engine lever has reopened.
8. Decide whether to keep monitoring or open a new research study.

The scoreboard should be read first because it compresses the live state into one place. The winner-recall dashboard answers the separate question: are historical misses actually closing?

## Step 1: Data freshness

Minimum checks:

- OHLCV max date matches the latest expected market session.
- refresh coverage stays high (`>= 99%` of tickers refreshed is the target range).
- failures are isolated and non-structural.

If data freshness is weak, stop the review there and fix data first.

## Step 2: Live overlay

Read:

- [dynamic_universe_actions_summary.md](C:/Users/nicol/Downloads/214/data/dynamic_universe/dynamic_universe_actions_summary.md)

Look for:

- new approved standalone adds
- new approved swaps
- new approved removals

Interpretation:

- if repaired families reach approved actions, the scan/governance repair is working
- if the live overlay stays static while the forward monitor improves, the bottleneck may still be action selection or portfolio logic

## Step 3: Winner recall closure

Read:

- [WINNER_RECALL_DASHBOARD_LATEST.md](C:/Users/nicol/Downloads/214/WINNER_RECALL_DASHBOARD_LATEST.md)

Primary signals:

- `missing_universe` closure:
  - how many old misses are now at least in scan
  - how many reached `watch+`
  - how many reached `targeted+`
- `missing_selection` closure:
  - how many old misses are now being promoted
  - which families remain unresolved
- whether unresolved names are isolated or concentrated in one family
- blockage labels:
  - `scan_candidate_still_weak`
  - `governance_stall`
  - `portfolio_like_gap`

Interpretation:

- if `missing_universe` names now reach `targeted+`, the scan/universe repair is working
- if `missing_selection` names stay trapped below `targeted+`, the gap may still be portfolio-like
- if unresolved misses are scattered, avoid overreacting
- if unresolved misses cluster by family for several weeks, that becomes a valid research trigger
- if a family starts producing multiple `portfolio_like_gap` names, it becomes an escalation candidate

## Step 4: Forward monitor progression

Read:

- [dynamic_universe_quality_compounder_forward_monitor.md](C:/Users/nicol/Downloads/214/data/dynamic_universe/dynamic_universe_quality_compounder_forward_monitor.md)

Primary signals:

- repeated presence across snapshots
- promotion progression:
  - `watch_queue -> targeted_integration`
  - `targeted_integration -> probation_live`
  - `probation_live -> approved_live`
- strong persistence score
- broad family behavior, not one isolated ticker

Positive evidence:

- several names from the same family progress together
- names improve without manual intervention
- names eventually appear in approved adds or approved swaps

## Step 5: Classify misses correctly

Use the monitor buckets:

- `legacy_universe_gap`
- `legacy_selection_gap`
- `quality_lane`

Interpretation:

- if a name now progresses through governance, it was mainly a scan/universe problem
- if a name remains stuck in low stages despite repeated healthy scan behavior, it is more likely a core ranking / portfolio problem
- if a name is now tagged `portfolio_like_gap`, it is no longer a pure discovery miss
- if a name is tagged `governance_stall`, the bottleneck is still earlier than the final portfolio layer

Do not treat retrospective famous winners as mandatory targets.

## Step 6: Family escalation check

Read:

- [RESEARCH_SCOREBOARD_LATEST.md](C:/Users/nicol/Downloads/214/RESEARCH_SCOREBOARD_LATEST.md)

Primary signals:

- `Families To Escalate If They Persist`
- number of names in the same cluster
- persistence score
- core rank support (`core_top30_flag` / `core_latest_rank`)

Interpretation:

- one isolated blocked name is still just monitoring
- a family with `2+` persistent blocked names and core support is the first real candidate for a focused study

## Step 7: Engine pressure check

Read:

- [ENGINE_RESEARCH_SCOREBOARD_LATEST.md](C:/Users/nicol/Downloads/214/ENGINE_RESEARCH_SCOREBOARD_LATEST.md)

Interpretation:

- if the engine scoreboard still says `closed_no_upgrade` on slot-3, switches and reentry, do not reopen those studies
- if concentration remains the only open structural pressure, prefer observation over patching
- only reopen core engine work if:
  - a family becomes `ready_if_family_repeats`
  - or a broad structural pressure survives several snapshots and remains economically coherent

## Step 8: Decision rule

Default weekly outcome:

- **keep monitoring**

Open a new research study only if all of the following are true:

1. the same family persists for at least `3-5` snapshots
2. more than one name in that family progresses
3. the family still fails to reach portfolio relevance
4. the failure is not explained by poor timing, poor event quality, or weak persistence

If one of these is missing, do not reopen the engine.

## Hard rules

- do not force names because they are known hindsight winners
- do not promote a new baseline from forward-monitor observations alone
- do not reopen score-tuning because one family looks attractive in hindsight
- do not override the anti-overfit rules in [RESEARCH_GOVERNANCE.md](C:/Users/nicol/Downloads/214/RESEARCH_GOVERNANCE.md)

## Weekly output expectation

At the end of each weekly review, produce a short note with:

- data freshness status
- live overlay changes
- top progressing families
- unresolved ranking/portfolio misses
- recommendation:
  - `continue monitoring`
  - `open research study`
  - `data issue first`

## Current watch families

As of `2026-04-08`, the main forward families to watch are:

- Asia tech supply-chain
- optical / networking
- industrial compounders
- quality compounders

The main unresolved core-ranking miss bucket still includes names such as:

- `AVGO`
- `NOW`
- `PANW`
- `CDNS`
- `KLAC`
- `AMAT`
- `LRCX`
- `META`

These stay in monitoring until the system itself proves that the bottleneck is structural and persistent.
