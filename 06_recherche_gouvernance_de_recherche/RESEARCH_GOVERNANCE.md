# Research Governance

Project research governance frozen on `2026-04-05`.

## Baseline policy

- active production baseline: `r8`
- robust fallback baseline: `r7`
- current research phase: **stabilization and forward validation**
- default posture: **do not promote**

The project has already harvested the main structural gains:

- `r2`: anti-churn / held-release
- `r4`: state-dependent reentry
- `r5`: fragile cluster concentration guard
- `r6`: weak-technology sizing cap
- `r7`: removal of legacy `MIE`
- `r8`: narrow uranium non-PP exit guard

From this point, every new idea starts in the category **suspect until proven robust**.

## What research is still encouraged

Allowed themes:

- structural portfolio-risk controls
- forward validation and monitoring
- governance replay and event attribution
- holdings-aware overnight risk measurement
- execution realism and stress slippage
- point-in-time taxonomy improvements
- data quality improvements
- scan/governance diagnostics that improve understanding without forcing promotion

These studies remain useful even when they produce no new baseline.

## Promotion gate

A change may only be promoted if it passes **all** of the following:

1. improves `full` replay
2. improves `OOS`
3. does not materially worsen drawdown
4. has a clear economic explanation
5. remains credible across more than one regime
6. does not depend mainly on:
   - one ticker
   - one narrow cluster
   - one isolated year
   - a handful of trades

If one of these conditions is missing, the change stays in research.

## Automatic rejection rules

The following may still be studied, but should not be promoted without extraordinary evidence:

- ticker-specific patches
- ultra-narrow cluster rules after `r8`
- micro-tuning of score weights
- historical patches that improve only one window
- fragile entry/exit tweaks that work only on a tiny pocket of history
- improvements that come mostly from a single name, theme, or episode

## What must be shown before promotion

Each promotable study should document:

- baseline compared against (`r8` by default, `r7` as robustness check)
- full metrics
- OOS metrics
- drawdown impact
- regime/year distribution of the gain
- economic explanation in plain language
- concentration of the gain:
  - by ticker
  - by cluster
  - by year

If the gain is too concentrated, the candidate is research-only.

## Monitoring policy

Baseline management should monitor:

- live behavior of `r8`
- relative drift between `r8` and `r7`
- forward drawdown behavior
- concentration and overnight gap risk
- persistence and quality of scan promotions
- whether recent gains continue to come from structural effects rather than narrow patches

## Decision rule

If a new idea is:

- local
- narrow
- hard to explain
- mostly historical
- or dependent on a small pocket of trades

then it should be:

- tested
- documented
- archived

but **not promoted**.

## Summary

The project is no longer in free-form optimization mode.

The working rule is:

- keep `r8` active
- keep `r7` as the robust fallback
- stop promoting micro-patches
- only promote future changes that are more general, more explainable, and more robust than `r8`
