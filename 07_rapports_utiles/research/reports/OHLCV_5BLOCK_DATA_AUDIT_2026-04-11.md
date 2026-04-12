# OHLCV 5-Block Data Audit

- as_of: `2026-04-11`
- ohlcv_max_date: `2026-04-10`
- scope: `380` tickers in the core OHLCV universe

## Block 1: Market Structure History

- output: `data/context/market_structure_daily.csv`
- output: `data/context/market_structure_latest.csv`
- summary: `data/extracts/market_structure_history_summary.csv`
- daily rows: `1,026,351`
- current shares coverage: `374 / 380`
- current float coverage: `361 / 380`
- daily shares coverage mean: `98.66%`

What it adds:
- historical `shares_outstanding`
- current `float_shares`
- estimated historical `market_cap`
- free-float ratio and market-cap-to-liquidity context

Why it matters:
- helps separate true institutional leaders from squeezier / low-float profiles
- adds structure for analyzing convex names like `AXTI`, `272210.KS`, `ONDS`

## Block 2: Earnings Point-in-Time

- output: `data/earnings/earnings_events_latest.csv`
- summary: `data/extracts/earnings_events_refresh_summary.csv`
- event rows: `10,678`
- unique tickers: `3,059`
- source mix:
  - `snapshot_next`: `6,186`
  - `snapshot_last`: `4,492`

What it adds:
- point-in-time earnings calendar memory from our own historical snapshots
- event dates seen from the vantage point of each snapshot

Important limitation:
- `yfinance.get_earnings_dates()` returned no usable detailed event history in this environment
- so this block is currently strong on calendar timing, but still weak on EPS surprise / reported EPS

## Block 3: Sector / Industry / Cluster Point-in-Time

- output: `data/context/taxonomy_point_in_time.csv`
- output: `data/context/taxonomy_point_in_time_latest.csv`
- summary: `data/extracts/taxonomy_point_in_time_summary.csv`
- snapshots available: `4`
- latest tickers covered: `3,595`
- latest clusters covered: `23`

Largest clusters inside the OHLCV core universe today:
- `financial-services`: `43`
- `oil-gas`: `43`
- `semiconductors`: `30`
- `space-defense`: `29`
- `healthcare`: `29`
- `industrial-compounders`: `28`
- `cloud-software`: `20`
- `optical-networking`: `11`
- `uranium`: `5`
- `crypto-beta`: `3`

What it adds:
- a usable point-in-time taxonomy layer for forward analysis
- a common language between scan, governance, drawdown analysis, and holdings concentration

Important limitation:
- this is genuinely point-in-time only from our saved snapshots onward
- older history is still reconstructed with heuristics, not a full archival taxonomy

## Block 4: FX Context

- output: `data/benchmarks/fx_reference_ohlcv.csv`
- summary: `data/benchmarks/fx_reference_summary.csv`
- currencies with full coverage: `22 / 23`
- only missing currency coverage: `AUD` (Yahoo rate limit on `AUDUSD=X`)

What it adds:
- `fx_to_usd`
- `fx_to_eur`
- short / medium horizon FX returns
- simple FX trend state via `above_sma200`

Why it matters:
- separates true local-stock strength from pure currency translation
- especially useful for `KRW`, `HKD`, `TWD`, `JPY`, `EUR` names in scan and ranking reviews

## Block 5: Listing Age / Corporate Actions / Metadata

- output: `data/extracts/listing_corporate_metadata.csv`
- refreshed corporate-action tickers this run: `108`
- action rows total: `20,373`
- tickers with dividend history: `235`
- tickers with split history: `202`
- median listing age: `4,148` days

What it adds:
- `first_ohlcv_date`
- `last_ohlcv_date`
- `bars_total`
- `listing_age_days`
- split / dividend recency and counts

Why it matters:
- lets us distinguish young, still-forming listings from seasoned names
- improves interpretation of violent moves after splits and other corporate-action effects

## Quant Takeaways

Most immediately useful for scan quality:
1. Market structure history
2. Taxonomy point-in-time
3. FX context

Most immediately useful for understanding drawdowns:
1. Taxonomy point-in-time
2. Market structure history
3. Listing / corporate actions metadata

Most useful but still partially blocked:
1. Earnings point-in-time with real surprise data

Net conclusion:
- the OHLCV file itself was already rich enough on pure price/volume behavior
- the biggest missing edge was not another technical indicator
- it was the missing context around market structure, taxonomy, FX, earnings timing, and listing/corporate-action state
