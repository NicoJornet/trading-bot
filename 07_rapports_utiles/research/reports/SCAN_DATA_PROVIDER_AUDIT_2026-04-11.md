# Scan Data Provider Audit

## Question

How far can we push the scan without missing real leaders, and is Yahoo the right long-term database for this system?

## Short Answer

- **Yahoo / yfinance is not the best primary database** for this project.
- **Yahoo remains useful as a free discovery-expansion layer** because it gives us screeners, sector/industry helpers, ETF holdings, lookup, and broad international symbol reach.
- **The best pragmatic upgrade path is hybrid**:
  - keep Yahoo for discovery breadth;
  - promote an official vendor to primary master data for OHLCV + corporate actions + company/fundamental context;
  - keep a second official vendor as fallback / validation layer.

## What Our Local Evidence Says

Current local evidence already shows where Yahoo is strong and where it is weak:

- latest OHLCV refresh:
  - [ohlcv_refresh_summary.csv](C:/Users/nicol/Downloads/214/data/extracts/ohlcv_refresh_summary.csv)
  - `379 / 380` tickers refreshed;
  - one persistent failure remains: `MMC`
- 5-block audit:
  - [OHLCV_5BLOCK_DATA_AUDIT_2026-04-11.md](C:/Users/nicol/Downloads/214/research/reports/OHLCV_5BLOCK_DATA_AUDIT_2026-04-11.md)
  - `yfinance.get_earnings_dates()` produced no usable detailed event history in this environment;
  - FX context had one Yahoo rate-limit hole on `AUDUSD=X`

This is enough to say:

- Yahoo is **good enough to keep us moving**;
- Yahoo is **not robust enough to be the only foundation** for a global research stack we want to trust deeply.

## Official Provider Reading

### Yahoo / yfinance

Official repo:
- [yfinance GitHub](https://github.com/ranaroussi/yfinance)

What matters for us:
- the project explicitly says it is **not affiliated, endorsed, or vetted by Yahoo**;
- it uses Yahoo’s publicly available APIs;
- it is intended for **research and educational purposes**

This is the key governance problem for a primary database.

### Polygon

Official docs:
- [Polygon day aggregates](https://polygon.io/docs/flat-files/stocks/day-aggregates/2025)
- [Polygon REST quickstart](https://polygon.io/docs/rest/quickstart)

What matters for us:
- strong official product;
- daily aggregates across **all U.S. equities**;
- EOD and full history available on higher plans;
- good for prices, dividends, splits, and U.S. reference / fundamentals

Conclusion:
- excellent if we were more U.S.-centric;
- **not enough alone** for our current multi-country equity stack.

### Financial Modeling Prep

Official pages:
- [FMP home](https://site.financialmodelingprep.com/)
- [FMP docs](https://site.financialmodelingprep.com/developer/docs)

What matters for us:
- `70,000+ securities`
- `30+ years of historical daily price data`
- `30+ years of fundamentals`
- `10+ years of earnings and call transcripts`
- search & directory, market calendar, company profile, analyst data, bulk downloads
- claims direct work with major exchanges and primary / regulatory sources

Conclusion:
- **best fit for our current architecture** if we want one official primary vendor for a global stock universe.

### Alpha Vantage

Official docs:
- [Alpha Vantage documentation](https://www.alphavantage.co/documentation/)

What matters for us:
- global equity daily time series with `20+ years`
- adjusted / raw price variants
- `shares outstanding`
- `listing & delisting status`
- `earnings calendar`

Conclusion:
- strong official fallback / validation provider;
- likely better as **backup / cross-check layer** than as the only primary source for our full workflow.

## Provider Verdict

### Best free exploration layer

- **Yahoo / yfinance**

Why:
- broad discovery tooling;
- low friction;
- good for iterative scan expansion and idea generation

Why not primary:
- unofficial;
- legal/support ambiguity;
- local evidence of rate-limit / completeness issues

### Best primary replacement for our current global setup

- **FMP**

Why:
- global scope matches our KR / TW / HK / JP / EU / US workflow better than Polygon;
- covers the exact missing context blocks we care about:
  - OHLCV
  - company profile
  - market calendar
  - fundamentals
  - search / directory

### Best U.S. premium satellite

- **Polygon**

Why:
- very strong official U.S. equities stack;
- useful if later we want a higher-trust U.S. reference layer for price / corp actions / intraday quality

### Best fallback / validator

- **Alpha Vantage**

Why:
- official global daily coverage;
- useful to verify or backfill price / listing / earnings-calendar issues

## New Scan Hardening Result

We also hardened the scan engine itself:
- added a new **local structural sentinel layer** based on our own OHLCV + point-in-time context;
- added `source_profile` modes:
  - `hybrid`
  - `local_structural`
  - `yahoo_external`

Study:
- [SCAN_ENGINE_HARDENING_2026-04-11.md](C:/Users/nicol/Downloads/214/research/reports/SCAN_ENGINE_HARDENING_2026-04-11.md)
- [scan_engine_hardening_summary_20260411.csv](C:/Users/nicol/Downloads/214/research/exports/scan_engine_hardening_summary_20260411.csv)

Key numbers:

- `hybrid`
  - missing-universe hits: `89 / 104`
  - missing-selection hits: `48 / 76`
- `local_structural`
  - missing-universe hits: `58 / 104`
  - missing-selection hits: `17 / 76`
- `yahoo_external`
  - missing-universe hits: `70 / 104`
  - missing-selection hits: `45 / 76`

Important nuance:

- local-only is **not** a replacement for Yahoo today;
- but local-only is now a **credible resilience layer**
- and inside `hybrid`, the new local sentinel sources already contribute:
  - `14` missing-universe hits
  - `5` missing-selection hits

That is the real improvement:
- the scan is now **less fragile** to Yahoo itself;
- and we have a better way to know whether a leader was missed because:
  - it never entered our OHLCV universe;
  - Yahoo discovery missed it;
  - governance stalled it;
  - or the portfolio rejected it later.

## How To Know If We Missed A True Leader

The current objective chain is now:

1. retrospective compatible misses
   - [SCAN_ALGO_RETROSPECTIVE_MISSING_WINNERS_20260406.md](C:/Users/nicol/Downloads/214/research/reports/SCAN_ALGO_RETROSPECTIVE_MISSING_WINNERS_20260406.md)

2. forward recall dashboard
   - [WINNER_RECALL_DASHBOARD_LATEST.md](C:/Users/nicol/Downloads/214/WINNER_RECALL_DASHBOARD_LATEST.md)

3. research scoreboard / blockage diagnosis
   - [RESEARCH_SCOREBOARD_LATEST.md](C:/Users/nicol/Downloads/214/RESEARCH_SCOREBOARD_LATEST.md)

4. engine scoreboard
   - [ENGINE_RESEARCH_SCOREBOARD_LATEST.md](C:/Users/nicol/Downloads/214/ENGINE_RESEARCH_SCOREBOARD_LATEST.md)

5. source-profile resilience audit
   - [SCAN_ENGINE_HARDENING_2026-04-11.md](C:/Users/nicol/Downloads/214/research/reports/SCAN_ENGINE_HARDENING_2026-04-11.md)

A “true leader missed” is now much easier to classify:

- **universe miss**: not in the research universe early enough
- **scan miss**: in universe, but not discovered strongly enough
- **governance miss**: discovered, but stalled in review/watch
- **portfolio miss**: promoted, but never converted into real portfolio preference

## Recommendation

1. keep `hybrid` as the default scan engine
2. keep the new `local_structural` layer active to harden recall
3. do **not** replace Yahoo with nothing
4. over the medium term, migrate the master data stack toward:
   - **FMP primary**
   - **Yahoo discovery secondary**
   - **Alpha Vantage validation fallback**
   - **Polygon optional U.S. premium satellite**

## Final Conclusion

The real answer is not “Yahoo or not Yahoo”.

The right answer for this system is:

- **Yahoo is useful, but not sufficient**
- **the scan should be hybrid**
- **the master database should move toward an official provider**
- and **the way to know if we missed real leaders is now measurable, not intuitive**
