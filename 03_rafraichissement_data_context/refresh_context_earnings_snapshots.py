from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import pandas as pd
import yfinance as yf
import yfinance.cache as yf_cache


ROOT = Path(__file__).resolve().parent
ACTIVE_PATH = ROOT / "data" / "extracts" / "apex_tickers_active.csv"
RESERVE_PATH = ROOT / "data" / "extracts" / "apex_tickers_reserve.csv"
DYNAMIC_DB_PATH = ROOT / "data" / "dynamic_universe" / "dynamic_universe_current.csv"
PROMOTION_QUEUE_PATH = ROOT / "data" / "dynamic_universe" / "dynamic_universe_promotion_queue.csv"
CONTEXT_CACHE_PATH = ROOT / "data" / "dynamic_universe" / "ticker_context_cache.json"
OHLCV_PATH = ROOT / "apex_ohlcv_full_2015_2026.csv"
SECTOR_BENCHMARKS_PATH = ROOT / "data" / "benchmarks" / "sector_benchmarks_ohlcv.csv"

CONTEXT_DIR = ROOT / "data" / "context"
CONTEXT_HISTORY_DIR = CONTEXT_DIR / "history"
EARNINGS_DIR = ROOT / "data" / "earnings"
EARNINGS_HISTORY_DIR = EARNINGS_DIR / "history"

CONTEXT_FIELDS = [
    "marketCap",
    "averageDailyVolume3Month",
    "sector",
    "sectorKey",
    "industry",
    "industryKey",
    "exchange",
    "fullExchangeName",
    "beta",
    "quoteType",
    "regularMarketPrice",
    "currency",
]
CONTEXT_REQUIRED_FIELDS = ["marketCap", "sectorKey", "exchange", "quoteType"]
EARNINGS_FIELDS = [
    "last_earnings_date",
    "next_earnings_date",
    "next_earnings_end_date",
    "days_since_last_earnings",
    "days_to_next_earnings",
    "mostRecentQuarter",
    "earningsQuarterlyGrowth",
    "earnings_blackout_5d",
    "earnings_blackout_10d",
    "post_earnings_5d",
    "post_earnings_10d",
]
PRIORITY_PROMOTION_STAGES = {"approved_live", "probation_live", "targeted_integration", "watch_queue"}
PRIORITY_DYNAMIC_STATUSES = {"approved", "prime_watch", "watch"}
PRIORITY_REFRESH_LIMIT = 1200
STALE_PRIORITY_DAYS = 3
STALE_STANDARD_DAYS = 14


def setup_yf_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    if hasattr(yf_cache, "set_cache_location"):
        yf_cache.set_cache_location(str(cache_dir))


def read_tickers_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return []
    return df["ticker"].dropna().astype(str).tolist()


def dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip().upper()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_json_dict(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, object]] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            out[str(key).strip().upper()] = dict(value)
    return out


def save_json_dict(path: Path, mapping: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")


def load_latest_row_map(path: Path) -> dict[str, dict[str, object]]:
    df = read_optional_csv(path)
    if df.empty or "ticker" not in df.columns:
        return {}
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return {
        str(row["ticker"]).strip().upper(): row.to_dict()
        for _, row in df.drop_duplicates("ticker", keep="last").iterrows()
    }


def load_db_priority_tickers() -> tuple[list[str], list[str]]:
    if not DYNAMIC_DB_PATH.exists():
        return [], []
    db = pd.read_csv(DYNAMIC_DB_PATH, low_memory=False)
    if "ticker" not in db.columns:
        return [], []
    db["ticker"] = db["ticker"].astype(str).str.upper()
    for col in ("promotion_stage", "dynamic_status"):
        if col not in db.columns:
            db[col] = ""
        db[col] = db[col].fillna("").astype(str)
    for col in ("promotion_score", "dynamic_conviction_score", "recent_score", "priority_score"):
        if col not in db.columns:
            db[col] = 0.0
        db[col] = pd.to_numeric(db[col], errors="coerce").fillna(0.0)

    ordered = db.sort_values(
        ["promotion_score", "dynamic_conviction_score", "recent_score", "priority_score"],
        ascending=[False, False, False, False],
    )
    priority = ordered.loc[
        ordered["promotion_stage"].isin(PRIORITY_PROMOTION_STAGES)
        | ordered["dynamic_status"].isin(PRIORITY_DYNAMIC_STATUSES),
        "ticker",
    ].dropna().astype(str).tolist()
    all_tickers = ordered["ticker"].dropna().astype(str).tolist()
    return dedupe_keep_order(priority), dedupe_keep_order(all_tickers)


def load_ohlcv_tickers() -> list[str]:
    if not OHLCV_PATH.exists():
        return []
    try:
        series = pd.read_csv(OHLCV_PATH, usecols=["ticker"])["ticker"]
    except Exception:
        return []
    return dedupe_keep_order(series.dropna().astype(str).str.upper().tolist())


def load_sector_benchmark_tickers() -> list[str]:
    df = read_optional_csv(SECTOR_BENCHMARKS_PATH)
    if df.empty or "ticker" not in df.columns:
        return []
    return dedupe_keep_order(df["ticker"].dropna().astype(str).str.upper().tolist())


def load_target_tickers() -> tuple[list[str], set[str]]:
    priority: list[str] = []
    broad: list[str] = []

    active = read_tickers_csv(ACTIVE_PATH)
    reserve = read_tickers_csv(RESERVE_PATH)
    promotion = read_tickers_csv(PROMOTION_QUEUE_PATH)
    db_priority, db_all = load_db_priority_tickers()
    context_latest = list(load_latest_row_map(CONTEXT_DIR / "ticker_context_latest.csv").keys())
    earnings_latest = list(load_latest_row_map(EARNINGS_DIR / "earnings_latest.csv").keys())
    context_cache = list(load_json_dict(CONTEXT_CACHE_PATH).keys())
    ohlcv_tickers = load_ohlcv_tickers()
    benchmark_tickers = load_sector_benchmark_tickers()

    priority.extend(active)
    priority.extend(reserve)
    priority.extend(promotion)
    priority.extend(db_priority)
    priority.extend(["SPY", "QQQ", "IWM"])

    broad.extend(db_all)
    broad.extend(context_cache)
    broad.extend(context_latest)
    broad.extend(earnings_latest)
    broad.extend(ohlcv_tickers)
    broad.extend(benchmark_tickers)
    broad.extend(["SPY", "QQQ", "IWM"])

    ordered = dedupe_keep_order(priority + broad)
    return ordered, set(dedupe_keep_order(priority))


def _parse_ts(value: object) -> pd.Timestamp | pd.NaT:
    if isinstance(value, (list, tuple, pd.Series)):
        for item in value:
            parsed = _parse_ts(item)
            if not pd.isna(parsed):
                return parsed
        return pd.NaT
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            items = value.tolist()
        except Exception:
            items = None
        if isinstance(items, list):
            return _parse_ts(items)
    if value in (None, "", 0):
        return pd.NaT
    try:
        return pd.to_datetime(int(float(value)), unit="s", utc=True).tz_convert(None).normalize()
    except Exception:
        try:
            return pd.to_datetime(value).normalize()
        except Exception:
            return pd.NaT


def _parse_calendar_date(value: object) -> pd.Timestamp | pd.NaT:
    if isinstance(value, list) and value:
        value = value[0]
    try:
        return pd.to_datetime(value).normalize()
    except Exception:
        return pd.NaT


def _has_ts(value: object) -> bool:
    return isinstance(value, pd.Timestamp) and not pd.isna(value)


def _merge_base_row(*rows: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if value is None:
                continue
            if isinstance(value, float) and pd.isna(value):
                continue
            if isinstance(value, str) and value == "":
                continue
            out[key] = value
    return out


def _normalize_ticker_map(mapping: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(k).strip().upper(): dict(v) for k, v in mapping.items() if isinstance(v, dict)}


def _days_since_snapshot(value: object, today: pd.Timestamp) -> int:
    try:
        ts = pd.to_datetime(value)
    except Exception:
        return 10_000
    if pd.isna(ts):
        return 10_000
    return abs((today.normalize() - ts.normalize()).days)


def _context_missing_key(row: dict[str, object]) -> bool:
    return any(pd.isna(row.get(col)) or str(row.get(col) or "").strip() == "" for col in CONTEXT_REQUIRED_FIELDS)


def _earnings_missing_key(row: dict[str, object]) -> bool:
    next_dt = str(row.get("next_earnings_date") or "").strip()
    last_dt = str(row.get("last_earnings_date") or "").strip()
    return next_dt == "" and last_dt == ""


def _context_row_from_base(ticker: str, today: pd.Timestamp, base: dict[str, object]) -> dict[str, object]:
    row = {"as_of": str(today.date()), "ticker": ticker}
    for field in CONTEXT_FIELDS:
        row[field] = base.get(field, pd.NA)
    return row


def _earnings_row_from_base(ticker: str, today: pd.Timestamp, base: dict[str, object]) -> dict[str, object]:
    row = {"as_of": str(today.date()), "ticker": ticker}
    for field in EARNINGS_FIELDS:
        row[field] = base.get(field, pd.NA if field.startswith("days_") else "")
    for field in ("earnings_blackout_5d", "earnings_blackout_10d", "post_earnings_5d", "post_earnings_10d"):
        value = row.get(field, 0)
        try:
            row[field] = 0 if pd.isna(value) or str(value).strip() == "" else int(float(value))
        except (TypeError, ValueError):
            row[field] = 0
    return row


def _apply_context_info(base: dict[str, object], info: dict[str, object]) -> None:
    for field in CONTEXT_FIELDS:
        value = info.get(field)
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if isinstance(value, str) and value == "":
            continue
        base[field] = value


def _apply_earnings_info(base: dict[str, object], info: dict[str, object], cal: dict[str, object], today: pd.Timestamp) -> None:
    last_earnings = _parse_ts(info.get("earningsTimestamp"))
    next_earnings = _parse_ts(info.get("earningsTimestampStart"))
    if not _has_ts(next_earnings):
        next_earnings = _parse_calendar_date(cal.get("Earnings Date"))
    next_earnings_end = _parse_ts(info.get("earningsTimestampEnd"))
    if not _has_ts(next_earnings_end) and _has_ts(next_earnings):
        next_earnings_end = next_earnings
    days_to_next = (next_earnings - today).days if _has_ts(next_earnings) else pd.NA
    days_since_last = (today - last_earnings).days if _has_ts(last_earnings) else pd.NA
    most_recent_quarter = _parse_ts(info.get("mostRecentQuarter"))

    updates = {
        "last_earnings_date": str(last_earnings.date()) if _has_ts(last_earnings) else "",
        "next_earnings_date": str(next_earnings.date()) if _has_ts(next_earnings) else "",
        "next_earnings_end_date": str(next_earnings_end.date()) if _has_ts(next_earnings_end) else "",
        "days_since_last_earnings": days_since_last,
        "days_to_next_earnings": days_to_next,
        "mostRecentQuarter": str(most_recent_quarter.date()) if _has_ts(most_recent_quarter) else "",
        "earningsQuarterlyGrowth": info.get("earningsQuarterlyGrowth"),
        "earnings_blackout_5d": int(pd.notna(days_to_next) and 0 <= int(days_to_next) <= 5),
        "earnings_blackout_10d": int(pd.notna(days_to_next) and 0 <= int(days_to_next) <= 10),
        "post_earnings_5d": int(pd.notna(days_since_last) and 0 <= int(days_since_last) <= 5),
        "post_earnings_10d": int(pd.notna(days_since_last) and 0 <= int(days_since_last) <= 10),
    }
    base.update(updates)


def write_context_cache(context_df: pd.DataFrame) -> int:
    if context_df.empty or "ticker" not in context_df.columns:
        return 0
    cache = load_json_dict(CONTEXT_CACHE_PATH)
    updated = 0
    fields = ["ticker", *CONTEXT_FIELDS]
    available = [col for col in fields if col in context_df.columns]
    for row in context_df[available].to_dict("records"):
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        payload = {
            key: value
            for key, value in row.items()
            if key != "ticker" and value is not None and not (isinstance(value, float) and pd.isna(value)) and not (isinstance(value, str) and value == "")
        }
        if not payload:
            continue
        previous = cache.get(ticker, {})
        merged = dict(previous)
        merged["ticker"] = ticker
        merged.update(payload)
        if merged != previous:
            cache[ticker] = merged
            updated += 1
    if updated:
        save_json_dict(CONTEXT_CACHE_PATH, cache)
    return updated


def build_rows(
    tickers: list[str],
    *,
    priority_tickers: set[str],
    max_refresh: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    context_rows: list[dict[str, object]] = []
    earnings_rows: list[dict[str, object]] = []
    today = pd.Timestamp(date.today()).normalize()
    context_latest_map = _normalize_ticker_map(load_latest_row_map(CONTEXT_DIR / "ticker_context_latest.csv"))
    earnings_latest_map = _normalize_ticker_map(load_latest_row_map(EARNINGS_DIR / "earnings_latest.csv"))
    context_cache_map = _normalize_ticker_map(load_json_dict(CONTEXT_CACHE_PATH))
    refreshed = 0
    refreshed_context = 0
    refreshed_earnings = 0
    reused_context = 0
    reused_earnings = 0

    for ticker in tickers:
        context_base = _merge_base_row(context_latest_map.get(ticker, {}), context_cache_map.get(ticker, {}))
        earnings_base = _merge_base_row(earnings_latest_map.get(ticker, {}))
        priority = ticker in priority_tickers
        context_stale_days = _days_since_snapshot(context_base.get("as_of"), today)
        earnings_stale_days = _days_since_snapshot(earnings_base.get("as_of"), today)
        context_needs_refresh = (
            not context_base
            or _context_missing_key(context_base)
            or (priority and context_stale_days > STALE_PRIORITY_DAYS)
            or (not priority and context_stale_days > STALE_STANDARD_DAYS)
        )
        earnings_needs_refresh = (
            not earnings_base
            or _earnings_missing_key(earnings_base)
            or (priority and earnings_stale_days > STALE_PRIORITY_DAYS)
            or (not priority and earnings_stale_days > STALE_STANDARD_DAYS)
        )
        can_refresh = refreshed < max_refresh and (priority or context_needs_refresh or earnings_needs_refresh)
        info: dict[str, object] = {}
        cal: dict[str, object] = {}
        did_refresh = False

        if can_refresh:
            tk = yf.Ticker(ticker)
            try:
                info = tk.get_info() or {}
            except Exception:
                info = {}
            if info:
                _apply_context_info(context_base, info)
                refreshed_context += 1
                did_refresh = True
            if priority or earnings_needs_refresh:
                try:
                    cal = tk.calendar or {}
                except Exception:
                    cal = {}
            if info or cal:
                _apply_earnings_info(earnings_base, info, cal, today)
                refreshed_earnings += 1
                did_refresh = True
        if did_refresh:
            refreshed += 1
        else:
            reused_context += int(bool(context_base))
            reused_earnings += int(bool(earnings_base))

        context_rows.append(_context_row_from_base(ticker, today, context_base))
        earnings_rows.append(_earnings_row_from_base(ticker, today, earnings_base))

    stats = {
        "refreshed_total": refreshed,
        "refreshed_context": refreshed_context,
        "refreshed_earnings": refreshed_earnings,
        "reused_context": reused_context,
        "reused_earnings": reused_earnings,
    }
    return pd.DataFrame(context_rows), pd.DataFrame(earnings_rows), stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh latest ticker context and earnings snapshots for active + dynamic candidates.")
    parser.add_argument("--tickers", default="", help="Optional comma-separated explicit tickers.")
    parser.add_argument("--max-refresh", type=int, default=PRIORITY_REFRESH_LIMIT, help="Maximum number of tickers to refresh from Yahoo on this run.")
    parser.add_argument("--passes", type=int, default=1, help="Number of incremental refresh passes to run.")
    args = parser.parse_args()

    setup_yf_cache(ROOT / ".yf_cache")
    if args.tickers:
        tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
        priority_tickers = set(tickers)
    else:
        tickers, priority_tickers = load_target_tickers()
    if not tickers:
        raise RuntimeError("No tickers to snapshot.")

    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    EARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    EARNINGS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    today_str = date.today().isoformat()
    context_latest = CONTEXT_DIR / "ticker_context_latest.csv"
    context_snapshot = CONTEXT_HISTORY_DIR / f"ticker_context_snapshot_{today_str}.csv"
    earnings_latest = EARNINGS_DIR / "earnings_latest.csv"
    earnings_snapshot = EARNINGS_HISTORY_DIR / f"earnings_snapshot_{today_str}.csv"
    summary_path = ROOT / "data" / "extracts" / "context_earnings_refresh_summary.csv"
    aggregate_stats = {
        "refreshed_total": 0,
        "refreshed_context": 0,
        "refreshed_earnings": 0,
        "reused_context": 0,
        "reused_earnings": 0,
    }
    context_df = pd.DataFrame()
    earnings_df = pd.DataFrame()

    for _ in range(max(1, int(args.passes))):
        context_df, earnings_df, stats = build_rows(
            tickers,
            priority_tickers=priority_tickers,
            max_refresh=max(0, int(args.max_refresh)),
        )
        context_df.to_csv(context_latest, index=False)
        earnings_df.to_csv(earnings_latest, index=False)
        for key in aggregate_stats:
            aggregate_stats[key] += int(stats.get(key, 0))

    context_df.to_csv(context_snapshot, index=False)
    earnings_df.to_csv(earnings_snapshot, index=False)
    cache_updates = write_context_cache(context_df)

    summary = pd.DataFrame(
        [
            {
                "as_of": today_str,
                "tickers": int(len(tickers)),
                "context_rows": int(len(context_df)),
                "earnings_rows": int(len(earnings_df)),
                "with_market_cap": int(context_df["marketCap"].notna().sum()),
                "with_sector_key": int(context_df["sectorKey"].notna().sum()),
                "with_next_earnings": int(earnings_df["next_earnings_date"].astype(str).str.len().gt(0).sum()),
                "earnings_blackout_5d_count": int(earnings_df["earnings_blackout_5d"].sum()),
                "priority_tickers": int(len(priority_tickers)),
                "passes": int(max(1, int(args.passes))),
                "cache_updates": int(cache_updates),
                "refreshed_total": int(aggregate_stats["refreshed_total"]),
                "refreshed_context": int(aggregate_stats["refreshed_context"]),
                "refreshed_earnings": int(aggregate_stats["refreshed_earnings"]),
                "reused_context": int(aggregate_stats["reused_context"]),
                "reused_earnings": int(aggregate_stats["reused_earnings"]),
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    print(f"Context latest: {context_latest}")
    print(f"Earnings latest: {earnings_latest}")
    print(f"Summary: {summary_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
