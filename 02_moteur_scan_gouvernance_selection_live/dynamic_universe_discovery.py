from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yfinance as yf
import yfinance.cache as yf_cache


ROOT = Path(__file__).resolve().parent
RESEARCH_DIR = ROOT / "research"
EXPORTS_DIR = RESEARCH_DIR / "exports"
REPORTS_DIR = RESEARCH_DIR / "reports"
CONTEXT_CACHE_PATH = ROOT / "data" / "dynamic_universe" / "ticker_context_cache.json"
ENGINE_PATH_CANDIDATES = [
    ROOT / "engine_bundle" / "run_v11_slot_quality" / "v11_slot_quality" / "apex_engine_slot_quality.py",
    ROOT / "engine_bundle" / "run_v11_slot_quality" / "apex_engine_slot_quality.py",
    ROOT / "engine_bundle" / "apex_engine_slot_quality.py",
]
CFG_PATH = ROOT / "BEST_ALGO_184.json"
CSV_PATH = ROOT / "apex_ohlcv_full_2015_2026.csv"
ACTIVE_PATH = ROOT / "data" / "extracts" / "apex_tickers_active.csv"
RESERVE_PATH = ROOT / "data" / "extracts" / "apex_tickers_reserve.csv"
HARD_EXCLUSIONS_PATH = ROOT / "data" / "extracts" / "apex_tickers_hard_exclusions.csv"
SELECTED_ADDS_PATH = ROOT / "data" / "dynamic_universe" / "dynamic_universe_selected_additions.csv"
SELECTED_DEMS_PATH = ROOT / "data" / "dynamic_universe" / "dynamic_universe_selected_demotions.csv"
SECTOR_BENCHMARKS_PATH = ROOT / "data" / "benchmarks" / "sector_benchmarks_ohlcv.csv"
CONTEXT_SNAPSHOT_PATH = ROOT / "data" / "context" / "ticker_context_latest.csv"
EARNINGS_SNAPSHOT_PATH = ROOT / "data" / "earnings" / "earnings_latest.csv"
CLOSED_TRADES_PATH = ROOT / "research" / "exports" / "baseline_184_closed_trade_episodes.csv"
MARKET_STRUCTURE_LATEST_PATH = ROOT / "data" / "context" / "market_structure_latest.csv"
TAXONOMY_PIT_LATEST_PATH = ROOT / "data" / "context" / "taxonomy_point_in_time_latest.csv"
LISTING_CORP_METADATA_PATH = ROOT / "data" / "extracts" / "listing_corporate_metadata.csv"

EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_REGIONS = (
    "us", "ca",
    "de", "fr", "gb", "it", "se", "nl", "es", "be", "ch", "dk", "no", "fi",
    "au", "hk", "jp", "sg",
    "kr", "tw", "in",
    "br", "mx",
    "za", "il",
)
DEFAULT_PREDEFINED_SCREENS = (
    "day_gainers",
    "most_actives",
    "growth_technology_stocks",
    "aggressive_small_caps",
    "small_cap_gainers",
)
DEFAULT_SECTOR_NAMES = (
    "Technology",
    "Industrials",
    "Communication Services",
    "Utilities",
    "Energy",
    "Basic Materials",
    "Healthcare",
    "Financial Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Real Estate",
)
SECTOR_NAME_TO_KEY = {
    "Technology": "technology",
    "Industrials": "industrials",
    "Communication Services": "communication-services",
    "Utilities": "utilities",
    "Energy": "energy",
    "Basic Materials": "basic-materials",
    "Healthcare": "healthcare",
    "Financial Services": "financial-services",
    "Consumer Cyclical": "consumer-cyclical",
    "Consumer Defensive": "consumer-defensive",
    "Real Estate": "real-estate",
}
SECTOR_KEY_TO_BENCHMARK = {
    "technology": "XLK",
    "industrials": "XLI",
    "communication-services": "XLC",
    "utilities": "XLU",
    "energy": "XLE",
    "basic-materials": "XLB",
    "healthcare": "XLV",
    "financial-services": "XLF",
    "consumer-cyclical": "XLY",
    "consumer-defensive": "XLP",
    "real-estate": "XLRE",
    "semiconductors": "SMH",
    "metals-mining": "XME",
    "gold-miners": "GDX",
    "silver-miners": "SIL",
    "market": "SPY",
}
BROKER_PROFILE = "xtb"
DEFAULT_MIN_MARKET_CAP = 500_000_000
DEFAULT_MIN_PRICE = 5.0
DEFAULT_MIN_ADV = 1_000_000
METADATA_REPAIR_LIMIT = 250
CONTEXT_CACHE_SAVE_EVERY = 20
RETEST_RECENT_ABS_DELTA = 0.75
RETEST_RECENT_REL_DELTA = 0.25
RETEST_COMPAT_DELTA = 0.75
RETEST_RANK_DELTA = 5.0
RETEST_TOP5_SHARE_DELTA = 0.08
RETEST_RELATIVE_SCORE_DELTA = 0.25
DATA_QUALITY_PENALTY = 1.2
BROKER_UNSUPPORTED_EXCHANGE_CODES = {"PNK", "OQB", "OQX", "OTC", "GREY"}
BROKER_SUPPORTED_EXCHANGE_CODES = {
    "NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "BTS", "BZX", "IEX",
    "TOR", "VAN",
    "PAR", "AMS", "BRU", "LSE", "MIL", "MC", "SWX", "VTX", "STO", "HEL", "CPH", "OSL",
    "ETR", "FRA", "STU", "HAM", "MUN", "BER", "DUS",
    "WSE", "VIE", "PRA", "BUD",
    "HKG", "TSE", "JPX", "SES", "ASX",
}
BROKER_SUPPORTED_EXCHANGE_NAME_PARTS = (
    "Nasdaq", "NYSE", "NYSEArca", "BATS", "IEX",
    "Toronto", "TSX",
    "Paris", "Amsterdam", "Brussels", "London", "Milan", "Madrid",
    "Swiss", "Stockholm", "Helsinki", "Copenhagen", "Oslo",
    "Xetra", "Frankfurt", "Tradegate", "Warsaw", "Vienna", "Prague", "Budapest",
    "Tokyo", "Hong Kong", "Singapore", "Australian",
)
CONSTITUENT_PROXY_ETFS: tuple[tuple[str, str], ...] = (
    ("SPY", "us_sp500_core"),
    ("IVV", "us_sp500_alt"),
    ("VOO", "us_sp500_alt2"),
    ("QQQ", "us_nasdaq100_core"),
    ("IWM", "us_russell2000_core"),
    ("MDY", "us_midcap400_core"),
    ("VTI", "us_total_market"),
    ("VGK", "europe_broad"),
    ("FEZ", "eurozone_broad"),
    ("EWJ", "japan_core"),
    ("EWH", "hongkong_core"),
    ("EWT", "taiwan_core"),
    ("EWY", "korea_core"),
    ("INDA", "india_core"),
    ("EWA", "australia_core"),
    ("EWC", "canada_core"),
    ("EWZ", "brazil_core"),
    ("EWW", "mexico_core"),
    ("EZA", "south_africa_core"),
    ("EIS", "israel_core"),
    ("VWO", "em_broad"),
    ("EEM", "em_broad_alt"),
    ("FXI", "china_largecap"),
    ("MCHI", "china_broad"),
)


@dataclass
class CandidateData:
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    volume: pd.Series
    info: dict[str, object]


_TICKER_CONTEXT_CACHE: dict[str, dict[str, object]] | None = None
_TICKER_CONTEXT_CACHE_DIRTY = 0
_CONTEXT_SNAPSHOT_MAP: dict[str, dict[str, object]] | None = None
_EARNINGS_SNAPSHOT_MAP: dict[str, dict[str, object]] | None = None
_SECTOR_BENCHMARK_CLOSE_MAP: dict[str, pd.Series] | None = None
_LATEST_FEATURE_SNAPSHOT: pd.DataFrame | None = None
_SETUP_CACHE: dict[str, object] | None = None
_MARKET_STRUCTURE_LATEST: pd.DataFrame | None = None
_TAXONOMY_PIT_LATEST: pd.DataFrame | None = None
_LISTING_CORP_METADATA: pd.DataFrame | None = None
_STRUCTURAL_SCAN_FRAME: pd.DataFrame | None = None


def setup_yf_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    if hasattr(yf_cache, "set_cache_location"):
        yf_cache.set_cache_location(str(cache_dir))


def normalize_downloaded_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df.rename(columns=lambda c: str(c).lower().replace(" ", "_"))


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_market_structure_latest() -> pd.DataFrame:
    global _MARKET_STRUCTURE_LATEST
    if _MARKET_STRUCTURE_LATEST is not None:
        return _MARKET_STRUCTURE_LATEST
    df = read_optional_csv(MARKET_STRUCTURE_LATEST_PATH)
    if df.empty or "ticker" not in df.columns:
        _MARKET_STRUCTURE_LATEST = pd.DataFrame()
        return _MARKET_STRUCTURE_LATEST
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str)
    rename_map = {
        "date": "pit_market_structure_date",
        "market_cap_current": "pit_market_cap_current",
        "shares_outstanding_current": "pit_shares_outstanding_current",
        "float_shares_current": "pit_float_shares_current",
        "implied_free_float_ratio_current": "pit_free_float_ratio_current",
        "coverage_ratio": "pit_market_structure_coverage_ratio",
        "currency": "pit_market_structure_currency",
    }
    keep = ["ticker", *rename_map.keys()]
    available = [col for col in keep if col in out.columns]
    out = out[available].rename(columns=rename_map)
    _MARKET_STRUCTURE_LATEST = out.drop_duplicates("ticker", keep="last")
    return _MARKET_STRUCTURE_LATEST


def load_taxonomy_pit_latest() -> pd.DataFrame:
    global _TAXONOMY_PIT_LATEST
    if _TAXONOMY_PIT_LATEST is not None:
        return _TAXONOMY_PIT_LATEST
    df = read_optional_csv(TAXONOMY_PIT_LATEST_PATH)
    if df.empty or "ticker" not in df.columns:
        _TAXONOMY_PIT_LATEST = pd.DataFrame()
        return _TAXONOMY_PIT_LATEST
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str)
    rename_map = {
        "as_of": "pit_taxonomy_as_of",
        "cluster_key": "pit_cluster_key",
        "cluster_source": "pit_cluster_source",
        "sectorKey": "pit_sector_key",
        "industryKey": "pit_industry_key",
    }
    keep = ["ticker", *rename_map.keys()]
    available = [col for col in keep if col in out.columns]
    out = out[available].rename(columns=rename_map)
    _TAXONOMY_PIT_LATEST = out.drop_duplicates("ticker", keep="last")
    return _TAXONOMY_PIT_LATEST


def load_listing_corporate_metadata() -> pd.DataFrame:
    global _LISTING_CORP_METADATA
    if _LISTING_CORP_METADATA is not None:
        return _LISTING_CORP_METADATA
    df = read_optional_csv(LISTING_CORP_METADATA_PATH)
    if df.empty or "ticker" not in df.columns:
        _LISTING_CORP_METADATA = pd.DataFrame()
        return _LISTING_CORP_METADATA
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str)
    rename_map = {
        "listing_age_days": "pit_listing_age_days",
        "bars_total": "pit_bars_total",
        "recent_split_365d": "pit_recent_split_365d",
        "recent_dividend_365d": "pit_recent_dividend_365d",
    }
    keep = ["ticker", *rename_map.keys()]
    available = [col for col in keep if col in out.columns]
    out = out[available].rename(columns=rename_map)
    _LISTING_CORP_METADATA = out.drop_duplicates("ticker", keep="last")
    return _LISTING_CORP_METADATA


def _load_context_snapshot_map() -> dict[str, dict[str, object]]:
    global _CONTEXT_SNAPSHOT_MAP
    if _CONTEXT_SNAPSHOT_MAP is not None:
        return _CONTEXT_SNAPSHOT_MAP
    df = read_optional_csv(CONTEXT_SNAPSHOT_PATH)
    if df.empty or "ticker" not in df.columns:
        _CONTEXT_SNAPSHOT_MAP = {}
        return _CONTEXT_SNAPSHOT_MAP
    df["ticker"] = df["ticker"].astype(str)
    _CONTEXT_SNAPSHOT_MAP = {
        str(row["ticker"]): row.to_dict()
        for _, row in df.drop_duplicates("ticker", keep="last").iterrows()
    }
    return _CONTEXT_SNAPSHOT_MAP


def _load_earnings_snapshot_map() -> dict[str, dict[str, object]]:
    global _EARNINGS_SNAPSHOT_MAP
    if _EARNINGS_SNAPSHOT_MAP is not None:
        return _EARNINGS_SNAPSHOT_MAP
    df = read_optional_csv(EARNINGS_SNAPSHOT_PATH)
    if df.empty or "ticker" not in df.columns:
        _EARNINGS_SNAPSHOT_MAP = {}
        return _EARNINGS_SNAPSHOT_MAP
    df["ticker"] = df["ticker"].astype(str)
    _EARNINGS_SNAPSHOT_MAP = {
        str(row["ticker"]): row.to_dict()
        for _, row in df.drop_duplicates("ticker", keep="last").iterrows()
    }
    return _EARNINGS_SNAPSHOT_MAP


def load_sector_benchmark_close_map() -> dict[str, pd.Series]:
    global _SECTOR_BENCHMARK_CLOSE_MAP
    if _SECTOR_BENCHMARK_CLOSE_MAP is not None:
        return _SECTOR_BENCHMARK_CLOSE_MAP
    df = read_optional_csv(SECTOR_BENCHMARKS_PATH)
    if df.empty or "ticker" not in df.columns or "date" not in df.columns or "adj_close" not in df.columns:
        _SECTOR_BENCHMARK_CLOSE_MAP = {}
        return _SECTOR_BENCHMARK_CLOSE_MAP
    df["date"] = pd.to_datetime(df["date"])
    out: dict[str, pd.Series] = {}
    for ticker, group in df.groupby("ticker", sort=False):
        series = (
            group.sort_values("date")
            .drop_duplicates("date", keep="last")
            .set_index("date")["adj_close"]
            .astype(float)
        )
        out[str(ticker)] = series
    _SECTOR_BENCHMARK_CLOSE_MAP = out
    return _SECTOR_BENCHMARK_CLOSE_MAP


def load_latest_feature_snapshot() -> pd.DataFrame:
    global _LATEST_FEATURE_SNAPSHOT
    if _LATEST_FEATURE_SNAPSHOT is not None:
        return _LATEST_FEATURE_SNAPSHOT
    if not CSV_PATH.exists():
        _LATEST_FEATURE_SNAPSHOT = pd.DataFrame()
        return _LATEST_FEATURE_SNAPSHOT
    usecols = [
        "date",
        "ticker",
        "close",
        "open",
        "high",
        "low",
        "volume",
        "return_21d",
        "return_63d",
        "return_126d",
        "return_252d",
        "dist_high_252_pct",
        "dist_sma50_pct",
        "dist_sma200_pct",
        "adv_20",
        "adv_60",
        "dollar_volume",
        "atr_pct",
        "volatility_60d",
        "momentum_score",
        "above_sma200",
        "rel_return_63d_spy",
        "rel_return_126d_spy",
        "rel_return_252d_spy",
        "residual_return_63d",
        "residual_return_126d",
        "residual_return_252d",
        "residual_momentum_score",
        "overnight_return_1d",
        "intraday_return_1d",
        "gap_pct",
        "gap_zscore_20",
        "gap_fill_share",
        "rel_volume_20",
        "volume_zscore_20",
        "dollar_volume_zscore_20",
        "close_in_range",
        "efficiency_ratio_20",
        "efficiency_ratio_60",
        "corr_63_spy",
        "downside_beta_63_spy",
        "red_flags_count",
    ]
    try:
        df = pd.read_csv(CSV_PATH, usecols=usecols)
    except Exception:
        _LATEST_FEATURE_SNAPSHOT = pd.DataFrame()
        return _LATEST_FEATURE_SNAPSHOT
    if df.empty or "ticker" not in df.columns:
        _LATEST_FEATURE_SNAPSHOT = pd.DataFrame()
        return _LATEST_FEATURE_SNAPSHOT
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str)
    df = df.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
    latest = df.groupby("ticker", sort=False).tail(1).copy()
    age_stats = (
        df.groupby("ticker", sort=False)
        .agg(
            bars_total=("close", "count"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
    )
    age_stats["listing_age_days"] = (age_stats["last_date"] - age_stats["first_date"]).dt.days
    latest = latest.merge(age_stats[["ticker", "bars_total", "first_date", "listing_age_days"]], on="ticker", how="left")
    _LATEST_FEATURE_SNAPSHOT = latest
    return _LATEST_FEATURE_SNAPSHOT


def load_context_cache_frame() -> pd.DataFrame:
    cache = _load_ticker_context_cache()
    if not cache:
        return pd.DataFrame()
    rows = [dict(value) for value in cache.values() if isinstance(value, dict)]
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if "ticker" not in out.columns:
        return pd.DataFrame()
    out["ticker"] = out["ticker"].astype(str)
    return out.drop_duplicates("ticker", keep="last")


def load_sector_regime_frame() -> pd.DataFrame:
    df = read_optional_csv(SECTOR_BENCHMARKS_PATH)
    if df.empty or "ticker" not in df.columns or "date" not in df.columns:
        return pd.DataFrame()
    use_cols = [
        "date",
        "ticker",
        "benchmark_group",
        "sector_key",
        "return_21d",
        "return_63d",
        "return_126d",
        "return_252d",
        "above_sma200",
    ]
    available = [col for col in use_cols if col in df.columns]
    df = df[available].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["ticker", "date"])
    latest = df.groupby("ticker", sort=False).tail(1).copy()
    if "benchmark_group" in latest.columns:
        latest = latest.loc[latest["benchmark_group"].fillna("").ne("market")].copy()
    for col in ("return_21d", "return_63d", "return_126d", "return_252d"):
        if col not in latest.columns:
            latest[col] = 0.0
        latest[col] = pd.to_numeric(latest[col], errors="coerce").fillna(0.0)
    latest["above_sma200"] = pd.to_numeric(latest.get("above_sma200"), errors="coerce").fillna(0.0)
    latest["sector_regime_score"] = (
        1.6 * latest["return_21d"]
        + 2.2 * latest["return_63d"]
        + 2.0 * latest["return_126d"]
        + 1.4 * latest["return_252d"]
        + 0.15 * latest["above_sma200"]
    )
    return latest.sort_values("sector_regime_score", ascending=False)


def load_historical_winner_frame(topn: int = 24) -> pd.DataFrame:
    df = read_optional_csv(CLOSED_TRADES_PATH)
    if df.empty or "Ticker" not in df.columns:
        return pd.DataFrame()
    df["PnLEUR"] = pd.to_numeric(df.get("PnLEUR"), errors="coerce").fillna(0.0)
    df["PnLPct"] = pd.to_numeric(df.get("PnLPct"), errors="coerce").fillna(0.0)
    winners = df.loc[df["PnLEUR"] > 0].copy()
    if winners.empty:
        return pd.DataFrame()
    winners["winner_score"] = (
        0.65 * winners["PnLEUR"].rank(pct=True, method="average")
        + 0.35 * winners["PnLPct"].rank(pct=True, method="average")
    )
    winners = (
        winners.sort_values(["winner_score", "PnLEUR", "PnLPct"], ascending=[False, False, False])
        .drop_duplicates("Ticker", keep="first")
        .head(topn)
        .copy()
    )
    winners["ticker"] = winners["Ticker"].astype(str)
    return winners


def broker_exchange_supported(row: pd.Series | dict) -> bool:
    exchange = str(row.get("exchange") or "").strip().upper()
    full_name = str(row.get("fullExchangeName") or "").strip()
    if exchange in BROKER_UNSUPPORTED_EXCHANGE_CODES or "OTC" in full_name.upper():
        return False
    if exchange in BROKER_SUPPORTED_EXCHANGE_CODES:
        return True
    if any(part.lower() in full_name.lower() for part in BROKER_SUPPORTED_EXCHANGE_NAME_PARTS):
        return True
    return exchange == "" and full_name == ""


def sector_benchmark_for_row(row: pd.Series | dict) -> str:
    sector_key = str(row.get("sectorKey") or row.get("sector_key") or "").strip()
    if sector_key in SECTOR_KEY_TO_BENCHMARK:
        return SECTOR_KEY_TO_BENCHMARK[sector_key]
    sector_name = str(row.get("sector") or "").strip()
    normalized = SECTOR_NAME_TO_KEY.get(sector_name)
    if normalized and normalized in SECTOR_KEY_TO_BENCHMARK:
        return SECTOR_KEY_TO_BENCHMARK[normalized]
    return "SPY"


def _load_ticker_context_cache() -> dict[str, dict[str, object]]:
    global _TICKER_CONTEXT_CACHE
    if _TICKER_CONTEXT_CACHE is not None:
        return _TICKER_CONTEXT_CACHE
    if CONTEXT_CACHE_PATH.exists():
        try:
            raw = json.loads(CONTEXT_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                _TICKER_CONTEXT_CACHE = {
                    str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)
                }
                return _TICKER_CONTEXT_CACHE
        except json.JSONDecodeError:
            pass
    _TICKER_CONTEXT_CACHE = {}
    return _TICKER_CONTEXT_CACHE


def _save_ticker_context_cache(force: bool = False) -> None:
    global _TICKER_CONTEXT_CACHE_DIRTY
    cache = _load_ticker_context_cache()
    if not force and _TICKER_CONTEXT_CACHE_DIRTY < CONTEXT_CACHE_SAVE_EVERY:
        return
    CONTEXT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTEXT_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    _TICKER_CONTEXT_CACHE_DIRTY = 0


def _store_ticker_context(ticker: str, ctx: dict[str, object]) -> dict[str, object]:
    global _TICKER_CONTEXT_CACHE_DIRTY
    cache = _load_ticker_context_cache()
    cache[str(ticker)] = dict(ctx)
    _TICKER_CONTEXT_CACHE_DIRTY += 1
    _save_ticker_context_cache()
    return cache[str(ticker)]


def read_tickers_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return []
    return df["ticker"].dropna().astype(str).tolist()


def dedupe_keep_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def resolve_engine_path() -> Path:
    for candidate in ENGINE_PATH_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find apex_engine_slot_quality.py in engine_bundle. "
        f"Tried: {[str(p) for p in ENGINE_PATH_CANDIDATES]}"
    )


def load_engine():
    engine_path = resolve_engine_path()
    spec = importlib.util.spec_from_file_location("apex_engine_slot_quality", engine_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_fallback_universe(exclusions: Sequence[str]) -> list[str]:
    active = read_tickers_csv(ACTIVE_PATH)
    adds = read_tickers_csv(SELECTED_ADDS_PATH)
    dems = set(read_tickers_csv(SELECTED_DEMS_PATH))
    base = active or []
    universe = []
    seen: set[str] = set()
    for ticker in list(base) + list(adds) + ["SPY"]:
        if ticker in dems or ticker in exclusions or ticker in seen:
            continue
        seen.add(ticker)
        universe.append(ticker)
    return universe


def load_prices_from_yfinance(engine, tickers: Sequence[str]):
    data = yf.download(
        tickers=list(tickers),
        group_by="column",
        auto_adjust=False,
        threads=True,
        progress=False,
        interval="1d",
        start="2014-01-01",
    )
    if data is None or len(data) == 0 or not isinstance(data.columns, pd.MultiIndex):
        raise RuntimeError("yfinance fallback returned empty or unexpected dataset.")
    lvl0 = list(data.columns.get_level_values(0).unique())
    if "Open" not in lvl0 and "Close" not in lvl0:
        data = data.swaplevel(axis=1).sort_index(axis=1)

    def _get_field(field: str) -> pd.DataFrame:
        if field not in data.columns.get_level_values(0):
            return pd.DataFrame(index=data.index)
        df_f = data[field].copy()
        df_f.columns = [str(c) for c in df_f.columns]
        return df_f

    o = _get_field("Open")
    c = _get_field("Close")
    cols = sorted(list(set(o.columns) | set(c.columns)))
    o = o.reindex(columns=cols)
    c = c.reindex(columns=cols)
    return engine.Prices(open=o, close=c)


def download_history_batch(tickers: Sequence[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    symbols = dedupe_keep_order(tickers)
    if not symbols:
        return {}
    try:
        data = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        return {}
    if data is None or len(data) == 0:
        return {}
    if not isinstance(data.columns, pd.MultiIndex):
        if len(symbols) == 1:
            return {symbols[0]: normalize_downloaded_columns(data.copy())}
        return {}

    out: dict[str, pd.DataFrame] = {}
    lvl0 = set(data.columns.get_level_values(0))
    lvl1 = set(data.columns.get_level_values(1))
    if any(symbol in lvl0 for symbol in symbols):
        for symbol in symbols:
            if symbol in lvl0:
                out[symbol] = normalize_downloaded_columns(data[symbol].copy())
    elif any(symbol in lvl1 for symbol in symbols):
        swapped = data.swaplevel(axis=1).sort_index(axis=1)
        for symbol in symbols:
            if symbol in swapped.columns.get_level_values(0):
                out[symbol] = normalize_downloaded_columns(swapped[symbol].copy())
    return out


def load_setup():
    global _SETUP_CACHE
    cache_key = {
        "cfg_mtime_ns": CFG_PATH.stat().st_mtime_ns if CFG_PATH.exists() else None,
        "csv_mtime_ns": CSV_PATH.stat().st_mtime_ns if CSV_PATH.exists() else None,
        "active_mtime_ns": ACTIVE_PATH.stat().st_mtime_ns if ACTIVE_PATH.exists() else None,
    }
    if _SETUP_CACHE and _SETUP_CACHE.get("key") == cache_key:
        return _SETUP_CACHE["value"]
    engine = load_engine()
    cfg_doc = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    cfg = dict(cfg_doc["cfg"])
    pp = dict(cfg_doc["pp"])
    exclusions = cfg_doc.get("exclusions", [])
    if CSV_PATH.exists():
        prices = engine.load_prices_from_csv(str(CSV_PATH))
    else:
        prices = load_prices_from_yfinance(engine, build_fallback_universe(exclusions))
    prices = engine.Prices(
        open=prices.open.drop(columns=exclusions, errors="ignore"),
        close=prices.close.drop(columns=exclusions, errors="ignore"),
    )
    baseline_active = current_universe_state()[0]
    staged_additions = set(cfg_doc.get("staged_activation_dates", {}).keys())
    allowed = set(baseline_active) | staged_additions | {"SPY"}
    keep_cols = [col for col in prices.close.columns if col in allowed]
    prices = engine.Prices(
        open=prices.open.reindex(columns=keep_cols),
        close=prices.close.reindex(columns=keep_cols),
    )
    result = (engine, cfg_doc, cfg, pp, prices)
    _SETUP_CACHE = {"key": cache_key, "value": result}
    return result


def run_metrics(engine, prices, cfg: dict, pp: dict, start: str, end: str):
    local_cfg = dict(cfg)
    for key in (
        "pp_enabled",
        "mie_enabled",
        "pp_mfe_trigger",
        "pp_trail_dd",
        "pp_min_days_after_arm",
        "mie_rs_th",
        "mie_min_hold",
    ):
        local_cfg.pop(key, None)
    eq, trades, out = engine.run_backtest(
        prices,
        start,
        end,
        pp_enabled=True,
        mie_enabled=True,
        pp_mfe_trigger=float(pp["pp_mfe_trigger"]),
        pp_trail_dd=float(pp["pp_trail_dd"]),
        pp_min_days_after_arm=int(pp["pp_min_days_after_arm"]),
        mie_rs_th=float(pp["mie_rs_th"]),
        mie_min_hold=int(pp["mie_min_hold"]),
        **local_cfg,
    )
    return eq, trades, dict(out)


def row_from_run(name: str, full: dict, oos: dict, extra: dict | None = None) -> dict:
    row = {
        "name": name,
        "full_roi_pct": float(full["ROI_%"]),
        "full_cagr_pct": float(full["CAGR_%"]),
        "full_maxdd_pct": float(full["MaxDD_%"]),
        "full_sharpe": float(full["Sharpe"]),
        "full_orders": int(full["Orders"]),
        "oos_roi_pct": float(oos["ROI_%"]),
        "oos_cagr_pct": float(oos["CAGR_%"]),
        "oos_maxdd_pct": float(oos["MaxDD_%"]),
        "oos_sharpe": float(oos["Sharpe"]),
        "oos_orders": int(oos["Orders"]),
    }
    if extra:
        row.update(extra)
    return row


def compute_universe_features(close: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sma_win = int(cfg["sma_win"])
    r63 = close / close.shift(63) - 1.0
    r126 = close / close.shift(126) - 1.0
    r252 = close / close.shift(252) - 1.0
    score = cfg["w_r63"] * r63 + cfg["w_r126"] * r126 + cfg["w_r252"] * r252
    sma220 = close.rolling(sma_win, min_periods=sma_win).mean()
    above = close > sma220
    rank = score.rank(axis=1, ascending=False, method="min")
    eligible = score.notna() & above

    latest_date = close.index.max()
    rows = []
    for ticker in close.columns:
        latest_close = close.at[latest_date, ticker] if ticker in close.columns else np.nan
        rows.append(
            {
                "ticker": ticker,
                "bars": int(close[ticker].notna().sum()),
                "days_top15_trend": int((eligible[ticker] & (rank[ticker] <= 15)).sum()),
                "days_top5_trend": int((eligible[ticker] & (rank[ticker] <= 5)).sum()),
                "latest_close": float(latest_close) if pd.notna(latest_close) else np.nan,
                "latest_score": float(score.at[latest_date, ticker]) if pd.notna(score.at[latest_date, ticker]) else np.nan,
                "latest_rank": float(rank.at[latest_date, ticker]) if pd.notna(rank.at[latest_date, ticker]) else np.nan,
                "latest_r63": float(r63.at[latest_date, ticker]) if pd.notna(r63.at[latest_date, ticker]) else np.nan,
                "latest_r252": float(r252.at[latest_date, ticker]) if pd.notna(r252.at[latest_date, ticker]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_realized_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    positions: dict[str, dict[str, float]] = {}
    realized = defaultdict(float)
    buys = defaultdict(int)
    sells = defaultdict(int)

    for row in trades.itertuples(index=False):
        ticker = str(row.Ticker)
        side = str(row.Side)
        shares = float(row.Shares)
        price = float(row.Price)
        fee = float(row.Fee)
        if side == "BUY":
            state = positions.setdefault(ticker, {"shares": 0.0, "cost": 0.0})
            state["shares"] += shares
            state["cost"] += shares * price
            buys[ticker] += 1
            continue

        state = positions.setdefault(ticker, {"shares": 0.0, "cost": 0.0})
        avg_cost = state["cost"] / state["shares"] if state["shares"] > 0 else 0.0
        realized[ticker] += shares * price - fee - shares * avg_cost
        state["shares"] -= shares
        state["cost"] -= shares * avg_cost
        sells[ticker] += 1

    rows = []
    tickers = set(realized) | set(buys) | set(sells) | set(positions)
    for ticker in sorted(tickers):
        rows.append(
            {
                "ticker": ticker,
                "realized_pnl_eur": float(realized[ticker]),
                "buy_count": int(buys[ticker]),
                "sell_count": int(sells[ticker]),
            }
        )
    return pd.DataFrame(rows)


def parse_ticker_csv(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return set()
    return set(df["ticker"].dropna().astype(str))


def current_universe_state() -> tuple[set[str], set[str], set[str]]:
    return parse_ticker_csv(ACTIVE_PATH), parse_ticker_csv(RESERVE_PATH), parse_ticker_csv(HARD_EXCLUSIONS_PATH)


def latest_raw_leaders(prices, cfg: dict, topn: int = 10) -> list[str]:
    feats = compute_universe_features(prices.close, cfg)
    ranked = feats.dropna(subset=["latest_rank"]).sort_values(["latest_rank", "latest_score"], ascending=[True, False])
    return ranked["ticker"].head(topn).tolist()


def rank_within_series(series: pd.Series) -> pd.Series:
    valid = series.fillna(series.median(skipna=True))
    if valid.nunique(dropna=True) <= 1:
        return pd.Series(np.ones(len(series)) * 0.5, index=series.index)
    return valid.rank(pct=True, method="average")


def remember_candidate(records: dict[str, dict], symbol: str, source_type: str, source_name: str, row: dict | pd.Series | None = None, extra: dict | None = None) -> None:
    if not symbol or not isinstance(symbol, str):
        return
    symbol = symbol.strip().upper()
    if not symbol:
        return
    rec = records.setdefault(
        symbol,
        {
            "ticker": symbol,
            "source_names": set(),
            "source_types": set(),
            "seed_hits": set(),
            "etf_hits": set(),
        },
    )
    rec["source_names"].add(source_name)
    rec["source_types"].add(source_type)
    if source_type == "seed_peer":
        rec["seed_hits"].add(source_name)
    if source_type == "etf_holding":
        rec["etf_hits"].add(source_name)
    data = {}
    if row is not None:
        data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    if extra:
        data.update(extra)
    for key, value in data.items():
        if key in {"ticker", "source_names", "source_types", "seed_hits", "etf_hits"}:
            continue
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if key not in rec or rec[key] in (None, "", np.nan):
            rec[key] = value
        elif key in {"marketCap", "averageDailyVolume3Month"} and value:
            rec[key] = max(float(rec.get(key, 0.0) or 0.0), float(value))


def quote_is_candidate(row: dict, min_market_cap: float, min_price: float, min_adv: float) -> bool:
    if not row:
        return False
    if str(row.get("quoteType", "")).upper() != "EQUITY":
        return False
    symbol = str(row.get("symbol", "")).strip().upper()
    if not symbol or symbol.startswith("^") or "=" in symbol:
        return False
    market_cap = float(row.get("marketCap") or 0.0)
    price = float(row.get("regularMarketPrice") or 0.0)
    adv = float(row.get("averageDailyVolume3Month") or row.get("regularMarketVolume") or 0.0)
    return market_cap >= min_market_cap and price >= min_price and adv >= min_adv


def fetch_screen_rows(query: str | object, *, count: int, max_pages: int, sort_field: str | None = None, sort_asc: bool | None = None) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    for _ in range(max_pages):
        if isinstance(query, str):
            response = yf.screen(query, count=count)
        else:
            response = yf.screen(query, offset=offset, size=count, sortField=sort_field, sortAsc=sort_asc)
        quotes = response.get("quotes", []) if isinstance(response, dict) else []
        if not quotes:
            break
        rows.extend(quotes)
        offset += len(quotes)
        total = int(response.get("total", len(rows))) if isinstance(response, dict) else len(rows)
        if isinstance(query, str) or offset >= total:
            break
    return rows


def discover_predefined_screens(records: dict[str, dict], screen_names: Sequence[str], min_market_cap: float, min_price: float, min_adv: float, count: int) -> None:
    if count <= 0:
        return
    for screen_name in screen_names:
        try:
            rows = fetch_screen_rows(screen_name, count=count, max_pages=1)
        except Exception:
            continue
        for row in rows:
            if quote_is_candidate(row, min_market_cap, min_price, min_adv):
                remember_candidate(records, row.get("symbol", ""), "predefined_screen", screen_name, row=row)


def discover_custom_region_sector_screens(
    records: dict[str, dict],
    regions: Sequence[str],
    sector_names: Sequence[str],
    min_market_cap: float,
    min_price: float,
    min_adv: float,
    count: int,
    max_pages: int,
) -> None:
    if count <= 0 or max_pages <= 0:
        return
    for region in regions:
        for sector_name in sector_names:
            query = yf.EquityQuery(
                "and",
                [
                    yf.EquityQuery("eq", ["region", region]),
                    yf.EquityQuery("eq", ["sector", sector_name]),
                    yf.EquityQuery("gte", ["intradayprice", min_price]),
                    yf.EquityQuery("gte", ["intradaymarketcap", min_market_cap]),
                ],
            )
            try:
                rows = fetch_screen_rows(query, count=count, max_pages=max_pages, sort_field="dayvolume", sort_asc=False)
            except Exception:
                continue
            for row in rows:
                if quote_is_candidate(row, min_market_cap, min_price, min_adv):
                    remember_candidate(
                        records,
                        row.get("symbol", ""),
                        "custom_screen",
                        f"{region}:{sector_name}",
                        row=row,
                    )


def discover_sector_expansion(records: dict[str, dict], sector_keys: Sequence[str], top_etfs_per_sector: int, etf_holding_count: int) -> None:
    seen_etfs: set[str] = set()
    for sector_key in sector_keys:
        try:
            sector = yf.Sector(sector_key)
            top_companies = sector.top_companies
        except Exception:
            continue
        if top_companies is not None and not top_companies.empty:
            for symbol, row in top_companies.reset_index().rename(columns={"index": "symbol"}).set_index("symbol").iterrows():
                remember_candidate(
                    records,
                    symbol,
                    "sector_top_company",
                    sector_key,
                    row=row,
                    extra={"sector_key": sector_key},
                )
        top_etfs = getattr(sector, "top_etfs", {}) or {}
        for etf_symbol in list(top_etfs.keys())[:top_etfs_per_sector]:
            if etf_symbol in seen_etfs:
                continue
            seen_etfs.add(etf_symbol)
            try:
                holdings = yf.Ticker(etf_symbol).funds_data.top_holdings
            except Exception:
                continue
            if holdings is None or holdings.empty:
                continue
            holding_rows = holdings.reset_index().rename(columns={"Symbol": "symbol"}).head(etf_holding_count)
            for row in holding_rows.to_dict("records"):
                remember_candidate(
                    records,
                    str(row.get("symbol", "")),
                    "etf_holding",
                    etf_symbol,
                    row=row,
                    extra={"sector_key": sector_key, "etf_name": top_etfs.get(etf_symbol, "")},
                )


def ticker_context(ticker: str) -> dict[str, object]:
    snapshot = _load_context_snapshot_map().get(ticker, {})
    cache = _load_ticker_context_cache()
    if ticker in cache or snapshot:
        merged = dict(snapshot)
        merged.update(cache.get(ticker, {}))
        merged.setdefault("ticker", ticker)
        return merged
    try:
        info = yf.Ticker(ticker).get_info() or {}
    except Exception:
        return dict(snapshot)
    ctx = {
        "ticker": ticker,
        "sector": info.get("sector"),
        "sectorKey": info.get("sectorKey"),
        "industry": info.get("industry"),
        "industryKey": info.get("industryKey"),
        "exchange": info.get("exchange"),
        "fullExchangeName": info.get("fullExchangeName"),
        "marketCap": info.get("marketCap"),
        "averageDailyVolume3Month": info.get("averageDailyVolume3Month"),
        "beta": info.get("beta"),
        "quoteType": info.get("quoteType"),
    }
    merged = dict(snapshot)
    merged.update(ctx)
    return dict(_store_ticker_context(ticker, merged))


def bulk_ticker_context(tickers: Sequence[str]) -> dict[str, dict[str, object]]:
    out = {ticker: ticker_context(ticker) for ticker in dedupe_keep_order(tickers)}
    _save_ticker_context_cache(force=True)
    return out


def discover_seed_expansion(records: dict[str, dict], seeds: Sequence[str], etf_holding_count: int, top_etfs_per_sector: int) -> pd.DataFrame:
    rows: list[dict] = []
    seen_sector_etfs: set[str] = set()
    for seed in seeds:
        ctx = ticker_context(seed)
        if not ctx:
            continue
        rows.append(ctx)
        sector_key = str(ctx.get("sectorKey") or "").strip()
        industry_key = str(ctx.get("industryKey") or "").strip()
        if sector_key:
            try:
                sector = yf.Sector(sector_key)
                top_companies = sector.top_companies
            except Exception:
                top_companies = None
                sector = None
            if top_companies is not None and not top_companies.empty:
                for symbol, row in top_companies.reset_index().rename(columns={"index": "symbol"}).set_index("symbol").iterrows():
                    remember_candidate(
                        records,
                        symbol,
                        "seed_peer",
                        seed,
                        row=row,
                        extra={"sector_key": sector_key, "industry_key": industry_key},
                    )
            top_etfs = getattr(sector, "top_etfs", {}) or {}
            for etf_symbol in list(top_etfs.keys())[:top_etfs_per_sector]:
                if etf_symbol in seen_sector_etfs:
                    continue
                seen_sector_etfs.add(etf_symbol)
                try:
                    holdings = yf.Ticker(etf_symbol).funds_data.top_holdings
                except Exception:
                    continue
                if holdings is None or holdings.empty:
                    continue
                holding_rows = holdings.reset_index().rename(columns={"Symbol": "symbol"}).head(etf_holding_count)
                for row in holding_rows.to_dict("records"):
                    remember_candidate(records, str(row.get("symbol", "")), "etf_holding", etf_symbol, row=row, extra={"seed": seed})
        if industry_key:
            try:
                industry = yf.Industry(industry_key)
            except Exception:
                continue
            for attr_name, source_type in (
                ("top_performing_companies", "industry_top_performing"),
                ("top_growth_companies", "industry_top_growth"),
            ):
                frame = getattr(industry, attr_name, None)
                if frame is None or frame.empty:
                    continue
                for symbol, row in frame.reset_index().rename(columns={"index": "symbol"}).set_index("symbol").iterrows():
                    remember_candidate(
                        records,
                        symbol,
                        source_type,
                        seed,
                        row=row,
                        extra={"sector_key": sector_key, "industry_key": industry_key},
                    )
    return pd.DataFrame(rows)


def discover_lookup_queries(records: dict[str, dict], keywords: Sequence[str], max_results: int) -> None:
    for keyword in keywords:
        try:
            frame = yf.Lookup(keyword).get_stock(count=max_results)
        except Exception:
            continue
        if frame is None or frame.empty:
            continue
        for symbol, row in frame.iterrows():
            remember_candidate(records, str(symbol), "lookup_query", keyword, row=row)


def load_structural_scan_frame() -> pd.DataFrame:
    global _STRUCTURAL_SCAN_FRAME
    if _STRUCTURAL_SCAN_FRAME is not None:
        return _STRUCTURAL_SCAN_FRAME
    context = read_optional_csv(CONTEXT_SNAPSHOT_PATH)
    cache_context = load_context_cache_frame()
    features = load_latest_feature_snapshot()
    if not context.empty:
        context = context.copy()
        context["ticker"] = context["ticker"].astype(str)
    if not cache_context.empty:
        if context.empty:
            context = cache_context.copy()
        else:
            context = context.merge(cache_context, on="ticker", how="outer", suffixes=("", "_cache"))
            for base_col in (
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
            ):
                cache_col = f"{base_col}_cache"
                if cache_col in context.columns:
                    if base_col not in context.columns:
                        context[base_col] = context[cache_col]
                    else:
                        context[base_col] = context[base_col].fillna(context[cache_col])
            context = context.drop(columns=[c for c in context.columns if c.endswith("_cache")], errors="ignore")
    if context.empty and features.empty:
        return pd.DataFrame()
    if context.empty:
        out = features.copy()
    elif features.empty:
        out = context.copy()
    else:
        out = context.merge(features, on="ticker", how="outer", suffixes=("", "_feat"))
    if out.empty:
        _STRUCTURAL_SCAN_FRAME = out
        return out
    market_structure = load_market_structure_latest()
    if not market_structure.empty:
        out = out.merge(market_structure, on="ticker", how="left")
    taxonomy = load_taxonomy_pit_latest()
    if not taxonomy.empty:
        out = out.merge(taxonomy, on="ticker", how="left")
    listing = load_listing_corporate_metadata()
    if not listing.empty:
        out = out.merge(listing, on="ticker", how="left")
    if "pit_market_cap_current" in out.columns:
        out["marketCap"] = pd.to_numeric(out.get("marketCap"), errors="coerce").fillna(
            pd.to_numeric(out["pit_market_cap_current"], errors="coerce")
        )
    if "pit_sector_key" in out.columns:
        if "sectorKey" not in out.columns:
            out["sectorKey"] = out["pit_sector_key"]
        else:
            out["sectorKey"] = out["sectorKey"].fillna(out["pit_sector_key"])
    if "pit_industry_key" in out.columns:
        if "industryKey" not in out.columns:
            out["industryKey"] = out["pit_industry_key"]
        else:
            out["industryKey"] = out["industryKey"].fillna(out["pit_industry_key"])
    if "pit_listing_age_days" in out.columns:
        out["listing_age_days"] = pd.to_numeric(out.get("listing_age_days"), errors="coerce").fillna(
            pd.to_numeric(out["pit_listing_age_days"], errors="coerce")
        )
    for col in (
        "marketCap",
        "averageDailyVolume3Month",
        "regularMarketPrice",
        "close",
        "bars_total",
        "listing_age_days",
        "adv_20",
        "adv_60",
        "residual_momentum_score",
        "overnight_return_1d",
        "intraday_return_1d",
        "gap_pct",
        "gap_zscore_20",
        "gap_fill_share",
        "rel_volume_20",
        "volume_zscore_20",
        "dollar_volume_zscore_20",
        "close_in_range",
        "efficiency_ratio_20",
        "efficiency_ratio_60",
        "corr_63_spy",
        "downside_beta_63_spy",
        "return_21d",
        "return_63d",
        "return_126d",
        "return_252d",
        "rel_return_63d_spy",
        "rel_return_126d_spy",
        "rel_return_252d_spy",
        "dist_high_252_pct",
        "dist_sma50_pct",
        "dist_sma200_pct",
        "red_flags_count",
        "pit_market_cap_current",
        "pit_shares_outstanding_current",
        "pit_float_shares_current",
        "pit_free_float_ratio_current",
        "pit_market_structure_coverage_ratio",
        "pit_cluster_key",
        "pit_sector_key",
        "pit_industry_key",
        "pit_listing_age_days",
        "pit_bars_total",
        "pit_recent_split_365d",
        "pit_recent_dividend_365d",
    ):
        if col not in out.columns:
            out[col] = np.nan
    if "regularMarketPrice" in out.columns and "close" in out.columns:
        out["scan_price"] = pd.to_numeric(out["regularMarketPrice"], errors="coerce").fillna(pd.to_numeric(out["close"], errors="coerce"))
    else:
        out["scan_price"] = pd.to_numeric(out.get("close"), errors="coerce")
    out["scan_adv_proxy"] = pd.to_numeric(out.get("averageDailyVolume3Month"), errors="coerce").fillna(
        pd.to_numeric(out.get("adv_20"), errors="coerce").fillna(pd.to_numeric(out.get("adv_60"), errors="coerce"))
    )
    out["marketCap"] = pd.to_numeric(out["marketCap"], errors="coerce")
    out["scan_price"] = pd.to_numeric(out["scan_price"], errors="coerce")
    out["scan_adv_proxy"] = pd.to_numeric(out["scan_adv_proxy"], errors="coerce")
    out["bars_total"] = pd.to_numeric(out["bars_total"], errors="coerce")
    out["listing_age_days"] = pd.to_numeric(out["listing_age_days"], errors="coerce")
    out["above_sma200"] = pd.to_numeric(out.get("above_sma200"), errors="coerce").fillna(0).astype(int)
    out["red_flags_count"] = pd.to_numeric(out.get("red_flags_count"), errors="coerce").fillna(0.0)
    quote_type = out["quoteType"] if "quoteType" in out.columns else pd.Series(["EQUITY"] * len(out), index=out.index)
    out = out.loc[(quote_type == "EQUITY") | quote_type.isna()].copy()
    _STRUCTURAL_SCAN_FRAME = out
    return out


def structural_candidate_filter(df: pd.DataFrame, *, min_market_cap: float, min_price: float, min_adv: float) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[
        (pd.to_numeric(df["marketCap"], errors="coerce").fillna(0.0) >= min_market_cap)
        & (pd.to_numeric(df["scan_price"], errors="coerce").fillna(0.0) >= min_price)
        & (pd.to_numeric(df["scan_adv_proxy"], errors="coerce").fillna(0.0) >= min_adv)
    ].copy()


def discover_filtered_universe_bases(
    records: dict[str, dict],
    *,
    min_market_cap: float,
    min_price: float,
    min_adv: float,
    per_sector: int,
    global_count: int,
) -> None:
    frame = structural_candidate_filter(load_structural_scan_frame(), min_market_cap=min_market_cap, min_price=min_price, min_adv=min_adv)
    if frame.empty:
        return
    frame["base_structural_score"] = (
        2.5 * rank_within_series(np.log1p(frame["marketCap"].fillna(0.0)))
        + 2.0 * rank_within_series(np.log1p(frame["scan_adv_proxy"].fillna(0.0)))
        + 2.0 * rank_within_series(frame["return_252d"].fillna(0.0))
        + 1.0 * rank_within_series(frame["rel_return_126d_spy"].fillna(0.0))
    )
    large_cap = frame.loc[frame["marketCap"].fillna(0.0) >= max(10_000_000_000, min_market_cap * 3)].copy()
    mid_cap = frame.loc[
        (frame["marketCap"].fillna(0.0) >= min_market_cap)
        & (frame["marketCap"].fillna(0.0) < max(25_000_000_000, min_market_cap * 12))
    ].copy()
    for subset, source_type, label in (
        (large_cap, "universe_base_largecap", "base_largecap"),
        (mid_cap, "universe_base_midcap", "base_midcap"),
    ):
        if subset.empty:
            continue
        leaders = subset.sort_values(["base_structural_score", "marketCap", "scan_adv_proxy"], ascending=[False, False, False]).head(global_count)
        for row in leaders.to_dict("records"):
            remember_candidate(records, str(row.get("ticker", "")), source_type, label, row=row)
        if "sector" in subset.columns:
            for sector_name, group in subset.groupby("sector", dropna=True, sort=False):
                if not str(sector_name or "").strip():
                    continue
                top = group.sort_values(["base_structural_score", "marketCap", "scan_adv_proxy"], ascending=[False, False, False]).head(per_sector)
                for row in top.to_dict("records"):
                    remember_candidate(records, str(row.get("ticker", "")), source_type, f"{label}:{sector_name}", row=row)


def discover_multi_horizon_candidates(
    records: dict[str, dict],
    *,
    min_market_cap: float,
    min_price: float,
    min_adv: float,
    per_horizon: int,
) -> None:
    frame = structural_candidate_filter(load_structural_scan_frame(), min_market_cap=min_market_cap, min_price=min_price, min_adv=min_adv)
    if frame.empty:
        return
    tradable = frame.loc[(frame["above_sma200"] > 0) & (frame["red_flags_count"] <= 1)].copy()
    if tradable.empty:
        return
    emerging = tradable.loc[
        (tradable["return_21d"].fillna(0.0) > 0.10)
        & (tradable["return_63d"].fillna(0.0) > 0.05)
        & (tradable["dist_high_252_pct"].fillna(-100.0) > -20.0)
    ].copy()
    emerging["horizon_score"] = (
        2.5 * rank_within_series(emerging["return_21d"].fillna(0.0))
        + 1.8 * rank_within_series(emerging["rel_return_63d_spy"].fillna(0.0))
        + 1.5 * rank_within_series(emerging["residual_momentum_score"].fillna(0.0))
        + 0.5 * rank_within_series(np.log1p(emerging["scan_adv_proxy"].fillna(0.0)))
    )
    breakout = tradable.loc[
        (tradable["dist_high_252_pct"].fillna(-100.0) > -6.0)
        & (tradable["return_126d"].fillna(0.0) > 0.10)
        & (tradable["rel_return_126d_spy"].fillna(0.0) > 0.0)
    ].copy()
    breakout["horizon_score"] = (
        2.2 * rank_within_series(breakout["residual_momentum_score"].fillna(0.0))
        + 1.6 * rank_within_series(breakout["return_126d"].fillna(0.0))
        + 1.2 * rank_within_series(-breakout["dist_high_252_pct"].fillna(-100.0))
        + 0.6 * rank_within_series(breakout["rel_return_252d_spy"].fillna(0.0))
    )
    pullback = tradable.loc[
        (tradable["return_252d"].fillna(0.0) > 0.20)
        & (tradable["dist_sma50_pct"].fillna(100.0).between(-8.0, 2.0))
        & (tradable["dist_high_252_pct"].fillna(-100.0).between(-18.0, -4.0))
    ].copy()
    pullback["horizon_score"] = (
        2.0 * rank_within_series(pullback["residual_momentum_score"].fillna(0.0))
        + 1.6 * rank_within_series(pullback["return_252d"].fillna(0.0))
        + 1.2 * rank_within_series(-pullback["dist_sma50_pct"].abs().fillna(100.0))
        + 0.8 * rank_within_series(-pullback["dist_high_252_pct"].fillna(-100.0))
    )
    for subset, source_type, label in (
        (emerging, "horizon_emerging", "emerging"),
        (breakout, "horizon_breakout", "breakout"),
        (pullback, "horizon_pullback", "pullback"),
    ):
        if subset.empty:
            continue
        top = subset.sort_values(["horizon_score", "marketCap", "scan_adv_proxy"], ascending=[False, False, False]).head(per_horizon)
        for row in top.to_dict("records"):
            remember_candidate(records, str(row.get("ticker", "")), source_type, label, row=row)


def discover_recent_listing_leaders(
    records: dict[str, dict],
    *,
    min_market_cap: float,
    min_price: float,
    min_adv: float,
    count: int,
) -> None:
    frame = structural_candidate_filter(load_structural_scan_frame(), min_market_cap=min_market_cap, min_price=min_price, min_adv=min_adv)
    if frame.empty:
        return
    bars = pd.to_numeric(frame["bars_total"], errors="coerce").fillna(0.0)
    age_days = pd.to_numeric(frame["listing_age_days"], errors="coerce").fillna(10_000.0)
    truly_fresh = frame.loc[
        (bars >= 120)
        & (bars <= 420)
        & (age_days <= 700)
        & (frame["red_flags_count"] <= 1)
        & (frame["scan_price"].fillna(0.0) >= min_price)
    ].copy()
    young_leaders = frame.loc[
        (bars >= 120)
        & (bars <= 1_600)
        & (age_days <= 2_400)
        & (frame["red_flags_count"] <= 1)
        & (frame["scan_price"].fillna(0.0) >= min_price)
        & (frame["dist_sma200_pct"].fillna(-100.0) > -6.0)
    ].copy()
    if truly_fresh.empty and young_leaders.empty:
        return

    breakout = truly_fresh.loc[
        (truly_fresh["return_21d"].fillna(0.0) > 0.05)
        & (truly_fresh["return_63d"].fillna(0.0) > 0.18)
        & (truly_fresh["rel_return_63d_spy"].fillna(0.0) > 0.02)
        & (truly_fresh["dist_high_252_pct"].fillna(-100.0) > -18.0)
        & (truly_fresh["dist_sma50_pct"].fillna(100.0).between(-6.0, 9.0))
    ].copy()
    breakout["fresh_listing_score"] = (
        2.4 * rank_within_series(breakout["return_63d"].fillna(0.0))
        + 1.8 * rank_within_series(breakout["rel_return_63d_spy"].fillna(0.0))
        + 1.4 * rank_within_series(breakout["residual_momentum_score"].fillna(0.0))
        + 0.8 * rank_within_series(-breakout["dist_high_252_pct"].fillna(-100.0))
        + 0.6 * rank_within_series(np.log1p(breakout["scan_adv_proxy"].fillna(0.0)))
        + 0.4 * rank_within_series(-(bars.loc[breakout.index] / 420.0))
    )

    pullback = truly_fresh.loc[
        (truly_fresh["return_126d"].fillna(0.0) > 0.15)
        & (truly_fresh["rel_return_126d_spy"].fillna(0.0) > 0.0)
        & (truly_fresh["dist_sma50_pct"].fillna(100.0).between(-9.0, 2.0))
        & (truly_fresh["dist_high_252_pct"].fillna(-100.0).between(-24.0, -4.0))
    ].copy()
    pullback["fresh_listing_score"] = (
        2.0 * rank_within_series(pullback["residual_momentum_score"].fillna(0.0))
        + 1.6 * rank_within_series(pullback["return_126d"].fillna(0.0))
        + 1.2 * rank_within_series(-pullback["dist_sma50_pct"].abs().fillna(100.0))
        + 0.8 * rank_within_series(-pullback["dist_high_252_pct"].fillna(-100.0))
        + 0.5 * rank_within_series(np.log1p(pullback["scan_adv_proxy"].fillna(0.0)))
    )

    young_breakout = young_leaders.loc[
        (young_leaders["return_63d"].fillna(0.0) > 0.18)
        & (young_leaders["return_126d"].fillna(0.0) > 0.12)
        & (young_leaders["rel_return_63d_spy"].fillna(0.0) > 0.04)
        & (young_leaders["dist_high_252_pct"].fillna(-100.0) > -28.0)
        & (young_leaders["dist_sma50_pct"].fillna(100.0).between(-8.0, 12.0))
    ].copy()
    young_breakout["fresh_listing_score"] = (
        2.1 * rank_within_series(young_breakout["return_63d"].fillna(0.0))
        + 1.7 * rank_within_series(young_breakout["rel_return_63d_spy"].fillna(0.0))
        + 1.4 * rank_within_series(young_breakout["residual_momentum_score"].fillna(0.0))
        + 0.8 * rank_within_series(-young_breakout["dist_high_252_pct"].fillna(-100.0))
        + 0.5 * rank_within_series(np.log1p(young_breakout["scan_adv_proxy"].fillna(0.0)))
        + 0.3 * rank_within_series(-(age_days.loc[young_breakout.index] / 2_400.0))
    )

    young_pullback = young_leaders.loc[
        (young_leaders["return_126d"].fillna(0.0) > 0.18)
        & (young_leaders["rel_return_126d_spy"].fillna(0.0) > 0.03)
        & (young_leaders["dist_sma50_pct"].fillna(100.0).between(-12.0, 4.0))
        & (young_leaders["dist_high_252_pct"].fillna(-100.0).between(-32.0, -6.0))
    ].copy()
    young_pullback["fresh_listing_score"] = (
        1.9 * rank_within_series(young_pullback["residual_momentum_score"].fillna(0.0))
        + 1.5 * rank_within_series(young_pullback["return_126d"].fillna(0.0))
        + 1.0 * rank_within_series(young_pullback["rel_return_126d_spy"].fillna(0.0))
        + 1.0 * rank_within_series(-young_pullback["dist_sma50_pct"].abs().fillna(100.0))
        + 0.7 * rank_within_series(-young_pullback["dist_high_252_pct"].fillna(-100.0))
        + 0.4 * rank_within_series(np.log1p(young_pullback["scan_adv_proxy"].fillna(0.0)))
    )

    for subset, source_type, label in (
        (breakout, "fresh_listing_leader", "fresh_breakout"),
        (pullback, "fresh_listing_pullback", "fresh_pullback"),
        (young_breakout, "recent_listing_leader", "young_breakout"),
        (young_pullback, "recent_listing_pullback", "young_pullback"),
    ):
        if subset.empty:
            continue
        top = subset.sort_values(["fresh_listing_score", "marketCap", "scan_adv_proxy"], ascending=[False, False, False]).head(count)
        for row in top.to_dict("records"):
            remember_candidate(records, str(row.get("ticker", "")), source_type, label, row=row)


def discover_sector_industry_relative_leaders(
    records: dict[str, dict],
    *,
    min_market_cap: float,
    min_price: float,
    min_adv: float,
    per_sector: int,
    per_industry: int,
) -> None:
    frame = structural_candidate_filter(load_structural_scan_frame(), min_market_cap=min_market_cap, min_price=min_price, min_adv=min_adv)
    if frame.empty:
        return
    frame = frame.loc[(frame["above_sma200"] > 0) & (frame["red_flags_count"] <= 1)].copy()
    if frame.empty:
        return
    frame["relative_strength_score"] = (
        2.4 * rank_within_series(frame["residual_momentum_score"].fillna(0.0))
        + 1.8 * rank_within_series(frame["rel_return_63d_spy"].fillna(0.0))
        + 1.2 * rank_within_series(frame["rel_return_126d_spy"].fillna(0.0))
        + 0.8 * rank_within_series(frame["return_252d"].fillna(0.0))
    )
    if "sector" in frame.columns:
        for sector_name, group in frame.groupby("sector", dropna=True, sort=False):
            if not str(sector_name or "").strip():
                continue
            top = group.sort_values(["relative_strength_score", "marketCap"], ascending=[False, False]).head(per_sector)
            for row in top.to_dict("records"):
                remember_candidate(records, str(row.get("ticker", "")), "sector_relative_leader", str(sector_name), row=row)
    if "industry" in frame.columns:
        counts = frame.groupby("industry")["ticker"].nunique()
        valid_industries = counts.loc[counts >= 3].index
        for industry_name, group in frame.loc[frame["industry"].isin(valid_industries)].groupby("industry", dropna=True, sort=False):
            if not str(industry_name or "").strip():
                continue
            top = group.sort_values(["relative_strength_score", "marketCap"], ascending=[False, False]).head(per_industry)
            for row in top.to_dict("records"):
                remember_candidate(records, str(row.get("ticker", "")), "industry_relative_leader", str(industry_name), row=row)


def discover_local_leader_sentinels(
    records: dict[str, dict],
    *,
    min_market_cap: float,
    min_price: float,
    min_adv: float,
    global_count: int,
    per_cluster: int,
) -> None:
    frame = structural_candidate_filter(load_structural_scan_frame(), min_market_cap=min_market_cap, min_price=min_price, min_adv=min_adv)
    if frame.empty:
        return
    tradable = frame.loc[
        (frame["above_sma200"] > 0)
        & (pd.to_numeric(frame["bars_total"], errors="coerce").fillna(pd.to_numeric(frame["pit_bars_total"], errors="coerce")).fillna(0.0) >= 260)
        & (frame["red_flags_count"] <= 1)
    ].copy()
    if tradable.empty:
        return

    free_float = pd.to_numeric(tradable.get("pit_free_float_ratio_current"), errors="coerce").clip(lower=0.05, upper=1.0).fillna(0.80)
    efficiency60 = pd.to_numeric(tradable.get("efficiency_ratio_60"), errors="coerce").clip(lower=0.0, upper=1.0).fillna(0.25)
    dist_sma50_abs = pd.to_numeric(tradable.get("dist_sma50_pct"), errors="coerce").abs().clip(upper=30.0).fillna(30.0)
    dist_sma200 = pd.to_numeric(tradable.get("dist_sma200_pct"), errors="coerce").clip(lower=-1.0, upper=3.0).fillna(-1.0)
    dist_high = pd.to_numeric(tradable.get("dist_high_252_pct"), errors="coerce").clip(lower=-100.0, upper=0.0).fillna(-100.0)
    rel63 = pd.to_numeric(tradable.get("rel_return_63d_spy"), errors="coerce").fillna(0.0)
    rel126 = pd.to_numeric(tradable.get("rel_return_126d_spy"), errors="coerce").fillna(0.0)
    ret63 = pd.to_numeric(tradable.get("return_63d"), errors="coerce").fillna(0.0)
    ret126 = pd.to_numeric(tradable.get("return_126d"), errors="coerce").fillna(0.0)
    ret252 = pd.to_numeric(tradable.get("return_252d"), errors="coerce").fillna(0.0)
    residual = pd.to_numeric(tradable.get("residual_momentum_score"), errors="coerce").fillna(0.0)
    adv = pd.to_numeric(tradable.get("scan_adv_proxy"), errors="coerce").fillna(0.0)
    market_cap = pd.to_numeric(tradable.get("marketCap"), errors="coerce").fillna(0.0)
    split_recent = pd.to_numeric(tradable.get("pit_recent_split_365d"), errors="coerce").fillna(0.0)

    tradable["local_sentinel_score"] = (
        2.2 * rank_within_series(ret63)
        + 2.0 * rank_within_series(ret126)
        + 1.8 * rank_within_series(rel63)
        + 1.4 * rank_within_series(rel126)
        + 1.6 * rank_within_series(residual)
        + 0.8 * rank_within_series(efficiency60)
        + 0.8 * rank_within_series(np.log1p(adv))
        + 0.7 * rank_within_series(np.log1p(market_cap))
        + 0.5 * rank_within_series(-dist_sma50_abs)
        + 0.5 * rank_within_series(dist_sma200)
        + 0.4 * rank_within_series(dist_high)
        + 0.4 * rank_within_series(free_float)
        - 0.4 * (free_float < 0.18).astype(float)
        - 0.2 * (split_recent > 0).astype(float)
    )
    tradable["local_quality_sentinel_score"] = (
        2.2 * rank_within_series(ret252)
        + 1.8 * rank_within_series(rel126)
        + 1.6 * rank_within_series(efficiency60)
        + 1.2 * rank_within_series(residual)
        + 0.9 * rank_within_series(np.log1p(adv))
        + 0.8 * rank_within_series(np.log1p(market_cap))
        + 0.7 * rank_within_series(free_float)
        + 0.5 * rank_within_series(-dist_sma50_abs)
        + 0.5 * rank_within_series(dist_sma200)
        - 0.2 * (free_float < 0.18).astype(float)
    )
    tradable["sentinel_cluster"] = (
        tradable.get("pit_cluster_key")
        .fillna(tradable.get("sectorKey"))
        .fillna(tradable.get("sector"))
        .astype(str)
        .str.strip()
    )

    global_leaders = tradable.loc[
        (ret63 > 0.08)
        & (ret126 > 0.12)
        & (rel63 > 0.0)
        & (dist_high > -30.0)
    ].copy()
    if not global_leaders.empty:
        top = global_leaders.sort_values(["local_sentinel_score", "marketCap", "scan_adv_proxy"], ascending=[False, False, False]).head(global_count)
        for row in top.to_dict("records"):
            remember_candidate(records, str(row.get("ticker", "")), "local_leader_sentinel", "sentinel_global", row=row)

    quality_leaders = tradable.loc[
        (ret252 > 0.18)
        & (rel126 > 0.02)
        & (efficiency60 >= 0.25)
        & (dist_sma200 > -0.05)
        & (dist_sma200 < 1.20)
    ].copy()
    if not quality_leaders.empty:
        top = quality_leaders.sort_values(["local_quality_sentinel_score", "marketCap", "scan_adv_proxy"], ascending=[False, False, False]).head(max(6, global_count // 2))
        for row in top.to_dict("records"):
            remember_candidate(records, str(row.get("ticker", "")), "local_quality_sentinel", "sentinel_quality", row=row)

    cluster_frame = tradable.loc[
        tradable["sentinel_cluster"].ne("")
        & tradable["sentinel_cluster"].ne("nan")
    ].copy()
    if cluster_frame.empty or per_cluster <= 0:
        return
    cluster_counts = cluster_frame.groupby("sentinel_cluster")["ticker"].nunique()
    valid_clusters = cluster_counts.loc[cluster_counts >= 2].index
    for cluster_name, group in cluster_frame.loc[cluster_frame["sentinel_cluster"].isin(valid_clusters)].groupby("sentinel_cluster", dropna=True, sort=False):
        top = group.sort_values(["local_sentinel_score", "marketCap", "scan_adv_proxy"], ascending=[False, False, False]).head(per_cluster)
        for row in top.to_dict("records"):
            remember_candidate(records, str(row.get("ticker", "")), "local_cluster_leader", f"cluster:{cluster_name}", row=row)


def discover_region_universe(records: dict[str, dict], regions: Sequence[str], min_market_cap: float, min_price: float, min_adv: float, count: int, max_pages: int) -> None:
    if count <= 0 or max_pages <= 0:
        return
    for region in regions:
        query = yf.EquityQuery(
            "and",
            [
                yf.EquityQuery("eq", ["region", region]),
                yf.EquityQuery("gte", ["intradayprice", min_price]),
                yf.EquityQuery("gte", ["intradaymarketcap", min_market_cap]),
            ],
        )
        for sort_field, suffix in (("dayvolume", "liquid"), ("intradaymarketcap", "largecap")):
            try:
                rows = fetch_screen_rows(query, count=count, max_pages=max_pages, sort_field=sort_field, sort_asc=False)
            except Exception:
                continue
            for row in rows:
                if quote_is_candidate(row, min_market_cap, min_price, min_adv):
                    remember_candidate(records, str(row.get("symbol", "")), "regional_universe", f"{region}:{suffix}", row=row)


def discover_sector_regime_leaders(
    records: dict[str, dict],
    *,
    top_sector_count: int,
    top_companies_per_sector: int,
    etf_holding_count: int,
) -> None:
    regime = load_sector_regime_frame()
    if regime.empty or "sector_key" not in regime.columns:
        return
    seen_etfs: set[str] = set()
    for row in regime.head(top_sector_count).itertuples(index=False):
        sector_key = str(getattr(row, "sector_key", "") or "").strip()
        if not sector_key:
            continue
        sector_key = {
            "semiconductors": "technology",
            "metals-mining": "basic-materials",
            "gold-miners": "basic-materials",
            "silver-miners": "basic-materials",
        }.get(sector_key, sector_key)
        try:
            sector = yf.Sector(sector_key)
            top_companies = sector.top_companies
        except Exception:
            continue
        if top_companies is not None and not top_companies.empty:
            top_frame = (
                top_companies.reset_index()
                .rename(columns={"index": "symbol"})
                .head(top_companies_per_sector)
            )
            for record in top_frame.to_dict("records"):
                remember_candidate(records, str(record.get("symbol", "")), "sector_regime_leader", sector_key, row=record)
        top_etfs = getattr(sector, "top_etfs", {}) or {}
        for etf_symbol in list(top_etfs.keys())[:2]:
            if etf_symbol in seen_etfs:
                continue
            seen_etfs.add(etf_symbol)
            try:
                holdings = yf.Ticker(etf_symbol).funds_data.top_holdings
            except Exception:
                continue
            if holdings is None or holdings.empty:
                continue
            holding_rows = holdings.reset_index().rename(columns={"Symbol": "symbol"}).head(etf_holding_count)
            for record in holding_rows.to_dict("records"):
                remember_candidate(records, str(record.get("symbol", "")), "sector_regime_etf", sector_key, row=record)


def discover_post_earnings_winners(
    records: dict[str, dict],
    *,
    min_market_cap: float,
    min_price: float,
    min_adv: float,
    count: int,
) -> None:
    frame = structural_candidate_filter(load_structural_scan_frame(), min_market_cap=min_market_cap, min_price=min_price, min_adv=min_adv)
    if frame.empty:
        return
    days_since = pd.to_numeric(frame["days_since_last_earnings"], errors="coerce") if "days_since_last_earnings" in frame.columns else pd.Series(np.nan, index=frame.index)
    post5 = pd.to_numeric(frame["post_earnings_5d"], errors="coerce").fillna(0.0) if "post_earnings_5d" in frame.columns else pd.Series(0.0, index=frame.index)
    post10 = pd.to_numeric(frame["post_earnings_10d"], errors="coerce").fillna(0.0) if "post_earnings_10d" in frame.columns else pd.Series(0.0, index=frame.index)
    growth = pd.to_numeric(frame["earningsQuarterlyGrowth"], errors="coerce").fillna(0.0) if "earningsQuarterlyGrowth" in frame.columns else pd.Series(0.0, index=frame.index)
    subset = frame.loc[
        (
            (post5 > 0)
            | (post10 > 0)
            | (days_since.between(1, 18, inclusive="both"))
        )
        & (frame["above_sma200"] > 0)
        & (frame["return_21d"].fillna(0.0) > 0.04)
        & (frame["rel_return_63d_spy"].fillna(0.0) >= 0.0)
    ].copy()
    if subset.empty:
        return
    subset["post_earnings_score"] = (
        2.4 * rank_within_series(subset["return_21d"].fillna(0.0))
        + 1.8 * rank_within_series(subset["rel_return_63d_spy"].fillna(0.0))
        + 1.2 * rank_within_series(subset["residual_momentum_score"].fillna(0.0))
        + 0.8 * rank_within_series(growth.loc[subset.index].fillna(0.0))
        + 0.4 * rank_within_series(np.log1p(subset["scan_adv_proxy"].fillna(0.0)))
    )
    for row in subset.sort_values(["post_earnings_score", "marketCap"], ascending=[False, False]).head(count).to_dict("records"):
        remember_candidate(records, str(row.get("ticker", "")), "post_earnings_winner", "post_earnings", row=row)


def discover_reversal_candidates(
    records: dict[str, dict],
    *,
    min_market_cap: float,
    min_price: float,
    min_adv: float,
    count: int,
) -> None:
    frame = structural_candidate_filter(load_structural_scan_frame(), min_market_cap=min_market_cap, min_price=min_price, min_adv=min_adv)
    if frame.empty:
        return
    tradable = frame.loc[(frame["above_sma200"] > 0) & (frame["red_flags_count"] <= 1)].copy()
    if tradable.empty:
        return
    base_breakout = tradable.loc[
        (tradable["return_252d"].fillna(0.0).between(-0.20, 0.35))
        & (tradable["return_63d"].fillna(0.0) > 0.18)
        & (tradable["dist_high_252_pct"].fillna(-100.0).between(-22.0, -4.0))
        & (tradable["dist_sma50_pct"].fillna(100.0).between(-4.0, 6.0))
    ].copy()
    base_breakout["reversal_score"] = (
        2.2 * rank_within_series(base_breakout["return_63d"].fillna(0.0))
        + 1.6 * rank_within_series(base_breakout["residual_momentum_score"].fillna(0.0))
        + 1.2 * rank_within_series(-base_breakout["dist_high_252_pct"].fillna(-100.0))
        + 0.8 * rank_within_series(-base_breakout["dist_sma50_pct"].abs().fillna(100.0))
    )
    recovery = tradable.loc[
        (tradable["return_252d"].fillna(0.0) < 0.20)
        & (tradable["return_21d"].fillna(0.0) > 0.10)
        & (tradable["rel_return_63d_spy"].fillna(0.0) > 0.02)
        & (tradable["dist_sma200_pct"].fillna(-100.0) > -2.0)
    ].copy()
    recovery["reversal_score"] = (
        2.0 * rank_within_series(recovery["return_21d"].fillna(0.0))
        + 1.8 * rank_within_series(recovery["rel_return_63d_spy"].fillna(0.0))
        + 1.2 * rank_within_series(recovery["residual_momentum_score"].fillna(0.0))
        + 0.6 * rank_within_series(np.log1p(recovery["scan_adv_proxy"].fillna(0.0)))
    )
    for subset, source_type, label in (
        (base_breakout, "reversal_base_breakout", "base_breakout"),
        (recovery, "reversal_recovery", "recovery"),
    ):
        if subset.empty:
            continue
        for row in subset.sort_values(["reversal_score", "marketCap"], ascending=[False, False]).head(count).to_dict("records"):
            remember_candidate(records, str(row.get("ticker", "")), source_type, label, row=row)


def discover_historical_winner_cousins(
    records: dict[str, dict],
    *,
    min_market_cap: float,
    min_price: float,
    min_adv: float,
    peers_per_winner: int,
) -> None:
    winners = load_historical_winner_frame()
    frame = structural_candidate_filter(load_structural_scan_frame(), min_market_cap=min_market_cap, min_price=min_price, min_adv=min_adv)
    if winners.empty or frame.empty:
        return
    ctx_map = _load_context_snapshot_map()
    frame = frame.copy()
    for winner in winners["ticker"].astype(str).tolist():
        ctx = ctx_map.get(winner, {}) or ticker_context(winner)
        sector = str(ctx.get("sector") or "").strip()
        industry = str(ctx.get("industry") or "").strip()
        subset = frame.copy()
        if industry:
            subset = subset.loc[subset["industry"].fillna("").astype(str) == industry].copy()
        elif sector:
            subset = subset.loc[subset["sector"].fillna("").astype(str) == sector].copy()
        else:
            continue
        subset = subset.loc[subset["ticker"].astype(str) != winner].copy()
        if subset.empty:
            continue
        subset["winner_cousin_score"] = (
            2.0 * rank_within_series(subset["residual_momentum_score"].fillna(0.0))
            + 1.6 * rank_within_series(subset["return_126d"].fillna(0.0))
            + 1.2 * rank_within_series(subset["rel_return_63d_spy"].fillna(0.0))
            + 0.8 * rank_within_series(np.log1p(subset["marketCap"].fillna(0.0)))
        )
        for row in subset.sort_values(["winner_cousin_score", "marketCap"], ascending=[False, False]).head(peers_per_winner).to_dict("records"):
            remember_candidate(records, str(row.get("ticker", "")), "historical_winner_cousin", winner, row=row)


def discover_constituent_proxy_universes(records: dict[str, dict], *, holdings_per_etf: int) -> None:
    seen_pairs: set[tuple[str, str]] = set()
    for etf_symbol, label in CONSTITUENT_PROXY_ETFS:
        try:
            holdings = yf.Ticker(etf_symbol).funds_data.top_holdings
        except Exception:
            continue
        if holdings is None or holdings.empty:
            continue
        holding_rows = holdings.reset_index().rename(columns={"Symbol": "symbol"}).head(holdings_per_etf)
        for row in holding_rows.to_dict("records"):
            symbol = str(row.get("symbol", "")).strip()
            if not symbol:
                continue
            pair = (etf_symbol, symbol)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            remember_candidate(
                records,
                symbol,
                "constituent_proxy",
                label,
                row=row,
                extra={"etf_name": etf_symbol},
            )


def repair_candidate_metadata(df: pd.DataFrame, *, max_repairs: int = METADATA_REPAIR_LIMIT) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return df
    out = df.copy()
    for col in ("sector", "industry", "exchange", "fullExchangeName", "marketCap", "averageDailyVolume3Month", "quoteType", "sectorKey", "industryKey"):
        if col not in out.columns:
            out[col] = np.nan
    missing_mask = (
        out["sector"].isna()
        | out["industry"].isna()
        | out["exchange"].isna()
        | pd.to_numeric(out["marketCap"], errors="coerce").isna()
    )
    if "candidate_status" in out.columns:
        missing_mask &= out["candidate_status"].fillna("").ne("hard_exclusion")
    if not missing_mask.any():
        return out
    missing = out.loc[missing_mask, ["ticker", "source_count", "marketCap"]].copy()
    missing["source_count"] = pd.to_numeric(missing["source_count"], errors="coerce").fillna(0.0)
    missing["marketCap"] = pd.to_numeric(missing["marketCap"], errors="coerce").fillna(0.0)
    tickers = (
        missing.sort_values(["source_count", "marketCap"], ascending=[False, False])["ticker"]
        .dropna()
        .astype(str)
        .head(max_repairs)
        .tolist()
    )
    if not tickers:
        return out
    repaired = pd.DataFrame(bulk_ticker_context(tickers).values())
    if repaired.empty or "ticker" not in repaired.columns:
        return out
    repaired["ticker"] = repaired["ticker"].astype(str)
    repaired = repaired.drop_duplicates("ticker", keep="last")
    out = out.merge(repaired, on="ticker", how="left", suffixes=("", "_repair"))
    for col in ("sector", "sectorKey", "industry", "industryKey", "exchange", "fullExchangeName", "marketCap", "averageDailyVolume3Month", "beta", "quoteType"):
        repair_col = f"{col}_repair"
        if repair_col in out.columns:
            out[col] = out[col].fillna(out[repair_col])
    out = out.drop(columns=[c for c in out.columns if c.endswith("_repair")], errors="ignore")
    return out


def build_candidate_frame(records: dict[str, dict], active: set[str], reserve: set[str], hard_exclusions: set[str]) -> pd.DataFrame:
    rows = []
    for rec in records.values():
        row = dict(rec)
        row["source_count"] = len(rec["source_names"])
        row["source_type_count"] = len(rec["source_types"])
        row["sources"] = "|".join(sorted(rec["source_names"]))
        row["source_types"] = "|".join(sorted(rec["source_types"]))
        row["seed_hits"] = "|".join(sorted(rec["seed_hits"]))
        row["etf_hits"] = "|".join(sorted(rec["etf_hits"]))
        row["is_active"] = int(row["ticker"] in active)
        row["is_reserve"] = int(row["ticker"] in reserve)
        row["is_hard_exclusion"] = int(row["ticker"] in hard_exclusions)
        row["candidate_status"] = (
            "active" if row["is_active"] else "reserve" if row["is_reserve"] else "hard_exclusion" if row["is_hard_exclusion"] else "new"
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in (
        "marketCap",
        "averageDailyVolume3Month",
        "regularMarketPrice",
        "currency",
        "fiftyDayAverageChangePercent",
        "twoHundredDayAverageChangePercent",
        "regularMarketChangePercent",
        "sector",
        "industry",
        "fullExchangeName",
        "beta",
        "quoteType",
        "sectorKey",
        "industryKey",
        "sector_key",
        "industry_key",
    ):
        if col not in df.columns:
            df[col] = np.nan
    context_snapshot = read_optional_csv(CONTEXT_SNAPSHOT_PATH)
    if not context_snapshot.empty and "ticker" in context_snapshot.columns:
        ctx_cols = [
            "ticker",
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
            "currency",
        ]
        available = [col for col in ctx_cols if col in context_snapshot.columns]
        df = df.merge(context_snapshot[available].drop_duplicates("ticker", keep="last"), on="ticker", how="left", suffixes=("", "_ctx"))
        fill_pairs = {
            "marketCap": "marketCap_ctx",
            "averageDailyVolume3Month": "averageDailyVolume3Month_ctx",
            "sector": "sector_ctx",
            "industry": "industry_ctx",
            "exchange": "exchange_ctx",
            "fullExchangeName": "fullExchangeName_ctx",
            "beta": "beta_ctx",
            "quoteType": "quoteType_ctx",
            "currency": "currency_ctx",
        }
        for left, right in fill_pairs.items():
            if right in df.columns:
                df[left] = df[left].fillna(df[right])
        if "sectorKey_ctx" in df.columns:
            df["sectorKey"] = df.get("sectorKey").fillna(df["sectorKey_ctx"]) if "sectorKey" in df.columns else df["sectorKey_ctx"]
            df["sector_key"] = df["sector_key"].fillna(df["sectorKey_ctx"])
        if "industryKey_ctx" in df.columns:
            df["industryKey"] = df.get("industryKey").fillna(df["industryKey_ctx"]) if "industryKey" in df.columns else df["industryKey_ctx"]
            df["industry_key"] = df["industry_key"].fillna(df["industryKey_ctx"])
        df = df.drop(columns=[c for c in df.columns if c.endswith("_ctx")], errors="ignore")
    df = repair_candidate_metadata(df)
    earnings_snapshot = read_optional_csv(EARNINGS_SNAPSHOT_PATH)
    if not earnings_snapshot.empty and "ticker" in earnings_snapshot.columns:
        earn_cols = [
            "ticker",
            "next_earnings_date",
            "days_to_next_earnings",
            "days_since_last_earnings",
            "earnings_blackout_5d",
            "earnings_blackout_10d",
            "post_earnings_5d",
            "post_earnings_10d",
            "earningsQuarterlyGrowth",
        ]
        available = [col for col in earn_cols if col in earnings_snapshot.columns]
        df = df.merge(earnings_snapshot[available].drop_duplicates("ticker", keep="last"), on="ticker", how="left")
    df["sector_benchmark"] = df.apply(sector_benchmark_for_row, axis=1)
    market_cap_num = pd.to_numeric(df["marketCap"], errors="coerce")
    adv_num = pd.to_numeric(df["averageDailyVolume3Month"], errors="coerce")
    price_num = pd.to_numeric(df.get("regularMarketPrice"), errors="coerce")
    df["marketCap"] = market_cap_num
    df["averageDailyVolume3Month"] = adv_num
    df["regularMarketPrice"] = price_num
    exchange_supported = df.apply(broker_exchange_supported, axis=1).astype(int)
    data_quality_flag = (
        market_cap_num.isna().astype(int)
        + adv_num.isna().astype(int)
        + df["sector"].isna().astype(int)
        + df["industry"].isna().astype(int)
        + price_num.isna().astype(int)
        + df["exchange"].isna().astype(int)
    )
    data_quality_flag += (
        (df["candidate_status"] == "new")
        & (df["source_count"].astype(float) <= 1.0)
        & market_cap_num.isna()
    ).astype(int)
    df["broker_exchange_supported"] = exchange_supported
    df["data_quality_flag"] = data_quality_flag.astype(int)
    df["priority_score"] = (
        6.0 * df["source_count"].astype(float)
        + 3.0 * df["source_types"].str.contains("seed_peer", na=False).astype(float)
        + 2.6 * df["source_types"].str.contains("local_leader_sentinel", na=False).astype(float)
        + 2.4 * df["source_types"].str.contains("local_quality_sentinel", na=False).astype(float)
        + 2.5 * df["source_types"].str.contains("industry_top_performing|industry_top_growth", na=False).astype(float)
        + 2.4 * df["source_types"].str.contains("industry_relative_leader", na=False).astype(float)
        + 2.2 * df["source_types"].str.contains("local_cluster_leader", na=False).astype(float)
        + 2.2 * df["source_types"].str.contains("sector_relative_leader", na=False).astype(float)
        + 2.0 * df["source_types"].str.contains("horizon_breakout|horizon_pullback|horizon_emerging", na=False).astype(float)
        + 2.0 * df["source_types"].str.contains("post_earnings_winner", na=False).astype(float)
        + 1.8 * df["source_types"].str.contains("reversal_base_breakout|reversal_recovery", na=False).astype(float)
        + 1.8 * df["source_types"].str.contains("historical_winner_cousin", na=False).astype(float)
        + 1.8 * df["source_types"].str.contains("fresh_listing_leader|fresh_listing_pullback|recent_listing_leader|recent_listing_pullback", na=False).astype(float)
        + 1.6 * df["source_types"].str.contains("sector_regime_leader|sector_regime_etf", na=False).astype(float)
        + 1.4 * df["source_types"].str.contains("regional_universe", na=False).astype(float)
        + 2.0 * df["source_types"].str.contains("etf_holding", na=False).astype(float)
        + 1.8 * df["source_types"].str.contains("constituent_proxy", na=False).astype(float)
        + 1.5 * df["source_types"].str.contains("sector_top_company", na=False).astype(float)
        + 1.2 * df["source_types"].str.contains("universe_base_largecap|universe_base_midcap", na=False).astype(float)
        + 1.0 * df["source_types"].str.contains("predefined_screen|custom_screen", na=False).astype(float)
        + 2.0 * rank_within_series(np.log1p(market_cap_num.fillna(0.0)))
        + 1.5 * rank_within_series(np.log1p(adv_num.fillna(0.0)))
        + 1.5 * rank_within_series(df["twoHundredDayAverageChangePercent"].fillna(0.0))
        + 1.0 * rank_within_series(df["fiftyDayAverageChangePercent"].fillna(0.0))
        + 0.4 * df["broker_exchange_supported"].astype(float)
        - 0.6 * pd.to_numeric(df.get("earnings_blackout_5d", 0), errors="coerce").fillna(0.0)
        - 0.3 * pd.to_numeric(df.get("earnings_blackout_10d", 0), errors="coerce").fillna(0.0)
        + 0.2 * pd.to_numeric(df.get("post_earnings_5d", 0), errors="coerce").fillna(0.0)
        - DATA_QUALITY_PENALTY * df["data_quality_flag"].astype(float)
    )
    return df.sort_values(
        ["candidate_status", "priority_score", "marketCap", "averageDailyVolume3Month"],
        ascending=[True, False, False, False],
    )


def safe_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(columns))
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[list(columns)]


def download_candidate_data(tickers: Iterable[str], start: str, end: str) -> dict[str, CandidateData]:
    data: dict[str, CandidateData] = {}
    symbols = dedupe_keep_order(list(tickers))
    batch = download_history_batch(symbols, start, end)
    missing: list[str] = []
    for ticker in symbols:
        df = normalize_downloaded_columns(batch.get(ticker, pd.DataFrame()))
        if df.empty or "open" not in df.columns or "close" not in df.columns:
            missing.append(ticker)
            continue
        df = df.reset_index()
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        data[ticker] = CandidateData(
            open=df.set_index("date")["open"].astype(float),
            high=df.set_index("date")["high"].astype(float),
            low=df.set_index("date")["low"].astype(float),
            close=df.set_index("date")["close"].astype(float),
            volume=df.set_index("date")["volume"].astype(float) if "volume" in df.columns else pd.Series(dtype=float),
            info={},
        )

    for ticker in missing:
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        except Exception:
            df = pd.DataFrame()
        df = normalize_downloaded_columns(df)
        if df.empty or "open" not in df.columns or "close" not in df.columns:
            continue
        df = df.reset_index()
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        data[ticker] = CandidateData(
            open=df.set_index("date")["open"].astype(float),
            high=df.set_index("date")["high"].astype(float),
            low=df.set_index("date")["low"].astype(float),
            close=df.set_index("date")["close"].astype(float),
            volume=df.set_index("date")["volume"].astype(float) if "volume" in df.columns else pd.Series(dtype=float),
            info={},
        )
    return data


def _recent_momentum_metrics(close: pd.Series) -> tuple[int, dict[str, float], float]:
    close = pd.to_numeric(close, errors="coerce").dropna()
    bars = int(close.shape[0])
    metrics: dict[str, float] = {}
    for lookback, name in ((63, "recent_r63"), (126, "recent_r126"), (252, "recent_r252")):
        metrics[name] = float(close.iloc[-1] / close.iloc[-1 - lookback] - 1.0) if bars > lookback else np.nan
    weights = {"recent_r63": 0.20, "recent_r126": 0.40, "recent_r252": 0.40}
    valid = {k: v for k, v in metrics.items() if pd.notna(v)}
    if not valid:
        return bars, metrics, np.nan
    weight_sum = sum(weights[k] for k in valid)
    score = sum(weights[k] * valid[k] for k in valid) / weight_sum
    return bars, metrics, float(score)


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def _efficiency_ratio(close: pd.Series, lookback: int) -> pd.Series:
    numer = (close - close.shift(lookback)).abs()
    denom = close.diff().abs().rolling(lookback, min_periods=max(lookback // 2, 5)).sum()
    return numer / denom.replace(0, np.nan)


def _candidate_session_metrics(cand: CandidateData) -> dict[str, float]:
    close = pd.to_numeric(cand.close, errors="coerce").dropna()
    if close.empty:
        return {
            "overnight_return_1d": np.nan,
            "intraday_return_1d": np.nan,
            "gap_pct": np.nan,
            "gap_zscore_20": np.nan,
            "gap_fill_share": np.nan,
            "rel_volume_20": np.nan,
            "volume_zscore_20": np.nan,
            "dollar_volume_zscore_20": np.nan,
            "close_in_range": np.nan,
            "efficiency_ratio_20": np.nan,
            "efficiency_ratio_60": np.nan,
        }
    idx = close.index
    px_open = pd.to_numeric(cand.open.reindex(idx), errors="coerce")
    px_high = pd.to_numeric(cand.high.reindex(idx), errors="coerce")
    px_low = pd.to_numeric(cand.low.reindex(idx), errors="coerce")
    px_volume = pd.to_numeric(cand.volume.reindex(idx), errors="coerce")
    prev_close = close.shift(1)

    gap_pct = px_open / prev_close - 1.0
    intraday_ret = close / px_open - 1.0
    gap_z = _rolling_zscore(gap_pct, 20, 10)

    gap_fill = pd.Series(np.nan, index=idx, dtype=float)
    pos_gap = (px_open > prev_close) & prev_close.notna()
    neg_gap = (px_open < prev_close) & prev_close.notna()
    pos_denom = (px_open - prev_close).replace(0, np.nan)
    neg_denom = (prev_close - px_open).replace(0, np.nan)
    gap_fill.loc[pos_gap] = ((px_open - px_low) / pos_denom).loc[pos_gap]
    gap_fill.loc[neg_gap] = ((px_high - px_open) / neg_denom).loc[neg_gap]

    rel_volume_20 = px_volume / px_volume.rolling(20, min_periods=10).mean().replace(0, np.nan)
    volume_zscore_20 = _rolling_zscore(np.log1p(px_volume.clip(lower=0.0)), 20, 10)
    dollar_volume = (close * px_volume).replace([np.inf, -np.inf], np.nan)
    dollar_volume_zscore_20 = _rolling_zscore(np.log1p(dollar_volume.clip(lower=0.0)), 20, 10)

    range_size = (px_high - px_low).replace(0, np.nan)
    close_in_range = ((close - px_low) / range_size).clip(lower=0.0, upper=1.0).fillna(0.5)
    efficiency_ratio_20 = _efficiency_ratio(close, 20)
    efficiency_ratio_60 = _efficiency_ratio(close, 60)

    return {
        "overnight_return_1d": float(gap_pct.iloc[-1]) if len(gap_pct) else np.nan,
        "intraday_return_1d": float(intraday_ret.iloc[-1]) if len(intraday_ret) else np.nan,
        "gap_pct": float(gap_pct.iloc[-1]) if len(gap_pct) else np.nan,
        "gap_zscore_20": float(gap_z.iloc[-1]) if len(gap_z) else np.nan,
        "gap_fill_share": float(gap_fill.clip(lower=0.0, upper=1.0).fillna(0.0).iloc[-1]) if len(gap_fill) else np.nan,
        "rel_volume_20": float(rel_volume_20.iloc[-1]) if len(rel_volume_20) else np.nan,
        "volume_zscore_20": float(volume_zscore_20.iloc[-1]) if len(volume_zscore_20) else np.nan,
        "dollar_volume_zscore_20": float(dollar_volume_zscore_20.iloc[-1]) if len(dollar_volume_zscore_20) else np.nan,
        "close_in_range": float(close_in_range.iloc[-1]) if len(close_in_range) else np.nan,
        "efficiency_ratio_20": float(efficiency_ratio_20.iloc[-1]) if len(efficiency_ratio_20) else np.nan,
        "efficiency_ratio_60": float(efficiency_ratio_60.iloc[-1]) if len(efficiency_ratio_60) else np.nan,
    }


def _candidate_corr_beta(close: pd.Series, benchmark_close: pd.Series, *, downside: bool = False) -> tuple[float, float]:
    if close.empty or benchmark_close.empty:
        return np.nan, np.nan
    aligned = pd.concat(
        [close.rename("close"), benchmark_close.reindex(close.index).ffill().rename("bench")],
        axis=1,
    ).dropna()
    if aligned.shape[0] < 63:
        return np.nan, np.nan
    own_ret = aligned["close"].pct_change(fill_method=None)
    bench_ret = aligned["bench"].pct_change(fill_method=None)
    min_periods = 20 if downside else 40
    if downside:
        own_ret = own_ret.where(bench_ret < 0)
        bench_ret = bench_ret.where(bench_ret < 0)
    corr = own_ret.rolling(63, min_periods=min_periods).corr(bench_ret)
    cov = own_ret.rolling(63, min_periods=min_periods).cov(bench_ret)
    var = bench_ret.rolling(63, min_periods=min_periods).var()
    corr_now = float(corr.iloc[-1]) if len(corr) else np.nan
    beta_now = float((cov / var.replace(0, np.nan)).iloc[-1]) if len(cov) else np.nan
    return corr_now, beta_now


def compute_recent_momentum_from_candidate_map(candidate_map: dict[str, CandidateData]) -> pd.DataFrame:
    rows = []
    for ticker, cand in candidate_map.items():
        bars, metrics, score = _recent_momentum_metrics(cand.close)
        rows.append(
            {
                "ticker": ticker,
                "recent_bars": bars,
                "recent_score": score,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def compute_algo_fit_for_candidates(base_prices, cfg: dict, candidate_map: dict[str, CandidateData], tickers: Sequence[str]) -> pd.DataFrame:
    rows = []
    min_bars_required = int(cfg["min_bars_required"])
    spy_close = base_prices.close["SPY"] if "SPY" in base_prices.close.columns else pd.Series(dtype=float)
    context_map = bulk_ticker_context(tickers)
    earnings_map = _load_earnings_snapshot_map()
    benchmark_close_map = load_sector_benchmark_close_map()
    for ticker in tickers:
        cand = candidate_map.get(ticker)
        if cand is None:
            continue
        close = pd.to_numeric(cand.close, errors="coerce").dropna()
        bars_recent = int(close.shape[0])
        session_metrics = _candidate_session_metrics(cand)
        recent_metrics = {}
        for lookback, name in ((63, "recent_r63"), (126, "recent_r126"), (252, "recent_r252")):
            if bars_recent > lookback:
                recent_metrics[name] = float(close.iloc[-1] / close.iloc[-1 - lookback] - 1.0)
            else:
                recent_metrics[name] = np.nan
        weights = {"recent_r63": 0.20, "recent_r126": 0.40, "recent_r252": 0.40}
        valid_recent = {k: v for k, v in recent_metrics.items() if pd.notna(v)}
        if valid_recent:
            weight_sum = sum(weights[k] for k in valid_recent)
            recent_score = sum(weights[k] * valid_recent[k] for k in valid_recent) / weight_sum
        else:
            recent_score = np.nan
        rel_metrics = {"recent_rel_r63": np.nan, "recent_rel_r126": np.nan, "recent_rel_r252": np.nan}
        rel_score = np.nan
        corr_63_spy = np.nan
        downside_beta_63_spy = np.nan
        if not spy_close.empty:
            aligned = pd.concat([close.rename("close"), spy_close.reindex(close.index).ffill().rename("spy")], axis=1).dropna()
            bars_rel = int(aligned.shape[0])
            if bars_rel > 252:
                for lookback, own_key, rel_key in (
                    (63, "recent_r63", "recent_rel_r63"),
                    (126, "recent_r126", "recent_rel_r126"),
                    (252, "recent_r252", "recent_rel_r252"),
                ):
                    own_ret = recent_metrics.get(own_key, np.nan)
                    if bars_rel > lookback:
                        spy_ret = float(aligned["spy"].iloc[-1] / aligned["spy"].iloc[-1 - lookback] - 1.0)
                        if pd.notna(own_ret) and pd.notna(spy_ret):
                            rel_metrics[rel_key] = float((1.0 + own_ret) / (1.0 + spy_ret) - 1.0)
                valid_rel = {k: v for k, v in rel_metrics.items() if pd.notna(v)}
                if valid_rel:
                    rel_weight_map = {
                        "recent_rel_r63": "recent_r63",
                        "recent_rel_r126": "recent_r126",
                        "recent_rel_r252": "recent_r252",
                    }
                    weight_sum = sum(weights[rel_weight_map[k]] for k in valid_rel)
                    rel_score = sum(weights[rel_weight_map[k]] * valid_rel[k] for k in valid_rel) / weight_sum
            corr_63_spy, _ = _candidate_corr_beta(close, spy_close, downside=False)
            _, downside_beta_63_spy = _candidate_corr_beta(close, spy_close, downside=True)
        sector_rel_metrics = {
            "recent_sector_rel_r63": np.nan,
            "recent_sector_rel_r126": np.nan,
            "recent_sector_rel_r252": np.nan,
        }
        sector_rel_score = np.nan
        sector_rr_score = np.nan
        sector_corr_63 = np.nan
        sector_beta_63 = np.nan
        ctx = context_map.get(ticker, {})
        sector_benchmark = sector_benchmark_for_row(ctx)
        bench_close = benchmark_close_map.get(sector_benchmark, pd.Series(dtype=float))
        if not bench_close.empty:
            aligned = pd.concat([close.rename("close"), bench_close.reindex(close.index).ffill().rename("bench")], axis=1).dropna()
            bars_rel = int(aligned.shape[0])
            if bars_rel > 252:
                for lookback, own_key, rel_key in (
                    (63, "recent_r63", "recent_sector_rel_r63"),
                    (126, "recent_r126", "recent_sector_rel_r126"),
                    (252, "recent_r252", "recent_sector_rel_r252"),
                ):
                    own_ret = recent_metrics.get(own_key, np.nan)
                    if bars_rel > lookback:
                        bench_ret = float(aligned["bench"].iloc[-1] / aligned["bench"].iloc[-1 - lookback] - 1.0)
                        if pd.notna(own_ret) and pd.notna(bench_ret):
                            sector_rel_metrics[rel_key] = float((1.0 + own_ret) / (1.0 + bench_ret) - 1.0)
                valid_sector = {k: v for k, v in sector_rel_metrics.items() if pd.notna(v)}
                if valid_sector:
                    sector_weight_map = {
                        "recent_sector_rel_r63": "recent_r63",
                        "recent_sector_rel_r126": "recent_r126",
                        "recent_sector_rel_r252": "recent_r252",
                    }
                    weight_sum = sum(weights[sector_weight_map[k]] for k in valid_sector)
                    sector_rel_score = sum(weights[sector_weight_map[k]] * valid_sector[k] for k in valid_sector) / weight_sum
            sector_corr_63, sector_beta_63 = _candidate_corr_beta(close, bench_close, downside=False)
        ret1 = close.pct_change(fill_method=None)
        vol63 = ret1.rolling(63, min_periods=40).std() * np.sqrt(252.0)
        vol63_now = float(vol63.iloc[-1]) if vol63.shape[0] else np.nan
        rr_score = np.nan
        if pd.notna(recent_score) and pd.notna(vol63_now) and vol63_now > 0:
            rr_score = float(recent_score / vol63_now)
        if pd.notna(sector_rel_score) and pd.notna(vol63_now) and vol63_now > 0:
            sector_rr_score = float(sector_rel_score / vol63_now)
        breakout_252 = np.nan
        trend200 = np.nan
        dist_sma220 = np.nan
        dd60_now = np.nan
        r21_now = np.nan
        entry_heat_flag = 0
        pullback_quality = 0.0
        if bars_recent >= 252:
            high252 = float(close.iloc[-252:].max())
            if high252 > 0:
                dd252 = float(close.iloc[-1] / high252 - 1.0)
                breakout_252 = float(np.clip(1.0 + dd252 / 0.35, 0.0, 1.0))
        if bars_recent >= 200:
            sma200 = float(close.iloc[-200:].mean())
            if sma200 > 0:
                trend200 = float(np.clip((close.iloc[-1] / sma200 - 1.0) / 1.5, 0.0, 1.0))
        if bars_recent >= 220:
            sma220_now = float(close.iloc[-220:].mean())
            if sma220_now > 0:
                dist_sma220 = float(close.iloc[-1] / sma220_now - 1.0)
        if bars_recent >= 60:
            high60 = float(close.iloc[-60:].max())
            if high60 > 0:
                dd60_now = float(close.iloc[-1] / high60 - 1.0)
        if bars_recent > 21:
            r21_now = float(close.iloc[-1] / close.iloc[-22] - 1.0)
        aug_prices = merge_candidates(base_prices, [ticker], candidate_map)
        feat_df = compute_universe_features(aug_prices.close, cfg)
        feat = feat_df.loc[feat_df["ticker"] == ticker]
        if feat.empty:
            continue
        feat_row = feat.iloc[0]
        latest_rank = feat_row.get("latest_rank", np.nan)
        days_top15 = float(feat_row.get("days_top15_trend", 0.0) or 0.0)
        days_top5 = float(feat_row.get("days_top5_trend", 0.0) or 0.0)
        bars = int(feat_row.get("bars", 0) or 0)
        rank_component = (
            1.0 if pd.notna(latest_rank) and latest_rank <= 5 else
            0.7 if pd.notna(latest_rank) and latest_rank <= 10 else
            0.45 if pd.notna(latest_rank) and latest_rank <= 15 else
            0.15 if pd.notna(latest_rank) and latest_rank <= 25 else
            0.0
        )
        persistence_component = min(days_top15 / 200.0, 1.0) * 0.7 + min(days_top5 / 100.0, 1.0) * 0.3
        bars_component = min(bars / max(min_bars_required, 1), 1.0)
        top5_share = float(days_top5 / max(days_top15, 1.0)) if days_top15 > 0 else 0.0
        latest_score = float(feat_row.get("latest_score", np.nan)) if pd.notna(feat_row.get("latest_score", np.nan)) else np.nan
        if pd.notna(dist_sma220) and pd.notna(dd60_now) and pd.notna(r21_now):
            gap_zscore_20 = safe_float_value(session_metrics.get("gap_zscore_20"), 0.0)
            rel_volume_20 = safe_float_value(session_metrics.get("rel_volume_20"), 1.0)
            gap_fill_share = safe_float_value(session_metrics.get("gap_fill_share"), 0.0)
            close_in_range = safe_float_value(session_metrics.get("close_in_range"), 0.5)
            efficiency_ratio_60 = safe_float_value(session_metrics.get("efficiency_ratio_60"), 0.0)
            heat_core = (dist_sma220 >= 0.70) and (dd60_now >= -0.05) and (r21_now >= 0.10)
            heat_confirmation = (
                (abs(gap_zscore_20) >= 1.25)
                or (rel_volume_20 >= 1.60)
                or ((efficiency_ratio_60 >= 0.35) and (close_in_range >= 0.65))
            )
            if heat_core and heat_confirmation:
                entry_heat_flag = 1
            if (dd60_now <= -0.05) and (dd60_now >= -0.20) and (r21_now <= 0.05):
                pullback_quality = 0.60
                if efficiency_ratio_60 >= 0.25:
                    pullback_quality += 0.15
                if close_in_range >= 0.55:
                    pullback_quality += 0.15
                if gap_fill_share <= 0.40 and abs(gap_zscore_20) <= 1.50:
                    pullback_quality += 0.10
                if (rel_volume_20 >= 0.80) and (rel_volume_20 <= 2.20):
                    pullback_quality += 0.10
                pullback_quality = float(np.clip(pullback_quality, 0.0, 1.0))
        earnings = earnings_map.get(ticker, {})
        earnings_blackout_5d = safe_int_value(earnings.get("earnings_blackout_5d"), 0)
        earnings_blackout_10d = safe_int_value(earnings.get("earnings_blackout_10d"), 0)
        post_earnings_5d = safe_int_value(earnings.get("post_earnings_5d"), 0)
        post_earnings_10d = safe_int_value(earnings.get("post_earnings_10d"), 0)
        days_to_next_earnings = safe_float_value(earnings.get("days_to_next_earnings"), np.nan)
        rows.append(
            {
                "ticker": ticker,
                "scan_bars_if_added": bars,
                "scan_latest_rank_if_added": latest_rank,
                "scan_latest_score_if_added": latest_score,
                "scan_days_top15_if_added": days_top15,
                "scan_days_top5_if_added": days_top5,
                "scan_top15_now_if_added": int(pd.notna(latest_rank) and latest_rank <= 15),
                "scan_top5_now_if_added": int(pd.notna(latest_rank) and latest_rank <= 5),
                "recent_score": float(recent_score) if pd.notna(recent_score) else np.nan,
                "scan_rank_component": rank_component,
                "scan_persistence_component": persistence_component,
                "scan_bars_component": bars_component,
                "scan_top5_share": top5_share,
                "scan_rel_recent_score": float(rel_score) if pd.notna(rel_score) else np.nan,
                "scan_rr_score": float(rr_score) if pd.notna(rr_score) else np.nan,
                "scan_sector_benchmark": sector_benchmark,
                "scan_sector_rel_recent_score": float(sector_rel_score) if pd.notna(sector_rel_score) else np.nan,
                "scan_sector_rr_score": float(sector_rr_score) if pd.notna(sector_rr_score) else np.nan,
                "scan_breakout252_component": float(breakout_252) if pd.notna(breakout_252) else np.nan,
                "scan_trend200_component": float(trend200) if pd.notna(trend200) else np.nan,
                "scan_dist_sma220": float(dist_sma220) if pd.notna(dist_sma220) else np.nan,
                "scan_dd60": float(dd60_now) if pd.notna(dd60_now) else np.nan,
                "scan_r21": float(r21_now) if pd.notna(r21_now) else np.nan,
                "scan_overnight_return_1d": session_metrics.get("overnight_return_1d", np.nan),
                "scan_intraday_return_1d": session_metrics.get("intraday_return_1d", np.nan),
                "scan_gap_pct": session_metrics.get("gap_pct", np.nan),
                "scan_gap_zscore_20": session_metrics.get("gap_zscore_20", np.nan),
                "scan_gap_fill_share": session_metrics.get("gap_fill_share", np.nan),
                "scan_rel_volume_20": session_metrics.get("rel_volume_20", np.nan),
                "scan_volume_zscore_20": session_metrics.get("volume_zscore_20", np.nan),
                "scan_dollar_volume_zscore_20": session_metrics.get("dollar_volume_zscore_20", np.nan),
                "scan_close_in_range": session_metrics.get("close_in_range", np.nan),
                "scan_efficiency_ratio_20": session_metrics.get("efficiency_ratio_20", np.nan),
                "scan_efficiency_ratio_60": session_metrics.get("efficiency_ratio_60", np.nan),
                "scan_corr_63_spy": float(corr_63_spy) if pd.notna(corr_63_spy) else np.nan,
                "scan_downside_beta_63_spy": float(downside_beta_63_spy) if pd.notna(downside_beta_63_spy) else np.nan,
                "scan_sector_corr_63": float(sector_corr_63) if pd.notna(sector_corr_63) else np.nan,
                "scan_sector_beta_63": float(sector_beta_63) if pd.notna(sector_beta_63) else np.nan,
                "scan_entry_heat_flag": int(entry_heat_flag),
                "scan_pullback_quality": float(pullback_quality),
                "scan_days_to_next_earnings": days_to_next_earnings,
                "scan_earnings_blackout_5d": int(earnings_blackout_5d),
                "scan_earnings_blackout_10d": int(earnings_blackout_10d),
                "scan_post_earnings_5d": int(post_earnings_5d),
                "scan_post_earnings_10d": int(post_earnings_10d),
                **rel_metrics,
                **sector_rel_metrics,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    latest_score_pct = rank_within_series(df["scan_latest_score_if_added"].fillna(df["scan_latest_score_if_added"].median(skipna=True)))
    recent_score_pct = rank_within_series(df["recent_score"].fillna(df["recent_score"].median(skipna=True)))
    rel_recent_score_pct = rank_within_series(df["scan_rel_recent_score"].fillna(df["scan_rel_recent_score"].median(skipna=True)))
    rr_score_pct = rank_within_series(df["scan_rr_score"].fillna(df["scan_rr_score"].median(skipna=True)))
    sector_rel_recent_score_pct = rank_within_series(df["scan_sector_rel_recent_score"].fillna(df["scan_sector_rel_recent_score"].median(skipna=True)))
    sector_rr_score_pct = rank_within_series(df["scan_sector_rr_score"].fillna(df["scan_sector_rr_score"].median(skipna=True)))
    breakout_pct = df["scan_breakout252_component"].fillna(0.5)
    trend200_pct = df["scan_trend200_component"].fillna(0.5)
    gap_zscore = df["scan_gap_zscore_20"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gap_abs = gap_zscore.abs().clip(upper=4.0)
    gap_fill_share = df["scan_gap_fill_share"].fillna(0.0).clip(lower=0.0, upper=1.0)
    rel_volume = df["scan_rel_volume_20"].fillna(1.0).clip(lower=0.0, upper=6.0)
    dollar_volume_z = df["scan_dollar_volume_zscore_20"].fillna(0.0).clip(lower=-3.0, upper=3.0)
    close_in_range = df["scan_close_in_range"].fillna(0.5).clip(lower=0.0, upper=1.0)
    efficiency20 = df["scan_efficiency_ratio_20"].fillna(df["scan_efficiency_ratio_20"].median(skipna=True)).clip(lower=0.0, upper=1.0)
    efficiency60 = df["scan_efficiency_ratio_60"].fillna(df["scan_efficiency_ratio_60"].median(skipna=True)).clip(lower=0.0, upper=1.0)
    corr_63_spy = df["scan_corr_63_spy"].fillna(0.0).clip(lower=-1.0, upper=1.0)
    downside_beta_63_spy = df["scan_downside_beta_63_spy"].fillna(0.0).clip(lower=-3.0, upper=4.0)
    sector_corr_63 = df["scan_sector_corr_63"].fillna(0.0).clip(lower=-1.0, upper=1.0)
    sector_beta_63 = df["scan_sector_beta_63"].fillna(0.0).clip(lower=-3.0, upper=4.0)
    gap_discipline = (0.6 * (1.0 - gap_abs / 4.0) + 0.4 * (1.0 - gap_fill_share)).clip(lower=0.0, upper=1.0)
    session_quality_pct = rank_within_series(0.50 * close_in_range + 0.25 * gap_discipline + 0.25 * efficiency20)
    volume_confirmation_pct = rank_within_series(np.log1p(rel_volume) + 0.35 * dollar_volume_z)
    trend_efficiency_pct = rank_within_series(0.35 * efficiency20 + 0.65 * efficiency60)
    crowding_penalty = (
        0.30 * (corr_63_spy >= 0.85).astype(float)
        + 0.20 * (downside_beta_63_spy >= 1.20).astype(float)
        + 0.30 * (sector_corr_63 >= 0.92).astype(float)
        + 0.20 * (sector_beta_63 >= 1.40).astype(float)
    )
    blowoff_penalty = (
        0.40 * (gap_abs >= 2.00).astype(float)
        + 0.30 * (gap_fill_share >= 0.75).astype(float)
        + 0.30 * ((rel_volume >= 3.0) & (close_in_range <= 0.45)).astype(float)
    )
    heat_penalty = df["scan_entry_heat_flag"].fillna(0).astype(float)
    pullback_bonus = df["scan_pullback_quality"].fillna(0.0)
    earnings_blackout_5d = df["scan_earnings_blackout_5d"].fillna(0).astype(float)
    earnings_blackout_10d = df["scan_earnings_blackout_10d"].fillna(0).astype(float)
    post_earnings_5d = df["scan_post_earnings_5d"].fillna(0).astype(float)
    df["scan_algo_compat_score"] = (
        5.0 * df["scan_rank_component"]
        + 3.0 * df["scan_persistence_component"]
        + 2.0 * latest_score_pct
        + 1.0 * df["scan_bars_component"]
    )
    df["scan_algo_compat_score_v2"] = (
        4.0 * df["scan_rank_component"]
        + 2.2 * df["scan_persistence_component"]
        + 1.5 * latest_score_pct
        + 1.0 * df["scan_bars_component"]
        + 1.0 * df["scan_top5_share"]
        + 1.0 * rel_recent_score_pct
        + 0.8 * rr_score_pct
        + 0.6 * sector_rel_recent_score_pct
        + 0.4 * sector_rr_score_pct
        + 0.6 * breakout_pct
        + 0.6 * trend200_pct
        + 0.4 * session_quality_pct
        + 0.3 * volume_confirmation_pct
        + 0.4 * trend_efficiency_pct
        + 0.3 * pullback_bonus
        - 0.8 * heat_penalty
        - 0.4 * blowoff_penalty
        - 0.3 * crowding_penalty
        - 0.4 * earnings_blackout_5d
        - 0.2 * earnings_blackout_10d
        + 0.2 * post_earnings_5d
    )
    df["scan_recent_score_pct"] = recent_score_pct
    df["scan_emerging_score"] = (
        4.0 * df["scan_rank_component"]
        + 3.0 * recent_score_pct
        + 2.0 * latest_score_pct
        + 1.0 * df["scan_bars_component"]
    )
    df["scan_emerging_score_v2"] = (
        3.0 * df["scan_rank_component"]
        + 2.5 * recent_score_pct
        + 1.5 * rel_recent_score_pct
        + 1.0 * rr_score_pct
        + 1.0 * breakout_pct
        + 1.0 * latest_score_pct
        + 0.5 * df["scan_bars_component"]
        + 0.4 * session_quality_pct
        + 0.4 * volume_confirmation_pct
        + 0.4 * trend_efficiency_pct
        + 0.3 * pullback_bonus
        - 0.8 * heat_penalty
        - 0.5 * blowoff_penalty
        - 0.2 * crowding_penalty
    )
    early_leader_young_bonus = (1.0 - df["scan_bars_component"]).clip(lower=0.0, upper=0.75)
    df["scan_early_leader_score"] = (
        2.8 * df["scan_rank_component"]
        + 2.2 * recent_score_pct
        + 1.6 * rel_recent_score_pct
        + 1.1 * rr_score_pct
        + 0.9 * breakout_pct
        + 0.6 * trend200_pct
        + 0.5 * session_quality_pct
        + 0.5 * volume_confirmation_pct
        + 0.5 * trend_efficiency_pct
        + 0.7 * pullback_bonus
        + 0.5 * df["scan_top5_share"]
        + 0.4 * early_leader_young_bonus
        - 1.0 * heat_penalty
        - 0.5 * blowoff_penalty
        - 0.25 * crowding_penalty
        - 0.25 * earnings_blackout_5d
        + 0.2 * post_earnings_5d
    )
    df["scan_early_leader_fit"] = np.select(
        [
            df["scan_early_leader_score"] >= 8.5,
            df["scan_early_leader_score"] >= 6.5,
            df["scan_early_leader_score"] >= 4.5,
        ],
        [
            "high",
            "medium",
            "low",
        ],
        default="weak",
    )
    compounder_extension_bonus = (
        ((df["scan_dist_sma220"].fillna(0.0) >= 0.05) & (df["scan_dist_sma220"].fillna(0.0) <= 0.85)).astype(float)
        + ((df["scan_r21"].fillna(0.0) >= -0.08) & (df["scan_r21"].fillna(0.0) <= 0.28)).astype(float)
    ) / 2.0
    compounder_stability = (
        0.55 * session_quality_pct
        + 0.45 * trend_efficiency_pct
    )
    df["scan_quality_compounder_score"] = (
        2.4 * df["scan_rank_component"]
        + 2.6 * df["scan_persistence_component"]
        + 1.3 * rel_recent_score_pct
        + 0.9 * sector_rel_recent_score_pct
        + 0.9 * trend200_pct
        + 0.6 * breakout_pct
        + 0.9 * compounder_stability
        + 0.3 * volume_confirmation_pct
        + 0.5 * df["scan_bars_component"]
        + 0.4 * df["scan_top5_share"]
        + 0.4 * pullback_bonus
        + 0.4 * compounder_extension_bonus
        - 0.45 * heat_penalty
        - 0.45 * blowoff_penalty
        - 0.15 * crowding_penalty
        - 0.25 * earnings_blackout_5d
        + 0.15 * post_earnings_5d
    )
    df["scan_quality_compounder_fit"] = np.select(
        [
            df["scan_quality_compounder_score"] >= 8.2,
            df["scan_quality_compounder_score"] >= 6.3,
            df["scan_quality_compounder_score"] >= 4.8,
        ],
        [
            "high",
            "medium",
            "low",
        ],
        default="weak",
    )
    hot_constructive_mask = (
        (heat_penalty > 0)
        & (df["scan_rel_recent_score"].fillna(0.0) >= 4.5)
        & (df["scan_rr_score"].fillna(0.0) >= 6.5)
        & (df["scan_r21"].fillna(0.0) <= 0.30)
        & (df["scan_dist_sma220"].fillna(0.0) >= 0.25)
        & (df["scan_dist_sma220"].fillna(0.0) <= 1.00)
        & (gap_abs <= 1.75)
        & (gap_fill_share <= 0.60)
        & (close_in_range >= 0.55)
        & (efficiency60 >= 0.25)
        & (sector_corr_63 <= 0.93)
    )
    hot_late_mask = (
        (heat_penalty > 0)
        & (
            (df["scan_dist_sma220"].fillna(0.0) >= 1.20)
            | (df["scan_r21"].fillna(0.0) >= 0.60)
            | (gap_abs >= 2.25)
            | (gap_fill_share >= 0.75)
            | ((rel_volume >= 3.0) & (close_in_range <= 0.45))
            | ((sector_corr_63 >= 0.95) & (sector_beta_63 >= 1.50) & (df["scan_dist_sma220"].fillna(0.0) >= 0.80))
        )
    )
    df["scan_hot_candidate_score"] = (
        0.8 * rel_recent_score_pct
        + 0.8 * rr_score_pct
        + 0.4 * breakout_pct
        + 0.3 * trend200_pct
        + 0.3 * session_quality_pct
        + 0.2 * volume_confirmation_pct
        + 0.2 * trend_efficiency_pct
        + 0.2 * df["scan_bars_component"]
        - 1.0 * heat_penalty
        - 0.8 * blowoff_penalty
        - 0.6 * crowding_penalty
        - 1.2 * (df["scan_dist_sma220"].fillna(0.0) >= 1.20).astype(float)
        - 1.0 * (df["scan_r21"].fillna(0.0) >= 0.60).astype(float)
        + 1.0 * hot_constructive_mask.astype(float)
        - 1.0 * hot_late_mask.astype(float)
    )
    df["scan_hot_archetype"] = np.select(
        [
            hot_constructive_mask,
            hot_late_mask,
            heat_penalty > 0,
        ],
        [
            "hot_constructive",
            "hot_late",
            "hot_watch",
        ],
        default="not_hot",
    )
    df["scan_algo_fit"] = np.select(
        [
            df["scan_algo_compat_score_v2"] >= 9.5,
            df["scan_algo_compat_score_v2"] >= 7.0,
            df["scan_algo_compat_score_v2"] >= 5.0,
        ],
        [
            "high",
            "medium",
            "low",
        ],
        default="weak",
    )
    df["scan_candidate_track"] = np.select(
        [
            (
                df["scan_quality_compounder_score"] >= 7.0
            )
            & (df["scan_persistence_component"] >= 0.40)
            & (df["scan_trend200_component"].fillna(0.0) >= 0.55),
            (df["scan_rank_component"] >= 0.7) & (df["scan_persistence_component"] >= 0.45),
            (df["scan_rank_component"] >= 0.7) & (df["scan_recent_score_pct"] >= 0.75),
            df["scan_recent_score_pct"] >= 0.85,
        ],
        [
            "quality_compounder",
            "persistent_leader",
            "emerging_leader",
            "emerging_leader",
        ],
        default="fringe",
    )
    return df.drop(columns=["recent_score"], errors="ignore")


def compute_recent_momentum_snapshot(tickers: Sequence[str], end_date: pd.Timestamp | None = None) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", "recent_bars", "recent_r63", "recent_r126", "recent_r252", "recent_score"])
    if end_date is None:
        end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=500)
    rows = []
    batch = download_history_batch(
        tickers,
        str(start_date.date()),
        str((end_date + pd.Timedelta(days=1)).date()),
    )
    for ticker in dedupe_keep_order(tickers):
        df = normalize_downloaded_columns(batch.get(ticker, pd.DataFrame()))
        if df.empty or "close" not in df.columns:
            continue
        bars, metrics, score = _recent_momentum_metrics(df["close"])
        rows.append({"ticker": ticker, "recent_bars": bars, "recent_score": score, **metrics})
    return pd.DataFrame(rows)


def merge_candidates(base_prices, picked: list[str], candidate_map: dict[str, CandidateData]):
    union_index = base_prices.close.index
    for ticker in picked:
        union_index = union_index.union(candidate_map[ticker].close.index)
    union_index = union_index.sort_values()

    open_df = base_prices.open.reindex(union_index).ffill()
    close_df = base_prices.close.reindex(union_index).ffill()
    for ticker in picked:
        cand = candidate_map[ticker]
        open_df[ticker] = cand.open.reindex(union_index).ffill()
        close_df[ticker] = cand.close.reindex(union_index).ffill()
    return type(base_prices)(open=open_df, close=close_df)


def safe_float_value(value: object, default: float = np.nan) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int_value(value: object, default: int = 0) -> int:
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_cached_rows(path: Path | None, key_col: str = "name") -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty or key_col not in df.columns:
        return {}
    df[key_col] = df[key_col].astype(str)
    df = df.drop_duplicates(key_col, keep="last")
    return {str(row[key_col]): row.to_dict() for _, row in df.iterrows()}


def should_retest_single(row, cached: dict | None) -> tuple[bool, str]:
    if not cached:
        return True, "new_candidate"
    if str(cached.get("candidate_status") or "") in {"download_failed", "baseline"}:
        return True, "retry_after_failure"
    prev_recent = safe_float_value(cached.get("recent_score"), 0.0)
    curr_recent = safe_float_value(getattr(row, "recent_score", np.nan), 0.0)
    recent_jump = curr_recent - prev_recent
    recent_needed = max(RETEST_RECENT_ABS_DELTA, abs(prev_recent) * RETEST_RECENT_REL_DELTA)
    if recent_jump >= recent_needed:
        return True, "recent_score_jump"
    prev_compat = safe_float_value(
        cached.get("scan_algo_compat_score_v2"),
        safe_float_value(cached.get("scan_algo_compat_score"), 0.0),
    )
    curr_compat = safe_float_value(
        getattr(row, "scan_algo_compat_score_v2", np.nan),
        safe_float_value(getattr(row, "scan_algo_compat_score", np.nan), 0.0),
    )
    if curr_compat - prev_compat >= RETEST_COMPAT_DELTA:
        return True, "compat_score_jump"
    prev_rank = safe_float_value(cached.get("scan_latest_rank_if_added"), np.nan)
    curr_rank = safe_float_value(getattr(row, "scan_latest_rank_if_added", np.nan), np.nan)
    if pd.notna(prev_rank) and pd.notna(curr_rank) and curr_rank <= (prev_rank - RETEST_RANK_DELTA):
        return True, "rank_jump"
    prev_top5 = safe_float_value(cached.get("scan_top5_share"), 0.0)
    curr_top5 = safe_float_value(getattr(row, "scan_top5_share", np.nan), 0.0)
    if curr_top5 - prev_top5 >= RETEST_TOP5_SHARE_DELTA:
        return True, "top5_share_jump"
    prev_rel = safe_float_value(cached.get("scan_rel_recent_score"), 0.0)
    curr_rel = safe_float_value(getattr(row, "scan_rel_recent_score", np.nan), 0.0)
    if curr_rel - prev_rel >= RETEST_RELATIVE_SCORE_DELTA:
        return True, "relative_score_jump"
    prev_track = str(cached.get("scan_candidate_track") or "")
    curr_track = str(getattr(row, "scan_candidate_track", "") or "")
    if curr_track == "persistent_leader" and prev_track != "persistent_leader":
        return True, "track_upgrade"
    return False, "cached_no_positive_variation"


def current_scan_payload(row, feat: dict, backtestable: int) -> dict[str, object]:
    return {
        "priority_score": float(getattr(row, "priority_score", 0.0) or 0.0),
        "source_count": int(getattr(row, "source_count", 0) or 0),
        "source_types": getattr(row, "source_types", ""),
        "sector": getattr(row, "sector", ""),
        "industry": getattr(row, "industry", ""),
        "recent_score": getattr(row, "recent_score", np.nan),
        "recent_r63": getattr(row, "recent_r63", np.nan),
        "recent_r126": getattr(row, "recent_r126", np.nan),
        "recent_r252": getattr(row, "recent_r252", np.nan),
        "scan_algo_compat_score": getattr(row, "scan_algo_compat_score", np.nan),
        "scan_algo_compat_score_v2": getattr(row, "scan_algo_compat_score_v2", np.nan),
        "scan_algo_fit": getattr(row, "scan_algo_fit", ""),
        "scan_latest_rank_if_added": getattr(row, "scan_latest_rank_if_added", np.nan),
        "scan_days_top15_if_added": getattr(row, "scan_days_top15_if_added", np.nan),
        "scan_days_top5_if_added": getattr(row, "scan_days_top5_if_added", np.nan),
        "scan_top5_share": getattr(row, "scan_top5_share", np.nan),
        "scan_rel_recent_score": getattr(row, "scan_rel_recent_score", np.nan),
        "scan_rr_score": getattr(row, "scan_rr_score", np.nan),
        "scan_sector_benchmark": getattr(row, "scan_sector_benchmark", ""),
        "scan_sector_rel_recent_score": getattr(row, "scan_sector_rel_recent_score", np.nan),
        "scan_sector_rr_score": getattr(row, "scan_sector_rr_score", np.nan),
        "scan_breakout252_component": getattr(row, "scan_breakout252_component", np.nan),
        "scan_trend200_component": getattr(row, "scan_trend200_component", np.nan),
        "scan_dist_sma220": getattr(row, "scan_dist_sma220", np.nan),
        "scan_dd60": getattr(row, "scan_dd60", np.nan),
        "scan_r21": getattr(row, "scan_r21", np.nan),
        "scan_overnight_return_1d": getattr(row, "scan_overnight_return_1d", np.nan),
        "scan_intraday_return_1d": getattr(row, "scan_intraday_return_1d", np.nan),
        "scan_gap_pct": getattr(row, "scan_gap_pct", np.nan),
        "scan_gap_zscore_20": getattr(row, "scan_gap_zscore_20", np.nan),
        "scan_gap_fill_share": getattr(row, "scan_gap_fill_share", np.nan),
        "scan_rel_volume_20": getattr(row, "scan_rel_volume_20", np.nan),
        "scan_volume_zscore_20": getattr(row, "scan_volume_zscore_20", np.nan),
        "scan_dollar_volume_zscore_20": getattr(row, "scan_dollar_volume_zscore_20", np.nan),
        "scan_close_in_range": getattr(row, "scan_close_in_range", np.nan),
        "scan_efficiency_ratio_20": getattr(row, "scan_efficiency_ratio_20", np.nan),
        "scan_efficiency_ratio_60": getattr(row, "scan_efficiency_ratio_60", np.nan),
        "scan_corr_63_spy": getattr(row, "scan_corr_63_spy", np.nan),
        "scan_downside_beta_63_spy": getattr(row, "scan_downside_beta_63_spy", np.nan),
        "scan_sector_corr_63": getattr(row, "scan_sector_corr_63", np.nan),
        "scan_sector_beta_63": getattr(row, "scan_sector_beta_63", np.nan),
        "scan_entry_heat_flag": int(getattr(row, "scan_entry_heat_flag", 0) or 0),
        "scan_pullback_quality": getattr(row, "scan_pullback_quality", np.nan),
        "scan_quality_compounder_score": getattr(row, "scan_quality_compounder_score", np.nan),
        "scan_quality_compounder_fit": getattr(row, "scan_quality_compounder_fit", ""),
        "scan_days_to_next_earnings": getattr(row, "scan_days_to_next_earnings", np.nan),
        "scan_earnings_blackout_5d": int(getattr(row, "scan_earnings_blackout_5d", 0) or 0),
        "scan_earnings_blackout_10d": int(getattr(row, "scan_earnings_blackout_10d", 0) or 0),
        "scan_post_earnings_5d": int(getattr(row, "scan_post_earnings_5d", 0) or 0),
        "scan_post_earnings_10d": int(getattr(row, "scan_post_earnings_10d", 0) or 0),
        "scan_candidate_track": getattr(row, "scan_candidate_track", ""),
        "backtestable_now": int(backtestable),
        "bars": int(feat["bars"]),
        "latest_rank_if_added": float(feat["latest_rank"]) if pd.notna(feat["latest_rank"]) else np.nan,
        "days_top15_trend_if_added": int(feat["days_top15_trend"]),
        "days_top5_trend_if_added": int(feat["days_top5_trend"]),
    }


def build_reused_single_row(ticker: str, cached: dict, row, feat: dict, backtestable: int, reason: str) -> dict[str, object]:
    out = dict(cached)
    out["name"] = ticker
    out.update(current_scan_payload(row, feat, backtestable))
    out["test_reused"] = 1
    out["retest_reason"] = reason
    out["tested_as_of"] = str(cached.get("tested_as_of") or "")
    return out


def evaluate_candidates(
    engine,
    cfg_doc: dict,
    cfg: dict,
    pp: dict,
    base_prices,
    discovery_df: pd.DataFrame,
    max_single_backtests: int,
    max_combo_backtests: int,
    single_cache_path: Path | None = None,
    combo_cache_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if discovery_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    full_start = cfg_doc["metrics"]["full_period"]["start"]
    full_end = cfg_doc["metrics"]["full_period"]["end"]
    oos_start = cfg_doc["metrics"]["oos_2022_2026"]["start"]
    oos_end = cfg_doc["metrics"]["oos_2022_2026"]["end"]

    _, _, baseline_full = run_metrics(engine, base_prices, cfg, pp, full_start, full_end)
    _, _, baseline_oos = run_metrics(engine, base_prices, cfg, pp, oos_start, oos_end)
    baseline_row = row_from_run("baseline_184", baseline_full, baseline_oos, {"candidate_status": "baseline"})
    baseline_row["tested_as_of"] = date.today().isoformat()
    baseline_row["test_reused"] = 0
    baseline_row["retest_reason"] = "baseline"

    shortlist = discovery_df.loc[
        (discovery_df["candidate_status"] == "new")
        & (~discovery_df["ticker"].isin(base_prices.close.columns))
    ].copy()
    preselect_pool_size = max(max_single_backtests * 10, 40)
    preselect = shortlist.sort_values(["priority_score", "marketCap", "averageDailyVolume3Month"], ascending=[False, False, False]).head(preselect_pool_size)
    download_start = str(pd.to_datetime(base_prices.close.index.min()).date())
    download_end = str((pd.to_datetime(base_prices.close.index.max()) + pd.Timedelta(days=1)).date())
    candidate_data = download_candidate_data(preselect["ticker"].tolist(), download_start, download_end)
    recent = compute_recent_momentum_from_candidate_map(candidate_data)
    compat = compute_algo_fit_for_candidates(base_prices, cfg, candidate_data, preselect["ticker"].tolist())
    preselect = preselect.merge(recent, on="ticker", how="left").merge(compat, on="ticker", how="left")
    min_bars_required = int(cfg["min_bars_required"])
    top_compat = preselect.sort_values(
        ["scan_algo_compat_score_v2", "scan_algo_compat_score", "recent_score", "priority_score", "marketCap"],
        ascending=[False, False, False, False, False],
    ).head(max_single_backtests * 2)
    top_emerging = preselect.loc[
        (preselect["scan_bars_if_added"].fillna(0) >= min_bars_required)
        & (preselect["scan_latest_rank_if_added"].fillna(999) <= 15)
        & (preselect["recent_score"].fillna(-999) > 0)
    ].sort_values(
        ["scan_emerging_score_v2", "scan_emerging_score", "recent_score", "priority_score"],
        ascending=[False, False, False, False],
    ).head(max_single_backtests * 2)
    top_priority = preselect.sort_values(
        ["priority_score", "marketCap", "averageDailyVolume3Month"],
        ascending=[False, False, False],
    ).head(max_single_backtests)
    shortlist = (
        pd.concat([top_compat, top_emerging, top_priority], ignore_index=True)
        .drop_duplicates("ticker")
        .sort_values(
            ["scan_algo_compat_score_v2", "scan_emerging_score_v2", "scan_algo_compat_score", "recent_score", "priority_score", "marketCap"],
            ascending=[False, False, False, False, False, False],
        )
        .head(max(max_single_backtests * 3, 12))
    )
    if shortlist.empty:
        return pd.DataFrame([baseline_row]), pd.DataFrame(), preselect

    single_cache = load_cached_rows(single_cache_path, key_col="name")
    single_rows = [baseline_row]
    today_str = date.today().isoformat()
    eval_queue: list[tuple[object, str, dict | None]] = []
    reuse_queue: list[tuple[object, str, dict]] = []
    retested_names: set[str] = set()

    for row in shortlist.itertuples(index=False):
        ticker = str(row.ticker)
        cached = single_cache.get(ticker)
        retest, reason = should_retest_single(row, cached)
        if retest and len(eval_queue) < max_single_backtests:
            eval_queue.append((row, reason, cached))
            retested_names.add(ticker)
        elif cached:
            reuse_queue.append((row, reason, cached))

    featured_reuse_names = {
        str(row.ticker)
        for row in shortlist.head(max_single_backtests).itertuples(index=False)
        if str(row.ticker) not in retested_names and str(row.ticker) in single_cache
    }

    for row, retest_reason, _cached in eval_queue:
        ticker = str(row.ticker)
        cand = candidate_data.get(ticker)
        if cand is None:
            single_rows.append(
                {
                    "name": ticker,
                    "candidate_status": "download_failed",
                    "priority_score": float(row.priority_score),
                    "source_count": int(row.source_count),
                    "tested_as_of": today_str,
                    "test_reused": 0,
                    "retest_reason": "download_failed",
                }
            )
            continue

        aug_prices = merge_candidates(base_prices, [ticker], candidate_data)
        aug_features = compute_universe_features(aug_prices.close, cfg)
        feat = aug_features.loc[aug_features["ticker"] == ticker].iloc[0].to_dict()
        backtestable = int(feat["bars"]) >= min_bars_required
        if backtestable:
            _, trades_full, full = run_metrics(engine, aug_prices, cfg, pp, full_start, full_end)
            _, trades_oos, oos = run_metrics(engine, aug_prices, cfg, pp, oos_start, oos_end)
            pnl_full = compute_realized_pnl(trades_full)
            pnl_oos = compute_realized_pnl(trades_oos)
            row_full = pnl_full.loc[pnl_full["ticker"] == ticker]
            row_oos = pnl_oos.loc[pnl_oos["ticker"] == ticker]
            realized_full = float(row_full["realized_pnl_eur"].iloc[0]) if not row_full.empty else 0.0
            realized_oos = float(row_oos["realized_pnl_eur"].iloc[0]) if not row_oos.empty else 0.0
            buys_full = int(row_full["buy_count"].iloc[0]) if not row_full.empty else 0
            buys_oos = int(row_oos["buy_count"].iloc[0]) if not row_oos.empty else 0
        else:
            full = baseline_full
            oos = baseline_oos
            realized_full = 0.0
            realized_oos = 0.0
            buys_full = 0
            buys_oos = 0

        add_row = row_from_run(
            ticker,
            full,
            oos,
            {
                **current_scan_payload(row, feat, backtestable),
                "full_realized_pnl_if_added": realized_full,
                "oos_realized_pnl_if_added": realized_oos,
                "full_buy_count_if_added": buys_full,
                "oos_buy_count_if_added": buys_oos,
                "full_delta_roi_pct": float(full["ROI_%"] - baseline_full["ROI_%"]),
                "full_delta_sharpe": float(full["Sharpe"] - baseline_full["Sharpe"]),
                "full_delta_maxdd_pct": float(full["MaxDD_%"] - baseline_full["MaxDD_%"]),
                "oos_delta_roi_pct": float(oos["ROI_%"] - baseline_oos["ROI_%"]),
                "oos_delta_sharpe": float(oos["Sharpe"] - baseline_oos["Sharpe"]),
                "oos_delta_maxdd_pct": float(oos["MaxDD_%"] - baseline_oos["MaxDD_%"]),
                "tested_as_of": today_str,
                "test_reused": 0,
                "retest_reason": retest_reason,
            },
        )
        add_row["recommendation"] = (
            "add"
            if backtestable
            and add_row["full_delta_roi_pct"] > 0
            and add_row["oos_delta_roi_pct"] > 0
            and add_row["oos_delta_sharpe"] >= 0
            and add_row["oos_delta_maxdd_pct"] > -1.0
            else "watch"
            if backtestable and (add_row["oos_delta_roi_pct"] > 0 or add_row["full_delta_roi_pct"] > 0)
            else "reject"
        )
        single_rows.append(add_row)

    for row, reuse_reason, cached in reuse_queue:
        ticker = str(row.ticker)
        if ticker not in featured_reuse_names:
            continue
        cand = candidate_data.get(ticker)
        if cand is not None:
            aug_prices = merge_candidates(base_prices, [ticker], candidate_data)
            aug_features = compute_universe_features(aug_prices.close, cfg)
            feat = aug_features.loc[aug_features["ticker"] == ticker].iloc[0].to_dict()
            backtestable = int(feat["bars"]) >= min_bars_required
        else:
            feat = {
                "bars": safe_int_value(cached.get("bars")),
                "latest_rank": safe_float_value(cached.get("latest_rank_if_added")),
                "days_top15_trend": safe_int_value(cached.get("days_top15_trend_if_added")),
                "days_top5_trend": safe_int_value(cached.get("days_top5_trend_if_added")),
            }
            backtestable = safe_int_value(cached.get("backtestable_now"))
        single_rows.append(build_reused_single_row(ticker, cached, row, feat, backtestable, reuse_reason))

    single_df = pd.DataFrame(single_rows)
    if single_df.empty:
        return single_df, pd.DataFrame(), preselect

    combo_cache = load_cached_rows(combo_cache_path, key_col="name")
    combo_rows = []
    combo_candidates = (
        single_df.loc[
            (single_df["name"] != "baseline_184")
            & (single_df["recommendation"].isin(["add", "watch"]))
            & (single_df["backtestable_now"] == 1)
        ]
        .sort_values(["oos_delta_roi_pct", "oos_delta_sharpe", "full_delta_roi_pct"], ascending=[False, False, False])
        .head(max_combo_backtests)["name"]
        .tolist()
    )

    tested = 0
    for i, left in enumerate(combo_candidates):
        for right in combo_candidates[i + 1 :]:
            combo = tuple(sorted((left, right)))
            combo_name = "+".join(combo)
            cached_combo = combo_cache.get(combo_name)
            if cached_combo and not any(name in retested_names for name in combo):
                reused_combo = dict(cached_combo)
                reused_combo["name"] = combo_name
                reused_combo["members"] = ",".join(combo)
                reused_combo["tested_as_of"] = str(cached_combo.get("tested_as_of") or "")
                reused_combo["test_reused"] = 1
                reused_combo["retest_reason"] = "cached_no_member_retest"
                combo_rows.append(reused_combo)
                tested += 1
                if tested >= max_combo_backtests:
                    break
                continue
            if any(name not in candidate_data for name in combo):
                continue
            aug_prices = merge_candidates(base_prices, list(combo), candidate_data)
            _, _, full = run_metrics(engine, aug_prices, cfg, pp, full_start, full_end)
            _, _, oos = run_metrics(engine, aug_prices, cfg, pp, oos_start, oos_end)
            combo_rows.append(
                row_from_run(
                    combo_name,
                    full,
                    oos,
                    {
                        "members": ",".join(combo),
                        "full_delta_roi_pct": float(full["ROI_%"] - baseline_full["ROI_%"]),
                        "full_delta_sharpe": float(full["Sharpe"] - baseline_full["Sharpe"]),
                        "full_delta_maxdd_pct": float(full["MaxDD_%"] - baseline_full["MaxDD_%"]),
                        "oos_delta_roi_pct": float(oos["ROI_%"] - baseline_oos["ROI_%"]),
                        "oos_delta_sharpe": float(oos["Sharpe"] - baseline_oos["Sharpe"]),
                        "oos_delta_maxdd_pct": float(oos["MaxDD_%"] - baseline_oos["MaxDD_%"]),
                        "tested_as_of": today_str,
                        "test_reused": 0,
                        "retest_reason": "member_retested_or_new",
                    },
                )
            )
            tested += 1
            if tested >= max_combo_backtests:
                break
        if tested >= max_combo_backtests:
            break
    combo_df = pd.DataFrame(combo_rows)
    single_df = single_df.sort_values(["oos_delta_roi_pct", "oos_delta_sharpe", "full_delta_roi_pct"], ascending=[False, False, False])
    if not combo_df.empty:
        combo_df = combo_df.sort_values(["oos_delta_roi_pct", "oos_delta_sharpe", "full_delta_roi_pct"], ascending=[False, False, False])
    preselect = preselect.sort_values(
        ["scan_algo_compat_score", "recent_score", "priority_score", "marketCap"],
        ascending=[False, False, False, False],
    )
    return single_df, combo_df, preselect


def format_small(df: pd.DataFrame, rows: int = 10) -> str:
    if df.empty:
        return "(none)"
    return df.head(rows).to_string(index=False)


def write_report(
    report_path: Path,
    *,
    mode: str,
    source_profile: str,
    seed_tickers: Sequence[str],
    keywords: Sequence[str],
    discovery_df: pd.DataFrame,
    compat_df: pd.DataFrame,
    single_df: pd.DataFrame,
    combo_df: pd.DataFrame,
    current_leaders: Sequence[str],
) -> None:
    new_df = discovery_df.loc[discovery_df["candidate_status"] == "new"].copy()
    recommended = pd.DataFrame()
    watch = pd.DataFrame()
    if not single_df.empty and "recommendation" in single_df.columns:
        recommended = single_df.loc[single_df["recommendation"] == "add"].copy()
        watch = single_df.loc[single_df["recommendation"] == "watch"].copy()

    lines = [
        f"# Dynamic Universe Discovery ({mode})",
        "",
        "## Purpose",
        "",
        "- build a dynamic candidate pool without relying only on a fixed universe",
        "- combine broad discovery with local structural leader sentinels and, when enabled, Yahoo expansion layers",
        "- score candidates with the current APEX baseline through single-addition backtests and small combo tests",
        "",
        "## Source Profile",
        "",
        f"- `{source_profile}`",
        "",
        "## Current raw leaders",
        "",
        "- " + ", ".join(current_leaders),
        "",
        "## Seed tickers",
        "",
        "- " + (", ".join(seed_tickers) if seed_tickers else "(none)"),
        "",
        "## Keywords",
        "",
        "- " + (", ".join(keywords) if keywords else "(none)"),
        "",
        "## Discovery summary",
        "",
        f"- total discovered symbols: `{len(discovery_df)}`",
        f"- new symbols outside active/reserve/exclusions: `{int((discovery_df['candidate_status'] == 'new').sum()) if not discovery_df.empty else 0}`",
        "",
        "## Top new candidates by discovery score",
        "",
        format_small(
            safe_columns(
                new_df,
                [
                    "ticker",
                    "priority_score",
                    "source_count",
                    "source_types",
                    "marketCap",
                    "averageDailyVolume3Month",
                    "sector",
                    "industry",
                ],
            ).round(4),
            rows=25,
        ) if not new_df.empty else "(none)",
        "",
        "## Top algo-compatible candidates before backtest",
        "",
        format_small(
            safe_columns(
                compat_df.loc[compat_df["candidate_status"] == "new"] if not compat_df.empty and "candidate_status" in compat_df.columns else compat_df,
                [
                    "ticker",
                    "scan_algo_fit",
                    "scan_algo_compat_score",
                    "scan_quality_compounder_fit",
                    "scan_quality_compounder_score",
                    "scan_latest_rank_if_added",
                    "scan_days_top15_if_added",
                    "scan_days_top5_if_added",
                    "recent_score",
                    "priority_score",
                    "source_types",
                ],
            ).round(4),
            rows=25,
        ) if not compat_df.empty else "(none)",
        "",
        "## Recommended additions",
        "",
        format_small(
            safe_columns(
                recommended,
                [
                    "name",
                    "full_delta_roi_pct",
                    "full_delta_sharpe",
                    "full_delta_maxdd_pct",
                    "oos_delta_roi_pct",
                    "oos_delta_sharpe",
                    "oos_delta_maxdd_pct",
                    "days_top15_trend_if_added",
                    "full_buy_count_if_added",
                    "source_types",
                ],
            ).round(4),
            rows=20,
        ) if not recommended.empty else "(none)",
        "",
        "## Watchlist additions",
        "",
        format_small(
            safe_columns(
                watch,
                [
                    "name",
                    "full_delta_roi_pct",
                    "full_delta_sharpe",
                    "full_delta_maxdd_pct",
                    "oos_delta_roi_pct",
                    "oos_delta_sharpe",
                    "oos_delta_maxdd_pct",
                    "source_types",
                ],
            ).round(4),
            rows=20,
        ) if not watch.empty else "(none)",
        "",
        "## Combo tests",
        "",
        format_small(
            safe_columns(
                combo_df,
                [
                    "name",
                    "full_delta_roi_pct",
                    "full_delta_sharpe",
                    "full_delta_maxdd_pct",
                    "oos_delta_roi_pct",
                    "oos_delta_sharpe",
                    "oos_delta_maxdd_pct",
                ],
            ).round(4),
            rows=20,
        ) if not combo_df.empty else "(none)",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_discovery(
    *,
    mode: str,
    label: str,
    source_profile: str,
    seed_tickers: Sequence[str],
    keywords: Sequence[str],
    regions: Sequence[str],
    sector_names: Sequence[str],
    min_market_cap: float,
    min_price: float,
    min_adv: float,
    screen_count: int,
    screen_pages: int,
    top_etfs_per_sector: int,
    etf_holding_count: int,
    max_single_backtests: int,
    max_combo_backtests: int,
    skip_backtest: bool,
) -> dict[str, Path]:
    setup_yf_cache(ROOT / ".yf_cache")
    active, reserve, hard_exclusions = current_universe_state()
    engine, cfg_doc, cfg, pp, base_prices = load_setup()
    current_leaders = latest_raw_leaders(base_prices, cfg, topn=10)
    use_external = source_profile in {"hybrid", "yahoo_external"}
    use_local = source_profile in {"hybrid", "local_structural"}
    if mode == "targeted" and not seed_tickers:
        seed_tickers = current_leaders[:8]
    elif mode == "broad" and not seed_tickers:
        seed_tickers = current_leaders[:6]

    records: dict[str, dict] = {}
    if use_external:
        discover_lookup_queries(records, keywords, max_results=max(screen_count, 8))
        discover_predefined_screens(records, DEFAULT_PREDEFINED_SCREENS, min_market_cap, min_price, min_adv, screen_count)
        discover_region_universe(records, regions, min_market_cap, min_price, min_adv, max(screen_count, 12), max(1, screen_pages))
        discover_custom_region_sector_screens(
            records,
            regions,
            sector_names,
            min_market_cap,
            min_price,
            min_adv,
            screen_count,
            screen_pages,
        )
        discover_sector_regime_leaders(
            records,
            top_sector_count=4 if mode == "broad" else 2,
            top_companies_per_sector=6 if mode == "broad" else 3,
            etf_holding_count=max(4, etf_holding_count),
        )
        discover_constituent_proxy_universes(
            records,
            holdings_per_etf=8 if mode == "broad" else 4,
        )
    if use_local:
        discover_post_earnings_winners(
            records,
            min_market_cap=min_market_cap,
            min_price=min_price,
            min_adv=min_adv,
            count=18 if mode == "broad" else 10,
        )
        discover_reversal_candidates(
            records,
            min_market_cap=min_market_cap,
            min_price=min_price,
            min_adv=min_adv,
            count=18 if mode == "broad" else 10,
        )
        discover_historical_winner_cousins(
            records,
            min_market_cap=min_market_cap,
            min_price=min_price,
            min_adv=min_adv,
            peers_per_winner=2 if mode == "broad" else 1,
        )
        discover_recent_listing_leaders(
            records,
            min_market_cap=min_market_cap,
            min_price=min_price,
            min_adv=min_adv,
            count=18 if mode == "broad" else 8,
        )
        if mode == "targeted":
            discover_filtered_universe_bases(
                records,
                min_market_cap=min_market_cap,
                min_price=min_price,
                min_adv=min_adv,
                per_sector=1,
                global_count=12,
            )
            discover_multi_horizon_candidates(
                records,
                min_market_cap=min_market_cap,
                min_price=min_price,
                min_adv=min_adv,
                per_horizon=12,
            )
            discover_sector_industry_relative_leaders(
                records,
                min_market_cap=min_market_cap,
                min_price=min_price,
                min_adv=min_adv,
                per_sector=1,
                per_industry=1,
            )
            discover_local_leader_sentinels(
                records,
                min_market_cap=min_market_cap,
                min_price=min_price,
                min_adv=min_adv,
                global_count=10,
                per_cluster=1,
            )
        else:
            discover_filtered_universe_bases(
                records,
                min_market_cap=min_market_cap,
                min_price=min_price,
                min_adv=min_adv,
                per_sector=2,
                global_count=24,
            )
            discover_multi_horizon_candidates(
                records,
                min_market_cap=min_market_cap,
                min_price=min_price,
                min_adv=min_adv,
                per_horizon=24,
            )
            discover_sector_industry_relative_leaders(
                records,
                min_market_cap=min_market_cap,
                min_price=min_price,
                min_adv=min_adv,
                per_sector=2,
                per_industry=1,
            )
            discover_local_leader_sentinels(
                records,
                min_market_cap=min_market_cap,
                min_price=min_price,
                min_adv=min_adv,
                global_count=20,
                per_cluster=2,
            )
    if use_external and mode == "broad":
        sector_keys = [SECTOR_NAME_TO_KEY[name] for name in sector_names if name in SECTOR_NAME_TO_KEY]
        discover_sector_expansion(records, sector_keys, top_etfs_per_sector, etf_holding_count)
    seed_df = discover_seed_expansion(records, seed_tickers, etf_holding_count, top_etfs_per_sector) if use_external else pd.DataFrame()

    discovery_df = build_candidate_frame(records, active, reserve, hard_exclusions)
    discovery_path = EXPORTS_DIR / f"dynamic_universe_{label}_discovery.csv"
    seed_path = EXPORTS_DIR / f"dynamic_universe_{label}_seed_context.csv"
    compat_path = EXPORTS_DIR / f"dynamic_universe_{label}_compat_scan.csv"
    single_path = EXPORTS_DIR / f"dynamic_universe_{label}_single_additions.csv"
    combo_path = EXPORTS_DIR / f"dynamic_universe_{label}_combo_additions.csv"
    report_path = REPORTS_DIR / f"DYNAMIC_UNIVERSE_{label.upper()}.md"

    discovery_df.to_csv(discovery_path, index=False)
    seed_df.to_csv(seed_path, index=False)

    if skip_backtest:
        compat_df = pd.DataFrame()
        single_df = pd.DataFrame()
        combo_df = pd.DataFrame()
    else:
        single_df, combo_df, compat_df = evaluate_candidates(
            engine,
            cfg_doc,
            cfg,
            pp,
            base_prices,
            discovery_df,
            max_single_backtests=max_single_backtests,
            max_combo_backtests=max_combo_backtests,
            single_cache_path=single_path,
            combo_cache_path=combo_path,
        )
    if not compat_df.empty:
        compat_df.to_csv(compat_path, index=False)
    if not single_df.empty:
        single_df.to_csv(single_path, index=False)
    if not combo_df.empty:
        combo_df.to_csv(combo_path, index=False)

    write_report(
        report_path,
        mode=mode,
        source_profile=source_profile,
        seed_tickers=seed_tickers,
        keywords=keywords,
        discovery_df=discovery_df,
        compat_df=compat_df,
        single_df=single_df,
        combo_df=combo_df,
        current_leaders=current_leaders,
    )
    return {
        "discovery": discovery_path,
        "seed_context": seed_path,
        "compat": compat_path,
        "single": single_path,
        "combo": combo_path,
        "report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic universe discovery for APEX.")
    parser.add_argument("--mode", choices=["broad", "targeted"], default="broad")
    parser.add_argument("--source-profile", choices=["hybrid", "local_structural", "yahoo_external"], default="hybrid")
    parser.add_argument("--label", default="")
    parser.add_argument("--seed-tickers", default="", help="Comma-separated seed tickers for targeted peer discovery.")
    parser.add_argument("--keywords", default="", help="Comma-separated Yahoo lookup keywords/themes for targeted discovery.")
    parser.add_argument("--regions", default=",".join(DEFAULT_REGIONS), help="Comma-separated yfinance screener regions.")
    parser.add_argument("--sector-names", default=",".join(DEFAULT_SECTOR_NAMES), help="Comma-separated sector display names to scan/expand.")
    parser.add_argument("--min-market-cap", type=float, default=DEFAULT_MIN_MARKET_CAP)
    parser.add_argument("--min-price", type=float, default=DEFAULT_MIN_PRICE)
    parser.add_argument("--min-adv", type=float, default=DEFAULT_MIN_ADV)
    parser.add_argument("--screen-count", type=int, default=25)
    parser.add_argument("--screen-pages", type=int, default=1)
    parser.add_argument("--top-etfs-per-sector", type=int, default=3)
    parser.add_argument("--etf-holding-count", type=int, default=15)
    parser.add_argument("--max-single-backtests", type=int, default=30)
    parser.add_argument("--max-combo-backtests", type=int, default=10)
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()

    label = args.label.strip() or args.mode
    seed_tickers = [x.strip().upper() for x in args.seed_tickers.split(",") if x.strip()]
    keywords = [x.strip() for x in args.keywords.split(",") if x.strip()]
    regions = [x.strip().lower() for x in args.regions.split(",") if x.strip()]
    sector_names = [x.strip() for x in args.sector_names.split(",") if x.strip()]

    outputs = run_discovery(
        mode=args.mode,
        label=label,
        source_profile=str(args.source_profile),
        seed_tickers=seed_tickers,
        keywords=keywords,
        regions=regions,
        sector_names=sector_names,
        min_market_cap=float(args.min_market_cap),
        min_price=float(args.min_price),
        min_adv=float(args.min_adv),
        screen_count=int(args.screen_count),
        screen_pages=int(args.screen_pages),
        top_etfs_per_sector=int(args.top_etfs_per_sector),
        etf_holding_count=int(args.etf_holding_count),
        max_single_backtests=int(args.max_single_backtests),
        max_combo_backtests=int(args.max_combo_backtests),
        skip_backtest=bool(args.skip_backtest),
    )
    for name, path in outputs.items():
        if path.exists():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
