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

EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_REGIONS = ("us", "ca", "de", "fr", "gb", "it", "se", "nl")
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
DEFAULT_MIN_MARKET_CAP = 2_000_000_000
DEFAULT_MIN_PRICE = 5.0
DEFAULT_MIN_ADV = 1_000_000
CONTEXT_CACHE_SAVE_EVERY = 20
RETEST_RECENT_ABS_DELTA = 0.75
RETEST_RECENT_REL_DELTA = 0.25
RETEST_COMPAT_DELTA = 0.75
RETEST_RANK_DELTA = 5.0
RETEST_TOP5_SHARE_DELTA = 0.08
RETEST_RELATIVE_SCORE_DELTA = 0.25


@dataclass
class CandidateData:
    open: pd.Series
    close: pd.Series
    info: dict[str, object]


_TICKER_CONTEXT_CACHE: dict[str, dict[str, object]] | None = None
_TICKER_CONTEXT_CACHE_DIRTY = 0


def setup_yf_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    if hasattr(yf_cache, "set_cache_location"):
        yf_cache.set_cache_location(str(cache_dir))


def normalize_downloaded_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df.rename(columns=lambda c: str(c).lower().replace(" ", "_"))


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
    return engine, cfg_doc, cfg, pp, prices


def run_metrics(engine, prices, cfg: dict, pp: dict, start: str, end: str):
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
        **cfg,
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
    cache = _load_ticker_context_cache()
    if ticker in cache:
        return dict(cache[ticker])
    try:
        info = yf.Ticker(ticker).get_info() or {}
    except Exception:
        return {}
    ctx = {
        "ticker": ticker,
        "sector": info.get("sector"),
        "sectorKey": info.get("sectorKey"),
        "industry": info.get("industry"),
        "industryKey": info.get("industryKey"),
        "exchange": info.get("exchange"),
        "marketCap": info.get("marketCap"),
    }
    return dict(_store_ticker_context(ticker, ctx))


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
        "fiftyDayAverageChangePercent",
        "twoHundredDayAverageChangePercent",
        "regularMarketChangePercent",
        "sector",
        "industry",
        "sector_key",
        "industry_key",
    ):
        if col not in df.columns:
            df[col] = np.nan
    df["priority_score"] = (
        6.0 * df["source_count"].astype(float)
        + 3.0 * df["source_types"].str.contains("seed_peer", na=False).astype(float)
        + 2.5 * df["source_types"].str.contains("industry_top_performing|industry_top_growth", na=False).astype(float)
        + 2.0 * df["source_types"].str.contains("etf_holding", na=False).astype(float)
        + 1.5 * df["source_types"].str.contains("sector_top_company", na=False).astype(float)
        + 1.0 * df["source_types"].str.contains("predefined_screen|custom_screen", na=False).astype(float)
        + 2.0 * rank_within_series(np.log1p(df["marketCap"].fillna(0.0)))
        + 1.5 * rank_within_series(np.log1p(df["averageDailyVolume3Month"].fillna(0.0)))
        + 1.5 * rank_within_series(df["twoHundredDayAverageChangePercent"].fillna(0.0))
        + 1.0 * rank_within_series(df["fiftyDayAverageChangePercent"].fillna(0.0))
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
            close=df.set_index("date")["close"].astype(float),
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
            close=df.set_index("date")["close"].astype(float),
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
    for ticker in tickers:
        cand = candidate_map.get(ticker)
        if cand is None:
            continue
        close = pd.to_numeric(cand.close, errors="coerce").dropna()
        bars_recent = int(close.shape[0])
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
        ret1 = close.pct_change(fill_method=None)
        vol63 = ret1.rolling(63, min_periods=40).std() * np.sqrt(252.0)
        vol63_now = float(vol63.iloc[-1]) if vol63.shape[0] else np.nan
        rr_score = np.nan
        if pd.notna(recent_score) and pd.notna(vol63_now) and vol63_now > 0:
            rr_score = float(recent_score / vol63_now)
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
            if (dist_sma220 >= 0.70) and (dd60_now >= -0.05) and (r21_now >= 0.10):
                entry_heat_flag = 1
            if (dd60_now <= -0.05) and (dd60_now >= -0.20) and (r21_now <= 0.05):
                pullback_quality = 1.0
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
                "scan_breakout252_component": float(breakout_252) if pd.notna(breakout_252) else np.nan,
                "scan_trend200_component": float(trend200) if pd.notna(trend200) else np.nan,
                "scan_dist_sma220": float(dist_sma220) if pd.notna(dist_sma220) else np.nan,
                "scan_dd60": float(dd60_now) if pd.notna(dd60_now) else np.nan,
                "scan_r21": float(r21_now) if pd.notna(r21_now) else np.nan,
                "scan_entry_heat_flag": int(entry_heat_flag),
                "scan_pullback_quality": float(pullback_quality),
                **rel_metrics,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    latest_score_pct = rank_within_series(df["scan_latest_score_if_added"].fillna(df["scan_latest_score_if_added"].median(skipna=True)))
    recent_score_pct = rank_within_series(df["recent_score"].fillna(df["recent_score"].median(skipna=True)))
    rel_recent_score_pct = rank_within_series(df["scan_rel_recent_score"].fillna(df["scan_rel_recent_score"].median(skipna=True)))
    rr_score_pct = rank_within_series(df["scan_rr_score"].fillna(df["scan_rr_score"].median(skipna=True)))
    breakout_pct = df["scan_breakout252_component"].fillna(0.5)
    trend200_pct = df["scan_trend200_component"].fillna(0.5)
    heat_penalty = df["scan_entry_heat_flag"].fillna(0).astype(float)
    pullback_bonus = df["scan_pullback_quality"].fillna(0.0)
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
        + 0.6 * breakout_pct
        + 0.6 * trend200_pct
        + 0.3 * pullback_bonus
        - 0.8 * heat_penalty
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
        + 0.3 * pullback_bonus
        - 0.8 * heat_penalty
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
            (df["scan_rank_component"] >= 0.7) & (df["scan_persistence_component"] >= 0.45),
            (df["scan_rank_component"] >= 0.7) & (df["scan_recent_score_pct"] >= 0.75),
            df["scan_recent_score_pct"] >= 0.85,
        ],
        [
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
        "scan_breakout252_component": getattr(row, "scan_breakout252_component", np.nan),
        "scan_trend200_component": getattr(row, "scan_trend200_component", np.nan),
        "scan_dist_sma220": getattr(row, "scan_dist_sma220", np.nan),
        "scan_dd60": getattr(row, "scan_dd60", np.nan),
        "scan_r21": getattr(row, "scan_r21", np.nan),
        "scan_entry_heat_flag": int(getattr(row, "scan_entry_heat_flag", 0) or 0),
        "scan_pullback_quality": getattr(row, "scan_pullback_quality", np.nan),
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
        "- build a dynamic candidate pool from yfinance instead of relying only on a fixed universe",
        "- combine broad discovery with targeted peer expansion around current leaders",
        "- score candidates with the current APEX baseline through single-addition backtests and small combo tests",
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
    if mode == "targeted" and not seed_tickers:
        seed_tickers = current_leaders[:8]
    elif mode == "broad" and not seed_tickers:
        seed_tickers = current_leaders[:6]

    records: dict[str, dict] = {}
    discover_lookup_queries(records, keywords, max_results=max(screen_count, 8))
    discover_predefined_screens(records, DEFAULT_PREDEFINED_SCREENS, min_market_cap, min_price, min_adv, screen_count)
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
    if mode == "broad":
        sector_keys = [SECTOR_NAME_TO_KEY[name] for name in sector_names if name in SECTOR_NAME_TO_KEY]
        discover_sector_expansion(records, sector_keys, top_etfs_per_sector, etf_holding_count)
    seed_df = discover_seed_expansion(records, seed_tickers, etf_holding_count, top_etfs_per_sector)

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
    parser = argparse.ArgumentParser(description="Dynamic yfinance universe discovery for APEX.")
    parser.add_argument("--mode", choices=["broad", "targeted"], default="broad")
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
