from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from refresh_context_earnings_snapshots import (
    CONTEXT_DIR,
    dedupe_keep_order,
    load_latest_row_map,
    load_ohlcv_tickers,
    load_target_tickers,
    setup_yf_cache,
)


ROOT = Path(__file__).resolve().parent
OHLCV_PATH = ROOT / "apex_ohlcv_full_2015_2026.csv"
DAILY_OUT_PATH = CONTEXT_DIR / "market_structure_daily.csv"
LATEST_OUT_PATH = CONTEXT_DIR / "market_structure_latest.csv"
HISTORY_OUT_DIR = CONTEXT_DIR / "history"
SUMMARY_OUT_PATH = ROOT / "data" / "extracts" / "market_structure_history_summary.csv"


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        if np.isnan(value):
            return None
        return value
    except Exception:
        return None


def _target_tickers() -> list[str]:
    priority, _ = load_target_tickers()
    ohlcv = load_ohlcv_tickers()
    return dedupe_keep_order(priority + ohlcv)


def _load_ohlcv_slice() -> pd.DataFrame:
    usecols = ["date", "ticker", "close", "adj_close", "adv_20"]
    df = pd.read_csv(OHLCV_PATH, usecols=usecols, low_memory=False, parse_dates=["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df["adv_20"] = pd.to_numeric(df["adv_20"], errors="coerce")
    return df.sort_values(["ticker", "date"], kind="mergesort")


def _normalize_share_series(series: object) -> pd.DataFrame:
    if series is None:
        return pd.DataFrame(columns=["date", "shares_outstanding"])
    if isinstance(series, pd.Series):
        if series.empty:
            return pd.DataFrame(columns=["date", "shares_outstanding"])
        out = series.rename("shares_outstanding").reset_index()
    else:
        out = pd.DataFrame(series)
        if out.empty:
            return pd.DataFrame(columns=["date", "shares_outstanding"])
        if "shares_outstanding" not in out.columns:
            if out.shape[1] >= 2:
                out = out.iloc[:, :2]
                out.columns = ["date", "shares_outstanding"]
            else:
                return pd.DataFrame(columns=["date", "shares_outstanding"])
    out = out.rename(columns={out.columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize().astype("datetime64[ns]")
    out["shares_outstanding"] = pd.to_numeric(out["shares_outstanding"], errors="coerce")
    out = out.dropna(subset=["date", "shares_outstanding"]).sort_values("date")
    return out.drop_duplicates("date", keep="last").reset_index(drop=True)


def _load_current_context_map() -> dict[str, dict[str, object]]:
    return load_latest_row_map(CONTEXT_DIR / "ticker_context_latest.csv")


def _fetch_current_info(ticker: str, context_row: dict[str, object]) -> dict[str, object]:
    current_market_cap = _safe_float(context_row.get("marketCap"))
    current_currency = str(context_row.get("currency") or "").strip().upper()
    current_shares = None
    current_float = None
    info: dict[str, object] = {}

    try:
        info = yf.Ticker(ticker).get_info() or {}
    except Exception:
        info = {}

    current_market_cap = _safe_float(info.get("marketCap")) or current_market_cap
    current_shares = _safe_float(info.get("sharesOutstanding"))
    current_float = _safe_float(info.get("floatShares"))
    if not current_currency:
        current_currency = str(info.get("currency") or "").strip().upper()

    return {
        "ticker": ticker,
        "market_cap_current": current_market_cap,
        "shares_outstanding_current": current_shares,
        "float_shares_current": current_float,
        "currency": current_currency,
    }


def _build_daily_rows(
    ticker: str,
    ticker_ohlcv: pd.DataFrame,
    context_row: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    current_info = _fetch_current_info(ticker, context_row)
    start = pd.Timestamp(ticker_ohlcv["date"].min()).strftime("%Y-%m-%d")
    try:
        shares_df = _normalize_share_series(yf.Ticker(ticker).get_shares_full(start=start))
    except Exception:
        shares_df = pd.DataFrame(columns=["date", "shares_outstanding"])

    daily = ticker_ohlcv[["date", "ticker", "close", "adj_close", "adv_20"]].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").astype("datetime64[ns]")
    if not shares_df.empty:
        daily = pd.merge_asof(
            daily.sort_values("date"),
            shares_df.sort_values("date"),
            on="date",
            direction="backward",
        )
    else:
        daily["shares_outstanding"] = np.nan

    current_shares = current_info["shares_outstanding_current"]
    if current_shares is not None:
        daily["shares_outstanding"] = daily["shares_outstanding"].fillna(current_shares)

    float_current = current_info["float_shares_current"]
    if float_current is not None and current_shares:
        float_ratio = float_current / current_shares if current_shares else np.nan
        daily["float_shares"] = daily["shares_outstanding"] * float_ratio
    else:
        float_ratio = np.nan
        daily["float_shares"] = np.nan

    daily["free_float_ratio"] = daily["float_shares"] / daily["shares_outstanding"].replace(0, np.nan)
    daily["market_cap_estimate"] = daily["close"] * daily["shares_outstanding"]
    daily["market_cap_to_adv20"] = daily["market_cap_estimate"] / daily["adv_20"].replace(0, np.nan)

    latest = daily.iloc[-1].to_dict()
    latest.update(current_info)
    latest.update(
        {
            "as_of": date.today().isoformat(),
            "shares_history_points": int(len(shares_df)),
            "shares_history_start": shares_df["date"].min().strftime("%Y-%m-%d") if not shares_df.empty else "",
            "shares_history_end": shares_df["date"].max().strftime("%Y-%m-%d") if not shares_df.empty else "",
            "coverage_ratio": float(daily["shares_outstanding"].notna().mean()),
            "implied_free_float_ratio_current": float_ratio,
        }
    )
    return daily, latest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build market cap / shares / float history layers.")
    parser.add_argument("--tickers", default="", help="Optional comma-separated tickers.")
    args = parser.parse_args()

    setup_yf_cache(ROOT / ".yf_cache")
    context_map = _load_current_context_map()
    tickers = (
        [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
        if args.tickers
        else _target_tickers()
    )
    ohlcv = _load_ohlcv_slice()
    available_tickers = set(ohlcv["ticker"].unique())
    tickers = [ticker for ticker in tickers if ticker in available_tickers]

    daily_frames: list[pd.DataFrame] = []
    latest_rows: list[dict[str, object]] = []
    for ticker in tickers:
        ticker_ohlcv = ohlcv.loc[ohlcv["ticker"] == ticker].copy()
        if ticker_ohlcv.empty:
            continue
        daily, latest = _build_daily_rows(ticker, ticker_ohlcv, context_map.get(ticker, {}))
        daily_frames.append(daily)
        latest_rows.append(latest)

    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    latest_df = pd.DataFrame(latest_rows)
    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.strftime("%Y-%m-%d")
    if not latest_df.empty:
        latest_df = latest_df.sort_values("ticker", kind="mergesort").reset_index(drop=True)

    DAILY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(DAILY_OUT_PATH, index=False)
    latest_df.to_csv(LATEST_OUT_PATH, index=False)
    latest_df.to_csv(HISTORY_OUT_DIR / f"market_structure_snapshot_{date.today().isoformat()}.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "as_of": date.today().isoformat(),
                "tickers": int(latest_df["ticker"].nunique()) if not latest_df.empty else 0,
                "rows_daily": int(len(daily_df)),
                "with_shares_outstanding": int(latest_df["shares_outstanding_current"].notna().sum()) if not latest_df.empty else 0,
                "with_float_shares": int(latest_df["float_shares_current"].notna().sum()) if not latest_df.empty else 0,
                "with_market_cap_current": int(latest_df["market_cap_current"].notna().sum()) if not latest_df.empty else 0,
                "daily_coverage_mean": float(pd.to_numeric(latest_df.get("coverage_ratio"), errors="coerce").fillna(0.0).mean()) if not latest_df.empty else 0.0,
                "daily_coverage_median": float(pd.to_numeric(latest_df.get("coverage_ratio"), errors="coerce").fillna(0.0).median()) if not latest_df.empty else 0.0,
            }
        ]
    )
    SUMMARY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUT_PATH, index=False)

    print(f"Market structure daily: {DAILY_OUT_PATH}")
    print(f"Market structure latest: {LATEST_OUT_PATH}")
    print(f"Summary: {SUMMARY_OUT_PATH}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
