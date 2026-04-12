from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import yfinance.cache as yf_cache


SMA_PERIOD = 20
ATR_PERIOD = 14
HIGH_LOOKBACK = 60
RF_RSI_OVERBOUGHT = 78
RF_DIST_HIGH_52W_MIN = -30.0
RF_ATR_PCT_MAX = 7.0
RF_DIST_SMA20_MAX = 20.0

REQUIRED_COLS = ["open", "high", "low", "close", "volume", "adj_close"]


def setup_yf_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    if hasattr(yf_cache, "set_cache_location"):
        yf_cache.set_cache_location(str(cache_dir))


def normalize_downloaded_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
    return df


def download_ohlcv_data(tickers: list[str], start_date: str, end_date: str) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    all_data: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
            )
            df = normalize_downloaded_columns(df)
            if df.empty or len(df) < 60:
                failed.append((ticker, "Insufficient data"))
                continue
            if not all(c in df.columns for c in REQUIRED_COLS):
                failed.append((ticker, f"Missing columns: {sorted(set(REQUIRED_COLS) - set(df.columns))}"))
                continue
            df["ticker"] = ticker
            df = df.reset_index()
            df = df.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
            df["date"] = pd.to_datetime(df["date"])
            all_data.append(df)
        except Exception as exc:
            failed.append((ticker, str(exc)))

    if not all_data:
        raise RuntimeError(f"No data downloaded. Failures: {failed}")

    return pd.concat(all_data, ignore_index=True), failed


def compute_adjusted_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    result: list[pd.DataFrame] = []
    for ticker, group in df.groupby("ticker", sort=False):
        g = group.copy().sort_values("date")
        g["adj_ratio"] = g["adj_close"] / g["close"]
        g["adj_open"] = g["open"] * g["adj_ratio"]
        g["adj_high"] = g["high"] * g["adj_ratio"]
        g["adj_low"] = g["low"] * g["adj_ratio"]
        g["adj_volume"] = g["volume"] / g["adj_ratio"]
        result.append(g)
    return pd.concat(result, ignore_index=True)


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result: list[pd.DataFrame] = []

    for ticker, group in df.groupby("ticker", sort=False):
        g = group.sort_values("date").copy()
        close = g["adj_close"]
        open_ = g["adj_open"]
        high = g["adj_high"]
        low = g["adj_low"]

        g["return_1d"] = close.pct_change()
        g["return_5d"] = close.pct_change(5)
        g["return_21d"] = close.pct_change(21)
        g["return_63d"] = close.pct_change(63)
        g["return_126d"] = close.pct_change(126)
        g["return_252d"] = close.pct_change(252)

        g["sma_20"] = close.rolling(20).mean()
        g["sma_50"] = close.rolling(50).mean()
        g["sma_200"] = close.rolling(200).mean()
        g["ema_20"] = close.ewm(span=20, adjust=False).mean()
        g["ema_50"] = close.ewm(span=50, adjust=False).mean()

        g["dist_sma20_pct"] = (close / g["sma_20"] - 1.0) * 100.0
        g["dist_sma50_pct"] = (close / g["sma_50"] - 1.0) * 100.0
        g["dist_sma200_pct"] = (close / g["sma_200"] - 1.0) * 100.0

        g["high_60"] = high.rolling(60).max()
        g["high_90"] = high.rolling(90).max()
        g["high_252"] = high.rolling(252).max()

        g["dist_high_60_pct"] = (close / g["high_60"] - 1.0) * 100.0
        g["dist_high_90_pct"] = (close / g["high_90"] - 1.0) * 100.0
        g["dist_high_252_pct"] = (close / g["high_252"] - 1.0) * 100.0

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        g["atr_14"] = tr.rolling(ATR_PERIOD).mean()
        g["atr_pct"] = (g["atr_14"] / close) * 100.0

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
        rs = avg_gain / avg_loss
        g["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        g["volatility_20d"] = g["return_1d"].rolling(20).std() * np.sqrt(252) * 100.0
        g["volatility_60d"] = g["return_1d"].rolling(60).std() * np.sqrt(252) * 100.0

        g["dollar_volume"] = close * g["volume"]
        g["adv_20"] = g["dollar_volume"].rolling(20).mean()
        g["adv_60"] = g["dollar_volume"].rolling(60).mean()
        g["volume_avg_20"] = g["volume"].rolling(20).mean()

        prev_close = close.shift(1)
        g["overnight_return_1d"] = open_ / prev_close - 1.0
        g["intraday_return_1d"] = close / open_ - 1.0
        g["gap_pct"] = g["overnight_return_1d"] * 100.0
        g["intraday_return_pct"] = g["intraday_return_1d"] * 100.0
        g["rel_volume_20"] = g["volume"] / g["volume_avg_20"].replace(0, np.nan)
        g["rel_dollar_volume_20"] = g["dollar_volume"] / g["adv_20"].replace(0, np.nan)

        abs_path = close.diff().abs()
        g["efficiency_ratio_20"] = close.diff(20).abs() / abs_path.rolling(20).sum().replace(0, np.nan)
        g["efficiency_ratio_63"] = close.diff(63).abs() / abs_path.rolling(63).sum().replace(0, np.nan)

        sma = close.rolling(SMA_PERIOD).mean()
        atr = g["atr_14"]
        high_lookback = high.rolling(HIGH_LOOKBACK).max()
        base = (close - sma) / atr.replace(0, np.nan)
        penalty = high_lookback / close.replace(0, np.nan)
        g["momentum_score"] = (base / penalty).replace([np.inf, -np.inf], np.nan)

        g["rf_rsi_high"] = (g["rsi_14"] > RF_RSI_OVERBOUGHT).astype(int)
        g["rf_dist_high_low"] = (g["dist_high_252_pct"] < RF_DIST_HIGH_52W_MIN).astype(int)
        g["rf_atr_high"] = (g["atr_pct"] > RF_ATR_PCT_MAX).astype(int)
        g["rf_dist_sma_high"] = (g["dist_sma20_pct"] > RF_DIST_SMA20_MAX).astype(int)
        g["red_flags_count"] = (
            g["rf_rsi_high"] + g["rf_dist_high_low"] + g["rf_atr_high"] + g["rf_dist_sma_high"]
        )

        g["above_sma200"] = (close > g["sma_200"]).astype(int)
        result.append(g)

    return pd.concat(result, ignore_index=True)


def build_summary(existing: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        g = existing.loc[existing["ticker"] == ticker].copy()
        if g.empty:
            rows.append({"ticker": ticker, "rows": 0})
            continue
        rows.append(
            {
                "ticker": ticker,
                "rows": int(len(g)),
                "start_date": g["date"].min().strftime("%Y-%m-%d"),
                "end_date": g["date"].max().strftime("%Y-%m-%d"),
                "first_close": float(g.iloc[0]["close"]),
                "last_close": float(g.iloc[-1]["close"]),
                "has_momentum_score": int(g["momentum_score"].notna().any()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="apex_ohlcv_full_2015_2026.csv")
    ap.add_argument("--tickers", nargs="+", default=["SNDK", "BE", "WDC"])
    args = ap.parse_args()

    root = Path.cwd()
    csv_path = (root / args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    cache_dir = root / ".yf_cache"
    setup_yf_cache(cache_dir)

    existing = pd.read_csv(csv_path, parse_dates=["date"])
    columns = existing.columns.tolist()
    start_date = existing["date"].min().strftime("%Y-%m-%d")
    end_date_exclusive = (existing["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw_data, failed = download_ohlcv_data(args.tickers, start_date, end_date_exclusive)
    if failed:
        print("Download warnings:")
        for ticker, reason in failed:
            print(f"  - {ticker}: {reason}")

    appended = compute_all_indicators(compute_adjusted_ohlc(raw_data))
    for col in columns:
        if col not in appended.columns:
            appended[col] = pd.NA
    appended = appended[columns].copy()

    updated = existing.loc[~existing["ticker"].isin(args.tickers)].copy()
    updated = pd.concat([updated, appended], ignore_index=True)
    updated["date"] = pd.to_datetime(updated["date"])
    updated = updated.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)

    updated_to_write = updated.copy()
    updated_to_write["date"] = updated_to_write["date"].dt.strftime("%Y-%m-%d")
    updated_to_write.to_csv(csv_path, index=False)

    summary = build_summary(updated, args.tickers)
    slug = "_".join(t.lower().replace(".", "_").replace("-", "_") for t in args.tickers)
    summary_path = root / "data" / "extracts" / f"appended_tickers_{slug}_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print(f"Updated CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
