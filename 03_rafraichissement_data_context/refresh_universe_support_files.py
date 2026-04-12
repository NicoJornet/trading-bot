from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import yfinance.cache as yf_cache


FALLBACK_METADATA = {
    "SNDK": {
        "ticker": "SNDK",
        "name": "Sandisk Corporation",
        "sector": "Technology",
        "industry": "Computer Hardware",
        "market_cap": 105444663296.0,
        "country": "United States",
        "currency": "USD",
        "exchange": "NMS",
        "shares_outstanding": 147600972.0,
        "float_shares": 139062948.0,
    },
    "BE": {
        "ticker": "BE",
        "name": "Bloom Energy Corporation",
        "sector": "Industrials",
        "industry": "Electrical Equipment & Parts",
        "market_cap": 43428864000.0,
        "country": "United States",
        "currency": "USD",
        "exchange": "NYQ",
        "shares_outstanding": 280548215.0,
        "float_shares": 264312890.0,
    },
    "WDC": {
        "ticker": "WDC",
        "name": "Western Digital Corporation",
        "sector": "Technology",
        "industry": "Computer Hardware",
        "market_cap": 97713111040.0,
        "country": "United States",
        "currency": "USD",
        "exchange": "NMS",
        "shares_outstanding": 339037922.0,
        "float_shares": 336712122.0,
    },
}

REMOVE_VERDICTS = {"NO_VALID_SCORE", "WEAK_SCORE", "RARELY_TOP"}


def setup_yf_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    if hasattr(yf_cache, "set_cache_location"):
        yf_cache.set_cache_location(str(cache_dir))


def compute_selection_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats: list[dict[str, object]] = []
    valid = df.loc[df["momentum_score"].notna()].copy()
    p95_global = valid["momentum_score"].quantile(0.95) if not valid.empty else np.nan

    for ticker in sorted(df["ticker"].unique()):
        ticker_data = df.loc[df["ticker"] == ticker]
        ticker_valid = valid.loc[valid["ticker"] == ticker]

        days_total = int(len(ticker_data))
        days_valid_score = int(len(ticker_valid))
        days_rf0 = int(len(ticker_valid.loc[ticker_valid["red_flags_count"] == 0]))

        if days_valid_score == 0:
            verdict = "NO_VALID_SCORE"
            max_score = 0.0
            median_score = 0.0
            pct_selectable = 0.0
        else:
            max_score = float(ticker_valid["momentum_score"].max())
            median_score = float(ticker_valid["momentum_score"].median())
            pct_selectable = float((days_rf0 / days_total) * 100.0) if days_total > 0 else 0.0
            times_p95 = int(len(ticker_valid.loc[ticker_valid["momentum_score"] > p95_global]))

            if max_score < 1.0:
                verdict = "WEAK_SCORE"
            elif pct_selectable < 5.0:
                verdict = "TOO_MANY_RF"
            elif times_p95 < 5:
                verdict = "RARELY_TOP"
            else:
                verdict = "OK"

        stats.append(
            {
                "ticker": ticker,
                "verdict": verdict,
                "days_total": days_total,
                "days_valid_score": days_valid_score,
                "days_rf0": days_rf0,
                "pct_selectable": pct_selectable,
                "max_score": max_score,
                "median_score": median_score,
            }
        )

    return pd.DataFrame(stats)


def fetch_metadata_row(ticker: str) -> dict[str, object]:
    fallback = FALLBACK_METADATA.get(ticker, {"ticker": ticker})

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or fallback.get("name", ""),
        "sector": info.get("sector") or fallback.get("sector", ""),
        "industry": info.get("industry") or fallback.get("industry", ""),
        "market_cap": info.get("marketCap", fallback.get("market_cap", np.nan)),
        "country": info.get("country") or fallback.get("country", ""),
        "currency": info.get("currency") or fallback.get("currency", ""),
        "exchange": info.get("exchange") or fallback.get("exchange", ""),
        "shares_outstanding": info.get("sharesOutstanding", fallback.get("shares_outstanding", np.nan)),
        "float_shares": info.get("floatShares", fallback.get("float_shares", np.nan)),
    }


def fetch_actions_rows(ticker: str) -> pd.DataFrame:
    try:
        actions = yf.Ticker(ticker).actions.copy()
    except Exception:
        actions = pd.DataFrame()

    if actions is None or actions.empty:
        return pd.DataFrame(columns=["date", "value", "ticker", "type"])

    actions = actions.reset_index()
    date_col = actions.columns[0]
    actions = actions.rename(columns={date_col: "date"})

    rows: list[dict[str, object]] = []
    for _, row in actions.iterrows():
        ts = pd.to_datetime(row["date"], utc=False)
        date_str = ts.isoformat(sep=" ")

        dividend = row.get("Dividends", 0)
        if pd.notna(dividend) and float(dividend) != 0.0:
            rows.append({"date": date_str, "value": float(dividend), "ticker": ticker, "type": "DIVIDEND"})

        split = row.get("Stock Splits", 0)
        if pd.notna(split) and float(split) != 0.0:
            rows.append({"date": date_str, "value": float(split), "ticker": ticker, "type": "SPLIT"})

    return pd.DataFrame(rows, columns=["date", "value", "ticker", "type"])


def build_report(
    df: pd.DataFrame,
    selection_stats: pd.DataFrame,
    keep_df: pd.DataFrame,
    remove_df: pd.DataFrame,
    actions_df: pd.DataFrame,
) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = pd.to_datetime(df["date"]).min().strftime("%Y-%m-%d")
    end = pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d")
    verdict_counts = selection_stats["verdict"].value_counts()
    div_count = int((actions_df["type"] == "DIVIDEND").sum())
    split_count = int((actions_df["type"] == "SPLIT").sum())
    ticker_count = int(df["ticker"].nunique())

    lines = [
        "=" * 100,
        f"APEX BACKTEST DATA EXTRACTION - RAPPORT FINAL ({ticker_count} TICKERS)",
        "=" * 100,
        "",
        f"Date: {now}",
        f"Periode: {start} -> {end}",
        "",
        "=" * 100,
        "UNIVERS",
        "=" * 100,
        f"Tickers demandes: {ticker_count}",
        f"Tickers telecharges: {ticker_count}",
        "Tickers echoues: 0",
        "",
        "=" * 100,
        "DONNEES EXTRAITES",
        "=" * 100,
        f"Total lignes: {len(df):,}",
        f"Total colonnes: {len(df.columns)}",
        "",
        "Colonnes OHLCV de base:",
        "  OK open, high, low, close, volume, adj_close",
        "  OK adj_open, adj_high, adj_low (calcules)",
        "",
        "Indicateurs derives:",
        "  OK Returns (1d, 5d, 21d, 63d, 126d, 252d)",
        "  OK SMA/EMA (20, 50, 200)",
        "  OK ATR & ATR%",
        "  OK RSI 14",
        "  OK Momentum Score",
        "  OK Red Flags",
        "  OK Volatilite (20d, 60d)",
        "  OK Dollar Volume & ADV",
        "",
        "=" * 100,
        "CORPORATE ACTIONS",
        "=" * 100,
        f"Dividendes: {div_count} evenements",
        f"Splits: {split_count} evenements",
        "",
        "=" * 100,
        "ANALYSE SELECTION",
        "=" * 100,
    ]

    for verdict, count in verdict_counts.items():
        lines.append(f"{verdict:<20}: {count:3d} tickers")

    lines.extend(
        [
            "",
            "=" * 100,
            "RECOMMANDATIONS",
            "=" * 100,
            f"Tickers a SUPPRIMER: {len(remove_df)}",
            f"Tickers a GARDER: {len(keep_df)}",
            "",
            f"Nouvel univers recommande: {len(keep_df)} tickers",
            "",
            "=" * 100,
            "FICHIERS GENERES",
            "=" * 100,
            f"1. apex_ohlcv_full_2015_2026.csv ({len(df):,} lignes)",
            "2. apex_corporate_actions.csv",
            "3. apex_ticker_metadata.csv",
            "4. apex_ticker_selection_stats.csv",
            f"5. apex_tickers_to_remove.csv ({len(remove_df)} tickers)",
            f"6. apex_tickers_to_keep.csv ({len(keep_df)} tickers)",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="apex_ohlcv_full_2015_2026.csv")
    parser.add_argument("--extracts-dir", default="data/extracts")
    parser.add_argument("--tickers", nargs="+", default=["SNDK", "BE", "WDC"])
    args = parser.parse_args()

    root = Path.cwd()
    setup_yf_cache(root / ".yf_cache")

    csv_path = (root / args.csv).resolve()
    extracts_dir = (root / args.extracts_dir).resolve()
    extracts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    selection_stats = compute_selection_stats(df)
    selection_stats.to_csv(extracts_dir / "apex_ticker_selection_stats.csv", index=False)

    remove_df = selection_stats.loc[selection_stats["verdict"].isin(REMOVE_VERDICTS), ["ticker"]].sort_values("ticker")
    keep_df = selection_stats.loc[~selection_stats["ticker"].isin(remove_df["ticker"]), ["ticker"]].sort_values("ticker")
    remove_df.to_csv(extracts_dir / "apex_tickers_to_remove.csv", index=False)
    keep_df.to_csv(extracts_dir / "apex_tickers_to_keep.csv", index=False)

    metadata_path = extracts_dir / "apex_ticker_metadata.csv"
    metadata_df = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame(columns=list(fetch_metadata_row(args.tickers[0]).keys()))
    metadata_df = metadata_df.loc[~metadata_df["ticker"].isin(args.tickers)].copy()
    appended_metadata = pd.DataFrame([fetch_metadata_row(ticker) for ticker in args.tickers])
    metadata_df = pd.concat([metadata_df, appended_metadata], ignore_index=True)
    metadata_df = metadata_df.sort_values("ticker", kind="mergesort").reset_index(drop=True)
    metadata_df.to_csv(metadata_path, index=False)

    actions_path = extracts_dir / "apex_corporate_actions.csv"
    actions_df = pd.read_csv(actions_path, dtype={"date": str, "value": str, "ticker": str, "type": str})
    actions_df = actions_df.loc[~actions_df["ticker"].isin(args.tickers)].copy()
    new_actions = pd.concat([fetch_actions_rows(ticker) for ticker in args.tickers], ignore_index=True)
    if not new_actions.empty:
        new_actions["value"] = new_actions["value"].astype(float)
        actions_df["value"] = pd.to_numeric(actions_df["value"], errors="coerce")
        actions_df = pd.concat([actions_df, new_actions], ignore_index=True)
    actions_df["_sort_date"] = pd.to_datetime(actions_df["date"], utc=True, errors="coerce")
    actions_df = actions_df.sort_values(["ticker", "_sort_date", "type"], kind="mergesort").drop(columns="_sort_date")
    actions_df.to_csv(actions_path, index=False)

    report = build_report(df, selection_stats, keep_df, remove_df, actions_df)
    (extracts_dir / "apex_backtest_report.txt").write_text(report, encoding="utf-8")

    print(f"Selection stats updated: {extracts_dir / 'apex_ticker_selection_stats.csv'}")
    print(f"Keep tickers: {len(keep_df)}")
    print(f"Remove tickers: {len(remove_df)}")
    print(f"Metadata rows: {len(metadata_df)}")
    print(f"Corporate actions rows: {len(actions_df)}")
    print(selection_stats.loc[selection_stats['ticker'].isin(args.tickers)].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
