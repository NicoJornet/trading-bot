from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from append_tickers_to_ohlcv import (
    compute_adjusted_ohlc,
    download_ohlcv_data,
    normalize_downloaded_columns,
    setup_yf_cache,
)


BENCHMARKS = {
    "SPY": {"benchmark_name": "S&P 500", "benchmark_group": "market", "sector_key": "market"},
    "QQQ": {"benchmark_name": "Nasdaq 100", "benchmark_group": "market", "sector_key": "market"},
    "IWM": {"benchmark_name": "Russell 2000", "benchmark_group": "market", "sector_key": "market"},
    "XLK": {"benchmark_name": "Technology Select Sector SPDR", "benchmark_group": "sector", "sector_key": "technology"},
    "XLI": {"benchmark_name": "Industrial Select Sector SPDR", "benchmark_group": "sector", "sector_key": "industrials"},
    "XLC": {"benchmark_name": "Communication Services Select Sector SPDR", "benchmark_group": "sector", "sector_key": "communication-services"},
    "XLU": {"benchmark_name": "Utilities Select Sector SPDR", "benchmark_group": "sector", "sector_key": "utilities"},
    "XLE": {"benchmark_name": "Energy Select Sector SPDR", "benchmark_group": "sector", "sector_key": "energy"},
    "XLB": {"benchmark_name": "Materials Select Sector SPDR", "benchmark_group": "sector", "sector_key": "basic-materials"},
    "XLV": {"benchmark_name": "Health Care Select Sector SPDR", "benchmark_group": "sector", "sector_key": "healthcare"},
    "XLF": {"benchmark_name": "Financial Select Sector SPDR", "benchmark_group": "sector", "sector_key": "financial-services"},
    "XLY": {"benchmark_name": "Consumer Discretionary Select Sector SPDR", "benchmark_group": "sector", "sector_key": "consumer-cyclical"},
    "XLP": {"benchmark_name": "Consumer Staples Select Sector SPDR", "benchmark_group": "sector", "sector_key": "consumer-defensive"},
    "XLRE": {"benchmark_name": "Real Estate Select Sector SPDR", "benchmark_group": "sector", "sector_key": "real-estate"},
    "SMH": {"benchmark_name": "VanEck Semiconductor ETF", "benchmark_group": "theme", "sector_key": "semiconductors"},
    "SOXX": {"benchmark_name": "iShares Semiconductor ETF", "benchmark_group": "theme", "sector_key": "semiconductors"},
    "XSD": {"benchmark_name": "SPDR S&P Semiconductor ETF", "benchmark_group": "theme", "sector_key": "semiconductors"},
    "XME": {"benchmark_name": "SPDR S&P Metals & Mining ETF", "benchmark_group": "theme", "sector_key": "metals-mining"},
    "GDX": {"benchmark_name": "VanEck Gold Miners ETF", "benchmark_group": "theme", "sector_key": "gold-miners"},
    "SIL": {"benchmark_name": "Global X Silver Miners ETF", "benchmark_group": "theme", "sector_key": "silver-miners"},
}


def compute_benchmark_features(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker, group in df.groupby("ticker", sort=False):
        g = group.sort_values("date").copy()
        close = pd.to_numeric(g["adj_close"], errors="coerce")
        g["return_1d"] = close.pct_change()
        g["return_21d"] = close.pct_change(21)
        g["return_63d"] = close.pct_change(63)
        g["return_126d"] = close.pct_change(126)
        g["return_252d"] = close.pct_change(252)
        g["sma_50"] = close.rolling(50).mean()
        g["sma_200"] = close.rolling(200).mean()
        g["above_sma200"] = (close > g["sma_200"]).astype(int)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh sector and market benchmark OHLCV layers.")
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--through", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    root = Path.cwd()
    cache_dir = root / ".yf_cache"
    setup_yf_cache(cache_dir)

    tickers = list(BENCHMARKS.keys())
    raw, failed = download_ohlcv_data(
        tickers,
        args.start,
        (pd.Timestamp(args.through) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    enriched = compute_benchmark_features(compute_adjusted_ohlc(raw))
    meta = pd.DataFrame.from_dict(BENCHMARKS, orient="index").reset_index().rename(columns={"index": "ticker"})
    out = enriched.merge(meta, on="ticker", how="left")

    output_path = root / "data" / "benchmarks" / "sector_benchmarks_ohlcv.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_to_write = out.copy()
    out_to_write["date"] = pd.to_datetime(out_to_write["date"]).dt.strftime("%Y-%m-%d")
    out_to_write.to_csv(output_path, index=False)

    summary_rows = []
    for ticker in tickers:
        g = out.loc[out["ticker"] == ticker]
        if g.empty:
            summary_rows.append({"ticker": ticker, "rows": 0, "refreshed": 0})
            continue
        summary_rows.append(
            {
                "ticker": ticker,
                "rows": int(len(g)),
                "start_date": str(pd.Timestamp(g["date"].min()).date()),
                "end_date": str(pd.Timestamp(g["date"].max()).date()),
                "benchmark_group": BENCHMARKS[ticker]["benchmark_group"],
                "sector_key": BENCHMARKS[ticker]["sector_key"],
                "refreshed": 1,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = root / "data" / "benchmarks" / "sector_benchmarks_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Benchmarks saved: {output_path}")
    print(f"Summary saved: {summary_path}")
    if failed:
        print("Download warnings:")
        for ticker, reason in failed:
            print(f"  - {ticker}: {reason}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
