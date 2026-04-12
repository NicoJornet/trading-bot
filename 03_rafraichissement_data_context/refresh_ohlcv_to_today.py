from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from append_tickers_to_ohlcv import (
    compute_adjusted_ohlc,
    compute_all_indicators,
    download_ohlcv_data,
    setup_yf_cache,
)


BOOTSTRAP_TICKER_FILES = (
    Path("data/extracts/apex_tickers_active.csv"),
    Path("data/extracts/apex_tickers_reserve.csv"),
    Path("data/extracts/apex_tickers_hard_exclusions.csv"),
)


def read_ticker_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return []
    return [str(x).strip() for x in df["ticker"].dropna().astype(str) if str(x).strip()]


def collect_bootstrap_tickers(root: Path) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()
    for rel_path in BOOTSTRAP_TICKER_FILES:
        for ticker in read_ticker_list(root / rel_path):
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return tickers


def build_ohlcv_from_scratch(
    root: Path,
    csv_path: Path,
    through: str,
    bootstrap_start: str,
    bootstrap_max_tickers: int,
) -> tuple[pd.DataFrame, list[str], list[tuple[str, str]]]:
    tickers = collect_bootstrap_tickers(root)
    if bootstrap_max_tickers > 0:
        tickers = tickers[:bootstrap_max_tickers]
    if not tickers:
        raise RuntimeError("No bootstrap tickers available to create OHLCV master CSV.")

    end_exclusive = (pd.Timestamp(through).normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache_dir = root / ".yf_cache"
    setup_yf_cache(cache_dir)

    raw_data, failed = download_ohlcv_data(tickers, bootstrap_start, end_exclusive)
    success_tickers = sorted(raw_data["ticker"].unique().tolist()) if not raw_data.empty else []
    if not success_tickers:
        raise RuntimeError(f"No successful ticker downloads during bootstrap. Failures: {failed}")

    refreshed = compute_all_indicators(compute_adjusted_ohlc(raw_data))
    refreshed["date"] = pd.to_datetime(refreshed["date"])
    refreshed = refreshed.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    refreshed_to_write = refreshed.copy()
    refreshed_to_write["date"] = refreshed_to_write["date"].dt.strftime("%Y-%m-%d")
    refreshed_to_write.to_csv(csv_path, index=False)
    return refreshed, tickers, failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="apex_ohlcv_full_2015_2026.csv")
    parser.add_argument("--through", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--lookback-days", type=int, default=500)
    parser.add_argument("--bootstrap-start", default="2015-01-01")
    parser.add_argument("--bootstrap-max-tickers", type=int, default=0)
    args = parser.parse_args()

    root = Path.cwd()
    csv_path = (root / args.csv).resolve()
    if not csv_path.exists():
        print(f"Master OHLCV not found, bootstrapping from ticker lists: {csv_path}")
        updated, tickers, failed = build_ohlcv_from_scratch(
            root,
            csv_path,
            args.through,
            args.bootstrap_start,
            int(args.bootstrap_max_tickers),
        )
        summary_rows = []
        for ticker in tickers:
            g = updated.loc[updated["ticker"] == ticker]
            if g.empty:
                summary_rows.append({"ticker": ticker, "rows": 0})
                continue
            summary_rows.append(
                {
                    "ticker": ticker,
                    "rows": int(len(g)),
                    "start_date": g["date"].min().strftime("%Y-%m-%d"),
                    "end_date": g["date"].max().strftime("%Y-%m-%d"),
                    "last_close": float(g.iloc[-1]["close"]),
                    "refreshed": int(ticker in set(updated["ticker"].astype(str))),
                }
            )
        summary = pd.DataFrame(summary_rows)
        summary_path = root / "data" / "extracts" / "ohlcv_refresh_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)

        print(f"CSV bootstrapped: {csv_path}")
        print(f"Requested through: {pd.Timestamp(args.through).normalize().date()}")
        print(f"New max date: {pd.Timestamp(updated['date'].max()).date()}")
        print(f"Tickers total: {len(tickers)}")
        print(f"Tickers refreshed successfully: {updated['ticker'].nunique()}")
        print(f"Tickers failed: {len(failed)}")
        if failed:
            print("Failures:")
            for ticker, reason in failed[:20]:
                print(f"  - {ticker}: {reason}")
            if len(failed) > 20:
                print(f"  ... and {len(failed) - 20} more")
        print(f"Summary saved: {summary_path}")
        return 0

    existing = pd.read_csv(csv_path, parse_dates=["date"])
    existing["date"] = pd.to_datetime(existing["date"])
    columns = existing.columns.tolist()
    tickers = sorted(existing["ticker"].unique().tolist())

    current_max = pd.Timestamp(existing["date"].max()).normalize()
    requested_through = pd.Timestamp(args.through).normalize()
    end_exclusive = (requested_through + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    overlap_start = max(existing["date"].min(), current_max - pd.Timedelta(days=int(args.lookback_days)))
    overlap_start_str = pd.Timestamp(overlap_start).strftime("%Y-%m-%d")

    cache_dir = root / ".yf_cache"
    setup_yf_cache(cache_dir)

    raw_data, failed = download_ohlcv_data(tickers, overlap_start_str, end_exclusive)
    success_tickers = sorted(raw_data["ticker"].unique().tolist()) if not raw_data.empty else []
    if not success_tickers:
        raise RuntimeError(f"No successful ticker downloads. Failures: {failed}")

    refreshed = compute_all_indicators(compute_adjusted_ohlc(raw_data))
    for col in columns:
        if col not in refreshed.columns:
            refreshed[col] = pd.NA
    refreshed = refreshed[columns].copy()
    refreshed["date"] = pd.to_datetime(refreshed["date"])

    keep_old = existing.loc[~existing["ticker"].isin(success_tickers)].copy()
    keep_prefix = existing.loc[
        existing["ticker"].isin(success_tickers) & (existing["date"] < overlap_start)
    ].copy()
    updated = pd.concat([keep_old, keep_prefix, refreshed], ignore_index=True)
    updated = updated.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)

    updated_to_write = updated.copy()
    updated_to_write["date"] = updated_to_write["date"].dt.strftime("%Y-%m-%d")
    updated_to_write.to_csv(csv_path, index=False)

    summary_rows = []
    for ticker in tickers:
        g = updated.loc[updated["ticker"] == ticker]
        if g.empty:
            summary_rows.append({"ticker": ticker, "rows": 0})
            continue
        summary_rows.append(
            {
                "ticker": ticker,
                "rows": int(len(g)),
                "start_date": g["date"].min().strftime("%Y-%m-%d"),
                "end_date": g["date"].max().strftime("%Y-%m-%d"),
                "last_close": float(g.iloc[-1]["close"]),
                "refreshed": int(ticker in success_tickers),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = root / "data" / "extracts" / "ohlcv_refresh_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print(f"CSV updated: {csv_path}")
    print(f"Requested through: {requested_through.date()}")
    print(f"Previous max date: {current_max.date()}")
    print(f"New max date: {pd.Timestamp(updated['date'].max()).date()}")
    print(f"Tickers total: {len(tickers)}")
    print(f"Tickers refreshed successfully: {len(success_tickers)}")
    print(f"Tickers failed: {len(failed)}")
    if failed:
        print("Failures:")
        for ticker, reason in failed[:20]:
            print(f"  - {ticker}: {reason}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
    print(f"Summary saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
