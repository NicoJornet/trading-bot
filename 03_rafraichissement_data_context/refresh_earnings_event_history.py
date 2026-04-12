from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from refresh_context_earnings_snapshots import load_ohlcv_tickers, setup_yf_cache


ROOT = Path(__file__).resolve().parent
EARNINGS_DIR = ROOT / "data" / "earnings"
HISTORY_DIR = EARNINGS_DIR / "history"
LATEST_OUT_PATH = EARNINGS_DIR / "earnings_events_latest.csv"
SUMMARY_OUT_PATH = ROOT / "data" / "extracts" / "earnings_events_refresh_summary.csv"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "reported_eps",
        "Surprise(%)": "surprise_pct",
    }
    out = df.rename(columns=rename_map).copy()
    out.columns = [str(col).strip().lower().replace(" ", "_").replace("%", "pct").replace("/", "_") for col in out.columns]
    return out


def _fetch_events(ticker: str, limit: int) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = _normalize_columns(df.reset_index())
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "earnings_date"})
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    df["ticker"] = ticker
    df["as_of_pull"] = date.today().isoformat()
    for col in ("eps_estimate", "reported_eps", "surprise_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA
    return df


def _load_snapshot_proxy_events() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(HISTORY_DIR.glob("earnings_snapshot_*.csv")):
        snap = pd.read_csv(path, low_memory=False)
        if snap.empty or "ticker" not in snap.columns:
            continue
        snap["ticker"] = snap["ticker"].astype(str).str.upper()
        snap["as_of_snapshot"] = pd.to_datetime(snap["as_of"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "last_earnings_date" in snap.columns:
            last_df = snap.loc[snap["last_earnings_date"].notna() & snap["last_earnings_date"].astype(str).str.len().gt(0)].copy()
            if not last_df.empty:
                last_df = last_df.rename(columns={"last_earnings_date": "earnings_date"})
                last_df["event_source"] = "snapshot_last"
                last_df["event_role"] = "last_seen"
                frames.append(last_df)
        if "next_earnings_date" in snap.columns:
            next_df = snap.loc[snap["next_earnings_date"].notna() & snap["next_earnings_date"].astype(str).str.len().gt(0)].copy()
            if not next_df.empty:
                next_df = next_df.rename(columns={"next_earnings_date": "earnings_date"})
                next_df["event_source"] = "snapshot_next"
                next_df["event_role"] = "next_seen"
                frames.append(next_df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["earnings_date"] = pd.to_datetime(out["earnings_date"], errors="coerce")
    out = out.dropna(subset=["ticker", "earnings_date"]).copy()
    keep_cols = [
        "ticker",
        "earnings_date",
        "as_of_snapshot",
        "event_source",
        "event_role",
        "mostRecentQuarter",
        "earningsQuarterlyGrowth",
        "days_since_last_earnings",
        "days_to_next_earnings",
    ]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out[keep_cols]


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh earnings event history with surprise data.")
    parser.add_argument("--tickers", default="", help="Optional comma-separated tickers.")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--try-yfinance-events", action="store_true", help="Attempt detailed event pulls via yfinance get_earnings_dates().")
    args = parser.parse_args()

    setup_yf_cache(ROOT / ".yf_cache")
    tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()] if args.tickers else load_ohlcv_tickers()
    if LATEST_OUT_PATH.exists():
        try:
            existing = pd.read_csv(LATEST_OUT_PATH, low_memory=False)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    frames: list[pd.DataFrame] = []
    refreshed: list[str] = []
    failed: list[str] = []
    if args.try_yfinance_events:
        for ticker in tickers:
            df = _fetch_events(ticker, args.limit)
            if df.empty:
                failed.append(ticker)
                continue
            frames.append(df)
            refreshed.append(ticker)

    fetched_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    snapshot_proxy = _load_snapshot_proxy_events()
    if not fetched_df.empty:
        fetched_df["event_source"] = "yfinance_event"
        fetched_df["event_role"] = "reported_event"
        fetched_df["as_of_snapshot"] = fetched_df["as_of_pull"]
    new_df = pd.concat([snapshot_proxy, fetched_df], ignore_index=True) if not snapshot_proxy.empty or not fetched_df.empty else pd.DataFrame()
    if not new_df.empty:
        new_df = new_df.sort_values(["ticker", "earnings_date", "event_source"], kind="mergesort").drop_duplicates(
            ["ticker", "earnings_date", "event_role"], keep="last"
        )
    if not existing.empty and refreshed:
        existing["ticker"] = existing["ticker"].astype(str).str.upper()
        existing = existing.loc[~existing["ticker"].isin(refreshed)].copy()
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty or not new_df.empty else pd.DataFrame()
    if not combined.empty:
        combined["ticker"] = combined["ticker"].astype(str).str.upper()
        combined["earnings_date"] = pd.to_datetime(combined["earnings_date"], errors="coerce")
        combined = combined.sort_values(["ticker", "earnings_date"], kind="mergesort").reset_index(drop=True)
        combined["earnings_date"] = combined["earnings_date"].dt.strftime("%Y-%m-%d")

    EARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(LATEST_OUT_PATH, index=False)
    combined.to_csv(HISTORY_DIR / f"earnings_events_snapshot_{date.today().isoformat()}.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "as_of": date.today().isoformat(),
                "tickers_requested": int(len(tickers)),
                "tickers_refreshed": int(len(refreshed)),
                "tickers_failed": int(len(failed)),
                "events_rows": int(len(combined)),
                "with_surprise_pct": int(pd.to_numeric(combined["surprise_pct"], errors="coerce").notna().sum()) if (not combined.empty and "surprise_pct" in combined.columns) else 0,
                "with_reported_eps": int(pd.to_numeric(combined["reported_eps"], errors="coerce").notna().sum()) if (not combined.empty and "reported_eps" in combined.columns) else 0,
                "with_snapshot_proxy": int(combined["event_source"].astype(str).str.startswith("snapshot").sum()) if (not combined.empty and "event_source" in combined.columns) else 0,
                "with_yfinance_events": int((combined["event_source"] == "yfinance_event").sum()) if (not combined.empty and "event_source" in combined.columns) else 0,
            }
        ]
    )
    SUMMARY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUT_PATH, index=False)

    print(f"Earnings events latest: {LATEST_OUT_PATH}")
    print(f"Summary: {SUMMARY_OUT_PATH}")
    print(summary.to_string(index=False))
    if failed:
        print(f"Failures ({len(failed)}): {', '.join(failed[:20])}{' ...' if len(failed) > 20 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
