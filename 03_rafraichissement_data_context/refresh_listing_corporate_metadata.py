from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from refresh_context_earnings_snapshots import load_ohlcv_tickers, setup_yf_cache
from refresh_universe_support_files import fetch_actions_rows


ROOT = Path(__file__).resolve().parent
OHLCV_PATH = ROOT / "apex_ohlcv_full_2015_2026.csv"
ACTIONS_PATH = ROOT / "data" / "extracts" / "apex_corporate_actions.csv"
METADATA_PATH = ROOT / "data" / "extracts" / "listing_corporate_metadata.csv"
SUMMARY_PATH = ROOT / "data" / "extracts" / "listing_corporate_metadata_summary.csv"


def _load_ohlcv_summary() -> pd.DataFrame:
    usecols = ["date", "ticker"]
    df = pd.read_csv(OHLCV_PATH, usecols=usecols, low_memory=False, parse_dates=["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    summary = (
        df.groupby("ticker", dropna=False)
        .agg(first_ohlcv_date=("date", "min"), last_ohlcv_date=("date", "max"), bars_total=("date", "size"))
        .reset_index()
    )
    summary["listing_age_days"] = (summary["last_ohlcv_date"] - summary["first_ohlcv_date"]).dt.days
    return summary


def _load_actions() -> pd.DataFrame:
    if ACTIONS_PATH.exists():
        df = pd.read_csv(ACTIONS_PATH, low_memory=False)
    else:
        df = pd.DataFrame(columns=["date", "value", "ticker", "type"])
    if not df.empty:
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _refresh_actions(existing: pd.DataFrame, tickers: list[str], max_refresh: int) -> tuple[pd.DataFrame, list[str]]:
    refreshed: list[str] = []
    frames = [existing.loc[~existing["ticker"].isin(tickers[:max_refresh])].copy()] if not existing.empty else []
    for ticker in tickers[:max_refresh]:
        frame = fetch_actions_rows(ticker)
        if frame.empty:
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_convert(None)
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frames.append(frame)
        refreshed.append(ticker)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "value", "ticker", "type"])
    if not combined.empty:
        combined = combined.dropna(subset=["date", "ticker", "type"]).sort_values(["ticker", "date", "type"], kind="mergesort")
    return combined, refreshed


def _compute_metadata(ohlcv_summary: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        actions = pd.DataFrame(columns=["date", "value", "ticker", "type"])
    cutoff_5y = pd.Timestamp(date.today()) - pd.Timedelta(days=365 * 5)
    action_summary = (
        actions.assign(in_5y=actions["date"] >= cutoff_5y)
        .groupby("ticker", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "last_dividend_date": g.loc[g["type"] == "DIVIDEND", "date"].max(),
                    "last_split_date": g.loc[g["type"] == "SPLIT", "date"].max(),
                    "dividend_count_total": int((g["type"] == "DIVIDEND").sum()),
                    "split_count_total": int((g["type"] == "SPLIT").sum()),
                    "dividend_count_5y": int(((g["type"] == "DIVIDEND") & g["in_5y"]).sum()),
                    "split_count_5y": int(((g["type"] == "SPLIT") & g["in_5y"]).sum()),
                }
            )
        )
        .reset_index()
    )
    out = ohlcv_summary.merge(action_summary, on="ticker", how="left")
    for col in ("dividend_count_total", "split_count_total", "dividend_count_5y", "split_count_5y"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    out["recent_dividend_365d"] = (
        pd.to_datetime(out["last_dividend_date"], errors="coerce") >= (pd.Timestamp(date.today()) - pd.Timedelta(days=365))
    ).astype(int)
    out["recent_split_365d"] = (
        pd.to_datetime(out["last_split_date"], errors="coerce") >= (pd.Timestamp(date.today()) - pd.Timedelta(days=365))
    ).astype(int)
    for col in ("first_ohlcv_date", "last_ohlcv_date", "last_dividend_date", "last_split_date"):
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.sort_values("ticker", kind="mergesort").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh listing age and corporate actions metadata layers.")
    parser.add_argument("--max-refresh", type=int, default=120)
    args = parser.parse_args()

    setup_yf_cache(ROOT / ".yf_cache")
    ohlcv_summary = _load_ohlcv_summary()
    existing_actions = _load_actions()
    tickers = load_ohlcv_tickers()
    refreshed_actions, refreshed = _refresh_actions(existing_actions, tickers, max_refresh=max(0, int(args.max_refresh)))
    metadata = _compute_metadata(ohlcv_summary, refreshed_actions)

    ACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    actions_to_write = refreshed_actions.copy()
    if not actions_to_write.empty:
        actions_to_write["date"] = pd.to_datetime(actions_to_write["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    actions_to_write.to_csv(ACTIONS_PATH, index=False)
    metadata.to_csv(METADATA_PATH, index=False)

    summary = pd.DataFrame(
        [
            {
                "as_of": date.today().isoformat(),
                "tickers": int(metadata["ticker"].nunique()),
                "actions_rows": int(len(refreshed_actions)),
                "refreshed_actions_tickers": int(len(refreshed)),
                "with_dividend_history": int((metadata["dividend_count_total"] > 0).sum()),
                "with_split_history": int((metadata["split_count_total"] > 0).sum()),
                "median_listing_age_days": float(pd.to_numeric(metadata["listing_age_days"], errors="coerce").median()),
            }
        ]
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"Corporate actions: {ACTIONS_PATH}")
    print(f"Listing/corporate metadata: {METADATA_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
