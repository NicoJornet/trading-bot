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


ROOT = Path(__file__).resolve().parent
DYNAMIC_DB_PATH = ROOT / "data" / "dynamic_universe" / "dynamic_universe_current.csv"


def stage_rank(value: str) -> int:
    return {
        "approved_live": 5,
        "probation_live": 4,
        "targeted_integration": 3,
        "watch_queue": 2,
        "review_queue": 1,
        "reject_queue": 0,
        "blocked_broker": -1,
    }.get(str(value or ""), -2)


def status_rank(value: str) -> int:
    return {
        "approved": 5,
        "prime_watch": 4,
        "watch": 3,
        "review": 2,
        "discovered": 1,
        "reject": 0,
    }.get(str(value or ""), -1)


def fit_rank(value: str) -> int:
    return {"high": 3, "medium": 2, "low": 1, "weak": 0}.get(str(value or ""), -1)


def load_missing_candidates(csv_path: Path, max_count: int) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not DYNAMIC_DB_PATH.exists():
        return []

    existing = pd.read_csv(csv_path, usecols=["ticker"])
    covered = set(existing["ticker"].dropna().astype(str))

    db = pd.read_csv(DYNAMIC_DB_PATH, low_memory=False)
    if db.empty or "ticker" not in db.columns:
        return []

    for col in (
        "promotion_stage",
        "dynamic_status",
        "scan_algo_fit",
        "governance_role",
        "candidate_status",
        "broker_tradeable",
    ):
        if col not in db.columns:
            db[col] = ""
    for col in ("promotion_score", "dynamic_conviction_score", "recent_score", "scan_algo_compat_score_v2"):
        if col not in db.columns:
            db[col] = 0.0
        db[col] = pd.to_numeric(db[col], errors="coerce").fillna(0.0)

    db["ticker"] = db["ticker"].astype(str)
    db["stage_rank"] = db["promotion_stage"].map(stage_rank).fillna(-2)
    db["status_rank"] = db["dynamic_status"].map(status_rank).fillna(-1)
    db["fit_rank"] = db["scan_algo_fit"].map(fit_rank).fillna(-1)

    eligible = db.loc[
        ~db["ticker"].isin(covered)
        & ~db["promotion_stage"].fillna("").isin(["blocked_broker", "reject_queue"])
        & db["candidate_status"].fillna("").ne("hard_exclusion")
        & (
            (db["stage_rank"] >= 1)
            | (db["status_rank"] >= 2)
            | (db["fit_rank"] >= 1)
        )
    ].copy()
    if eligible.empty:
        return []

    eligible = eligible.sort_values(
        [
            "stage_rank",
            "status_rank",
            "fit_rank",
            "promotion_score",
            "dynamic_conviction_score",
            "scan_algo_compat_score_v2",
            "recent_score",
        ],
        ascending=[False, False, False, False, False, False, False],
    )
    return eligible["ticker"].dropna().astype(str).head(max_count).tolist()


def append_candidates(csv_path: Path, tickers: list[str]) -> tuple[pd.DataFrame, Path]:
    existing = pd.read_csv(csv_path, parse_dates=["date"])
    columns = existing.columns.tolist()
    start_date = existing["date"].min().strftime("%Y-%m-%d")
    end_date_exclusive = (existing["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw_data, failed = download_ohlcv_data(tickers, start_date, end_date_exclusive)
    if failed:
        print("Download warnings:")
        for ticker, reason in failed:
            print(f"  - {ticker}: {reason}")

    appended = compute_all_indicators(compute_adjusted_ohlc(raw_data))
    for col in columns:
        if col not in appended.columns:
            appended[col] = pd.NA
    appended = appended[columns].copy()

    updated = existing.loc[~existing["ticker"].isin(tickers)].copy()
    updated = pd.concat([updated, appended], ignore_index=True)
    updated["date"] = pd.to_datetime(updated["date"])
    updated = updated.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)

    updated_to_write = updated.copy()
    updated_to_write["date"] = updated_to_write["date"].dt.strftime("%Y-%m-%d")
    updated_to_write.to_csv(csv_path, index=False)

    summary_rows = []
    for ticker in tickers:
        group = updated.loc[updated["ticker"] == ticker].copy()
        if group.empty:
            summary_rows.append({"ticker": ticker, "rows": 0})
            continue
        summary_rows.append(
            {
                "ticker": ticker,
                "rows": int(len(group)),
                "start_date": group["date"].min().strftime("%Y-%m-%d"),
                "end_date": group["date"].max().strftime("%Y-%m-%d"),
                "first_close": float(group.iloc[0]["close"]),
                "last_close": float(group.iloc[-1]["close"]),
                "has_momentum_score": int(group["momentum_score"].notna().any()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = ROOT / "data" / "extracts" / "ohlcv_dynamic_candidate_extension_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Append top dynamic candidates missing from the OHLCV master CSV.")
    parser.add_argument("--csv", default="apex_ohlcv_full_2015_2026.csv")
    parser.add_argument("--max-count", type=int, default=60)
    parser.add_argument("--tickers", default="", help="Optional comma-separated explicit tickers.")
    args = parser.parse_args()

    csv_path = (ROOT / args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    setup_yf_cache(ROOT / ".yf_cache")
    if args.tickers:
        tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
    else:
        tickers = load_missing_candidates(csv_path, max_count=max(1, int(args.max_count)))
    if not tickers:
        print("No missing dynamic candidates selected for OHLCV extension.")
        return 0

    summary, summary_path = append_candidates(csv_path, tickers)
    print(f"Updated CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
