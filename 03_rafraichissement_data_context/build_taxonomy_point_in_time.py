from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
CONTEXT_DIR = ROOT / "data" / "context"
HISTORY_DIR = CONTEXT_DIR / "history"
OUT_PATH = CONTEXT_DIR / "taxonomy_point_in_time.csv"
LATEST_OUT_PATH = CONTEXT_DIR / "taxonomy_point_in_time_latest.csv"
SUMMARY_OUT_PATH = ROOT / "data" / "extracts" / "taxonomy_point_in_time_summary.csv"

URANIUM_TICKERS = {"CCJ", "DNN", "LEU", "NXE", "UEC"}
CRYPTO_BETA_TICKERS = {"MSTR", "MARA", "RIOT"}
SPACE_DEFENSE_TICKERS = {"RKLB", "LUNR", "PLTR", "AVAV", "KTOS"}


def infer_cluster_key(ticker: str, sector_key: str, industry_key: str, industry: str) -> str:
    ticker = ticker.upper()
    sector_key = "" if pd.isna(sector_key) else str(sector_key).strip().lower()
    industry_key = "" if pd.isna(industry_key) else str(industry_key).strip().lower()
    industry = "" if pd.isna(industry) else str(industry).strip().lower()

    if ticker in URANIUM_TICKERS:
        return "uranium"
    if ticker in CRYPTO_BETA_TICKERS:
        return "crypto-beta"
    if ticker in SPACE_DEFENSE_TICKERS or "aerospace" in industry_key or "defense" in industry_key:
        return "space-defense"
    if "semiconductor" in industry_key or "semiconductor" in industry:
        return "semiconductors"
    if "communication-equipment" in industry_key or "optical" in industry or "network" in industry:
        return "optical-networking"
    if "software" in industry_key or "software" in industry:
        return "cloud-software"
    if "electrical-equipment" in industry_key or "industrial" in industry_key:
        return "industrial-compounders"
    if "oil" in industry_key or "gas" in industry_key:
        return "oil-gas"
    if "gold" in industry_key:
        return "gold-miners"
    if "silver" in industry_key:
        return "silver-miners"
    if "copper" in industry_key:
        return "copper-miners"
    if sector_key:
        return sector_key
    return "unknown"


def load_context_snapshots() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(HISTORY_DIR.glob("ticker_context_snapshot_*.csv")):
        df = pd.read_csv(path, low_memory=False)
        if df.empty or "ticker" not in df.columns:
            continue
        frames.append(df)
    latest_path = CONTEXT_DIR / "ticker_context_latest.csv"
    if latest_path.exists():
        latest = pd.read_csv(latest_path, low_memory=False)
        if not latest.empty and "ticker" in latest.columns:
            frames.append(latest)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["as_of", "ticker"]).drop_duplicates(["as_of", "ticker"], keep="last")
    return df.sort_values(["ticker", "as_of"], kind="mergesort").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build point-in-time sector / industry / cluster taxonomy layers.")
    parser.parse_args()

    df = load_context_snapshots()
    if df.empty:
        raise RuntimeError("No context snapshots available to build taxonomy point-in-time.")

    df["cluster_key"] = [
        infer_cluster_key(ticker, sector_key, industry_key, industry)
        for ticker, sector_key, industry_key, industry in zip(
            df["ticker"],
            df.get("sectorKey", ""),
            df.get("industryKey", ""),
            df.get("industry", ""),
        )
    ]
    df["cluster_source"] = "heuristic"

    latest = df.sort_values(["ticker", "as_of"], kind="mergesort").drop_duplicates("ticker", keep="last").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    latest.to_csv(LATEST_OUT_PATH, index=False)

    summary = (
        df.groupby(["as_of", "cluster_key"], dropna=False)
        .size()
        .rename("tickers")
        .reset_index()
        .sort_values(["as_of", "tickers", "cluster_key"], ascending=[True, False, True], kind="mergesort")
    )
    SUMMARY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUT_PATH, index=False)

    print(f"Taxonomy PIT: {OUT_PATH}")
    print(f"Taxonomy latest: {LATEST_OUT_PATH}")
    print(f"Summary: {SUMMARY_OUT_PATH}")
    print(
        pd.DataFrame(
            [
                {
                    "as_of": date.today().isoformat(),
                    "snapshots": int(df["as_of"].nunique()),
                    "tickers_latest": int(latest["ticker"].nunique()),
                    "clusters_latest": int(latest["cluster_key"].nunique()),
                }
            ]
        ).to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
