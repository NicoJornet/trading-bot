from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from append_tickers_to_ohlcv import compute_adjusted_ohlc, download_ohlcv_data, setup_yf_cache


ROOT = Path(__file__).resolve().parent
CONTEXT_PATH = ROOT / "data" / "context" / "ticker_context_latest.csv"
OHLCV_PATH = ROOT / "apex_ohlcv_full_2015_2026.csv"
OUT_PATH = ROOT / "data" / "benchmarks" / "fx_reference_ohlcv.csv"
SUMMARY_PATH = ROOT / "data" / "benchmarks" / "fx_reference_summary.csv"

DIRECT_USD_PAIRS = {
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "AUD": "AUDUSD=X",
    "NZD": "NZDUSD=X",
}
INVERTED_USD_PAIRS = {
    "CAD": "CAD=X",
    "CHF": "CHF=X",
    "CLP": "CLP=X",
    "CNY": "CNY=X",
    "DKK": "DKK=X",
    "HKD": "HKD=X",
    "ILS": "ILS=X",
    "INR": "INR=X",
    "JPY": "JPY=X",
    "KRW": "KRW=X",
    "MXN": "MXN=X",
    "MYR": "MYR=X",
    "NOK": "NOK=X",
    "PLN": "PLN=X",
    "SEK": "SEK=X",
    "SGD": "SGD=X",
    "TWD": "TWD=X",
    "ZAR": "ZAR=X",
    "BRL": "BRL=X",
}
ALIASES = {"ILA": "ILS", "ZAC": "ZAR"}


def _load_needed_currencies() -> list[str]:
    context = pd.read_csv(CONTEXT_PATH, usecols=["currency"], low_memory=False)
    currencies = context["currency"].dropna().astype(str).str.upper().map(lambda x: ALIASES.get(x, x)).unique().tolist()
    return sorted([c for c in currencies if c and c != "USD"])


def _load_calendar() -> pd.DatetimeIndex:
    ohlcv = pd.read_csv(OHLCV_PATH, usecols=["date"], parse_dates=["date"], low_memory=False)
    return pd.DatetimeIndex(sorted(pd.to_datetime(ohlcv["date"]).dropna().unique()))


def _build_constant_series(calendar: pd.DatetimeIndex, value: float) -> pd.DataFrame:
    return pd.DataFrame({"date": calendar, "adj_close": value})


def _compute_fx_frame(calendar: pd.DatetimeIndex, currency: str, raw: pd.DataFrame | None, invert: bool) -> pd.DataFrame:
    if raw is None or raw.empty:
        frame = _build_constant_series(calendar, np.nan)
    else:
        frame = raw[["date", "adj_close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame = frame.drop_duplicates("date", keep="last").sort_values("date")
        frame = frame.set_index("date").reindex(calendar).ffill().reset_index().rename(columns={"index": "date"})
    if invert:
        frame["fx_to_usd"] = 1.0 / frame["adj_close"].replace(0, np.nan)
    else:
        frame["fx_to_usd"] = frame["adj_close"]
    frame["currency"] = currency
    return frame[["date", "currency", "fx_to_usd"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh FX reference layers for non-USD instruments.")
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--through", default=date.today().isoformat())
    args = parser.parse_args()

    setup_yf_cache(ROOT / ".yf_cache")
    calendar = _load_calendar()
    currencies = _load_needed_currencies()
    pair_requests = {**{c: (t, False) for c, t in DIRECT_USD_PAIRS.items()}, **{c: (t, True) for c, t in INVERTED_USD_PAIRS.items()}}
    download_tickers = sorted({pair for currency in currencies if currency in pair_requests for pair, _ in [pair_requests[currency]]} | {"EURUSD=X"})

    raw, failed = download_ohlcv_data(
        download_tickers,
        args.start,
        (pd.Timestamp(args.through) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    adjusted = compute_adjusted_ohlc(raw)
    by_pair = {ticker: df.copy() for ticker, df in adjusted.groupby("ticker", sort=False)}
    eurusd = _compute_fx_frame(calendar, "EUR", by_pair.get("EURUSD=X"), invert=False)[["date", "fx_to_usd"]].rename(columns={"fx_to_usd": "eur_to_usd"})

    frames = [_build_constant_series(calendar, 1.0).assign(currency="USD", fx_to_usd=1.0)[["date", "currency", "fx_to_usd"]]]
    for currency in currencies:
        spec = pair_requests.get(currency)
        if spec is None:
            frame = _build_constant_series(calendar, np.nan).assign(currency=currency, fx_to_usd=np.nan)[["date", "currency", "fx_to_usd"]]
        else:
            pair, invert = spec
            frame = _compute_fx_frame(calendar, currency, by_pair.get(pair), invert=invert)
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out = out.merge(eurusd, on="date", how="left")
    out["fx_to_eur"] = out["fx_to_usd"] / out["eur_to_usd"].replace(0, np.nan)
    out.loc[out["currency"] == "EUR", "fx_to_eur"] = 1.0
    out.loc[out["currency"] == "USD", "fx_to_eur"] = 1.0 / out.loc[out["currency"] == "USD", "eur_to_usd"].replace(0, np.nan)
    out["return_1d"] = out.groupby("currency", sort=False)["fx_to_usd"].pct_change()
    out["return_21d"] = out.groupby("currency", sort=False)["fx_to_usd"].pct_change(21)
    out["return_63d"] = out.groupby("currency", sort=False)["fx_to_usd"].pct_change(63)
    out["sma_50"] = out.groupby("currency", sort=False)["fx_to_usd"].transform(lambda s: s.rolling(50).mean())
    out["sma_200"] = out.groupby("currency", sort=False)["fx_to_usd"].transform(lambda s: s.rolling(200).mean())
    out["above_sma200"] = (out["fx_to_usd"] > out["sma_200"]).astype(int)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    summary = (
        out.groupby("currency", dropna=False)
        .agg(
            rows=("date", "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            coverage=("fx_to_usd", lambda s: float(pd.to_numeric(s, errors="coerce").notna().mean())),
            last_fx_to_usd=("fx_to_usd", "last"),
            last_fx_to_eur=("fx_to_eur", "last"),
        )
        .reset_index()
        .sort_values("currency", kind="mergesort")
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"FX reference OHLCV: {OUT_PATH}")
    print(f"FX summary: {SUMMARY_PATH}")
    if failed:
        print("Download warnings:")
        for ticker, reason in failed:
            print(f"  - {ticker}: {reason}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
