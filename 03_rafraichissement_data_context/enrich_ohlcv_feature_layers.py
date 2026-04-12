from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REL_COLUMNS = (
    "spy_return_63d",
    "spy_return_126d",
    "spy_return_252d",
    "rel_return_63d_spy",
    "rel_return_126d_spy",
    "rel_return_252d_spy",
    "beta_63_spy",
    "residual_return_63d",
    "residual_return_126d",
    "residual_return_252d",
    "residual_momentum_score",
)

BEHAVIOR_COLUMNS = (
    "overnight_return_1d",
    "intraday_return_1d",
    "gap_pct",
    "gap_zscore_20",
    "gap_fill_share",
    "rel_volume_20",
    "volume_zscore_20",
    "dollar_volume_zscore_20",
    "close_in_range",
    "efficiency_ratio_20",
    "efficiency_ratio_60",
    "corr_63_spy",
    "downside_beta_63_spy",
)


def _zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def _efficiency_ratio(close: pd.Series, lookback: int) -> pd.Series:
    abs_move = close.diff().abs()
    numer = (close - close.shift(lookback)).abs()
    denom = abs_move.rolling(lookback, min_periods=max(lookback // 2, 5)).sum()
    return numer / denom.replace(0, np.nan)


def compute_cross_ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)

    spy = out.loc[out["ticker"] == "SPY", ["date", "adj_close"]].dropna().sort_values("date")
    if spy.empty:
        for col in REL_COLUMNS:
            out[col] = np.nan
        return out

    spy = spy.assign(
        spy_return_1d=spy["adj_close"].pct_change(),
        spy_return_63d=spy["adj_close"].pct_change(63),
        spy_return_126d=spy["adj_close"].pct_change(126),
        spy_return_252d=spy["adj_close"].pct_change(252),
    )
    spy = spy.set_index("date")

    frames: list[pd.DataFrame] = []
    for ticker, group in out.groupby("ticker", sort=False):
        g = group.copy().sort_values("date")
        px_open = pd.to_numeric(g.get("adj_open", g.get("open")), errors="coerce")
        px_high = pd.to_numeric(g.get("adj_high", g.get("high")), errors="coerce")
        px_low = pd.to_numeric(g.get("adj_low", g.get("low")), errors="coerce")
        px_close = pd.to_numeric(g.get("adj_close", g.get("close")), errors="coerce")
        px_volume = pd.to_numeric(g.get("adj_volume", g.get("volume")), errors="coerce")
        prev_close = px_close.shift(1)

        g["overnight_return_1d"] = px_open / prev_close - 1.0
        g["intraday_return_1d"] = px_close / px_open - 1.0
        g["gap_pct"] = g["overnight_return_1d"]
        g["gap_zscore_20"] = _zscore(g["gap_pct"], 20, 10)

        gap_fill = pd.Series(np.nan, index=g.index, dtype=float)
        pos_gap = (px_open > prev_close) & prev_close.notna()
        neg_gap = (px_open < prev_close) & prev_close.notna()
        pos_denom = (px_open - prev_close).replace(0, np.nan)
        neg_denom = (prev_close - px_open).replace(0, np.nan)
        gap_fill.loc[pos_gap] = ((px_open - px_low) / pos_denom).loc[pos_gap]
        gap_fill.loc[neg_gap] = ((px_high - px_open) / neg_denom).loc[neg_gap]
        g["gap_fill_share"] = gap_fill.clip(lower=0.0, upper=1.0).fillna(0.0)

        g["rel_volume_20"] = px_volume / px_volume.rolling(20, min_periods=10).mean().replace(0, np.nan)
        g["volume_zscore_20"] = _zscore(np.log1p(px_volume), 20, 10)
        dollar_volume = (px_close * px_volume).replace([np.inf, -np.inf], np.nan)
        g["dollar_volume_zscore_20"] = _zscore(np.log1p(dollar_volume.clip(lower=0.0)), 20, 10)

        range_size = (px_high - px_low).replace(0, np.nan)
        g["close_in_range"] = ((px_close - px_low) / range_size).clip(lower=0.0, upper=1.0).fillna(0.5)
        g["efficiency_ratio_20"] = _efficiency_ratio(px_close, 20)
        g["efficiency_ratio_60"] = _efficiency_ratio(px_close, 60)

        g["spy_return_63d"] = g["date"].map(spy["spy_return_63d"])
        g["spy_return_126d"] = g["date"].map(spy["spy_return_126d"])
        g["spy_return_252d"] = g["date"].map(spy["spy_return_252d"])

        aligned_spy_1d = g["date"].map(spy["spy_return_1d"])
        own_ret_1d = pd.to_numeric(g["return_1d"], errors="coerce")
        cov63 = own_ret_1d.rolling(63, min_periods=40).cov(aligned_spy_1d)
        var63 = aligned_spy_1d.rolling(63, min_periods=40).var()
        g["beta_63_spy"] = cov63 / var63.replace(0, np.nan)
        g["corr_63_spy"] = own_ret_1d.rolling(63, min_periods=40).corr(aligned_spy_1d)

        spy_down = aligned_spy_1d.where(aligned_spy_1d < 0)
        own_on_down = own_ret_1d.where(aligned_spy_1d < 0)
        cov_down = own_on_down.rolling(63, min_periods=20).cov(spy_down)
        var_down = spy_down.rolling(63, min_periods=20).var()
        g["downside_beta_63_spy"] = cov_down / var_down.replace(0, np.nan)

        for lookback in (63, 126, 252):
            own_col = f"return_{lookback}d"
            spy_col = f"spy_return_{lookback}d"
            rel_col = f"rel_return_{lookback}d_spy"
            resid_col = f"residual_return_{lookback}d"
            own = pd.to_numeric(g[own_col], errors="coerce")
            spy_ret = pd.to_numeric(g[spy_col], errors="coerce")
            g[rel_col] = (1.0 + own) / (1.0 + spy_ret) - 1.0
            g[resid_col] = own - g["beta_63_spy"] * spy_ret

        g["residual_momentum_score"] = (
            0.20 * g["residual_return_63d"]
            + 0.40 * g["residual_return_126d"]
            + 0.40 * g["residual_return_252d"]
        )
        frames.append(g)

    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich the OHLCV master CSV with cross-ticker SPY-relative features.")
    parser.add_argument("--csv", default="apex_ohlcv_full_2015_2026.csv")
    args = parser.parse_args()

    root = Path.cwd()
    csv_path = (root / args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    enriched = compute_cross_ticker_features(df)
    enriched_to_write = enriched.copy()
    enriched_to_write["date"] = enriched_to_write["date"].dt.strftime("%Y-%m-%d")
    enriched_to_write.to_csv(csv_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "rows": int(len(enriched)),
                "tickers": int(enriched["ticker"].nunique()),
                "max_date": str(pd.Timestamp(enriched["date"].max()).date()),
                "spy_present": int((enriched["ticker"] == "SPY").any()),
                "non_null_beta_63_spy": int(enriched["beta_63_spy"].notna().sum()),
                "non_null_residual_score": int(enriched["residual_momentum_score"].notna().sum()),
                "non_null_gap_zscore_20": int(enriched["gap_zscore_20"].notna().sum()),
                "non_null_rel_volume_20": int(enriched["rel_volume_20"].notna().sum()),
                "non_null_efficiency_ratio_60": int(enriched["efficiency_ratio_60"].notna().sum()),
                "non_null_corr_63_spy": int(enriched["corr_63_spy"].notna().sum()),
            }
        ]
    )
    summary_path = root / "data" / "extracts" / "ohlcv_feature_enrichment_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print(f"Updated CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
