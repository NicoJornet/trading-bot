from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

import dynamic_universe_discovery as dud


ROOT = Path(__file__).resolve().parent
EXPORTS_DIR = ROOT / "research" / "exports"
REPORTS_DIR = ROOT / "research" / "reports"

EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_EXPORT = EXPORTS_DIR / "dynamic_universe_swap_candidates.csv"
DEMOTION_EXPORT = EXPORTS_DIR / "dynamic_universe_swap_demotions.csv"
SWAP_EXPORT = EXPORTS_DIR / "dynamic_universe_swap_single_summary.csv"
WALK_EXPORT = EXPORTS_DIR / "dynamic_universe_swap_walkforward.csv"
WALK_SUMMARY_EXPORT = EXPORTS_DIR / "dynamic_universe_swap_walkforward_summary.csv"
REPORT_PATH = REPORTS_DIR / "DYNAMIC_UNIVERSE_SWAP_STUDY_184.md"
SELECTED_ADDS_PATH = ROOT / "data" / "dynamic_universe" / "dynamic_universe_selected_additions.csv"
SELECTED_DEMS_PATH = ROOT / "data" / "dynamic_universe" / "dynamic_universe_selected_demotions.csv"


def safe_rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    return series.rank(pct=True, ascending=ascending, method="average")


def read_ticker_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return set()
    return set(df["ticker"].dropna().astype(str))


def aggregate_candidate_sources(base_prices) -> pd.DataFrame:
    dynamic_db_path = ROOT / "data" / "dynamic_universe" / "dynamic_universe_current.csv"
    dyn = pd.read_csv(dynamic_db_path) if dynamic_db_path.exists() else pd.DataFrame(columns=["ticker"])
    selected_adds = read_ticker_set(SELECTED_ADDS_PATH)

    rows = []
    for path in sorted(EXPORTS_DIR.glob("dynamic_universe*_single_additions.csv")):
        df = pd.read_csv(path)
        if "name" not in df.columns:
            continue
        df = df[df["name"] != "baseline_184"].copy()
        if df.empty:
            continue
        df["ticker"] = df["name"].astype(str)
        df["source_file"] = path.name
        rows.append(df)

    hist = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ticker"])
    if not hist.empty:
        hist = (
            hist.sort_values(
                ["ticker", "oos_delta_roi_pct", "oos_delta_sharpe", "full_delta_roi_pct"],
                ascending=[True, False, False, False],
            )
            .drop_duplicates("ticker")
            .rename(columns={"name": "backtest_name"})
        )

    merged = dyn.merge(
        hist[
            [
                "ticker",
                "source_file",
                "recent_score",
                "recent_r63",
                "recent_r126",
                "recent_r252",
                "scan_algo_fit",
                "scan_algo_compat_score",
                "scan_latest_rank_if_added",
                "scan_days_top15_if_added",
                "scan_days_top5_if_added",
                "recommendation",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
                "full_delta_sharpe",
                "oos_delta_sharpe",
                "full_delta_maxdd_pct",
                "oos_delta_maxdd_pct",
            ]
        ]
        if not hist.empty
        else pd.DataFrame(columns=["ticker"]),
        on="ticker",
        how="outer",
    )

    for name in (
        "source_file",
        "recent_score",
        "recent_r63",
        "recent_r126",
        "recent_r252",
        "scan_algo_fit",
        "scan_algo_compat_score",
        "scan_latest_rank_if_added",
        "scan_days_top15_if_added",
        "scan_days_top5_if_added",
        "recommendation",
        "full_delta_roi_pct",
        "oos_delta_roi_pct",
        "full_delta_sharpe",
        "oos_delta_sharpe",
        "full_delta_maxdd_pct",
        "oos_delta_maxdd_pct",
    ):
        left = f"{name}_x"
        right = f"{name}_y"
        if left in merged.columns or right in merged.columns:
            merged[name] = merged.get(left)
            if name not in merged.columns or merged[name] is None:
                merged[name] = merged.get(right)
            else:
                merged[name] = merged[name].where(merged[name].notna(), merged.get(right))

    for col in ("is_active", "is_reserve", "is_hard_exclusion"):
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = merged[col].fillna(0).astype(int)

    for col in ("scan_algo_fit", "dynamic_status", "recommendation"):
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].fillna("")

    for col in (
        "recent_score",
        "scan_algo_compat_score",
        "full_delta_roi_pct",
        "oos_delta_roi_pct",
        "oos_delta_sharpe",
    ):
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    has_hist_signal = merged["source_file"].notna() | merged["recommendation"].notna()
    allow_selected_adds = merged["ticker"].isin(selected_adds)
    merged = merged[
        ((~merged["ticker"].isin(base_prices.close.columns)) | allow_selected_adds | has_hist_signal)
        & ((merged["is_active"] == 0) | allow_selected_adds | has_hist_signal)
        & (merged["is_reserve"] == 0)
        & (merged["is_hard_exclusion"] == 0)
    ].copy()

    verdict_priority = {"add": 3, "watch": 2, "reject": 1, "": 0}
    fit_priority = {"high": 3, "medium": 2, "weak": 1, "low": 0, "": 0}
    status_priority = {"watch": 3, "review": 2, "discovered": 1, "reject": 0, "": 0}
    merged["verdict_priority"] = merged["recommendation"].map(verdict_priority).fillna(0)
    merged["fit_priority"] = merged["scan_algo_fit"].map(fit_priority).fillna(0)
    merged["status_priority"] = merged["dynamic_status"].map(status_priority).fillna(0)

    merged = merged.sort_values(
        [
            "verdict_priority",
            "oos_delta_roi_pct",
            "oos_delta_sharpe",
            "full_delta_roi_pct",
            "fit_priority",
            "scan_algo_compat_score",
            "recent_score",
            "status_priority",
        ],
        ascending=[False, False, False, False, False, False, False, False],
    )
    return merged


def baseline_diagnostics(prices, cfg: dict, trades_full: pd.DataFrame) -> pd.DataFrame:
    feats = dud.compute_universe_features(prices.close, cfg)
    pnl = dud.compute_realized_pnl(trades_full)
    df = feats.merge(pnl, on="ticker", how="left").fillna(
        {"realized_pnl_eur": 0.0, "buy_count": 0, "sell_count": 0}
    )

    close = prices.close
    sma220 = close.rolling(int(cfg["sma_win"]), min_periods=int(cfg["sma_win"])).mean()
    r63 = close / close.shift(63) - 1.0
    r126 = close / close.shift(126) - 1.0
    r252 = close / close.shift(252) - 1.0
    score = cfg["w_r63"] * r63 + cfg["w_r126"] * r126 + cfg["w_r252"] * r252
    rank = score.rank(axis=1, ascending=False, method="min")
    elig = score.notna() & (close > sma220)
    recent126 = close.index[-126:]
    recent252 = close.index[-252:]
    last_date = close.index.max()

    recent_rows = []
    for ticker in close.columns:
        recent_rows.append(
            {
                "ticker": ticker,
                "days_top15_126": int((elig.loc[recent126, ticker] & (rank.loc[recent126, ticker] <= 15)).sum()) if len(recent126) else 0,
                "days_top5_126": int((elig.loc[recent126, ticker] & (rank.loc[recent126, ticker] <= 5)).sum()) if len(recent126) else 0,
                "days_top15_252": int((elig.loc[recent252, ticker] & (rank.loc[recent252, ticker] <= 15)).sum()) if len(recent252) else 0,
                "above220_now": bool(close.at[last_date, ticker] > sma220.at[last_date, ticker]) if pd.notna(sma220.at[last_date, ticker]) else False,
            }
        )
    recent_df = pd.DataFrame(recent_rows)
    df = df.merge(recent_df, on="ticker", how="left")

    df["retain_score"] = (
        0.30 * safe_rank_pct(df["latest_score"], ascending=True)
        + 0.25 * safe_rank_pct(df["days_top15_trend"], ascending=True)
        + 0.15 * safe_rank_pct(df["days_top5_trend"], ascending=True)
        + 0.20 * safe_rank_pct(df["realized_pnl_eur"], ascending=True)
        + 0.10 * safe_rank_pct(df["buy_count"], ascending=True)
    )
    rank_bad = df["latest_rank"].fillna(999.0).clip(upper=200.0) / 200.0
    recent_bad = 1.0 - (df["days_top15_126"].fillna(0.0) / 126.0).clip(lower=0.0, upper=1.0)
    pnl_bad = 1.0 - safe_rank_pct(df["realized_pnl_eur"], ascending=True)
    r63_bad = (-(df["latest_r63"].fillna(-1.0))).clip(lower=0.0, upper=1.0)
    all_bad = 1.0 - (df["days_top15_trend"].fillna(0.0) / max(float(df["days_top15_trend"].max() or 1.0), 1.0)).clip(lower=0.0, upper=1.0)
    df["dead_score"] = (
        0.30 * rank_bad
        + 0.30 * recent_bad
        + 0.20 * pnl_bad
        + 0.10 * r63_bad
        + 0.10 * all_bad
        + 0.10 * (~df["above220_now"].fillna(False)).astype(float)
    )
    return df


def select_demotion_shortlist(diag: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    harmful_traded = diag.loc[diag["buy_count"] > 0].sort_values(
        ["dead_score", "realized_pnl_eur", "latest_score"],
        ascending=[False, True, True],
    ).head(limit * 2)

    manual_order = ["MELI", "UEC", "PAAS", "AXON", "NET", "WPM", "ZS", "RACE"]
    selected_dems = read_ticker_set(SELECTED_DEMS_PATH)
    manual_order.extend([ticker for ticker in selected_dems if ticker not in manual_order])
    manual = diag.loc[diag["ticker"].isin(manual_order)].copy()
    manual["manual_order"] = manual["ticker"].map({t: i for i, t in enumerate(manual_order)})
    manual = manual.sort_values("manual_order")

    shortlist = pd.concat([manual, harmful_traded], ignore_index=True)
    shortlist = (
        shortlist.sort_values(
            ["manual_order", "dead_score", "realized_pnl_eur", "latest_score"],
            ascending=[True, False, True, True],
            na_position="last",
        )
        .drop_duplicates("ticker")
        .head(limit)
        .copy()
    )
    return shortlist


def swap_prices(engine, base_prices, remove_ticker: str, add_ticker: str, candidate_map: dict):
    swapped = dud.merge_candidates(base_prices, [add_ticker], candidate_map)
    return engine.Prices(
        open=swapped.open.drop(columns=[remove_ticker], errors="ignore"),
        close=swapped.close.drop(columns=[remove_ticker], errors="ignore"),
    )


def recommend_swap(row: pd.Series) -> str:
    if (
        row["full_delta_roi_pct"] > 0
        and row["oos_delta_roi_pct"] > 0
        and row["full_delta_sharpe"] >= 0
        and row["oos_delta_sharpe"] >= -0.01
        and row["full_delta_maxdd_pct"] > -1.0
        and row["oos_delta_maxdd_pct"] > -1.0
    ):
        return "promote"
    if (
        row["oos_delta_roi_pct"] > 0
        and row["oos_delta_sharpe"] >= -0.01
        and row["oos_delta_maxdd_pct"] > -1.0
    ):
        return "watch"
    return "reject"


def yearly_windows(end_date: str) -> list[tuple[str, str, str]]:
    windows = []
    for year in range(2017, 2026):
        windows.append((str(year), f"{year}-01-01", f"{year}-12-31"))
    windows.append(("2026_ytd", "2026-01-02", end_date))
    return windows


def main() -> None:
    t0 = time.time()
    dud.setup_yf_cache(ROOT / ".yf_cache")
    engine, cfg_doc, cfg, pp, base_prices = dud.load_setup()
    print("[swap-study] setup loaded")

    full_start = cfg_doc["metrics"]["full_period"]["start"]
    full_end = cfg_doc["metrics"]["full_period"]["end"]
    oos_start = cfg_doc["metrics"]["oos_2022_2026"]["start"]
    oos_end = cfg_doc["metrics"]["oos_2022_2026"]["end"]

    _, trades_full, baseline_full = dud.run_metrics(engine, base_prices, cfg, pp, full_start, full_end)
    _, _, baseline_oos = dud.run_metrics(engine, base_prices, cfg, pp, oos_start, oos_end)
    print("[swap-study] baseline full/oos computed")

    diag = baseline_diagnostics(base_prices, cfg, trades_full)
    demotions = select_demotion_shortlist(diag, limit=6)
    demotions.to_csv(DEMOTION_EXPORT, index=False)
    print(f"[swap-study] demotion shortlist ready count={len(demotions)}")

    candidates = aggregate_candidate_sources(base_prices)
    candidates = candidates.loc[
        (candidates["recommendation"].isin(["add", "watch"]))
        | (candidates["dynamic_status"].isin(["watch", "review"]))
        | (candidates["scan_algo_fit"].isin(["high", "medium"]))
    ].copy()
    candidates = candidates.loc[
        ~(
            (candidates["recommendation"] == "reject")
            & (candidates["oos_delta_roi_pct"].fillna(0) <= 0)
            & (candidates["full_delta_roi_pct"].fillna(0) <= 0)
        )
    ].copy()
    candidates = candidates.head(4).copy()
    candidates.to_csv(CANDIDATE_EXPORT, index=False)
    print(f"[swap-study] candidate shortlist ready count={len(candidates)}")

    download_start = str(pd.to_datetime(base_prices.close.index.min()).date())
    download_end = str((pd.to_datetime(base_prices.close.index.max()) + pd.Timedelta(days=1)).date())
    candidate_map = dud.download_candidate_data(candidates["ticker"].tolist(), download_start, download_end)
    min_bars_required = int(cfg["min_bars_required"])
    print(f"[swap-study] candidate data downloaded symbols={len(candidate_map)}")

    swap_rows: list[dict] = []
    total_pairs = int(len(candidates) * len(demotions))
    pair_idx = 0
    for cand in candidates.itertuples(index=False):
        ticker = str(cand.ticker)
        cand_data = candidate_map.get(ticker)
        if cand_data is None:
            print(f"[swap-study] skip candidate={ticker} reason=no_data")
            continue
        cand_bars = int(cand_data.close.notna().sum())
        if cand_bars < min_bars_required:
            print(f"[swap-study] skip candidate={ticker} reason=bars<{min_bars_required}")
            continue

        for rem in demotions.itertuples(index=False):
            pair_idx += 1
            removed = str(rem.ticker)
            print(f"[swap-study] evaluate pair={pair_idx}/{total_pairs} add={ticker} remove={removed}")
            variant_prices = swap_prices(engine, base_prices, removed, ticker, candidate_map)
            _, _, full = dud.run_metrics(engine, variant_prices, cfg, pp, full_start, full_end)
            _, _, oos = dud.run_metrics(engine, variant_prices, cfg, pp, oos_start, oos_end)

            row = dud.row_from_run(
                f"{ticker}_for_{removed}",
                full,
                oos,
                {
                    "candidate": ticker,
                    "removed": removed,
                    "candidate_recent_score": getattr(cand, "recent_score", np.nan),
                    "candidate_scan_fit": getattr(cand, "scan_algo_fit", ""),
                    "candidate_scan_compat": getattr(cand, "scan_algo_compat_score", np.nan),
                    "candidate_hist_reco": getattr(cand, "recommendation", ""),
                    "removed_latest_rank": float(rem.latest_rank) if pd.notna(rem.latest_rank) else np.nan,
                    "removed_latest_score": float(rem.latest_score) if pd.notna(rem.latest_score) else np.nan,
                    "removed_days_top15": int(rem.days_top15_trend),
                    "removed_realized_pnl_eur": float(rem.realized_pnl_eur),
                    "removed_buy_count": int(rem.buy_count),
                    "full_delta_roi_pct": float(full["ROI_%"] - baseline_full["ROI_%"]),
                    "full_delta_sharpe": float(full["Sharpe"] - baseline_full["Sharpe"]),
                    "full_delta_maxdd_pct": float(full["MaxDD_%"] - baseline_full["MaxDD_%"]),
                    "oos_delta_roi_pct": float(oos["ROI_%"] - baseline_oos["ROI_%"]),
                    "oos_delta_sharpe": float(oos["Sharpe"] - baseline_oos["Sharpe"]),
                    "oos_delta_maxdd_pct": float(oos["MaxDD_%"] - baseline_oos["MaxDD_%"]),
                },
            )
            row["recommendation"] = recommend_swap(pd.Series(row))
            swap_rows.append(row)

    swap_df = pd.DataFrame(swap_rows)
    if swap_df.empty:
        REPORT_PATH.write_text("# Dynamic Universe Swap Study\n\nNo swap evaluated.\n", encoding="utf-8")
        print("No swaps evaluated.")
        return

    reco_priority = {"promote": 2, "watch": 1, "reject": 0}
    swap_df["reco_priority"] = swap_df["recommendation"].map(reco_priority).fillna(-1)
    swap_df = swap_df.sort_values(
        ["reco_priority", "oos_delta_roi_pct", "oos_delta_sharpe", "full_delta_roi_pct", "full_delta_sharpe"],
        ascending=[False, False, False, False, False],
    )
    swap_df.to_csv(SWAP_EXPORT, index=False)

    top_walk = swap_df.loc[swap_df["recommendation"].isin(["promote", "watch"])].head(6).copy()
    print(f"[swap-study] walkforward shortlist count={len(top_walk)}")
    walk_rows: list[dict] = []
    base_window_metrics = {
        label: dud.run_metrics(engine, base_prices, cfg, pp, win_start, win_end)[2]
        for label, win_start, win_end in yearly_windows(full_end)
    }
    total_walk = int(len(top_walk) * len(base_window_metrics))
    walk_idx = 0
    for row in top_walk.itertuples(index=False):
        variant_prices = swap_prices(engine, base_prices, str(row.removed), str(row.candidate), candidate_map)
        for label, win_start, win_end in yearly_windows(full_end):
            walk_idx += 1
            print(f"[swap-study] walkforward {walk_idx}/{total_walk} swap={row.name} window={label}")
            base_out = base_window_metrics[label]
            _, _, var_out = dud.run_metrics(engine, variant_prices, cfg, pp, win_start, win_end)
            walk_rows.append(
                {
                    "swap": row.name,
                    "candidate": row.candidate,
                    "removed": row.removed,
                    "window": label,
                    "delta_roi_pct": float(var_out["ROI_%"] - base_out["ROI_%"]),
                    "delta_sharpe": float(var_out["Sharpe"] - base_out["Sharpe"]),
                    "delta_maxdd_pct": float(var_out["MaxDD_%"] - base_out["MaxDD_%"]),
                }
            )

    walk_df = pd.DataFrame(walk_rows)
    walk_df.to_csv(WALK_EXPORT, index=False)

    summary_rows = []
    if not walk_df.empty:
        for swap_name, group in walk_df.groupby("swap"):
            yearly = group[group["window"] != "2026_ytd"]
            ytd = group[group["window"] == "2026_ytd"]
            summary_rows.append(
                {
                    "swap": swap_name,
                    "mean_delta_roi_2017_2025": float(yearly["delta_roi_pct"].mean()),
                    "mean_delta_sharpe_2017_2025": float(yearly["delta_sharpe"].mean()),
                    "mean_delta_maxdd_2017_2025": float(yearly["delta_maxdd_pct"].mean()),
                    "roi_wins_2017_2025": int((yearly["delta_roi_pct"] > 0).sum()),
                    "sharpe_wins_2017_2025": int((yearly["delta_sharpe"] > 0).sum()),
                    "delta_roi_2026_ytd": float(ytd["delta_roi_pct"].iloc[0]) if not ytd.empty else np.nan,
                    "delta_sharpe_2026_ytd": float(ytd["delta_sharpe"].iloc[0]) if not ytd.empty else np.nan,
                    "delta_maxdd_2026_ytd": float(ytd["delta_maxdd_pct"].iloc[0]) if not ytd.empty else np.nan,
                }
            )
    walk_summary = pd.DataFrame(summary_rows)
    if not walk_summary.empty:
        walk_summary = walk_summary.sort_values(
            ["mean_delta_sharpe_2017_2025", "mean_delta_roi_2017_2025", "delta_roi_2026_ytd"],
            ascending=[False, False, False],
        )
    else:
        walk_summary = pd.DataFrame(
            columns=[
                "swap",
                "mean_delta_roi_2017_2025",
                "mean_delta_sharpe_2017_2025",
                "mean_delta_maxdd_2017_2025",
                "roi_wins_2017_2025",
                "sharpe_wins_2017_2025",
                "delta_roi_2026_ytd",
                "delta_sharpe_2026_ytd",
                "delta_maxdd_2026_ytd",
            ]
        )
    walk_summary.to_csv(WALK_SUMMARY_EXPORT, index=False)

    lines = [
        "# Dynamic Universe Swap Study (184)",
        "",
        "## Baseline",
        "",
        f"- full ROI: `{baseline_full['ROI_%']:.2f}%`",
        f"- full Sharpe: `{baseline_full['Sharpe']:.5f}`",
        f"- full MaxDD: `{baseline_full['MaxDD_%']:.2f}%`",
        f"- OOS ROI: `{baseline_oos['ROI_%']:.2f}%`",
        f"- OOS Sharpe: `{baseline_oos['Sharpe']:.5f}`",
        f"- OOS MaxDD: `{baseline_oos['MaxDD_%']:.2f}%`",
        "",
        "## Candidate shortlist",
        "",
        candidates[
            [
                "ticker",
                "dynamic_status",
                "recommendation",
                "scan_algo_fit",
                "scan_algo_compat_score",
                "recent_score",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
            ]
        ].to_string(index=False),
        "",
        "## Demotion shortlist",
        "",
        demotions[
            [
                "ticker",
                "retain_score",
                "latest_rank",
                "latest_score",
                "days_top15_trend",
                "realized_pnl_eur",
                "buy_count",
            ]
        ].to_string(index=False),
        "",
        "## Top single swaps",
        "",
        swap_df[
            [
                "candidate",
                "removed",
                "recommendation",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
                "full_delta_sharpe",
                "oos_delta_sharpe",
                "full_delta_maxdd_pct",
                "oos_delta_maxdd_pct",
            ]
        ]
        .head(20)
        .to_string(index=False),
        "",
        "## Walk-forward summary",
        "",
        walk_summary.to_string(index=False) if not walk_summary.empty else "(none)",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {CANDIDATE_EXPORT}")
    print(f"Saved: {DEMOTION_EXPORT}")
    print(f"Saved: {SWAP_EXPORT}")
    print(f"Saved: {WALK_EXPORT}")
    print(f"Saved: {WALK_SUMMARY_EXPORT}")
    print(f"Saved: {REPORT_PATH}")
    print(f"[swap-study] completed elapsed_sec={time.time() - t0:.1f}")
    print(swap_df.head(10).to_string(index=False))
    if not walk_summary.empty:
        print("\nWalk-forward summary:")
        print(walk_summary.to_string(index=False))


if __name__ == "__main__":
    main()
