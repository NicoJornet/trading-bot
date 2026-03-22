from __future__ import annotations

from pathlib import Path

import pandas as pd

import dynamic_universe_discovery as dud
import dynamic_universe_swap_study as swap_study


ROOT = Path(__file__).resolve().parent
EXPORTS_DIR = ROOT / "research" / "exports"
REPORTS_DIR = ROOT / "research" / "reports"
ACTIVE_PATH = ROOT / "data" / "extracts" / "apex_tickers_active.csv"

EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REMOVE_EXPORT = EXPORTS_DIR / "dynamic_universe_standalone_removals.csv"
WALK_EXPORT = EXPORTS_DIR / "dynamic_universe_standalone_removal_walkforward.csv"
WALK_SUMMARY_EXPORT = EXPORTS_DIR / "dynamic_universe_standalone_removal_walkforward_summary.csv"
REPORT_PATH = REPORTS_DIR / "DYNAMIC_UNIVERSE_STANDALONE_REMOVAL_STUDY_184.md"


def yearly_windows(full_end: str) -> list[tuple[str, str, str]]:
    return [(str(year), f"{year}-01-02", f"{year}-12-31") for year in range(2017, 2026)] + [("2026_ytd", "2026-01-02", full_end)]


def classify_removal(row: pd.Series) -> str:
    if (
        row["full_delta_roi_pct"] > 0
        and row["oos_delta_roi_pct"] >= 0
        and row["full_delta_sharpe"] >= 0
        and row["oos_delta_sharpe"] >= -0.01
        and row["full_delta_maxdd_pct"] > -1.0
        and row["oos_delta_maxdd_pct"] > -0.5
        and row["mean_delta_roi_2017_2025"] >= 0
        and row["mean_delta_sharpe_2017_2025"] >= 0
        and row["roi_wins_2017_2025"] >= 2
        and row["sharpe_wins_2017_2025"] >= 2
    ):
        return "approved_remove"
    if (
        row["full_delta_roi_pct"] > 0
        or row["oos_delta_roi_pct"] > 0
        or row["mean_delta_roi_2017_2025"] > 0
    ):
        return "watch_remove"
    return "reject_remove"


def removal_score(row: pd.Series) -> float:
    return (
        1000.0 * float(row["mean_delta_sharpe_2017_2025"])
        + 10.0 * float(row["mean_delta_roi_2017_2025"])
        + 0.05 * float(row["oos_delta_roi_pct"])
        + 20.0 * float(row["oos_delta_sharpe"])
        + 5.0 * float(row["mean_delta_maxdd_2017_2025"])
        + 10.0 * float(row["dead_score"])
    )


def main() -> None:
    engine, _, cfg, pp, prices = dud.load_setup()
    active = pd.read_csv(ACTIVE_PATH)["ticker"].dropna().astype(str).tolist()
    active_cols = [col for col in prices.close.columns if col in active]
    prices = engine.Prices(open=prices.open[active_cols], close=prices.close[active_cols])

    full_start, full_end = "2015-01-02", "2026-03-21"
    oos_start = "2022-01-03"

    _, trades_full, baseline_full = dud.run_metrics(engine, prices, cfg, pp, full_start, full_end)
    _, _, baseline_oos = dud.run_metrics(engine, prices, cfg, pp, oos_start, full_end)

    diag = swap_study.baseline_diagnostics(prices, cfg, trades_full)
    demotions = swap_study.select_demotion_shortlist(diag, limit=8).copy()

    rows: list[dict] = []
    for rem in demotions.itertuples(index=False):
        ticker = str(rem.ticker)
        variant = engine.Prices(
            open=prices.open.drop(columns=[ticker], errors="ignore"),
            close=prices.close.drop(columns=[ticker], errors="ignore"),
        )
        _, _, full = dud.run_metrics(engine, variant, cfg, pp, full_start, full_end)
        _, _, oos = dud.run_metrics(engine, variant, cfg, pp, oos_start, full_end)
        rows.append(
            dud.row_from_run(
                ticker,
                full,
                oos,
                {
                    "ticker": ticker,
                    "dead_score": float(rem.dead_score),
                    "retain_score": float(rem.retain_score),
                    "latest_rank": float(rem.latest_rank) if pd.notna(rem.latest_rank) else float("nan"),
                    "latest_score": float(rem.latest_score) if pd.notna(rem.latest_score) else float("nan"),
                    "days_top15_trend": int(rem.days_top15_trend),
                    "realized_pnl_eur": float(rem.realized_pnl_eur),
                    "buy_count": int(rem.buy_count),
                    "full_delta_roi_pct": float(full["ROI_%"] - baseline_full["ROI_%"]),
                    "full_delta_sharpe": float(full["Sharpe"] - baseline_full["Sharpe"]),
                    "full_delta_maxdd_pct": float(full["MaxDD_%"] - baseline_full["MaxDD_%"]),
                    "oos_delta_roi_pct": float(oos["ROI_%"] - baseline_oos["ROI_%"]),
                    "oos_delta_sharpe": float(oos["Sharpe"] - baseline_oos["Sharpe"]),
                    "oos_delta_maxdd_pct": float(oos["MaxDD_%"] - baseline_oos["MaxDD_%"]),
                },
            )
        )

    remove_df = pd.DataFrame(rows)
    if remove_df.empty:
        REPORT_PATH.write_text("# Standalone Removal Study\n\nNo removal evaluated.\n", encoding="utf-8")
        return

    walk_rows: list[dict] = []
    top_walk = remove_df.sort_values(
        ["oos_delta_roi_pct", "oos_delta_sharpe", "full_delta_roi_pct"],
        ascending=[False, False, False],
    ).head(6)
    for row in top_walk.itertuples(index=False):
        variant = engine.Prices(
            open=prices.open.drop(columns=[str(row.ticker)], errors="ignore"),
            close=prices.close.drop(columns=[str(row.ticker)], errors="ignore"),
        )
        for label, win_start, win_end in yearly_windows(full_end):
            _, _, base_out = dud.run_metrics(engine, prices, cfg, pp, win_start, win_end)
            _, _, var_out = dud.run_metrics(engine, variant, cfg, pp, win_start, win_end)
            walk_rows.append(
                {
                    "ticker": str(row.ticker),
                    "window": label,
                    "delta_roi_pct": float(var_out["ROI_%"] - base_out["ROI_%"]),
                    "delta_sharpe": float(var_out["Sharpe"] - base_out["Sharpe"]),
                    "delta_maxdd_pct": float(var_out["MaxDD_%"] - base_out["MaxDD_%"]),
                }
            )

    walk_df = pd.DataFrame(walk_rows)
    walk_df.to_csv(WALK_EXPORT, index=False)

    summary_rows: list[dict] = []
    if not walk_df.empty:
        for ticker, group in walk_df.groupby("ticker"):
            yearly = group[group["window"] != "2026_ytd"]
            ytd = group[group["window"] == "2026_ytd"]
            summary_rows.append(
                {
                    "ticker": ticker,
                    "mean_delta_roi_2017_2025": float(yearly["delta_roi_pct"].mean()),
                    "mean_delta_sharpe_2017_2025": float(yearly["delta_sharpe"].mean()),
                    "mean_delta_maxdd_2017_2025": float(yearly["delta_maxdd_pct"].mean()),
                    "roi_wins_2017_2025": int((yearly["delta_roi_pct"] > 0).sum()),
                    "sharpe_wins_2017_2025": int((yearly["delta_sharpe"] > 0).sum()),
                    "maxdd_wins_2017_2025": int((yearly["delta_maxdd_pct"] >= 0).sum()),
                    "delta_roi_2026_ytd": float(ytd["delta_roi_pct"].mean()) if not ytd.empty else 0.0,
                    "delta_sharpe_2026_ytd": float(ytd["delta_sharpe"].mean()) if not ytd.empty else 0.0,
                    "delta_maxdd_2026_ytd": float(ytd["delta_maxdd_pct"].mean()) if not ytd.empty else 0.0,
                }
            )

    walk_summary = pd.DataFrame(summary_rows)
    if not walk_summary.empty:
        remove_df = remove_df.merge(walk_summary, on="ticker", how="left")
    for col in (
        "mean_delta_roi_2017_2025",
        "mean_delta_sharpe_2017_2025",
        "mean_delta_maxdd_2017_2025",
        "roi_wins_2017_2025",
        "sharpe_wins_2017_2025",
        "maxdd_wins_2017_2025",
        "delta_roi_2026_ytd",
        "delta_sharpe_2026_ytd",
        "delta_maxdd_2026_ytd",
    ):
        if col not in remove_df.columns:
            remove_df[col] = 0.0
        remove_df[col] = pd.to_numeric(remove_df[col], errors="coerce").fillna(0.0)

    remove_df["selection_status"] = remove_df.apply(classify_removal, axis=1)
    remove_df["selection_score"] = remove_df.apply(removal_score, axis=1)
    remove_df = remove_df.sort_values(
        ["selection_status", "selection_score", "oos_delta_roi_pct", "full_delta_roi_pct"],
        ascending=[False, False, False, False],
    )

    remove_df.to_csv(REMOVE_EXPORT, index=False)
    walk_summary.to_csv(WALK_SUMMARY_EXPORT, index=False)

    lines = [
        "# Dynamic Universe Standalone Removal Study",
        "",
        "## Demotion shortlist",
        "",
        demotions[
            [
                "ticker",
                "retain_score",
                "dead_score",
                "latest_rank",
                "latest_score",
                "days_top15_trend",
                "realized_pnl_eur",
                "buy_count",
            ]
        ].to_string(index=False),
        "",
        "## Standalone removals",
        "",
        remove_df[
            [
                "ticker",
                "selection_status",
                "selection_score",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
                "full_delta_sharpe",
                "oos_delta_sharpe",
                "full_delta_maxdd_pct",
                "oos_delta_maxdd_pct",
                "mean_delta_roi_2017_2025",
                "mean_delta_sharpe_2017_2025",
            ]
        ].to_string(index=False),
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {REMOVE_EXPORT}")
    print(f"Saved: {WALK_EXPORT}")
    print(f"Saved: {WALK_SUMMARY_EXPORT}")
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
