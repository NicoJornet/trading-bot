from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPORTS_DIR = ROOT / "research" / "exports"
REPORTS_DIR = ROOT / "research" / "reports"
CURRENT_PATH = ROOT / "data" / "dynamic_universe" / "dynamic_universe_current.csv"
BEFORE_PATH = EXPORTS_DIR / "dynamic_universe_current_before_quality_compounder_20260406.csv"
MISSING_UNIVERSE_PATH = EXPORTS_DIR / "scan_algo_missing_universe_20260406.csv"
MISSING_SELECTION_PATH = EXPORTS_DIR / "scan_algo_missing_selection_20260406.csv"
IMPACT_PATH = EXPORTS_DIR / "scan_quality_compounder_repair_20260406.csv"
REPORT_PATH = REPORTS_DIR / "SCAN_QUALITY_COMPOUNDER_REPAIR_20260406.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def safe_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out[cols]


def stage_rank(value: object) -> int:
    return {
        "approved_live": 5,
        "probation_live": 4,
        "targeted_integration": 3,
        "watch_queue": 2,
        "review_queue": 1,
        "reject_queue": 0,
        "blocked_broker": -1,
    }.get(str(value or ""), -2)


def status_rank(value: object) -> int:
    return {
        "approved": 5,
        "prime_watch": 4,
        "watch": 3,
        "review": 2,
        "discovered": 1,
        "reject": 0,
    }.get(str(value or ""), -1)


def fmt_table(df: pd.DataFrame, rows: int = 20) -> str:
    if df.empty:
        return "(none)"
    return df.head(rows).to_string(index=False)


def main() -> None:
    before = read_csv(BEFORE_PATH)
    after = read_csv(CURRENT_PATH)
    missing_universe = read_csv(MISSING_UNIVERSE_PATH)
    missing_selection = read_csv(MISSING_SELECTION_PATH)

    if before.empty or after.empty:
        raise SystemExit("Missing before/after dynamic universe snapshots.")

    universe_targets = missing_universe["ticker"].dropna().astype(str).tolist() if "ticker" in missing_universe.columns else []
    selection_targets = missing_selection["ticker"].dropna().astype(str).tolist() if "ticker" in missing_selection.columns else []
    all_targets = sorted(set(universe_targets) | set(selection_targets))

    compare_cols_before = [
        "ticker",
        "promotion_stage",
        "dynamic_status",
        "scan_algo_fit",
        "scan_candidate_track",
        "recent_score",
        "dynamic_conviction_score",
        "promotion_score",
        "entry_timing_zone",
    ]
    compare_cols_after = compare_cols_before + [
        "scan_quality_compounder_fit",
        "scan_quality_compounder_score",
    ]

    merged = safe_cols(before, compare_cols_before).merge(
        safe_cols(after, compare_cols_after),
        on="ticker",
        how="outer",
        suffixes=("_before", "_after"),
    )
    merged = merged.loc[merged["ticker"].isin(all_targets)].copy()
    merged["stage_rank_before"] = merged["promotion_stage_before"].map(stage_rank)
    merged["stage_rank_after"] = merged["promotion_stage_after"].map(stage_rank)
    merged["status_rank_before"] = merged["dynamic_status_before"].map(status_rank)
    merged["status_rank_after"] = merged["dynamic_status_after"].map(status_rank)
    merged["delta_stage_rank"] = merged["stage_rank_after"] - merged["stage_rank_before"]
    merged["delta_status_rank"] = merged["status_rank_after"] - merged["status_rank_before"]
    merged["delta_dynamic_conviction_score"] = pd.to_numeric(
        merged["dynamic_conviction_score_after"], errors="coerce"
    ).fillna(0.0) - pd.to_numeric(merged["dynamic_conviction_score_before"], errors="coerce").fillna(0.0)
    merged["delta_promotion_score"] = pd.to_numeric(
        merged["promotion_score_after"], errors="coerce"
    ).fillna(0.0) - pd.to_numeric(merged["promotion_score_before"], errors="coerce").fillna(0.0)

    merged["bucket"] = merged["ticker"].map(
        {
            **{ticker: "missing_universe" for ticker in universe_targets},
            **{ticker: "missing_selection" for ticker in selection_targets},
        }
    )

    repaired = merged.loc[
        (merged["delta_stage_rank"] > 0)
        | (merged["delta_status_rank"] > 0)
        | (merged["scan_quality_compounder_fit"].fillna("").isin(["high", "medium"]))
        | (pd.to_numeric(merged["delta_promotion_score"], errors="coerce").abs() >= 5.0)
    ].copy()
    repaired = repaired.sort_values(
        ["delta_stage_rank", "delta_status_rank", "delta_promotion_score", "scan_quality_compounder_score"],
        ascending=[False, False, False, False],
    )
    repaired.to_csv(IMPACT_PATH, index=False)

    repaired_universe = repaired.loc[repaired["bucket"] == "missing_universe"].copy()
    unresolved_universe = merged.loc[
        (merged["bucket"] == "missing_universe")
        & ~merged["ticker"].isin(repaired_universe["ticker"])
    ].copy()
    unresolved_universe = unresolved_universe.sort_values(
        ["promotion_score_after", "dynamic_conviction_score_after"],
        ascending=[False, False],
    )

    unresolved_selection = merged.loc[merged["bucket"] == "missing_selection"].copy()
    unresolved_selection = unresolved_selection.sort_values(
        ["promotion_score_after", "dynamic_conviction_score_after"],
        ascending=[False, False],
    )

    lines = [
        "# Scan Quality Compounder Repair",
        "",
        "This report covers the structural repair applied after the retrospective missing-winners audit.",
        "",
        "## What Was Fixed",
        "",
        "- discovery now evaluates additions against the true baseline universe (`active + staged additions + SPY`) instead of treating the whole OHLCV file as already admitted;",
        "- discovery now computes a structural `quality_compounder` lane for persistent, high-quality trend names;",
        "- governance now gives a moderate promotion bonus to that lane instead of patching tickers one by one.",
        "",
        "## Repaired Missing-Universe Names",
        "",
        fmt_table(
            safe_cols(
                repaired_universe,
                [
                    "ticker",
                    "promotion_stage_before",
                    "promotion_stage_after",
                    "dynamic_status_before",
                    "dynamic_status_after",
                    "scan_quality_compounder_fit",
                    "scan_quality_compounder_score",
                    "scan_candidate_track_after",
                    "delta_promotion_score",
                ],
            ),
            rows=25,
        ),
        "",
        "## Remaining Missing-Universe Names",
        "",
        fmt_table(
            safe_cols(
                unresolved_universe,
                [
                    "ticker",
                    "promotion_stage_after",
                    "dynamic_status_after",
                    "scan_algo_fit_after",
                    "scan_quality_compounder_fit",
                    "scan_quality_compounder_score",
                    "scan_candidate_track_after",
                    "promotion_score_after",
                ],
            ),
            rows=20,
        ),
        "",
        "## Still Core Algo / Ranking Misses",
        "",
        "These are names that remain mostly in the `missing_selection` bucket. They point to a ranking / portfolio archetype gap more than a scan-universe gap.",
        "",
        fmt_table(
            safe_cols(
                unresolved_selection,
                [
                    "ticker",
                    "promotion_stage_after",
                    "dynamic_status_after",
                    "scan_algo_fit_after",
                    "scan_quality_compounder_fit",
                    "promotion_score_after",
                ],
            ),
            rows=20,
        ),
        "",
        "## Conclusion",
        "",
        "- the repair clearly improves the scan-side capture of missing-universe families such as Asia tech supply-chain, optical/networking, and industrial compounders;",
        "- the biggest remaining misses (`AVGO`, `KLAC`, `AMAT`, `ASML`, `PWR`, `HWM`, `CDNS`, `NOW`, `PANW`, `META`) are still mostly core ranking / portfolio misses, not discovery misses;",
        "- so the scan is now materially less blind than before, but the next structural step for the remaining misses belongs to the core ranking lane, not to more discovery patches.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"impact: {IMPACT_PATH}")
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
