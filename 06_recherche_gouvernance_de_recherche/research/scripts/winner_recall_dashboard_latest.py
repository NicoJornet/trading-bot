from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = ROOT / "data" / "dynamic_universe"
REPORTS_DIR = ROOT / "research" / "reports"
EXPORTS_DIR = ROOT / "research" / "exports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

MISSING_UNIVERSE_PATH = EXPORTS_DIR / "scan_algo_missing_universe_20260406.csv"
MISSING_SELECTION_PATH = EXPORTS_DIR / "scan_algo_missing_selection_20260406.csv"
FORWARD_MONITOR_PATH = DATA_DIR / "dynamic_universe_quality_compounder_forward_monitor.csv"
CURRENT_DB_PATH = DATA_DIR / "dynamic_universe_current.csv"
SUMMARY_EXPORT_PATH = EXPORTS_DIR / "winner_recall_summary.csv"
FAMILY_EXPORT_PATH = EXPORTS_DIR / "winner_recall_family_summary.csv"
STAGE_EXPORT_PATH = EXPORTS_DIR / "winner_recall_stage_breakdown.csv"
UNRESOLVED_EXPORT_PATH = EXPORTS_DIR / "winner_recall_top_unresolved.csv"
PROGRESSED_EXPORT_PATH = EXPORTS_DIR / "winner_recall_top_progressed.csv"
REPORT_PATH = REPORTS_DIR / "WINNER_RECALL_DASHBOARD_LATEST.md"
ROOT_REPORT_PATH = ROOT / "WINNER_RECALL_DASHBOARD_LATEST.md"

ACTIVE_STAGES = {"approved_live", "probation_live", "targeted_integration"}
STAGE_ORDER = {
    "not_seen": 0,
    "review_queue": 1,
    "watch_queue": 2,
    "targeted_integration": 3,
    "probation_live": 4,
    "approved_live": 5,
}


def safe_num(series: pd.Series | object, default: float = 0.0) -> pd.Series | float:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    try:
        value = float(series)
    except (TypeError, ValueError):
        return default
    return value if pd.notna(value) else default


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.fillna("").itertuples(index=False, name=None)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def stage_bin(stage: str, in_current_scan: int) -> str:
    stage = str(stage or "").strip()
    if not stage or stage == "nan":
        return "not_in_current_scan"
    if stage == "approved_live":
        return "approved_live"
    if stage == "probation_live":
        return "probation_live"
    if stage == "targeted_integration":
        return "targeted_integration"
    if stage == "watch_queue":
        return "watch_queue"
    if int(in_current_scan or 0) == 1:
        return "review_or_discovered"
    return "not_in_current_scan"


def coalesce(df: pd.DataFrame, primary: str, secondary: str, default: str = "") -> pd.Series:
    left = df[primary] if primary in df.columns else pd.Series(index=df.index, dtype="object")
    right = df[secondary] if secondary in df.columns else pd.Series(index=df.index, dtype="object")
    return left.combine_first(right).fillna(default)


def blockage_diagnosis(row: pd.Series) -> str:
    stage = str(row.get("promotion_stage_latest") or "")
    in_scan = int(safe_num(row.get("in_current_scan_latest"), 0.0))
    snapshots = float(safe_num(row.get("snapshots_seen_latest"), 0.0))
    persistence = float(safe_num(row.get("persistence_score_latest"), 0.0))
    core_rank = float(safe_num(row.get("core_latest_rank"), 999.0))
    core_top30 = int(safe_num(row.get("core_top30_flag"), 0.0))

    if stage in ACTIVE_STAGES:
        return "active"
    if in_scan == 0:
        return "outside_active_scan"
    if (core_top30 == 1 or core_rank <= 30) and snapshots >= 5 and persistence >= 4.0:
        return "portfolio_like_gap"
    if snapshots >= 5 and persistence >= 4.0:
        return "governance_stall"
    if stage == "watch_queue":
        return "watch_not_promoted_yet"
    return "scan_candidate_still_weak"


def prepare_bucket(df: pd.DataFrame, bucket: str, forward: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    base = df.copy()
    base["ticker"] = base["ticker"].astype(str).str.strip()
    base["recall_bucket"] = bucket

    latest = pd.DataFrame({"ticker": []})
    if not forward.empty:
        latest = forward.copy()
        latest["ticker"] = latest["ticker"].astype(str).str.strip()
    if not current.empty:
        current_latest = current.copy()
        current_latest["ticker"] = current_latest["ticker"].astype(str).str.strip()
        current_latest = current_latest.sort_values(["ticker"], kind="stable").drop_duplicates("ticker", keep="last")
        keep_cols = [
            c
            for c in [
                "ticker",
                "promotion_stage",
                "dynamic_status",
                "pit_cluster_key",
                "recent_score",
                "promotion_score",
                "core_latest_rank",
                "core_top30_flag",
                "core_top15_flag",
                "core_rank_bridge_score",
            ]
            if c in current_latest.columns
        ]
        current_latest = current_latest[keep_cols].add_suffix("_current")
        current_latest = current_latest.rename(columns={"ticker_current": "ticker"})
        latest = latest.merge(current_latest, on="ticker", how="outer") if not latest.empty else current_latest

    merged = base.merge(latest, on="ticker", how="left", suffixes=("", "_forward"))
    merged["theme_cluster_latest"] = coalesce(merged, "pit_cluster_key", "theme_cluster", "unknown")
    current_stage = merged["promotion_stage_current"] if "promotion_stage_current" in merged.columns else pd.Series("", index=merged.index, dtype="object")
    retro_stage = merged["promotion_stage"] if "promotion_stage" in merged.columns else pd.Series("", index=merged.index, dtype="object")
    merged["promotion_stage_latest"] = current_stage.where(current_stage.fillna("").astype(str).ne(""), retro_stage).fillna("")
    current_status = merged["dynamic_status_current"] if "dynamic_status_current" in merged.columns else pd.Series("", index=merged.index, dtype="object")
    retro_status = merged["dynamic_status"] if "dynamic_status" in merged.columns else pd.Series("", index=merged.index, dtype="object")
    merged["dynamic_status_latest"] = current_status.where(current_status.fillna("").astype(str).ne(""), retro_status).fillna("")
    current_scan_col = merged["in_current_scan"] if "in_current_scan" in merged.columns else pd.Series(0, index=merged.index, dtype="float64")
    merged["in_current_scan_latest"] = pd.to_numeric(current_scan_col, errors="coerce").fillna(0).astype(int)
    merged["monitor_priority_score_latest"] = pd.to_numeric(merged.get("monitor_priority_score"), errors="coerce")
    merged["snapshots_seen_latest"] = pd.to_numeric(merged.get("snapshots_seen"), errors="coerce")
    merged["persistence_score_latest"] = pd.to_numeric(merged.get("history_emergence_persistence_score"), errors="coerce")
    merged["core_latest_rank"] = safe_num(merged.get("core_latest_rank_current"), 999.0)
    merged["core_top30_flag"] = safe_num(merged.get("core_top30_flag_current"), 0.0).astype(int)
    merged["core_top15_flag"] = safe_num(merged.get("core_top15_flag_current"), 0.0).astype(int)
    merged["core_rank_bridge_score"] = safe_num(merged.get("core_rank_bridge_score_current"), 0.0)
    merged["priority_score_retro"] = pd.to_numeric(merged.get("priority_score"), errors="coerce")
    merged["priority_score_combined"] = merged["monitor_priority_score_latest"].combine_first(merged["priority_score_retro"])
    merged["stage_bin"] = [
        stage_bin(stage, scan)
        for stage, scan in zip(merged["promotion_stage_latest"], merged["in_current_scan_latest"], strict=False)
    ]
    merged["blockage_diagnosis"] = merged.apply(blockage_diagnosis, axis=1)
    merged["stage_ord"] = merged["promotion_stage_latest"].map(STAGE_ORDER).fillna(0).astype(int)
    merged["watch_plus"] = merged["stage_ord"] >= STAGE_ORDER["watch_queue"]
    merged["targeted_plus"] = merged["stage_ord"] >= STAGE_ORDER["targeted_integration"]
    merged["approved_plus"] = merged["stage_ord"] >= STAGE_ORDER["approved_live"]
    merged["resolved_now"] = merged["targeted_plus"].astype(int)
    return merged


def pct(num: float, den: float) -> float:
    if not den:
        return 0.0
    return round(100.0 * num / den, 2)


def main() -> int:
    missing_universe = read_csv(MISSING_UNIVERSE_PATH)
    missing_selection = read_csv(MISSING_SELECTION_PATH)
    forward = read_csv(FORWARD_MONITOR_PATH)
    current = read_csv(CURRENT_DB_PATH)

    bucket_frames = [
        prepare_bucket(missing_universe, "missing_universe", forward, current),
        prepare_bucket(missing_selection, "missing_selection", forward, current),
    ]
    combined = pd.concat([df for df in bucket_frames if not df.empty], ignore_index=True, sort=False)

    summary_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    family_frames: list[pd.DataFrame] = []

    for bucket, frame in combined.groupby("recall_bucket"):
        total = int(len(frame))
        in_scan = int(frame["in_current_scan_latest"].sum())
        watch_plus = int(frame["watch_plus"].sum())
        targeted_plus = int(frame["targeted_plus"].sum())
        approved_plus = int(frame["approved_plus"].sum())
        summary_rows.append(
            {
                "bucket": bucket,
                "count": total,
                "in_current_scan": in_scan,
                "in_current_scan_pct": pct(in_scan, total),
                "watch_plus": watch_plus,
                "watch_plus_pct": pct(watch_plus, total),
                "targeted_plus": targeted_plus,
                "targeted_plus_pct": pct(targeted_plus, total),
                "approved_live": approved_plus,
                "approved_live_pct": pct(approved_plus, total),
                "median_priority": round(pd.to_numeric(frame["priority_score_combined"], errors="coerce").median(), 4),
            }
        )

        stage_counts = (
            frame["stage_bin"]
            .value_counts()
            .rename_axis("stage_bin")
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        stage_counts["bucket"] = bucket
        stage_counts["share_pct"] = stage_counts["count"].map(lambda x: pct(x, total))
        stage_rows.extend(stage_counts[["bucket", "stage_bin", "count", "share_pct"]].to_dict("records"))

        family = (
            frame.groupby("theme_cluster_latest", dropna=False)
            .agg(
                count=("ticker", "count"),
                in_scan=("in_current_scan_latest", "sum"),
                watch_plus=("watch_plus", "sum"),
                targeted_plus=("targeted_plus", "sum"),
                approved_live=("approved_plus", "sum"),
                avg_priority=("priority_score_combined", "mean"),
                avg_persistence=("persistence_score_latest", "mean"),
            )
            .reset_index()
            .rename(columns={"theme_cluster_latest": "family"})
            .sort_values(["targeted_plus", "approved_live", "avg_priority"], ascending=[False, False, False])
            .head(20)
        )
        family["bucket"] = bucket
        family["in_scan_pct"] = family["in_scan"].map(lambda x: pct(x, total))
        family["watch_plus_pct"] = family["watch_plus"].map(lambda x: pct(x, total))
        family["targeted_plus_pct"] = family["targeted_plus"].map(lambda x: pct(x, total))
        family["approved_live_pct"] = family["approved_live"].map(lambda x: pct(x, total))
        family["avg_priority"] = family["avg_priority"].round(4)
        family["avg_persistence"] = family["avg_persistence"].round(4)
        family_frames.append(
            family[
                [
                    "bucket",
                    "family",
                    "count",
                    "in_scan",
                    "watch_plus",
                    "targeted_plus",
                    "approved_live",
                    "avg_priority",
                    "avg_persistence",
                ]
            ]
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("bucket")
    stage_df = pd.DataFrame(stage_rows)
    family_df = pd.concat(family_frames, ignore_index=True, sort=False) if family_frames else pd.DataFrame()

    unresolved = combined.loc[~combined["targeted_plus"]].copy()
    unresolved = unresolved.sort_values(
        ["priority_score_combined", "in_current_scan_latest", "snapshots_seen_latest"],
        ascending=[False, False, False],
    ).head(25)
    unresolved_export = unresolved[
        [
            "ticker",
            "recall_bucket",
            "theme_cluster_latest",
            "in_current_scan_latest",
            "promotion_stage_latest",
            "dynamic_status_latest",
            "snapshots_seen_latest",
            "persistence_score_latest",
            "priority_score_combined",
            "core_latest_rank",
            "core_top30_flag",
            "blockage_diagnosis",
        ]
    ].rename(
        columns={
            "theme_cluster_latest": "family",
            "in_current_scan_latest": "in_current_scan",
            "promotion_stage_latest": "promotion_stage",
            "dynamic_status_latest": "dynamic_status",
            "snapshots_seen_latest": "snapshots_seen",
            "persistence_score_latest": "persistence_score",
            "priority_score_combined": "priority_score",
        }
    )
    unresolved_export["priority_score"] = pd.to_numeric(unresolved_export["priority_score"], errors="coerce").round(4)
    unresolved_export["persistence_score"] = pd.to_numeric(unresolved_export["persistence_score"], errors="coerce").round(4)

    progressed = combined.loc[combined["targeted_plus"]].copy()
    progressed = progressed.sort_values(
        ["stage_ord", "priority_score_combined", "snapshots_seen_latest"],
        ascending=[False, False, False],
    ).head(25)
    progressed_export = progressed[
        [
            "ticker",
            "recall_bucket",
            "theme_cluster_latest",
            "promotion_stage_latest",
            "dynamic_status_latest",
            "snapshots_seen_latest",
            "persistence_score_latest",
            "priority_score_combined",
            "core_latest_rank",
            "core_top30_flag",
            "blockage_diagnosis",
        ]
    ].rename(
        columns={
            "theme_cluster_latest": "family",
            "promotion_stage_latest": "promotion_stage",
            "dynamic_status_latest": "dynamic_status",
            "snapshots_seen_latest": "snapshots_seen",
            "persistence_score_latest": "persistence_score",
            "priority_score_combined": "priority_score",
        }
    )
    progressed_export["priority_score"] = pd.to_numeric(progressed_export["priority_score"], errors="coerce").round(4)
    progressed_export["persistence_score"] = pd.to_numeric(progressed_export["persistence_score"], errors="coerce").round(4)

    as_of_candidates = []
    if not forward.empty and "last_seen_snapshot" in forward.columns:
        as_of_candidates.append(str(forward["last_seen_snapshot"].max()))
    if not current.empty and "as_of" in current.columns:
        as_of_candidates.append(str(current["as_of"].max()))
    as_of = max([x for x in as_of_candidates if x and x != "nan"], default="unknown")

    summary_df.to_csv(SUMMARY_EXPORT_PATH, index=False)
    stage_df.to_csv(STAGE_EXPORT_PATH, index=False)
    family_df.to_csv(FAMILY_EXPORT_PATH, index=False)
    unresolved_export.to_csv(UNRESOLVED_EXPORT_PATH, index=False)
    progressed_export.to_csv(PROGRESSED_EXPORT_PATH, index=False)

    universe_summary = summary_df.loc[summary_df["bucket"] == "missing_universe"].iloc[0] if "missing_universe" in set(summary_df["bucket"]) else None
    selection_summary = summary_df.loc[summary_df["bucket"] == "missing_selection"].iloc[0] if "missing_selection" in set(summary_df["bucket"]) else None

    lines = [
        "# Winner Recall Dashboard",
        "",
        f"As of `{as_of}`.",
        "",
        "Purpose:",
        "- track whether historical missing winners are now being repaired by the current pipeline;",
        "- separate remaining misses into universe / scan-governance / portfolio-like bottlenecks;",
        "- quantify closure rates without opening a new backtest-heavy study.",
        "",
        "## High-Level Summary",
        "",
    ]
    if universe_summary is not None:
        lines.extend(
            [
                f"- `missing_universe`: `{int(universe_summary['count'])}` historical compatible misses;",
                f"  - now in current scan: `{int(universe_summary['in_current_scan'])}` (`{universe_summary['in_current_scan_pct']:.2f}%`)",
                f"  - now watch+: `{int(universe_summary['watch_plus'])}` (`{universe_summary['watch_plus_pct']:.2f}%`)",
                f"  - now targeted+: `{int(universe_summary['targeted_plus'])}` (`{universe_summary['targeted_plus_pct']:.2f}%`)",
                f"  - now approved live: `{int(universe_summary['approved_live'])}` (`{universe_summary['approved_live_pct']:.2f}%`)",
            ]
        )
    if selection_summary is not None:
        lines.extend(
            [
                f"- `missing_selection`: `{int(selection_summary['count'])}` historical compatible misses;",
                f"  - now in current scan: `{int(selection_summary['in_current_scan'])}` (`{selection_summary['in_current_scan_pct']:.2f}%`)",
                f"  - now watch+: `{int(selection_summary['watch_plus'])}` (`{selection_summary['watch_plus_pct']:.2f}%`)",
                f"  - now targeted+: `{int(selection_summary['targeted_plus'])}` (`{selection_summary['targeted_plus_pct']:.2f}%`)",
                f"  - now approved live: `{int(selection_summary['approved_live'])}` (`{selection_summary['approved_live_pct']:.2f}%`)",
            ]
        )

    lines.extend(
        [
            "",
            "## Bucket Summary",
            "",
            markdown_table(summary_df),
            "",
            "## Stage Breakdown",
            "",
            markdown_table(stage_df),
            "",
            "## Blockage Diagnosis",
            "",
            markdown_table(
                combined.groupby(["recall_bucket", "blockage_diagnosis"], dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values(["recall_bucket", "count"], ascending=[True, False])
            ),
            "",
            "## Families Repairing Best",
            "",
            markdown_table(family_df.head(20)),
            "",
            "## Top Historical Misses Still Unresolved",
            "",
            markdown_table(unresolved_export.head(15)),
            "",
            "## Historical Misses Already Progressing",
            "",
            markdown_table(progressed_export.head(15)),
            "",
            "## Strategic Read",
            "",
            "- If `missing_universe` closure keeps rising, the scan/universe repair is working.",
            "- If names become `watch` or `targeted`, the gap is no longer a pure universe miss.",
            "- If a whole family stays persistent but cannot reach `targeted+`, that starts to look like a scan/governance bottleneck.",
            "- If names get labelled `portfolio_like_gap`, they are no longer simple discovery misses: the system sees them, but the final chain still does not convert them.",
        ]
    )

    report_text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    ROOT_REPORT_PATH.write_text(report_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
