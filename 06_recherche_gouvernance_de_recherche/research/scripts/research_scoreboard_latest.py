from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = ROOT / "data" / "dynamic_universe"
REPORTS_DIR = ROOT / "research" / "reports"
EXPORTS_DIR = ROOT / "research" / "exports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

OHLCV_REFRESH_PATH = ROOT / "data" / "extracts" / "ohlcv_refresh_summary.csv"
CURRENT_DB_PATH = DATA_DIR / "dynamic_universe_current.csv"
FORWARD_MONITOR_PATH = DATA_DIR / "dynamic_universe_quality_compounder_forward_monitor.csv"
SELECTED_MOVES_PATH = DATA_DIR / "dynamic_universe_selected_moves.csv"
SELECTED_ADDS_PATH = DATA_DIR / "dynamic_universe_selected_additions.csv"
SELECTED_DEMS_PATH = DATA_DIR / "dynamic_universe_selected_demotions.csv"
PORTFOLIO_PATH = ROOT / "portfolio.json"
BASELINE_LATEST_PATH = ROOT / "SYSTEM_BASELINE_LATEST.md"
REPLAY_REPORT_PATH = REPORTS_DIR / "DYNAMIC_UNIVERSE_STAGED_REPLAY_184.md"

SUMMARY_EXPORT_PATH = EXPORTS_DIR / "research_scoreboard_summary.csv"
TOP_PRIORITIES_EXPORT_PATH = EXPORTS_DIR / "research_scoreboard_top_priorities.csv"
UNRESOLVED_EXPORT_PATH = EXPORTS_DIR / "research_scoreboard_unresolved_gaps.csv"
PROGRESSION_EXPORT_PATH = EXPORTS_DIR / "research_scoreboard_stage_progressions.csv"
FAMILY_EXPORT_PATH = EXPORTS_DIR / "research_scoreboard_family_summary.csv"
ESCALATION_EXPORT_PATH = EXPORTS_DIR / "research_scoreboard_escalation_families.csv"

REPORT_PATH = REPORTS_DIR / "RESEARCH_SCOREBOARD_LATEST.md"
ROOT_REPORT_PATH = ROOT / "RESEARCH_SCOREBOARD_LATEST.md"

ACTIVE_STAGES = {"approved_live", "probation_live", "targeted_integration"}


def safe_num(series: pd.Series | object, default: float = 0.0) -> pd.Series | float:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    try:
        value = float(series)
    except (TypeError, ValueError):
        return default
    return value if pd.notna(value) else default


def blockage_diagnosis(row: pd.Series) -> str:
    stage = str(row.get("promotion_stage") or "")
    in_scan = int(safe_num(row.get("in_current_scan"), 0.0))
    snapshots = float(safe_num(row.get("snapshots_seen"), 0.0))
    persistence = float(safe_num(row.get("history_emergence_persistence_score"), 0.0))
    core_rank = float(safe_num(row.get("core_latest_rank"), 999.0))
    core_top30 = int(safe_num(row.get("core_top30_flag"), 0.0))
    core_top15 = int(safe_num(row.get("core_top15_flag"), 0.0))

    if stage in ACTIVE_STAGES:
        return "active"
    if in_scan == 0:
        return "outside_active_scan"
    if (core_top15 == 1 or core_rank <= 15) and persistence >= 4.0:
        return "portfolio_like_gap"
    if (core_top30 == 1 or core_rank <= 30) and snapshots >= 5 and persistence >= 4.0:
        return "portfolio_like_gap"
    if snapshots >= 5 and persistence >= 4.0:
        return "governance_stall"
    if stage == "watch_queue":
        return "watch_not_promoted_yet"
    return "scan_candidate_still_weak"


def escalation_flag(row: pd.Series) -> int:
    diagnosis = str(row.get("blockage_diagnosis") or "")
    snapshots = float(safe_num(row.get("snapshots_seen"), 0.0))
    persistence = float(safe_num(row.get("history_emergence_persistence_score"), 0.0))
    in_scan = int(safe_num(row.get("in_current_scan"), 0.0))
    return int(
        diagnosis in {"portfolio_like_gap", "governance_stall"}
        and in_scan == 1
        and snapshots >= 5
        and persistence >= 4.0
    )


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


def parse_baseline_latest(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    label_match = re.search(r"- label: `([^`]+)`", text)
    frozen_match = re.search(r"- frozen on: `([^`]+)`", text)
    return {
        "label": label_match.group(1) if label_match else "unknown",
        "frozen_on": frozen_match.group(1) if frozen_match else "unknown",
    }


def parse_replay_metrics(path: Path) -> dict[str, float | str]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    rows: list[list[str]] = []
    for line in lines:
        if line.startswith("| ---"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        rows.append(parts)
    if len(rows) < 2:
        return {}
    header = rows[0]
    data = pd.DataFrame(rows[1:], columns=header)
    data.columns = [c.strip() for c in data.columns]
    metrics: dict[str, float | str] = {}
    for suffix, key in (("full", "staged_proxy_live_185_full"), ("oos", "staged_proxy_live_185_oos")):
        row = data.loc[data["label"] == key]
        if row.empty:
            continue
        r = row.iloc[0]
        metrics[f"{suffix}_start"] = r["start"]
        metrics[f"{suffix}_end"] = r["end"]
        for col in ("roi_pct", "cagr_pct", "maxdd_pct", "sharpe", "orders"):
            val = r[col]
            try:
                metrics[f"{suffix}_{col}"] = float(val)
            except ValueError:
                metrics[f"{suffix}_{col}"] = val
    return metrics


def load_portfolio_positions(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    positions = data.get("positions") or {}
    return sorted(str(ticker) for ticker in positions.keys())


def classify_recommendation(refresh_pct: float, forward_df: pd.DataFrame) -> str:
    if refresh_pct < 98.5:
        return "data issue first"
    if not forward_df.empty:
        candidate_attention = forward_df.loc[
            (pd.to_numeric(forward_df.get("needs_algo_attention"), errors="coerce").fillna(0).astype(int) == 1)
            & (~forward_df["promotion_stage"].isin(ACTIVE_STAGES))
            & (pd.to_numeric(forward_df.get("snapshots_seen"), errors="coerce").fillna(0) >= 5)
            & (pd.to_numeric(forward_df.get("history_emergence_persistence_score"), errors="coerce").fillna(0) >= 4.0)
            & (pd.to_numeric(forward_df.get("in_current_scan"), errors="coerce").fillna(0) == 1)
        ].copy()
        if not candidate_attention.empty and "pit_cluster_key" in candidate_attention.columns:
            cluster_counts = candidate_attention["pit_cluster_key"].fillna("unknown").value_counts()
            if int(cluster_counts.max()) >= 2:
                return "open research study"
    return "continue monitoring"


def main() -> int:
    baseline = parse_baseline_latest(BASELINE_LATEST_PATH)
    replay_metrics = parse_replay_metrics(REPLAY_REPORT_PATH)

    refresh = read_csv(OHLCV_REFRESH_PATH)
    current = read_csv(CURRENT_DB_PATH)
    forward = read_csv(FORWARD_MONITOR_PATH)
    moves = read_csv(SELECTED_MOVES_PATH)
    selected_adds = read_csv(SELECTED_ADDS_PATH)
    selected_dems = read_csv(SELECTED_DEMS_PATH)
    portfolio_positions = load_portfolio_positions(PORTFOLIO_PATH)

    refreshed_rows = int(pd.to_numeric(refresh.get("refreshed"), errors="coerce").fillna(0).sum()) if not refresh.empty else 0
    refresh_total = int(len(refresh))
    refresh_pct = 100.0 * refreshed_rows / refresh_total if refresh_total else 0.0
    failed_refresh = []
    ohlcv_max_date = ""
    if not refresh.empty:
        failed_refresh = refresh.loc[pd.to_numeric(refresh["refreshed"], errors="coerce").fillna(0) == 0, "ticker"].astype(str).tolist()
        ohlcv_max_date = str(refresh["end_date"].max())

    if not current.empty:
        current["core_latest_rank"] = safe_num(current.get("core_latest_rank"), 999.0)
        current["core_top30_flag"] = safe_num(current.get("core_top30_flag"), 0.0).astype(int)
        current["core_top15_flag"] = safe_num(current.get("core_top15_flag"), 0.0).astype(int)
        current["core_rank_bridge_score"] = safe_num(current.get("core_rank_bridge_score"), 0.0)
    approved_live_count = int((current.get("promotion_stage") == "approved_live").sum()) if not current.empty else 0
    targeted_count = int((current.get("promotion_stage") == "targeted_integration").sum()) if not current.empty else 0
    probation_count = int((current.get("promotion_stage") == "probation_live").sum()) if not current.empty else 0
    watch_count = int((current.get("promotion_stage") == "watch_queue").sum()) if not current.empty else 0
    current_as_of = str(current["as_of"].max()) if not current.empty and "as_of" in current.columns else ""

    tracked_names = int(len(forward))
    in_current_scan = int(pd.to_numeric(forward.get("in_current_scan"), errors="coerce").fillna(0).sum()) if not forward.empty else 0
    active_forward = forward.loc[forward["promotion_stage"].isin(ACTIVE_STAGES)].copy() if not forward.empty else pd.DataFrame()
    progressed = forward.loc[
        (pd.to_numeric(forward.get("stage_delta_vs_prev"), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(forward.get("status_delta_vs_prev"), errors="coerce").fillna(0) > 0)
    ].copy() if not forward.empty else pd.DataFrame()
    current_bridge = pd.DataFrame()
    if not current.empty:
        keep_cols = [
            "ticker",
            "core_latest_rank",
            "core_top30_flag",
            "core_top15_flag",
            "core_rank_bridge_fit",
            "core_rank_bridge_score",
            "pit_cluster_key",
            "promotion_stage",
            "dynamic_status",
            "recent_score",
        ]
        keep_cols = [c for c in keep_cols if c in current.columns]
        current_bridge = current[keep_cols].copy().drop_duplicates("ticker", keep="last")

    if not forward.empty and not current_bridge.empty:
        forward = forward.merge(
            current_bridge.add_suffix("_current").rename(columns={"ticker_current": "ticker"}),
            on="ticker",
            how="left",
        )
        for col in ("promotion_stage", "dynamic_status", "pit_cluster_key", "recent_score"):
            current_col = f"{col}_current"
            if current_col in forward.columns:
                forward[col] = forward[current_col].combine_first(forward.get(col))
        for col in ("core_latest_rank", "core_top30_flag", "core_top15_flag", "core_rank_bridge_score", "core_rank_bridge_fit"):
            current_col = f"{col}_current"
            if current_col in forward.columns:
                forward[col] = forward[current_col]

    if not forward.empty:
        for col in ("snapshots_seen", "history_emergence_persistence_score", "in_current_scan", "core_latest_rank", "core_top30_flag", "core_top15_flag", "core_rank_bridge_score"):
            if col not in forward.columns:
                forward[col] = 0.0
        forward["snapshots_seen"] = safe_num(forward["snapshots_seen"], 0.0)
        forward["history_emergence_persistence_score"] = safe_num(forward["history_emergence_persistence_score"], 0.0)
        forward["in_current_scan"] = safe_num(forward["in_current_scan"], 0.0).astype(int)
        forward["core_latest_rank"] = safe_num(forward["core_latest_rank"], 999.0)
        forward["core_top30_flag"] = safe_num(forward["core_top30_flag"], 0.0).astype(int)
        forward["core_top15_flag"] = safe_num(forward["core_top15_flag"], 0.0).astype(int)
        forward["core_rank_bridge_score"] = safe_num(forward["core_rank_bridge_score"], 0.0)
        forward["blockage_diagnosis"] = forward.apply(blockage_diagnosis, axis=1)
        forward["family_escalation_flag"] = forward.apply(escalation_flag, axis=1)

    unresolved = forward.loc[
        (forward["monitor_bucket"] == "legacy_selection_gap")
        & (~forward["promotion_stage"].isin(ACTIVE_STAGES))
    ].copy() if not forward.empty else pd.DataFrame()

    top_priorities = forward.sort_values("monitor_priority_score", ascending=False).head(15).copy() if not forward.empty else pd.DataFrame()
    if not top_priorities.empty:
        top_priorities = top_priorities[
            [
                "ticker",
                "monitor_bucket",
                "promotion_stage",
                "dynamic_status",
                "pit_cluster_key",
                "scan_candidate_track",
                "history_emergence_persistence_score",
                "pit_data_context_score",
                "recent_score",
                "monitor_priority_score",
            ]
        ]

    unresolved_export = unresolved.sort_values("monitor_priority_score", ascending=False).head(15).copy() if not unresolved.empty else pd.DataFrame()
    if not unresolved_export.empty:
        unresolved_export = unresolved_export[
            [
                "ticker",
                "promotion_stage",
                "dynamic_status",
                "pit_cluster_key",
                "core_latest_rank",
                "core_top30_flag",
                "scan_candidate_track",
                "scan_quality_compounder_fit",
                "snapshots_seen",
                "history_emergence_persistence_score",
                "blockage_diagnosis",
                "family_escalation_flag",
                "pit_data_context_score",
                "monitor_priority_score",
            ]
        ]

    progressed_export = progressed.sort_values(
        ["stage_delta_vs_prev", "status_delta_vs_prev", "monitor_priority_score"],
        ascending=[False, False, False],
    ).head(15).copy() if not progressed.empty else pd.DataFrame()
    if not progressed_export.empty:
        progressed_export = progressed_export[
            [
                "ticker",
                "monitor_bucket",
                "prior_promotion_stage",
                "promotion_stage",
                "prior_dynamic_status",
                "dynamic_status",
                "pit_cluster_key",
                "monitor_priority_score",
            ]
        ]

    family_summary = pd.DataFrame()
    if not active_forward.empty and "pit_cluster_key" in active_forward.columns:
        family_summary = (
            active_forward.groupby("pit_cluster_key", dropna=False)
            .agg(
                names=("ticker", "count"),
                approved_live=("promotion_stage", lambda s: int((s == "approved_live").sum())),
                targeted=("promotion_stage", lambda s: int((s == "targeted_integration").sum())),
                avg_priority=("monitor_priority_score", "mean"),
                avg_persistence=("history_emergence_persistence_score", "mean"),
            )
            .reset_index()
            .rename(columns={"pit_cluster_key": "cluster"})
            .sort_values(["approved_live", "targeted", "avg_priority"], ascending=[False, False, False])
            .head(12)
        )
        family_summary["avg_priority"] = family_summary["avg_priority"].round(4)
        family_summary["avg_persistence"] = family_summary["avg_persistence"].round(4)

    escalation_families = pd.DataFrame()
    if not unresolved.empty:
        candidates = unresolved.loc[unresolved["family_escalation_flag"] == 1].copy()
        if not candidates.empty and "pit_cluster_key" in candidates.columns:
            escalation_families = (
                candidates.groupby("pit_cluster_key", dropna=False)
                .agg(
                    blocked_names=("ticker", "count"),
                    top30_supported=("core_top30_flag", "sum"),
                    avg_core_rank=("core_latest_rank", "mean"),
                    avg_priority=("monitor_priority_score", "mean"),
                    avg_persistence=("history_emergence_persistence_score", "mean"),
                    example_names=("ticker", lambda s: ", ".join(sorted(map(str, s))[:4])),
                )
                .reset_index()
                .rename(columns={"pit_cluster_key": "cluster"})
                .sort_values(["blocked_names", "top30_supported", "avg_priority"], ascending=[False, False, False])
            )
            escalation_families["avg_core_rank"] = escalation_families["avg_core_rank"].round(2)
            escalation_families["avg_priority"] = escalation_families["avg_priority"].round(4)
            escalation_families["avg_persistence"] = escalation_families["avg_persistence"].round(4)

    selected_add_names = selected_adds["ticker"].astype(str).tolist() if not selected_adds.empty else []
    selected_dem_names = selected_dems["ticker"].astype(str).tolist() if not selected_dems.empty and "ticker" in selected_dems.columns else []
    swap_pairs = []
    standalone_remove_names: list[str] = []
    if not moves.empty:
        standalone_remove_names = (
            moves.loc[moves["action_type"] == "standalone_remove", "ticker"].astype(str).tolist()
            if "action_type" in moves.columns
            else []
        )
        single_moves = moves.loc[moves["action_type"] == "single", ["ticker", "paired_ticker", "side"]].copy()
        add_rows = single_moves.loc[single_moves["side"] == "ADD"]
        swap_pairs = [f"{row['paired_ticker']} -> {row['ticker']}" for _, row in add_rows.iterrows()]

    aligned_with_local_book = sorted(set(selected_add_names).intersection(portfolio_positions))
    local_book_outside_overlay = sorted(set(portfolio_positions) - set(selected_add_names))

    recommendation = classify_recommendation(refresh_pct, forward)

    summary_row = {
        "as_of": current_as_of or ohlcv_max_date,
        "baseline_label": baseline["label"],
        "baseline_frozen_on": baseline["frozen_on"],
        "ohlcv_max_date": ohlcv_max_date,
        "ohlcv_refreshed_rows": refreshed_rows,
        "ohlcv_total_rows": refresh_total,
        "ohlcv_refresh_pct": round(refresh_pct, 4),
        "ohlcv_failed_tickers": ",".join(failed_refresh),
        "approved_live_count": approved_live_count,
        "targeted_count": targeted_count,
        "probation_count": probation_count,
        "watch_count": watch_count,
        "selected_additions_count": len(selected_add_names),
        "selected_single_swaps_count": len(swap_pairs),
        "selected_demotions_count": len(selected_dem_names),
        "standalone_removes_count": len(standalone_remove_names),
        "selected_additions": ",".join(selected_add_names),
        "selected_swaps": ",".join(swap_pairs),
        "tracked_forward_names": tracked_names,
        "forward_in_current_scan": in_current_scan,
        "forward_active_names": int(len(active_forward)),
        "forward_progressions": int(len(progressed)),
        "unresolved_legacy_selection_gaps": int(len(unresolved)),
        "family_escalation_candidates": int(len(escalation_families)),
        "local_portfolio_names": ",".join(portfolio_positions),
        "local_portfolio_outside_overlay": ",".join(local_book_outside_overlay),
        "recommendation": recommendation,
        **replay_metrics,
    }
    pd.DataFrame([summary_row]).to_csv(SUMMARY_EXPORT_PATH, index=False)

    if not top_priorities.empty:
        top_priorities.to_csv(TOP_PRIORITIES_EXPORT_PATH, index=False)
    else:
        pd.DataFrame(columns=["ticker"]).to_csv(TOP_PRIORITIES_EXPORT_PATH, index=False)
    if not unresolved_export.empty:
        unresolved_export.to_csv(UNRESOLVED_EXPORT_PATH, index=False)
    else:
        pd.DataFrame(columns=["ticker"]).to_csv(UNRESOLVED_EXPORT_PATH, index=False)
    if not progressed_export.empty:
        progressed_export.to_csv(PROGRESSION_EXPORT_PATH, index=False)
    else:
        pd.DataFrame(columns=["ticker"]).to_csv(PROGRESSION_EXPORT_PATH, index=False)
    if not family_summary.empty:
        family_summary.to_csv(FAMILY_EXPORT_PATH, index=False)
    else:
        pd.DataFrame(columns=["cluster"]).to_csv(FAMILY_EXPORT_PATH, index=False)
    if not escalation_families.empty:
        escalation_families.to_csv(ESCALATION_EXPORT_PATH, index=False)
    else:
        pd.DataFrame(columns=["cluster"]).to_csv(ESCALATION_EXPORT_PATH, index=False)

    lines: list[str] = [
        "# Research Scoreboard Latest",
        "",
        f"- as_of: `{summary_row['as_of']}`",
        f"- active baseline: `{baseline['label']}`",
        f"- baseline frozen on: `{baseline['frozen_on']}`",
        f"- recommendation: `{recommendation}`",
        "",
        "## Baseline Snapshot",
        "",
        f"- full replay: `ROI {replay_metrics.get('full_roi_pct', 'n/a')}%`, `Sharpe {replay_metrics.get('full_sharpe', 'n/a')}`, `MaxDD {replay_metrics.get('full_maxdd_pct', 'n/a')}%`",
        f"- OOS replay: `ROI {replay_metrics.get('oos_roi_pct', 'n/a')}%`, `Sharpe {replay_metrics.get('oos_sharpe', 'n/a')}`, `MaxDD {replay_metrics.get('oos_maxdd_pct', 'n/a')}%`",
        "",
        "## Data Freshness",
        "",
        f"- OHLCV max date: `{ohlcv_max_date}`",
        f"- refreshed coverage: `{refreshed_rows}/{refresh_total}` = `{refresh_pct:.2f}%`",
        f"- failed tickers: `{', '.join(failed_refresh) if failed_refresh else 'none'}`",
        "",
        "## Live Overlay",
        "",
        f"- approved standalone adds: `{len(selected_add_names)}`",
        f"- approved single swaps: `{len(swap_pairs)}`",
        f"- approved standalone removes: `{len(standalone_remove_names)}`",
        f"- selected demotions: `{len(selected_dem_names)}`",
        f"- selected adds: `{', '.join(selected_add_names) if selected_add_names else 'none'}`",
        f"- selected swaps: `{', '.join(swap_pairs) if swap_pairs else 'none'}`",
        "",
        "## Pipeline State",
        "",
        f"- tracked forward names: `{tracked_names}`",
        f"- names still in current scan: `{in_current_scan}`",
        f"- approved / probation / targeted names: `{len(active_forward)}`",
        f"- stage or status progressions vs previous snapshot: `{len(progressed)}`",
        f"- unresolved legacy selection gaps below targeted: `{len(unresolved)}`",
        f"- family escalation candidates: `{len(escalation_families)}`",
        "",
        "## Top Forward Priorities",
        "",
        markdown_table(top_priorities),
        "",
        "## Family Summary",
        "",
        markdown_table(family_summary),
        "",
        "## Recent Progressions",
        "",
        markdown_table(progressed_export),
        "",
        "## Unresolved Portfolio-Ranking Gaps",
        "",
        markdown_table(unresolved_export),
        "",
        "## Families To Escalate If They Persist",
        "",
        markdown_table(escalation_families.head(12)),
        "",
        "## Local Portfolio Context",
        "",
        f"- local portfolio names: `{', '.join(portfolio_positions) if portfolio_positions else 'none'}`",
        f"- names already shared with live overlay: `{', '.join(aligned_with_local_book) if aligned_with_local_book else 'none'}`",
        f"- local holdings outside current live overlay: `{', '.join(local_book_outside_overlay) if local_book_outside_overlay else 'none'}`",
        "- note: this local book is useful for operational alignment, but it is not promotion evidence for the engine.",
        "",
        "## Interpretation",
        "",
        "- the scan repair is still doing useful work: multiple families now progress naturally through governance without forcing names into the engine.",
        "- the remaining misses are concentrated in the ranking / portfolio layer, but the new blockage labels now separate `scan_candidate_still_weak`, `governance_stall` and `portfolio_like_gap`.",
        "- the only families worth escalating are the ones that keep producing persistent blocked names with live scan support and core-rank support.",
        "- the right posture remains: keep `r8` stable, keep observing the repaired families, and only reopen engine work if a family persists across several snapshots and still fails structurally.",
        "",
        "## References",
        "",
        f"- [SYSTEM_BASELINE_LATEST.md]({BASELINE_LATEST_PATH.as_posix()})",
        f"- [dynamic_universe_actions_summary.md]({(DATA_DIR / 'dynamic_universe_actions_summary.md').as_posix()})",
        f"- [dynamic_universe_quality_compounder_forward_monitor.md]({(DATA_DIR / 'dynamic_universe_quality_compounder_forward_monitor.md').as_posix()})",
        f"- [WEEKLY_FORWARD_REVIEW_PROTOCOL.md]({(ROOT / 'WEEKLY_FORWARD_REVIEW_PROTOCOL.md').as_posix()})",
        "",
    ]

    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    ROOT_REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"saved_report: {REPORT_PATH}")
    print(f"saved_root_report: {ROOT_REPORT_PATH}")
    print(f"saved_summary: {SUMMARY_EXPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
