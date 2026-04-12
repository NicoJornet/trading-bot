from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
EXPORTS_DIR = ROOT / "research" / "exports"
REPORTS_DIR = ROOT / "research" / "reports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SLOT3_STUDY_PATH = EXPORTS_DIR / "r8_slot3_admission_sizing_study.csv"
SWITCH_AUDIT_PATH = EXPORTS_DIR / "r8_switch_trigger_audit.csv"
REENTRY_STUDY_PATH = EXPORTS_DIR / "r8_reason_aware_reentry_study.csv"
PORT_RISK_PATH = EXPORTS_DIR / "r4_portfolio_risk_daily.csv"
UNRESOLVED_GAPS_PATH = EXPORTS_DIR / "research_scoreboard_unresolved_gaps.csv"
WINNER_UNRESOLVED_PATH = EXPORTS_DIR / "winner_recall_top_unresolved.csv"
SUMMARY_EXPORT_PATH = EXPORTS_DIR / "engine_research_axes_summary.csv"
OPEN_CANDIDATES_EXPORT_PATH = EXPORTS_DIR / "engine_research_open_candidates.csv"
REPORT_PATH = REPORTS_DIR / "ENGINE_RESEARCH_SCOREBOARD_LATEST.md"
ROOT_REPORT_PATH = ROOT / "ENGINE_RESEARCH_SCOREBOARD_LATEST.md"


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


def safe_num(series: pd.Series | object, default: float = 0.0) -> pd.Series | float:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    try:
        value = float(series)
    except (TypeError, ValueError):
        return default
    return value if pd.notna(value) else default


def axis_status_from_deltas(full_roi_delta: float, oos_roi_delta: float, full_sharpe_delta: float, oos_sharpe_delta: float) -> str:
    if full_roi_delta > 0 and oos_roi_delta > 0 and full_sharpe_delta > 0 and oos_sharpe_delta > 0:
        return "promotion_candidate"
    if full_roi_delta > 0 and abs(oos_roi_delta) < 1e-9 and full_sharpe_delta >= 0 and abs(oos_sharpe_delta) < 1e-9:
        return "candidate_under_surveillance"
    return "closed_no_upgrade"


def build_axis_row(label: str, family: str, thesis: str, baseline: pd.Series, best: pd.Series) -> dict[str, object]:
    full_roi_delta = float(best["full_roi_pct"] - baseline["full_roi_pct"])
    oos_roi_delta = float(best["oos_roi_pct"] - baseline["oos_roi_pct"])
    full_sharpe_delta = float(best["full_sharpe"] - baseline["full_sharpe"])
    oos_sharpe_delta = float(best["oos_sharpe"] - baseline["oos_sharpe"])
    return {
        "axis": label,
        "family": family,
        "thesis": thesis,
        "best_variant": str(best["label"]),
        "full_roi_delta": round(full_roi_delta, 4),
        "oos_roi_delta": round(oos_roi_delta, 4),
        "full_sharpe_delta": round(full_sharpe_delta, 6),
        "oos_sharpe_delta": round(oos_sharpe_delta, 6),
        "status": axis_status_from_deltas(full_roi_delta, oos_roi_delta, full_sharpe_delta, oos_sharpe_delta),
    }


def slot3_axis() -> dict[str, object]:
    df = read_csv(SLOT3_STUDY_PATH)
    if df.empty:
        return {
            "axis": "slot3_architecture",
            "family": "portfolio_admission",
            "thesis": "third line should earn its size more clearly",
            "best_variant": "missing_data",
            "full_roi_delta": 0.0,
            "oos_roi_delta": 0.0,
            "full_sharpe_delta": 0.0,
            "oos_sharpe_delta": 0.0,
            "status": "missing_data",
        }
    baseline = df.loc[df["label"] == "baseline_current"].iloc[0]
    best = df.iloc[0]
    return build_axis_row(
        "slot3_architecture",
        "portfolio_admission",
        "third line should earn its size more clearly",
        baseline,
        best,
    )


def reentry_axis() -> dict[str, object]:
    df = read_csv(REENTRY_STUDY_PATH)
    if df.empty:
        return {
            "axis": "reason_aware_reentry",
            "family": "reentry_logic",
            "thesis": "reentries should depend on previous exit reason",
            "best_variant": "missing_data",
            "full_roi_delta": 0.0,
            "oos_roi_delta": 0.0,
            "full_sharpe_delta": 0.0,
            "oos_sharpe_delta": 0.0,
            "status": "missing_data",
        }
    baseline = df.loc[df["label"] == "baseline_current"].iloc[0]
    best = df.iloc[0]
    return build_axis_row(
        "reason_aware_reentry",
        "reentry_logic",
        "reentries should depend on previous exit reason",
        baseline,
        best,
    )


def switch_axis() -> dict[str, object]:
    df = read_csv(SWITCH_AUDIT_PATH)
    if df.empty:
        return {
            "axis": "switch_logic",
            "family": "rotation_logic",
            "thesis": "switch trigger thresholds may still be too loose or too strict",
            "best_variant": "missing_data",
            "full_roi_delta": 0.0,
            "oos_roi_delta": 0.0,
            "full_sharpe_delta": 0.0,
            "oos_sharpe_delta": 0.0,
            "status": "missing_data",
        }
    baseline = df.loc[df["label"] == "baseline_current"].iloc[0]
    best = df.iloc[0]
    row = build_axis_row(
        "switch_logic",
        "rotation_logic",
        "switch trigger thresholds may still be too loose or too strict",
        baseline,
        best,
    )
    row["switch_days_delta"] = int(best["switch_days"] - baseline["switch_days"])
    row["switch_pairs_delta"] = int(best["switch_pairs_lb"] - baseline["switch_pairs_lb"])
    return row


def concentration_axis() -> dict[str, object]:
    df = read_csv(PORT_RISK_PATH)
    if df.empty:
        return {
            "axis": "concentration_gap_risk",
            "family": "portfolio_risk",
            "thesis": "drawdowns are still dominated by carried concentration and overnight gaps",
            "best_variant": "structural_open",
            "avg_top1_weight": 0.0,
            "p95_top1_weight": 0.0,
            "avg_top3_share": 0.0,
            "days_dom_cluster_over_70": 0,
            "worst_overnight_gap_pct": 0.0,
            "status": "missing_data",
        }
    live = df.loc[safe_num(df["n_positions"], 0.0) > 0].copy()
    if live.empty:
        live = df.copy()
    dom_cluster_over_70 = int((safe_num(live["dominant_cluster_share"], 0.0) >= 0.70).sum())
    dom_sector_over_70 = int((safe_num(live["dominant_sector_share"], 0.0) >= 0.70).sum())
    worst_gap = float(safe_num(live["overnight_gap_return_pct"], 0.0).min())
    return {
        "axis": "concentration_gap_risk",
        "family": "portfolio_risk",
        "thesis": "drawdowns are still dominated by carried concentration and overnight gaps",
        "best_variant": "no_clean_patch_yet",
        "avg_top1_weight": round(float(safe_num(live["top1_weight"], 0.0).mean()), 4),
        "p95_top1_weight": round(float(safe_num(live["top1_weight"], 0.0).quantile(0.95)), 4),
        "avg_top3_share": round(float(safe_num(live["top3_share"], 0.0).mean()), 4),
        "days_dom_cluster_over_70": dom_cluster_over_70,
        "days_dom_sector_over_70": dom_sector_over_70,
        "worst_overnight_gap_pct": round(worst_gap, 4),
        "status": "open_structural_pressure",
    }


def engine_candidates() -> pd.DataFrame:
    current_unresolved = read_csv(UNRESOLVED_GAPS_PATH)
    winner_unresolved = read_csv(WINNER_UNRESOLVED_PATH)

    frames: list[pd.DataFrame] = []
    if not current_unresolved.empty:
        cur = current_unresolved.copy()
        cur["source"] = "current_unresolved"
        cur["family"] = cur["pit_cluster_key"]
        cur["persistence_score"] = safe_num(cur["history_emergence_persistence_score"], 0.0)
        cur["priority_score"] = safe_num(cur["monitor_priority_score"], 0.0)
        cur["core_rank"] = safe_num(cur["core_latest_rank"], 999.0)
        cur["core_top30"] = safe_num(cur["core_top30_flag"], 0.0).astype(int)
        frames.append(
            cur[
                [
                    "ticker",
                    "source",
                    "family",
                    "promotion_stage",
                    "dynamic_status",
                    "blockage_diagnosis",
                    "snapshots_seen",
                    "persistence_score",
                    "priority_score",
                    "core_rank",
                    "core_top30",
                ]
            ]
        )
    if not winner_unresolved.empty:
        win = winner_unresolved.copy()
        win["source"] = "winner_recall"
        win["family"] = win["family"]
        win["persistence_score"] = safe_num(win["persistence_score"], 0.0)
        win["priority_score"] = safe_num(win["priority_score"], 0.0)
        win["core_rank"] = safe_num(win["core_latest_rank"], 999.0)
        win["core_top30"] = safe_num(win["core_top30_flag"], 0.0).astype(int)
        frames.append(
            win[
                [
                    "ticker",
                    "source",
                    "family",
                    "promotion_stage",
                    "dynamic_status",
                    "blockage_diagnosis",
                    "snapshots_seen",
                    "persistence_score",
                    "priority_score",
                    "core_rank",
                    "core_top30",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(["ticker", "source"])
    df["_source_priority"] = df["source"].map({"current_unresolved": 0, "winner_recall": 1}).fillna(9)
    core_rank_norm = df["core_rank"].astype(float).replace(0.0, 999.0)
    df["_core_rank_norm"] = core_rank_norm
    df = df.sort_values(
        ["ticker", "_source_priority", "priority_score", "_core_rank_norm"],
        ascending=[True, True, False, True],
    )
    grouped = (
        df.groupby("ticker", as_index=False)
        .agg(
            source=("source", lambda s: ", ".join(dict.fromkeys(map(str, s)))),
            family=("family", "first"),
            promotion_stage=("promotion_stage", "first"),
            dynamic_status=("dynamic_status", "first"),
            blockage_diagnosis=("blockage_diagnosis", "first"),
            snapshots_seen=("snapshots_seen", "max"),
            persistence_score=("persistence_score", "max"),
            priority_score=("priority_score", "max"),
            core_rank=("_core_rank_norm", "min"),
            core_top30=("core_top30", "max"),
        )
    )
    grouped["core_rank"] = grouped["core_rank"].replace(999.0, 0.0)
    df = grouped
    df["engine_attention"] = (
        (df["core_top30"] == 1)
        & (safe_num(df["snapshots_seen"], 0.0) >= 5)
        & (safe_num(df["persistence_score"], 0.0) >= 4.0)
        & (df["blockage_diagnosis"].isin(["portfolio_like_gap", "governance_stall"]))
    ).astype(int)
    df["readiness"] = "monitor_only"
    df.loc[
        (df["core_top30"] == 1)
        & (safe_num(df["snapshots_seen"], 0.0) >= 5)
        & (safe_num(df["persistence_score"], 0.0) >= 2.0)
        & (df["blockage_diagnosis"].isin(["watch_not_promoted_yet", "portfolio_like_gap"])),
        "readiness",
    ] = "candidate_under_watch"
    df.loc[df["engine_attention"] == 1, "readiness"] = "ready_if_family_repeats"
    df = df.sort_values(
        ["engine_attention", "priority_score", "core_top30", "core_rank"],
        ascending=[False, False, False, True],
    )
    return df


def main() -> int:
    axis_rows = [
        slot3_axis(),
        reentry_axis(),
        switch_axis(),
        concentration_axis(),
    ]
    axis_df = pd.DataFrame(axis_rows)
    open_candidates = engine_candidates()

    axis_df.to_csv(SUMMARY_EXPORT_PATH, index=False)
    if not open_candidates.empty:
        open_candidates.to_csv(OPEN_CANDIDATES_EXPORT_PATH, index=False)
    else:
        pd.DataFrame(columns=["ticker"]).to_csv(OPEN_CANDIDATES_EXPORT_PATH, index=False)

    concentration = axis_df.loc[axis_df["axis"] == "concentration_gap_risk"].iloc[0]
    open_engine = open_candidates.loc[open_candidates["engine_attention"] == 1].copy() if not open_candidates.empty else pd.DataFrame()
    watch_engine = open_candidates.head(15).copy() if not open_candidates.empty else pd.DataFrame()

    lines = [
        "# Engine Research Scoreboard Latest",
        "",
        "Pragmatic scoreboard for the core `r8` engine: which structural levers are still open, which are already closed, and which candidates are only worth watching for now.",
        "",
        "## Axes Summary",
        "",
        markdown_table(axis_df),
        "",
        "## Core Reading",
        "",
        f"- `slot3_architecture`: `{axis_df.loc[axis_df['axis'] == 'slot3_architecture', 'status'].iloc[0]}`",
        f"- `reason_aware_reentry`: `{axis_df.loc[axis_df['axis'] == 'reason_aware_reentry', 'status'].iloc[0]}`",
        f"- `switch_logic`: `{axis_df.loc[axis_df['axis'] == 'switch_logic', 'status'].iloc[0]}`",
        f"- `concentration_gap_risk`: `{concentration['status']}`",
        "",
        "## Engine Candidates Under Watch",
        "",
        markdown_table(
            watch_engine[
                [
                    "ticker",
                    "source",
                    "family",
                    "promotion_stage",
                    "blockage_diagnosis",
                    "snapshots_seen",
                    "persistence_score",
                    "core_rank",
                    "core_top30",
                    "readiness",
                ]
            ].head(15)
        ),
        "",
        "## Ready To Reopen Engine Work",
        "",
        markdown_table(
            open_engine[
                [
                    "ticker",
                    "family",
                    "promotion_stage",
                    "blockage_diagnosis",
                    "snapshots_seen",
                    "persistence_score",
                    "core_rank",
                    "readiness",
                ]
            ].head(12)
        ),
        "",
        "## Structural Pressure That Remains",
        "",
        f"- average top-1 weight: `{concentration.get('avg_top1_weight', 'n/a')}`",
        f"- 95th percentile top-1 weight: `{concentration.get('p95_top1_weight', 'n/a')}`",
        f"- average top-3 share: `{concentration.get('avg_top3_share', 'n/a')}`",
        f"- days with dominant cluster above 70%: `{int(concentration.get('days_dom_cluster_over_70', 0))}`",
        f"- days with dominant sector above 70%: `{int(concentration.get('days_dom_sector_over_70', 0))}`",
        f"- worst overnight gap return: `{concentration.get('worst_overnight_gap_pct', 'n/a')}%`",
        "",
        "## Interpretation",
        "",
        "- The engine levers we already tested are mostly closed: reentry-by-reason, switch-trigger tuning, and slot-3 gating did not produce a clean upgrade to `r8`.",
        "- The only pressure that still looks genuinely structural is concentration and overnight gap risk, but we still do not have a broad, non-overfit patch that improves both full and OOS.",
        "- Engine work should not reopen because of isolated names. It should reopen only when the same family keeps generating persistent `ready_if_family_repeats` candidates or when a broad concentration overlay earns its keep without hindsight patching.",
        "",
        "## Recommendation",
        "",
        "- Keep `r8` stable.",
        "- Treat slot-3 as `candidate_under_surveillance`, not as a promotion.",
        "- Treat reentry and switch tuning as `closed_no_upgrade` for now.",
        "- Keep concentration / gap risk as the main open structural pressure, but do not patch it again until forward evidence accumulates.",
        "",
    ]

    report_text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    ROOT_REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"saved_report: {REPORT_PATH}")
    print(f"saved_root_report: {ROOT_REPORT_PATH}")
    print(f"saved_summary: {SUMMARY_EXPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
