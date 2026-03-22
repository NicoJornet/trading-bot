from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Iterable
import time

import pandas as pd

import dynamic_universe_discovery as dud
import run_dynamic_universe_cycle as cycle


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "dynamic_universe"
HISTORY_DIR = DATA_DIR / "history"
REPORTS_DIR = ROOT / "research" / "reports"
EXPORTS_DIR = ROOT / "research" / "exports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

BROKER_MIN_MARKET_CAP = 1_000_000_000


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [col for col in columns if col in df.columns]
    if not available:
        return pd.DataFrame()
    return df[available].copy()


def profile_paths(profile_name: str) -> dict[str, Path]:
    profile = cycle.PROFILES[profile_name]
    label = profile.label
    return {
        "discovery": EXPORTS_DIR / f"dynamic_universe_{label}_discovery.csv",
        "seed_context": EXPORTS_DIR / f"dynamic_universe_{label}_seed_context.csv",
        "compat": EXPORTS_DIR / f"dynamic_universe_{label}_compat_scan.csv",
        "single": EXPORTS_DIR / f"dynamic_universe_{label}_single_additions.csv",
        "combo": EXPORTS_DIR / f"dynamic_universe_{label}_combo_additions.csv",
        "report": REPORTS_DIR / f"DYNAMIC_UNIVERSE_{label.upper()}.md",
    }


def run_profiles(profile_names: Iterable[str], keywords: list[str]) -> dict[str, dict[str, Path]]:
    outputs: dict[str, dict[str, Path]] = {}
    for profile_name in profile_names:
        t0 = time.time()
        print(f"[dynamic-db] start profile={profile_name}")
        outputs[profile_name] = cycle.run_profile(cycle.PROFILES[profile_name], keywords)
        dt = time.time() - t0
        print(f"[dynamic-db] done profile={profile_name} elapsed_sec={dt:.1f}")
    return {k: {kk: Path(vv) for kk, vv in v.items()} for k, v in outputs.items()}


def load_profiles(profile_names: Iterable[str]) -> dict[str, dict[str, Path]]:
    return {profile_name: profile_paths(profile_name) for profile_name in profile_names}


def merge_profile_data(profile_name: str, paths: dict[str, Path]) -> pd.DataFrame:
    discovery = read_optional_csv(paths["discovery"])
    compat = read_optional_csv(paths["compat"])
    single = read_optional_csv(paths["single"])
    if discovery.empty:
        return pd.DataFrame()

    out = discovery.copy()
    out["profile"] = profile_name

    if not compat.empty:
        compat_cols = safe_columns(
            compat,
            [
                "ticker",
                "recent_score",
                "recent_r63",
                "recent_r126",
                "recent_r252",
                "scan_algo_fit",
                "scan_algo_compat_score",
                "scan_algo_compat_score_v2",
                "scan_latest_rank_if_added",
                "scan_days_top15_if_added",
                "scan_days_top5_if_added",
                "scan_top5_share",
                "scan_rel_recent_score",
                "scan_rr_score",
                "scan_breakout252_component",
                "scan_trend200_component",
                "scan_candidate_track",
            ],
        )
        out = out.merge(
            compat_cols,
            on="ticker",
            how="left",
        )
    if not single.empty:
        single_cols = safe_columns(
            single,
            [
                "name",
                "recommendation",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
                "full_delta_sharpe",
                "oos_delta_sharpe",
                "full_delta_maxdd_pct",
                "oos_delta_maxdd_pct",
                "backtestable_now",
            ],
        )
        out = out.merge(
            single_cols.rename(columns={"name": "ticker"}),
            on="ticker",
            how="left",
        )
    return out


def recommendation_rank(value: str) -> int:
    return {"add": 3, "watch": 2, "reject": 1}.get(str(value or ""), 0)


def safe_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fit_rank(value: str) -> int:
    return {"high": 3, "medium": 2, "low": 1, "weak": 0}.get(str(value or ""), 0)


def conviction_score(row: pd.Series) -> float:
    rec = str(row.get("recommendation") or "")
    algo_fit = str(row.get("scan_algo_fit") or "")
    full_roi = safe_float(row.get("full_delta_roi_pct"))
    oos_roi = safe_float(row.get("oos_delta_roi_pct"))
    oos_sharpe = safe_float(row.get("oos_delta_sharpe"))
    oos_maxdd = safe_float(row.get("oos_delta_maxdd_pct"))
    compat = max(
        0.0,
        safe_float(
            row.get("scan_algo_compat_score_v2"),
            safe_float(row.get("scan_algo_compat_score")),
        ),
    )
    recent = max(0.0, safe_float(row.get("recent_score")))
    source_type_count = min(6, safe_int(row.get("source_type_count")))
    profile_count = min(3, max(1, safe_int(row.get("profile_count"), default=1)))
    high_fit_profiles = min(3, safe_int(row.get("high_fit_profile_count")))
    positive_profiles = min(3, safe_int(row.get("positive_profile_count")))
    backtestable = safe_int(row.get("backtestable_now"))
    candidate_track = str(row.get("scan_candidate_track") or "")
    top5_share = max(0.0, safe_float(row.get("scan_top5_share")))
    breakout252 = max(0.0, safe_float(row.get("scan_breakout252_component"), 0.5))
    trend200 = max(0.0, safe_float(row.get("scan_trend200_component"), 0.5))
    rel_recent = max(0.0, safe_float(row.get("scan_rel_recent_score")))
    rr_score = max(0.0, safe_float(row.get("scan_rr_score")))
    broker_tradeable = bool(row.get("broker_tradeable", True))

    score = (
        0.75 * compat
        + 1.25 * min(recent, 3.5)
        + 0.90 * fit_rank(algo_fit)
        + 0.45 * source_type_count
        + 0.60 * profile_count
        + 0.45 * high_fit_profiles
        + 0.40 * positive_profiles
        + 0.30 * backtestable
        + 1.20 * top5_share
        + 0.35 * breakout252
        + 0.35 * trend200
        + 0.15 * min(rel_recent, 2.0)
        + 0.12 * min(rr_score, 3.0)
    )
    if rec == "add":
        score += 1.00
    elif rec == "watch":
        score += 0.50
    elif rec == "reject":
        score -= 2.50
    if candidate_track == "persistent_leader":
        score += 0.60
    elif candidate_track == "emerging_leader":
        score += 0.35
    if rec == "watch" and full_roi < 0:
        score -= 1.25
    if full_roi > 0:
        score += 0.40
    if oos_roi > 0:
        score += 0.90
    elif oos_roi >= 0 and full_roi > 0:
        score += 0.20
    if oos_sharpe >= 0:
        score += 0.50
    elif oos_sharpe > -0.02:
        score += 0.20
    if oos_maxdd <= -1.0:
        score -= 0.40
    if not broker_tradeable and str(row.get("candidate_status") or "") == "new":
        score -= 4.00
    return round(score, 4)


def classify_dynamic_status(row: pd.Series) -> str:
    rec = str(row.get("recommendation") or "")
    algo_fit = str(row.get("scan_algo_fit") or "")
    full_roi = safe_float(row.get("full_delta_roi_pct"))
    oos_roi = safe_float(row.get("oos_delta_roi_pct"))
    oos_sharpe = safe_float(row.get("oos_delta_sharpe"))
    oos_maxdd = safe_float(row.get("oos_delta_maxdd_pct"))
    recent = safe_float(row.get("recent_score"))
    profile_count = safe_int(row.get("profile_count"), default=1)
    score = safe_float(row.get("dynamic_conviction_score"))
    broker_tradeable = bool(row.get("broker_tradeable", True))

    if not broker_tradeable and str(row.get("candidate_status") or "") == "new":
        return "reject"

    if rec == "reject":
        return "reject"
    if rec == "add" and algo_fit in {"high", "medium"} and profile_count >= 2 and oos_roi > 0 and oos_sharpe >= 0 and oos_maxdd > -1.0 and score >= 11.0:
        return "approved"
    if (
        rec in {"add", "watch"}
        and algo_fit == "high"
        and profile_count >= 2
        and score >= 10.0
        and full_roi > 0
        and oos_roi >= 0
        and oos_sharpe >= -0.02
    ):
        return "prime_watch"
    if (
        (
            rec in {"add", "watch"}
            or (profile_count >= 2 and recent > 1.0 and full_roi >= 0 and oos_roi >= 0)
        )
        and algo_fit in {"high", "medium"}
        and score >= 8.0
    ):
        return "watch"
    if algo_fit in {"high", "medium"} or score >= 6.0:
        return "review"
    return "discovered"


def status_rank(value: str) -> int:
    return {
        "approved": 5,
        "prime_watch": 4,
        "watch": 3,
        "review": 2,
        "discovered": 1,
        "reject": 0,
    }.get(str(value or ""), -1)


def enrich_broker_tradeability(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ("marketCap", "exchange", "fullExchangeName"):
        if col not in out.columns:
            out[col] = pd.NA
    needs_lookup = (
        out["marketCap"].isna()
        & (
            out["candidate_status"].fillna("").eq("new")
            | out["recommendation"].fillna("").isin(["add", "watch"])
            | out["scan_algo_fit"].fillna("").isin(["high", "medium"])
        )
    )
    for idx, row in out.loc[needs_lookup].iterrows():
        ctx = dud.ticker_context(str(row.get("ticker") or ""))
        if pd.isna(out.at[idx, "marketCap"]):
            out.at[idx, "marketCap"] = ctx.get("marketCap")
        if pd.isna(out.at[idx, "exchange"]):
            out.at[idx, "exchange"] = ctx.get("exchange")
        if pd.isna(out.at[idx, "fullExchangeName"]):
            out.at[idx, "fullExchangeName"] = ctx.get("exchange")

    out["marketCap"] = pd.to_numeric(out["marketCap"], errors="coerce")
    out["broker_tradeable"] = ~(
        out["candidate_status"].fillna("").eq("new")
        & out["marketCap"].notna()
        & (out["marketCap"] < BROKER_MIN_MARKET_CAP)
    )
    out["broker_min_market_cap"] = float(BROKER_MIN_MARKET_CAP)
    return out


def aggregate_database(profile_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in profile_frames.values() if not df.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["recommendation_rank"] = combined["recommendation"].map(recommendation_rank).fillna(0)
    combined["scan_algo_compat_score"] = pd.to_numeric(combined["scan_algo_compat_score"], errors="coerce")
    if "scan_algo_compat_score_v2" not in combined.columns:
        combined["scan_algo_compat_score_v2"] = combined["scan_algo_compat_score"]
    combined["scan_algo_compat_score_v2"] = pd.to_numeric(combined["scan_algo_compat_score_v2"], errors="coerce")
    combined["recent_score"] = pd.to_numeric(combined["recent_score"], errors="coerce")
    combined["priority_score"] = pd.to_numeric(combined["priority_score"], errors="coerce")

    rows = []
    for ticker, grp in combined.groupby("ticker", dropna=False):
        grp_sorted = grp.sort_values(
            ["recommendation_rank", "scan_algo_compat_score_v2", "scan_algo_compat_score", "recent_score", "priority_score"],
            ascending=[False, False, False, False, False],
        )
        best = grp_sorted.iloc[0].copy()
        best["source_profiles"] = "|".join(sorted(set(grp["profile"].dropna().astype(str))))
        best["profile_count"] = int(grp["profile"].dropna().astype(str).nunique())
        best["high_fit_profile_count"] = int((grp["scan_algo_fit"].fillna("").astype(str) == "high").sum())
        best["positive_profile_count"] = int(grp["recommendation"].fillna("").astype(str).isin(["add", "watch"]).sum())
        best["dynamic_conviction_score"] = conviction_score(best)
        best["dynamic_status"] = classify_dynamic_status(best)
        best["dynamic_status_rank"] = status_rank(best["dynamic_status"])
        rows.append(best)

    out = pd.DataFrame(rows).sort_values(
        ["dynamic_status_rank", "dynamic_conviction_score", "recommendation_rank", "scan_algo_compat_score_v2", "scan_algo_compat_score", "recent_score", "priority_score"],
        ascending=[False, False, False, False, False, False, False],
    )
    out = enrich_broker_tradeability(out)
    out["dynamic_conviction_score"] = out.apply(conviction_score, axis=1)
    out["dynamic_status"] = out.apply(classify_dynamic_status, axis=1)
    out["dynamic_status_rank"] = out["dynamic_status"].map(status_rank).fillna(-1)
    out = out.sort_values(
        ["dynamic_status_rank", "dynamic_conviction_score", "recommendation_rank", "scan_algo_compat_score_v2", "scan_algo_compat_score", "recent_score", "priority_score"],
        ascending=[False, False, False, False, False, False, False],
    )
    out["as_of"] = str(date.today())
    return out


def write_database_outputs(db: pd.DataFrame) -> dict[str, Path]:
    current_path = DATA_DIR / "dynamic_universe_current.csv"
    approved_path = DATA_DIR / "dynamic_universe_approved_additions.csv"
    summary_path = DATA_DIR / "dynamic_universe_summary.md"
    snapshot_path = HISTORY_DIR / f"dynamic_universe_snapshot_{date.today().isoformat()}.csv"
    selected_adds_path = DATA_DIR / "dynamic_universe_selected_additions.csv"
    selected_dems_path = DATA_DIR / "dynamic_universe_selected_demotions.csv"

    db.to_csv(current_path, index=False)
    db.to_csv(snapshot_path, index=False)

    approved = db.loc[db["dynamic_status"] == "approved", ["ticker"]].drop_duplicates().sort_values("ticker")
    if approved.empty:
        approved = pd.DataFrame({"ticker": []})
    approved.to_csv(approved_path, index=False)

    selected_adds = read_optional_csv(selected_adds_path)
    selected_dems = read_optional_csv(selected_dems_path)

    lines = [
        "# Dynamic Universe Summary",
        "",
        f"- as_of: `{date.today().isoformat()}`",
        f"- total rows: `{len(db)}`",
        f"- approved: `{int((db['dynamic_status'] == 'approved').sum())}`",
        f"- prime_watch: `{int((db['dynamic_status'] == 'prime_watch').sum())}`",
        f"- watch: `{int((db['dynamic_status'] == 'watch').sum())}`",
        f"- review: `{int((db['dynamic_status'] == 'review').sum())}`",
        f"- reject: `{int((db['dynamic_status'] == 'reject').sum())}`",
        f"- selected additions live: `{len(selected_adds) if not selected_adds.empty else 0}`",
        f"- selected demotions live: `{len(selected_dems) if not selected_dems.empty else 0}`",
        "",
        "## Approved",
        "",
        approved.head(50).to_string(index=False) if not approved.empty else "(none)",
        "",
        "## Prime watch",
        "",
        db.loc[db["dynamic_status"] == "prime_watch", [
            "ticker",
            "source_profiles",
            "profile_count",
            "scan_algo_fit",
            "scan_algo_compat_score_v2",
            "recent_score",
            "dynamic_conviction_score",
            "recommendation",
            "full_delta_roi_pct",
            "oos_delta_roi_pct",
        ]].head(20).to_string(index=False) if not db.empty else "(none)",
        "",
        "## Selected live overlay",
        "",
        "Selected additions:",
        selected_adds.head(50).to_string(index=False) if not selected_adds.empty else "(none)",
        "",
        "Selected demotions:",
        selected_dems.head(50).to_string(index=False) if not selected_dems.empty else "(none)",
        "",
        "## Top watch/review",
        "",
        db.loc[db["dynamic_status"].isin(["watch", "review"]), [
            "ticker",
            "dynamic_status",
            "source_profiles",
            "profile_count",
            "scan_algo_fit",
            "scan_algo_compat_score_v2",
            "recent_score",
            "dynamic_conviction_score",
            "recommendation",
            "full_delta_roi_pct",
            "oos_delta_roi_pct",
        ]].head(30).to_string(index=False) if not db.empty else "(none)",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "current": current_path,
        "approved": approved_path,
        "summary": summary_path,
        "snapshot": snapshot_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the persisted dynamic universe database.")
    parser.add_argument("--profiles", default="targeted_current,broad_focus,broad_diversified", help="Comma-separated cycle profiles to aggregate.")
    parser.add_argument("--keywords", default="", help="Optional comma-separated keywords passed to the profiles when running.")
    parser.add_argument("--skip-cycle", action="store_true", help="Use existing cycle outputs instead of rerunning the profiles.")
    args = parser.parse_args()

    profile_names = [x.strip() for x in args.profiles.split(",") if x.strip()]
    keywords = [x.strip() for x in args.keywords.split(",") if x.strip()]

    if args.skip_cycle:
        paths = load_profiles(profile_names)
    else:
        paths = run_profiles(profile_names, keywords)

    profile_frames = {profile_name: merge_profile_data(profile_name, profile_paths) for profile_name, profile_paths in paths.items()}
    db = aggregate_database(profile_frames)
    outputs = write_database_outputs(db)

    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
