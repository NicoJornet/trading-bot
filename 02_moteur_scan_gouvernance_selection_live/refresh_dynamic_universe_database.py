from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import dynamic_universe_discovery as dud
import dynamic_universe_governance as dug
import run_dynamic_universe_cycle as cycle


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "dynamic_universe"
HISTORY_DIR = DATA_DIR / "history"
REPORTS_DIR = ROOT / "research" / "reports"
EXPORTS_DIR = ROOT / "research" / "exports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

BROKER_PROFILE = "xtb"
BROKER_MIN_MARKET_CAP = 300_000_000
HISTORY_LOOKBACK = 6
FLOAT_ZERO_EPS = 1e-9
PROMOTION_ADD_STUDY_PATHS = [
    EXPORTS_DIR / "dynamic_universe_targeted_watch_adds.csv",
    EXPORTS_DIR / "dynamic_universe_integration_adds_184.csv",
    EXPORTS_DIR / "dynamic_universe_new_families_adds_184.csv",
    EXPORTS_DIR / "dynamic_universe_gap_fill_adds_184.csv",
]
PROMOTION_ADD_WF_PATHS = [
    EXPORTS_DIR / "dynamic_universe_targeted_watch_adds_walkforward.csv",
    EXPORTS_DIR / "dynamic_universe_integration_adds_walkforward_184.csv",
    EXPORTS_DIR / "dynamic_universe_new_families_adds_walkforward_184.csv",
    EXPORTS_DIR / "dynamic_universe_gap_fill_adds_walkforward_184.csv",
]
RECENT_VALIDATION_PATH = EXPORTS_DIR / "recent_validation_hypothesis_candidates_184.csv"
RETRO_MISSING_UNIVERSE_PATH = EXPORTS_DIR / "scan_algo_missing_universe_20260406.csv"
RETRO_MISSING_SELECTION_PATH = EXPORTS_DIR / "scan_algo_missing_selection_20260406.csv"
QUALITY_FORWARD_MONITOR_PATH = DATA_DIR / "dynamic_universe_quality_compounder_forward_monitor.csv"
QUALITY_FORWARD_REPORT_PATH = DATA_DIR / "dynamic_universe_quality_compounder_forward_monitor.md"
MARKET_STRUCTURE_LATEST_PATH = ROOT / "data" / "context" / "market_structure_latest.csv"
TAXONOMY_PIT_LATEST_PATH = ROOT / "data" / "context" / "taxonomy_point_in_time_latest.csv"
FX_REFERENCE_OHLCV_PATH = ROOT / "data" / "benchmarks" / "fx_reference_ohlcv.csv"
LISTING_CORP_METADATA_PATH = ROOT / "data" / "extracts" / "listing_corporate_metadata.csv"
EARNINGS_EVENTS_LATEST_PATH = ROOT / "data" / "earnings" / "earnings_events_latest.csv"
CORE_RANK_BRIDGE_CONTEXT_MIN = 0.85

PERSISTENCE_FILL_COLUMNS = {
    "recommendation": "last_recommendation",
    "scan_algo_fit": "last_scan_algo_fit",
    "scan_algo_compat_score": "last_scan_algo_compat_score",
    "scan_algo_compat_score_v2": "last_scan_algo_compat_score_v2",
    "scan_early_leader_score": "last_scan_early_leader_score",
    "scan_early_leader_fit": "last_scan_early_leader_fit",
    "scan_quality_compounder_score": "last_scan_quality_compounder_score",
    "scan_quality_compounder_fit": "last_scan_quality_compounder_fit",
    "scan_hot_candidate_score": "last_scan_hot_candidate_score",
    "scan_hot_archetype": "last_scan_hot_archetype",
    "recent_score": "last_recent_score",
    "recent_r63": "last_recent_r63",
    "recent_r126": "last_recent_r126",
    "recent_r252": "last_recent_r252",
    "scan_top5_share": "last_scan_top5_share",
    "scan_rel_recent_score": "last_scan_rel_recent_score",
    "scan_rr_score": "last_scan_rr_score",
    "scan_sector_rel_recent_score": "last_scan_sector_rel_recent_score",
    "scan_sector_rr_score": "last_scan_sector_rr_score",
    "scan_breakout252_component": "last_scan_breakout252_component",
    "scan_trend200_component": "last_scan_trend200_component",
    "scan_dist_sma220": "last_scan_dist_sma220",
    "scan_dd60": "last_scan_dd60",
    "scan_r21": "last_scan_r21",
    "scan_overnight_return_1d": "last_scan_overnight_return_1d",
    "scan_intraday_return_1d": "last_scan_intraday_return_1d",
    "scan_gap_pct": "last_scan_gap_pct",
    "scan_gap_zscore_20": "last_scan_gap_zscore_20",
    "scan_gap_fill_share": "last_scan_gap_fill_share",
    "scan_rel_volume_20": "last_scan_rel_volume_20",
    "scan_volume_zscore_20": "last_scan_volume_zscore_20",
    "scan_dollar_volume_zscore_20": "last_scan_dollar_volume_zscore_20",
    "scan_close_in_range": "last_scan_close_in_range",
    "scan_efficiency_ratio_20": "last_scan_efficiency_ratio_20",
    "scan_efficiency_ratio_60": "last_scan_efficiency_ratio_60",
    "scan_corr_63_spy": "last_scan_corr_63_spy",
    "scan_downside_beta_63_spy": "last_scan_downside_beta_63_spy",
    "scan_sector_corr_63": "last_scan_sector_corr_63",
    "scan_sector_beta_63": "last_scan_sector_beta_63",
    "scan_entry_heat_flag": "last_scan_entry_heat_flag",
    "scan_pullback_quality": "last_scan_pullback_quality",
    "scan_candidate_track": "last_scan_candidate_track",
    "full_delta_roi_pct": "last_full_delta_roi_pct",
    "oos_delta_roi_pct": "last_oos_delta_roi_pct",
    "full_delta_sharpe": "last_full_delta_sharpe",
    "oos_delta_sharpe": "last_oos_delta_sharpe",
    "full_delta_maxdd_pct": "last_full_delta_maxdd_pct",
    "oos_delta_maxdd_pct": "last_oos_delta_maxdd_pct",
    "backtestable_now": "last_backtestable_now",
    "add_wf_available": "last_add_wf_available",
    "add_mean_delta_roi_2017_2025": "last_add_mean_delta_roi_2017_2025",
    "add_mean_delta_sharpe_2017_2025": "last_add_mean_delta_sharpe_2017_2025",
    "add_mean_delta_maxdd_2017_2025": "last_add_mean_delta_maxdd_2017_2025",
    "add_roi_wins_2017_2025": "last_add_roi_wins_2017_2025",
    "add_sharpe_wins_2017_2025": "last_add_sharpe_wins_2017_2025",
    "add_maxdd_wins_2017_2025": "last_add_maxdd_wins_2017_2025",
    "add_delta_roi_2026_ytd": "last_add_delta_roi_2026_ytd",
    "add_delta_sharpe_2026_ytd": "last_add_delta_sharpe_2026_ytd",
    "add_delta_maxdd_2026_ytd": "last_add_delta_maxdd_2026_ytd",
}


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def series_or_default(df: pd.DataFrame, column: str, default) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def load_market_structure_latest() -> pd.DataFrame:
    df = read_optional_csv(MARKET_STRUCTURE_LATEST_PATH)
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    rename_map = {
        "market_cap_current": "pit_market_cap_current",
        "shares_outstanding_current": "pit_shares_outstanding_current",
        "float_shares_current": "pit_float_shares_current",
        "implied_free_float_ratio_current": "pit_free_float_ratio_current",
        "market_cap_to_adv20": "pit_market_cap_to_adv20",
        "coverage_ratio": "pit_market_structure_coverage_ratio",
        "currency": "pit_market_structure_currency",
    }
    keep = ["ticker", *rename_map.keys()]
    available = [col for col in keep if col in out.columns]
    out = out[available].rename(columns=rename_map)
    return out.drop_duplicates("ticker", keep="last")


def load_taxonomy_pit_latest() -> pd.DataFrame:
    df = read_optional_csv(TAXONOMY_PIT_LATEST_PATH)
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    rename_map = {
        "as_of": "pit_taxonomy_as_of",
        "cluster_key": "pit_cluster_key",
        "cluster_source": "pit_cluster_source",
    }
    keep = ["ticker", "sectorKey", "industryKey", *rename_map.keys()]
    available = [col for col in keep if col in out.columns]
    out = out[available].rename(columns=rename_map)
    if "sectorKey" in out.columns:
        out = out.rename(columns={"sectorKey": "pit_sector_key"})
    if "industryKey" in out.columns:
        out = out.rename(columns={"industryKey": "pit_industry_key"})
    return out.drop_duplicates("ticker", keep="last")


def load_fx_latest_context() -> pd.DataFrame:
    df = read_optional_csv(FX_REFERENCE_OHLCV_PATH)
    if df.empty or "currency" not in df.columns or "date" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["currency"] = out["currency"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["currency", "date"]).sort_values(["currency", "date"], kind="mergesort")
    out = out.drop_duplicates("currency", keep="last")
    rename_map = {
        "date": "pit_fx_as_of",
        "fx_to_usd": "pit_fx_to_usd",
        "fx_to_eur": "pit_fx_to_eur",
        "return_21d": "pit_fx_return_21d",
        "return_63d": "pit_fx_return_63d",
        "above_sma200": "pit_fx_above_sma200",
    }
    keep = ["currency", *rename_map.keys()]
    available = [col for col in keep if col in out.columns]
    out = out[available].rename(columns=rename_map)
    return out


def load_listing_corporate_metadata() -> pd.DataFrame:
    df = read_optional_csv(LISTING_CORP_METADATA_PATH)
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    rename_map = {
        "listing_age_days": "pit_listing_age_days",
        "recent_split_365d": "pit_recent_split_365d",
        "recent_dividend_365d": "pit_recent_dividend_365d",
        "split_count_total": "pit_split_count_total",
        "dividend_count_total": "pit_dividend_count_total",
        "split_count_5y": "pit_split_count_5y",
        "dividend_count_5y": "pit_dividend_count_5y",
        "last_split_date": "pit_last_split_date",
        "last_dividend_date": "pit_last_dividend_date",
        "first_ohlcv_date": "pit_first_ohlcv_date",
        "last_ohlcv_date": "pit_last_ohlcv_date",
        "bars_total": "pit_bars_total",
    }
    keep = ["ticker", *rename_map.keys()]
    available = [col for col in keep if col in out.columns]
    out = out[available].rename(columns=rename_map)
    return out.drop_duplicates("ticker", keep="last")


def load_earnings_event_summary() -> pd.DataFrame:
    df = read_optional_csv(EARNINGS_EVENTS_LATEST_PATH)
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["earnings_date"] = pd.to_datetime(out.get("earnings_date"), errors="coerce")
    if "as_of_snapshot" in out.columns:
        out["as_of_snapshot"] = pd.to_datetime(out["as_of_snapshot"], errors="coerce")
    summary = (
        out.groupby("ticker", dropna=False)
        .agg(
            pit_earnings_event_rows=("ticker", "size"),
            pit_earnings_next_seen_count=("event_role", lambda s: int((s.astype(str) == "next_seen").sum())),
            pit_earnings_last_seen_count=("event_role", lambda s: int((s.astype(str) == "last_seen").sum())),
            pit_earnings_first_event_date=("earnings_date", "min"),
            pit_earnings_last_event_date=("earnings_date", "max"),
            pit_earnings_snapshot_count=("as_of_snapshot", "nunique") if "as_of_snapshot" in out.columns else ("ticker", "size"),
        )
        .reset_index()
    )
    return summary


def enrich_forward_data_blocks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return df
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()

    market_structure = load_market_structure_latest()
    if not market_structure.empty:
        out = out.merge(market_structure, on="ticker", how="left")

    taxonomy = load_taxonomy_pit_latest()
    if not taxonomy.empty:
        out = out.merge(taxonomy, on="ticker", how="left")

    listing = load_listing_corporate_metadata()
    if not listing.empty:
        out = out.merge(listing, on="ticker", how="left")

    earnings_events = load_earnings_event_summary()
    if not earnings_events.empty:
        out = out.merge(earnings_events, on="ticker", how="left")

    if "currency" in out.columns:
        fx = load_fx_latest_context()
        if not fx.empty:
            out["currency"] = out["currency"].fillna("").astype(str).str.upper()
            out = out.merge(fx, on="currency", how="left")

    for col in (
        "pit_market_cap_current",
        "pit_shares_outstanding_current",
        "pit_float_shares_current",
        "pit_free_float_ratio_current",
        "pit_market_cap_to_adv20",
        "pit_market_structure_coverage_ratio",
        "pit_listing_age_days",
        "pit_bars_total",
        "pit_fx_to_usd",
        "pit_fx_to_eur",
        "pit_fx_return_21d",
        "pit_fx_return_63d",
        "pit_earnings_event_rows",
        "pit_earnings_next_seen_count",
        "pit_earnings_last_seen_count",
        "pit_earnings_snapshot_count",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in (
        "pit_recent_split_365d",
        "pit_recent_dividend_365d",
        "pit_split_count_total",
        "pit_dividend_count_total",
        "pit_split_count_5y",
        "pit_dividend_count_5y",
        "pit_fx_above_sma200",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    free_float_series = pd.to_numeric(series_or_default(out, "pit_free_float_ratio_current", 1.0), errors="coerce").fillna(1.0)
    listing_age_series = pd.to_numeric(series_or_default(out, "pit_listing_age_days", 10_000), errors="coerce").fillna(10_000)
    fx_return_21d_series = pd.to_numeric(series_or_default(out, "pit_fx_return_21d", 0.0), errors="coerce").fillna(0.0)
    fx_above_sma200_series = pd.to_numeric(series_or_default(out, "pit_fx_above_sma200", 1), errors="coerce").fillna(1).astype(int)
    market_structure_coverage_series = pd.to_numeric(series_or_default(out, "pit_market_structure_coverage_ratio", 0.0), errors="coerce").fillna(0.0)
    earnings_snapshot_series = pd.to_numeric(series_or_default(out, "pit_earnings_snapshot_count", 0.0), errors="coerce").fillna(0.0)
    fx_to_usd_series = pd.to_numeric(series_or_default(out, "pit_fx_to_usd", np.nan), errors="coerce")
    cluster_key_present = series_or_default(out, "pit_cluster_key", "").fillna("").astype(str).ne("").astype(float)

    out["pit_low_float_flag"] = (free_float_series < 0.40).astype(int)
    out["pit_micro_float_flag"] = (free_float_series < 0.15).astype(int)
    out["pit_new_listing_flag"] = (listing_age_series < 756).astype(int)
    out["pit_fx_headwind_flag"] = (
        out.get("currency", pd.Series("", index=out.index)).fillna("").astype(str).str.upper().ne("USD")
        & (fx_return_21d_series < -0.03)
        & (fx_above_sma200_series == 0)
    ).astype(int)
    out["pit_data_context_score"] = (
        0.25 * market_structure_coverage_series.clip(0, 1)
        + 0.20 * cluster_key_present
        + 0.15 * earnings_snapshot_series.clip(0, 4) / 4.0
        + 0.15 * listing_age_series.clip(0, 2520) / 2520.0
        + 0.25 * fx_to_usd_series.notna().astype(float)
    ).round(4)

    for col in (
        "pit_taxonomy_as_of",
        "pit_fx_as_of",
        "pit_last_split_date",
        "pit_last_dividend_date",
        "pit_first_ohlcv_date",
        "pit_last_ohlcv_date",
        "pit_earnings_first_event_date",
        "pit_earnings_last_event_date",
    ):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")

    return out


def load_core_rank_bridge_frame() -> pd.DataFrame:
    try:
        _engine, _cfg_doc, cfg, _pp, base_prices = dud.load_setup()
        feats = dud.compute_universe_features(base_prices.close, cfg)
    except Exception:
        return pd.DataFrame()
    if feats.empty or "ticker" not in feats.columns:
        return pd.DataFrame()
    out = feats.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out = out.rename(
        columns={
            "latest_rank": "core_latest_rank",
            "latest_score": "core_latest_score",
            "latest_r63": "core_latest_r63",
            "latest_r252": "core_latest_r252",
            "days_top15_trend": "core_days_top15_trend",
            "days_top5_trend": "core_days_top5_trend",
        }
    )
    rank = pd.to_numeric(out.get("core_latest_rank"), errors="coerce")
    score = pd.to_numeric(out.get("core_latest_score"), errors="coerce")
    r63 = pd.to_numeric(out.get("core_latest_r63"), errors="coerce")
    r252 = pd.to_numeric(out.get("core_latest_r252"), errors="coerce")
    out["core_rank_bridge_fit"] = np.select(
        [
            rank.le(15) & score.gt(0.0) & r63.gt(0.0) & r252.gt(0.0),
            rank.le(30) & score.gt(0.0) & r252.gt(0.0),
            rank.le(50) & score.gt(0.0),
        ],
        ["high", "medium", "low"],
        default="weak",
    )
    rank_bonus = np.where(rank.notna(), np.clip((31.0 - rank) / 10.0, 0.0, 2.5), 0.0)
    trend_bonus = np.clip(pd.to_numeric(out.get("core_days_top15_trend"), errors="coerce").fillna(0.0) / 40.0, 0.0, 1.5)
    out["core_rank_bridge_score"] = np.round(rank_bonus + trend_bonus, 4)
    out["core_top30_flag"] = rank.le(30).fillna(False).astype(int)
    out["core_top15_flag"] = rank.le(15).fillna(False).astype(int)
    return out[
        [
            "ticker",
            "core_latest_rank",
            "core_latest_score",
            "core_latest_r63",
            "core_latest_r252",
            "core_days_top15_trend",
            "core_days_top5_trend",
            "core_rank_bridge_fit",
            "core_rank_bridge_score",
            "core_top30_flag",
            "core_top15_flag",
        ]
    ].drop_duplicates("ticker", keep="last")


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
        outputs[profile_name] = cycle.run_profile(cycle.PROFILES[profile_name], keywords)
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
                "scan_early_leader_score",
                "scan_early_leader_fit",
                "scan_quality_compounder_score",
                "scan_quality_compounder_fit",
                "scan_hot_candidate_score",
                "scan_hot_archetype",
                "scan_latest_rank_if_added",
                "scan_days_top15_if_added",
                "scan_days_top5_if_added",
                "scan_top5_share",
                "scan_rel_recent_score",
                "scan_rr_score",
                "scan_sector_benchmark",
                "scan_sector_rel_recent_score",
                "scan_sector_rr_score",
                "scan_breakout252_component",
                "scan_trend200_component",
                "scan_dist_sma220",
                "scan_dd60",
                "scan_r21",
                "scan_overnight_return_1d",
                "scan_intraday_return_1d",
                "scan_gap_pct",
                "scan_gap_zscore_20",
                "scan_gap_fill_share",
                "scan_rel_volume_20",
                "scan_volume_zscore_20",
                "scan_dollar_volume_zscore_20",
                "scan_close_in_range",
                "scan_efficiency_ratio_20",
                "scan_efficiency_ratio_60",
                "scan_corr_63_spy",
                "scan_downside_beta_63_spy",
                "scan_sector_corr_63",
                "scan_sector_beta_63",
                "scan_entry_heat_flag",
                "scan_pullback_quality",
                "scan_days_to_next_earnings",
                "scan_earnings_blackout_5d",
                "scan_earnings_blackout_10d",
                "scan_post_earnings_5d",
                "scan_post_earnings_10d",
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


def safe_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default
    text = str(value)
    return text if text and text.lower() != "nan" else default


def clip_small_float(value: object, eps: float = FLOAT_ZERO_EPS) -> float:
    out = safe_float(value)
    return 0.0 if abs(out) < eps else out


def fit_rank(value: str) -> int:
    return {"high": 3, "medium": 2, "low": 1, "weak": 0}.get(str(value or ""), 0)


def conviction_score(row: pd.Series) -> float:
    rec = str(row.get("recommendation") or "")
    add_validation_status = str(row.get("add_validation_status") or "")
    algo_fit = str(row.get("scan_algo_fit") or "")
    quality_fit = str(row.get("scan_quality_compounder_fit") or "")
    full_roi = clip_small_float(row.get("full_delta_roi_pct"))
    oos_roi = clip_small_float(row.get("oos_delta_roi_pct"))
    oos_sharpe = clip_small_float(row.get("oos_delta_sharpe"))
    oos_maxdd = clip_small_float(row.get("oos_delta_maxdd_pct"))
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
    entry_heat_flag = safe_int(row.get("scan_entry_heat_flag"))
    pullback_quality = max(0.0, safe_float(row.get("scan_pullback_quality")))
    quality_compounder_score = max(0.0, safe_float(row.get("scan_quality_compounder_score")))
    rel_recent = max(0.0, safe_float(row.get("scan_rel_recent_score")))
    rr_score = max(0.0, safe_float(row.get("scan_rr_score")))
    broker_tradeable = bool(row.get("broker_tradeable", True))
    broker_exchange_supported = bool(row.get("broker_exchange_supported", True))
    data_quality_flag = max(0, safe_int(row.get("data_quality_flag")))
    portfolio_crowding = max(0.0, safe_float(row.get("portfolio_crowding_score")))
    portfolio_diversification = max(0.0, safe_float(row.get("portfolio_diversification_score")))
    core_bridge_fit = str(row.get("core_rank_bridge_fit") or "")
    core_bridge_score = max(0.0, safe_float(row.get("core_rank_bridge_score")))

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
        + 0.20 * pullback_quality
        + 0.18 * min(quality_compounder_score, 9.0)
        + 0.15 * min(rel_recent, 2.0)
        + 0.12 * min(rr_score, 3.0)
        + 0.55 * min(core_bridge_score, 3.0)
    )
    if rec == "add":
        score += 1.00
    elif rec == "watch":
        score += 0.50
    elif rec == "reject":
        score -= 2.50
    if add_validation_status == "approved_add":
        score += 0.75
    elif add_validation_status == "watch_add":
        score += 0.20
    elif add_validation_status == "reject_add":
        score -= 1.50
    if candidate_track == "persistent_leader":
        score += 0.60
    elif candidate_track == "emerging_leader":
        score += 0.35
    elif candidate_track == "quality_compounder":
        score += 0.55
    if quality_fit == "high":
        score += 0.70
    elif quality_fit == "medium":
        score += 0.35
    if core_bridge_fit == "high":
        score += 1.10
    elif core_bridge_fit == "medium":
        score += 0.55
    elif core_bridge_fit == "low":
        score += 0.20
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
    if entry_heat_flag:
        score -= 0.60
    score += 0.70 * min(portfolio_diversification, 1.25)
    score -= 0.85 * min(portfolio_crowding, 1.50)
    score -= 0.40 * min(4, data_quality_flag)
    if not broker_exchange_supported and str(row.get("candidate_status") or "") == "new":
        score -= 0.35 if BROKER_PROFILE == "xtb" else 0.80
    if not broker_tradeable and str(row.get("candidate_status") or "") == "new":
        score -= 4.00
    return round(score, 4)


def classify_dynamic_status(row: pd.Series) -> str:
    rec = str(row.get("recommendation") or "")
    add_validation_status = str(row.get("add_validation_status") or "")
    algo_fit = str(row.get("scan_algo_fit") or "")
    quality_fit = str(row.get("scan_quality_compounder_fit") or "")
    full_roi = clip_small_float(row.get("full_delta_roi_pct"))
    oos_roi = clip_small_float(row.get("oos_delta_roi_pct"))
    oos_sharpe = clip_small_float(row.get("oos_delta_sharpe"))
    oos_maxdd = clip_small_float(row.get("oos_delta_maxdd_pct"))
    recent = safe_float(row.get("recent_score"))
    profile_count = safe_int(row.get("profile_count"), default=1)
    score = safe_float(row.get("dynamic_conviction_score"))
    broker_tradeable = bool(row.get("broker_tradeable", True))
    data_quality_flag = max(0, safe_int(row.get("data_quality_flag")))
    core_bridge_fit = str(row.get("core_rank_bridge_fit") or "")
    core_bridge_score = safe_float(row.get("core_rank_bridge_score"))
    core_rank = safe_float(row.get("core_latest_rank"), 999.0)
    context_score = safe_float(row.get("pit_data_context_score"))
    low_float_flag = safe_int(row.get("pit_low_float_flag")) > 0
    event_label = str(row.get("scan_event_quality_label") or "")
    entry_zone = str(row.get("entry_timing_zone") or "")

    if not broker_tradeable and str(row.get("candidate_status") or "") == "new":
        return "reject"
    if data_quality_flag >= 4 and str(row.get("candidate_status") or "") == "new":
        return "reject"

    if rec == "reject":
        return "reject"
    if add_validation_status == "reject_add" and full_roi <= 0 and oos_roi <= 0 and score < 18.0:
        return "review"
    strong_like = algo_fit in {"high", "medium"} or quality_fit in {"high", "medium"}
    if rec == "add" and strong_like and profile_count >= 2 and oos_roi > 0 and oos_sharpe >= 0 and oos_maxdd > -1.0 and score >= 11.0:
        return "approved"
    if (
        rec in {"add", "watch"}
        and (algo_fit == "high" or quality_fit == "high")
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
        and strong_like
        and score >= 8.0
    ):
        return "watch"
    if (
        rec not in {"add", "watch", "reject"}
        and core_bridge_fit == "high"
        and core_rank <= 20
        and context_score >= CORE_RANK_BRIDGE_CONTEXT_MIN
        and not low_float_flag
        and event_label not in {"speculative", "risky"}
        and entry_zone not in {"hot_late", "event_risk"}
        and profile_count >= 2
        and core_bridge_score >= 1.2
    ):
        return "watch"
    if (
        rec not in {"add", "watch", "reject"}
        and core_bridge_fit in {"high", "medium"}
        and core_rank <= 30
        and context_score >= CORE_RANK_BRIDGE_CONTEXT_MIN
        and not low_float_flag
        and event_label != "risky"
        and entry_zone != "event_risk"
        and profile_count >= 2
        and core_bridge_score >= 0.8
    ):
        return "review"
    if strong_like or score >= 6.0:
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


def promotion_stage_rank(value: str) -> int:
    return {
        "approved_live": 5,
        "probation_live": 4,
        "targeted_integration": 3,
        "watch_queue": 2,
        "review_queue": 1,
        "reject_queue": 0,
        "blocked_broker": -1,
    }.get(str(value or ""), -2)


def load_recent_snapshot_history(limit: int = HISTORY_LOOKBACK) -> pd.DataFrame:
    snapshots = sorted(HISTORY_DIR.glob("dynamic_universe_snapshot_*.csv"), reverse=True)[:limit]
    frames: list[pd.DataFrame] = []
    for path in snapshots:
        df = read_optional_csv(path)
        if df.empty or "ticker" not in df.columns:
            continue
        out = df.copy()
        out["snapshot_date"] = path.stem.replace("dynamic_universe_snapshot_", "")
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True, sort=False)
    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"], errors="coerce")
    history = history.dropna(subset=["snapshot_date"]).sort_values(["ticker", "snapshot_date"], ascending=[True, False])
    return history


def _history_rank_best(values: pd.Series, rank_fn) -> str:
    best_value = ""
    best_rank = -10_000
    for value in values.fillna("").astype(str):
        rank = rank_fn(value)
        if rank > best_rank:
            best_rank = rank
            best_value = value
    return best_value


def _history_consecutive(records: list[dict[str, object]], predicate) -> int:
    streak = 0
    for record in records:
        if not predicate(record):
            break
        streak += 1
    return streak


def _history_first_valid(grp: pd.DataFrame, column: str):
    if column not in grp.columns:
        return pd.NA
    series = grp[column]
    if series.dtype == "object":
        valid = series[series.notna() & series.astype(str).ne("")]
    else:
        valid = series[series.notna()]
    if valid.empty:
        return pd.NA
    return valid.iloc[0]


def _history_watch_like(record: dict[str, object]) -> bool:
    return status_rank(record.get("dynamic_status")) >= status_rank("watch") or promotion_stage_rank(record.get("promotion_stage")) >= promotion_stage_rank("watch_queue")


def _history_targeted_like(record: dict[str, object]) -> bool:
    return promotion_stage_rank(record.get("promotion_stage")) >= promotion_stage_rank("targeted_integration")


def _history_probation_like(record: dict[str, object]) -> bool:
    return promotion_stage_rank(record.get("promotion_stage")) >= promotion_stage_rank("probation_live")


def _history_constructive_entry(record: dict[str, object]) -> bool:
    entry_zone = str(record.get("entry_timing_zone") or "")
    entry_proto = str(record.get("entry_hot_proto") or "")
    return entry_zone in {"clean", "constructive", "hot_constructive"} or (
        entry_zone == "hot_watch" and entry_proto == "hot_constructive_proto"
    )


def _history_hot_late_entry(record: dict[str, object]) -> bool:
    entry_zone = str(record.get("entry_timing_zone") or "")
    entry_proto = str(record.get("entry_hot_proto") or "")
    return entry_zone == "hot_late" or (entry_zone == "hot_watch" and entry_proto == "hot_late_proto")


def _history_emergence_metrics(records: list[dict[str, object]], snapshot_limit: int) -> dict[str, float]:
    if not records:
        return {
            "history_constructive_scans": 0,
            "history_hot_late_scans": 0,
            "history_avg_support_count": 0.0,
            "history_stage_progression": 0.0,
            "history_emergence_persistence_score": 0.0,
        }

    total = max(1, len(records))
    watch_count = int(sum(1 for record in records if _history_watch_like(record)))
    targeted_count = int(sum(1 for record in records if _history_targeted_like(record)))
    probation_count = int(sum(1 for record in records if _history_probation_like(record)))
    constructive_count = int(sum(1 for record in records if _history_constructive_entry(record)))
    hot_late_count = int(sum(1 for record in records if _history_hot_late_entry(record)))
    avg_support = float(
        sum(
            max(
                safe_float(record.get("profile_count")),
                safe_float(record.get("source_type_count")),
            )
            for record in records
        )
        / total
    )

    latest = records[0]
    oldest = records[-1]
    stage_progression = max(
        0.0,
        float(promotion_stage_rank(latest.get("promotion_stage")) - promotion_stage_rank(oldest.get("promotion_stage")))
        + 0.5 * float(status_rank(latest.get("dynamic_status")) - status_rank(oldest.get("dynamic_status"))),
    )

    presence_rate = min(1.0, total / max(1, snapshot_limit))
    watch_rate = watch_count / total
    targeted_rate = targeted_count / total
    probation_rate = probation_count / total
    constructive_rate = constructive_count / total
    hot_late_rate = hot_late_count / total

    persistence_score = max(
        0.0,
        0.90 * presence_rate
        + 1.10 * watch_rate
        + 1.40 * targeted_rate
        + 1.10 * probation_rate
        + 0.80 * constructive_rate
        + 0.50 * min(1.0, avg_support / 3.0)
        + 0.45 * min(1.0, watch_count / 3.0)
        + 0.75 * min(1.0, targeted_count / 2.0)
        + 0.35 * min(1.0, stage_progression / 3.0)
        - 1.00 * hot_late_rate,
    )

    return {
        "history_constructive_scans": constructive_count,
        "history_hot_late_scans": hot_late_count,
        "history_avg_support_count": round(avg_support, 4),
        "history_stage_progression": round(stage_progression, 4),
        "history_emergence_persistence_score": round(persistence_score, 4),
    }


def build_history_memory(limit: int = HISTORY_LOOKBACK) -> pd.DataFrame:
    history = load_recent_snapshot_history(limit)
    if history.empty:
        return pd.DataFrame()

    for col in (
        "dynamic_status",
        "promotion_stage",
        "recommendation",
        "scan_algo_fit",
        "scan_candidate_track",
        "entry_timing_zone",
        "entry_hot_proto",
    ):
        if col not in history.columns:
            history[col] = ""
    for col in ("profile_count", "source_type_count"):
        if col not in history.columns:
            history[col] = 0.0

    rows: list[dict[str, object]] = []
    for ticker, grp in history.groupby("ticker", dropna=False):
        grp = grp.sort_values("snapshot_date", ascending=False)
        latest = grp.iloc[0]
        records = grp.to_dict("records")
        emergence_metrics = _history_emergence_metrics(records, limit)

        row: dict[str, object] = {
            "ticker": ticker,
            "history_scans_seen": int(len(grp)),
            "history_watch_scans": int(sum(1 for record in records if _history_watch_like(record))),
            "history_targeted_scans": int(sum(1 for record in records if _history_targeted_like(record))),
            "history_probation_scans": int(sum(1 for record in records if _history_probation_like(record))),
            "history_consecutive_watch": int(_history_consecutive(records, _history_watch_like)),
            "history_consecutive_targeted": int(_history_consecutive(records, _history_targeted_like)),
            "history_consecutive_probation": int(_history_consecutive(records, _history_probation_like)),
            "history_best_dynamic_status": _history_rank_best(grp["dynamic_status"], status_rank),
            "history_best_promotion_stage": _history_rank_best(grp["promotion_stage"], promotion_stage_rank),
            "history_best_promotion_score": float(pd.to_numeric(grp.get("promotion_score"), errors="coerce").max()) if "promotion_score" in grp.columns else 0.0,
            "history_best_conviction_score": float(pd.to_numeric(grp.get("dynamic_conviction_score"), errors="coerce").max()) if "dynamic_conviction_score" in grp.columns else 0.0,
        }
        row.update(emergence_metrics)
        for current_col, history_col in PERSISTENCE_FILL_COLUMNS.items():
            row[history_col] = _history_first_valid(grp, current_col)
        rows.append(row)
    return pd.DataFrame(rows)


def merge_history_memory(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    history = build_history_memory()
    if history.empty:
        out = df.copy()
        out["metrics_carried_from_history"] = 0
        return out

    out = df.merge(history, on="ticker", how="left")
    out["metrics_carried_from_history"] = 0
    for current_col, history_col in PERSISTENCE_FILL_COLUMNS.items():
        if history_col not in out.columns:
            continue
        if current_col not in out.columns:
            out[current_col] = out[history_col]
            out["metrics_carried_from_history"] = out["metrics_carried_from_history"].where(out[history_col].isna(), 1)
            continue
        missing = out[current_col].isna()
        if out[current_col].dtype == "object":
            missing = missing | out[current_col].fillna("").astype(str).eq("")
        elif current_col == "backtestable_now":
            missing = missing | pd.to_numeric(out[current_col], errors="coerce").fillna(0.0).eq(0.0)
        fillable = missing & out[history_col].notna()
        if fillable.any():
            out[current_col] = out[current_col].mask(fillable, out[history_col])
            out.loc[fillable, "metrics_carried_from_history"] = 1

    for col in (
        "history_scans_seen",
        "history_watch_scans",
        "history_targeted_scans",
        "history_probation_scans",
        "history_consecutive_watch",
        "history_consecutive_targeted",
        "history_consecutive_probation",
        "history_constructive_scans",
        "history_hot_late_scans",
        "history_avg_support_count",
        "history_stage_progression",
        "history_emergence_persistence_score",
        "history_best_promotion_score",
        "history_best_conviction_score",
    ):
        if col not in out.columns:
            out[col] = 0
    return out


def _ticker_set_from_csv(path: Path) -> set[str]:
    df = read_optional_csv(path)
    if df.empty or "ticker" not in df.columns:
        return set()
    return set(df["ticker"].dropna().astype(str))


def _monitor_table(df: pd.DataFrame, columns: list[str], rows: int = 20) -> str:
    if df.empty:
        return "(none)"
    available = [col for col in columns if col in df.columns]
    if not available:
        return "(none)"
    return df[available].head(rows).to_string(index=False)


def build_quality_compounder_forward_monitor(db: pd.DataFrame, history_limit: int = 60) -> pd.DataFrame:
    if db.empty or "ticker" not in db.columns:
        return pd.DataFrame()

    universe_gap = _ticker_set_from_csv(RETRO_MISSING_UNIVERSE_PATH)
    selection_gap = _ticker_set_from_csv(RETRO_MISSING_SELECTION_PATH)
    tracked_gap = universe_gap | selection_gap

    out = db.copy()
    for col in (
        "scan_quality_compounder_fit",
        "scan_candidate_track",
        "promotion_stage",
        "dynamic_status",
        "validation_lane",
        "governance_role",
        "recent_score",
        "scan_quality_compounder_score",
        "promotion_score",
        "dynamic_conviction_score",
        "scan_event_quality_score",
        "history_emergence_persistence_score",
        "profile_count",
        "pit_cluster_key",
        "pit_free_float_ratio_current",
        "pit_low_float_flag",
        "pit_listing_age_days",
        "pit_fx_return_21d",
        "pit_fx_headwind_flag",
        "pit_earnings_snapshot_count",
        "pit_data_context_score",
    ):
        if col not in out.columns:
            out[col] = pd.NA

    quality_like_mask = (
        out["scan_quality_compounder_fit"].fillna("").astype(str).isin(["high", "medium"])
        | out["scan_candidate_track"].fillna("").astype(str).eq("quality_compounder")
        | out["ticker"].fillna("").astype(str).isin(tracked_gap)
    )
    current_focus = out.loc[quality_like_mask].copy()
    current_focus["ticker"] = current_focus["ticker"].astype(str)

    history = load_recent_snapshot_history(limit=history_limit)
    if not history.empty:
        history["ticker"] = history["ticker"].astype(str)
        for col in (
            "scan_quality_compounder_fit",
            "scan_candidate_track",
            "promotion_stage",
            "dynamic_status",
            "promotion_score",
            "dynamic_conviction_score",
            "scan_quality_compounder_score",
            "scan_event_quality_score",
        ):
            if col not in history.columns:
                history[col] = pd.NA

    focus_tickers = set(current_focus["ticker"].tolist()) | tracked_gap
    if not history.empty:
        focus_tickers |= set(
            history.loc[
                history["scan_quality_compounder_fit"].fillna("").astype(str).isin(["high", "medium"])
                | history["scan_candidate_track"].fillna("").astype(str).eq("quality_compounder"),
                "ticker",
            ].dropna().astype(str).tolist()
        )
        history = history.loc[history["ticker"].isin(focus_tickers)].copy()

    current_by_ticker = (
        current_focus.drop_duplicates("ticker").set_index("ticker")
        if not current_focus.empty
        else pd.DataFrame()
    )

    rows: list[dict[str, object]] = []
    for ticker in sorted(focus_tickers):
        current_row = (
            current_by_ticker.loc[ticker]
            if not current_by_ticker.empty and ticker in current_by_ticker.index
            else None
        )
        hist_grp = (
            history.loc[history["ticker"] == ticker].sort_values("snapshot_date", ascending=False)
            if not history.empty
            else pd.DataFrame()
        )
        latest_hist = hist_grp.iloc[0] if not hist_grp.empty else None

        current_stage = str((current_row.get("promotion_stage") if current_row is not None else "") or "")
        current_status = str((current_row.get("dynamic_status") if current_row is not None else "") or "")
        prior_stage = str((latest_hist.get("promotion_stage") if latest_hist is not None else "") or "")
        prior_status = str((latest_hist.get("dynamic_status") if latest_hist is not None else "") or "")

        current_quality_fit = str(
            (
                current_row.get("scan_quality_compounder_fit")
                if current_row is not None
                else (latest_hist.get("scan_quality_compounder_fit") if latest_hist is not None else "")
            )
            or ""
        )
        current_track = str(
            (
                current_row.get("scan_candidate_track")
                if current_row is not None
                else (latest_hist.get("scan_candidate_track") if latest_hist is not None else "")
            )
            or ""
        )
        current_quality_score = safe_float(
            current_row.get("scan_quality_compounder_score")
            if current_row is not None
            else (latest_hist.get("scan_quality_compounder_score") if latest_hist is not None else 0.0)
        )
        current_recent_score = safe_float(
            current_row.get("recent_score")
            if current_row is not None
            else (latest_hist.get("recent_score") if latest_hist is not None else 0.0)
        )
        current_promotion_score = safe_float(
            current_row.get("promotion_score")
            if current_row is not None
            else (latest_hist.get("promotion_score") if latest_hist is not None else 0.0)
        )
        current_conviction_score = safe_float(
            current_row.get("dynamic_conviction_score")
            if current_row is not None
            else (latest_hist.get("dynamic_conviction_score") if latest_hist is not None else 0.0)
        )
        event_quality_score = safe_float(
            current_row.get("scan_event_quality_score")
            if current_row is not None
            else (latest_hist.get("scan_event_quality_score") if latest_hist is not None else 0.0)
        )
        persistence_score = safe_float(
            current_row.get("history_emergence_persistence_score") if current_row is not None else 0.0
        )
        profile_count = safe_float(current_row.get("profile_count") if current_row is not None else 0.0)
        current_cluster = safe_text(current_row.get("pit_cluster_key") if current_row is not None else "", "")
        current_free_float = safe_float(current_row.get("pit_free_float_ratio_current") if current_row is not None else np.nan, default=np.nan)
        current_listing_age = safe_float(current_row.get("pit_listing_age_days") if current_row is not None else np.nan, default=np.nan)
        current_fx_return_21d = safe_float(current_row.get("pit_fx_return_21d") if current_row is not None else 0.0)
        current_data_context_score = safe_float(current_row.get("pit_data_context_score") if current_row is not None else 0.0)
        current_earnings_snapshots = safe_float(current_row.get("pit_earnings_snapshot_count") if current_row is not None else 0.0)
        current_low_float_flag = safe_int(current_row.get("pit_low_float_flag") if current_row is not None else 0)
        current_fx_headwind_flag = safe_int(current_row.get("pit_fx_headwind_flag") if current_row is not None else 0)

        if ticker in selection_gap:
            monitor_bucket = "legacy_selection_gap"
        elif ticker in universe_gap:
            monitor_bucket = "legacy_universe_gap"
        elif current_track == "quality_compounder" or current_quality_fit in {"high", "medium"}:
            monitor_bucket = "quality_lane"
        else:
            monitor_bucket = "monitor_only"

        stage_delta = promotion_stage_rank(current_stage) - promotion_stage_rank(prior_stage)
        status_delta = status_rank(current_status) - status_rank(prior_status)
        monitor_priority = (
            1.50 * max(0.0, promotion_stage_rank(current_stage))
            + 0.80 * max(0.0, status_rank(current_status))
            + 0.70 * persistence_score
            + 0.18 * min(current_quality_score, 10.0)
            + 0.12 * min(event_quality_score, 10.0)
            + 0.08 * max(0.0, current_recent_score)
            + 0.08 * min(profile_count, 4.0)
            + (0.60 if stage_delta > 0 else 0.0)
            + (0.25 if status_delta > 0 else 0.0)
        )

        rows.append(
            {
                "ticker": ticker,
                "monitor_bucket": monitor_bucket,
                "in_current_scan": int(current_row is not None),
                "snapshots_seen": int(hist_grp["snapshot_date"].nunique()) if not hist_grp.empty else 0,
                "first_seen_snapshot": hist_grp["snapshot_date"].min().strftime("%Y-%m-%d") if not hist_grp.empty else "",
                "last_seen_snapshot": hist_grp["snapshot_date"].max().strftime("%Y-%m-%d") if not hist_grp.empty else "",
                "promotion_stage": current_stage or "not_seen",
                "prior_promotion_stage": prior_stage or "not_seen",
                "stage_delta_vs_prev": int(stage_delta),
                "dynamic_status": current_status or "not_seen",
                "prior_dynamic_status": prior_status or "not_seen",
                "status_delta_vs_prev": int(status_delta),
                "validation_lane": safe_text(current_row.get("validation_lane") if current_row is not None else "", ""),
                "governance_role": safe_text(current_row.get("governance_role") if current_row is not None else "", ""),
                "scan_candidate_track": current_track or "not_seen",
                "scan_quality_compounder_fit": current_quality_fit or "weak",
                "scan_quality_compounder_score": round(current_quality_score, 4),
                "history_emergence_persistence_score": round(persistence_score, 4),
                "scan_event_quality_score": round(event_quality_score, 4),
                "profile_count": int(profile_count) if abs(profile_count - round(profile_count)) < 1e-9 else round(profile_count, 4),
                "pit_cluster_key": current_cluster or "unknown",
                "pit_free_float_ratio_current": round(current_free_float, 4) if not pd.isna(current_free_float) else np.nan,
                "pit_low_float_flag": int(current_low_float_flag),
                "pit_listing_age_days": round(current_listing_age, 1) if not pd.isna(current_listing_age) else np.nan,
                "pit_fx_return_21d": round(current_fx_return_21d, 4),
                "pit_fx_headwind_flag": int(current_fx_headwind_flag),
                "pit_earnings_snapshot_count": int(current_earnings_snapshots) if abs(current_earnings_snapshots - round(current_earnings_snapshots)) < 1e-9 else round(current_earnings_snapshots, 4),
                "pit_data_context_score": round(current_data_context_score, 4),
                "recent_score": round(current_recent_score, 4),
                "promotion_score": round(current_promotion_score, 4),
                "dynamic_conviction_score": round(current_conviction_score, 4),
                "monitor_priority_score": round(monitor_priority, 4),
                "needs_algo_attention": int(
                    monitor_bucket == "legacy_selection_gap"
                    and promotion_stage_rank(current_stage) < promotion_stage_rank("targeted_integration")
                ),
            }
        )

    monitor = pd.DataFrame(rows)
    if monitor.empty:
        return monitor

    monitor = monitor.sort_values(
        [
            "monitor_priority_score",
            "promotion_score",
            "history_emergence_persistence_score",
            "scan_quality_compounder_score",
            "recent_score",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return monitor


def write_quality_compounder_forward_monitor(monitor: pd.DataFrame) -> dict[str, Path]:
    if monitor.empty:
        QUALITY_FORWARD_MONITOR_PATH.write_text("", encoding="utf-8")
        QUALITY_FORWARD_REPORT_PATH.write_text(
            "# Quality Compounder Forward Monitor\n\n(none)\n",
            encoding="utf-8",
        )
        return {
            "quality_monitor": QUALITY_FORWARD_MONITOR_PATH,
            "quality_monitor_report": QUALITY_FORWARD_REPORT_PATH,
        }

    monitor.to_csv(QUALITY_FORWARD_MONITOR_PATH, index=False)

    improved = monitor.loc[
        (monitor["stage_delta_vs_prev"] > 0) | (monitor["status_delta_vs_prev"] > 0)
    ].copy()
    unresolved_selection = monitor.loc[
        (monitor["monitor_bucket"] == "legacy_selection_gap")
        & (monitor["needs_algo_attention"] == 1)
    ].copy()

    lines = [
        "# Quality Compounder Forward Monitor",
        "",
        f"- as_of: `{date.today().isoformat()}`",
        f"- tracked names: `{len(monitor)}`",
        f"- in current scan: `{int(monitor['in_current_scan'].sum())}`",
        f"- approved/probation/targeted: `{int(monitor['promotion_stage'].isin(['approved_live', 'probation_live', 'targeted_integration']).sum())}`",
        f"- legacy selection gaps still below targeted: `{len(unresolved_selection)}`",
        "",
        "## Top Forward Priorities",
        "",
        _monitor_table(
            monitor,
            [
                "ticker",
                "monitor_bucket",
                "promotion_stage",
                "dynamic_status",
                "pit_cluster_key",
                "scan_quality_compounder_fit",
                "scan_candidate_track",
                "history_emergence_persistence_score",
                "pit_data_context_score",
                "pit_fx_headwind_flag",
                "recent_score",
                "promotion_score",
                "monitor_priority_score",
            ],
            rows=20,
        ),
        "",
        "## Improved Since Previous Snapshot",
        "",
        _monitor_table(
            improved,
            [
                "ticker",
                "monitor_bucket",
                "prior_promotion_stage",
                "promotion_stage",
                "stage_delta_vs_prev",
                "prior_dynamic_status",
                "dynamic_status",
                "status_delta_vs_prev",
                "monitor_priority_score",
            ],
            rows=20,
        ),
        "",
        "## Legacy Selection Gaps Still Missing Higher Promotion",
        "",
        _monitor_table(
            unresolved_selection,
            [
                "ticker",
                "promotion_stage",
                "dynamic_status",
                "pit_cluster_key",
                "scan_quality_compounder_fit",
                "scan_candidate_track",
                "pit_data_context_score",
                "recent_score",
                "promotion_score",
                "monitor_priority_score",
            ],
            rows=20,
        ),
    ]
    QUALITY_FORWARD_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return {
        "quality_monitor": QUALITY_FORWARD_MONITOR_PATH,
        "quality_monitor_report": QUALITY_FORWARD_REPORT_PATH,
    }


def summarize_walkforward(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(key_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(key_cols, keys)}
        yearly = group[group["window"] != "2026_ytd"]
        ytd = group[group["window"] == "2026_ytd"]
        row.update(
            {
                "add_wf_available": 1,
                "add_mean_delta_roi_2017_2025": clip_small_float(yearly["delta_roi_pct"].mean()) if not yearly.empty else 0.0,
                "add_mean_delta_sharpe_2017_2025": clip_small_float(yearly["delta_sharpe"].mean()) if not yearly.empty else 0.0,
                "add_mean_delta_maxdd_2017_2025": clip_small_float(yearly["delta_maxdd_pct"].mean()) if not yearly.empty else 0.0,
                "add_roi_wins_2017_2025": int((yearly["delta_roi_pct"] > 0).sum()) if not yearly.empty else 0,
                "add_sharpe_wins_2017_2025": int((yearly["delta_sharpe"] > 0).sum()) if not yearly.empty else 0,
                "add_maxdd_wins_2017_2025": int((yearly["delta_maxdd_pct"] >= 0).sum()) if not yearly.empty else 0,
                "add_delta_roi_2026_ytd": clip_small_float(ytd["delta_roi_pct"].mean()) if not ytd.empty else 0.0,
                "add_delta_sharpe_2026_ytd": clip_small_float(ytd["delta_sharpe"].mean()) if not ytd.empty else 0.0,
                "add_delta_maxdd_2026_ytd": clip_small_float(ytd["delta_maxdd_pct"].mean()) if not ytd.empty else 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_selection_rank(value: str) -> int:
    return {"approved_add": 3, "watch_add": 2, "reject_add": 1}.get(str(value or ""), 0)


def load_promotion_add_summary() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in PROMOTION_ADD_STUDY_PATHS:
        df = read_optional_csv(path)
        if df.empty:
            continue
        if "ticker" not in df.columns:
            if "name" in df.columns:
                df = df.rename(columns={"name": "ticker"})
            else:
                continue
        out = df.copy()
        out["ticker"] = out["ticker"].astype(str)
        if "selection_status" not in out.columns:
            out["selection_status"] = ""
        out["selection_status"] = out["selection_status"].fillna("")
        for col in (
            "full_delta_roi_pct",
            "oos_delta_roi_pct",
            "full_delta_sharpe",
            "oos_delta_sharpe",
            "full_delta_maxdd_pct",
            "oos_delta_maxdd_pct",
        ):
            if col not in out.columns:
                out[col] = 0.0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).map(clip_small_float)
        if "backtestable_now" not in out.columns:
            out["backtestable_now"] = 1
        out["backtestable_now"] = pd.to_numeric(out["backtestable_now"], errors="coerce").fillna(1).astype(int)
        out["add_validation_status"] = out["selection_status"].fillna("")
        out["add_validation_rank"] = out["add_validation_status"].map(add_selection_rank).fillna(0)
        frames.append(
            out[
                [
                    "ticker",
                    "add_validation_status",
                    "add_validation_rank",
                    "full_delta_roi_pct",
                    "oos_delta_roi_pct",
                    "full_delta_sharpe",
                    "oos_delta_sharpe",
                    "full_delta_maxdd_pct",
                    "oos_delta_maxdd_pct",
                    "backtestable_now",
                ]
            ].copy()
        )
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values(
        [
            "ticker",
            "add_validation_rank",
            "oos_delta_roi_pct",
            "oos_delta_sharpe",
            "full_delta_roi_pct",
            "full_delta_sharpe",
        ],
        ascending=[True, False, False, False, False, False],
    ).drop_duplicates("ticker", keep="first")
    return combined


def load_promotion_add_wf_summary() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in PROMOTION_ADD_WF_PATHS:
        df = read_optional_csv(path)
        if df.empty or "ticker" not in df.columns or "window" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(["ticker", "window"], keep="first")
    return summarize_walkforward(combined, ["ticker"])


def load_recent_validation_summary() -> pd.DataFrame:
    df = read_optional_csv(RECENT_VALIDATION_PATH)
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str)
    for col in (
        "delta_score_full",
        "delta_score_oos",
        "delta_score_recent_504d",
        "delta_score_recent_252d",
        "recent_gap_252_vs_oos",
        "recent_gap_504_vs_full",
    ):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).map(clip_small_float)
    if "recency_flag" not in out.columns:
        out["recency_flag"] = ""
    out["recency_flag"] = out["recency_flag"].fillna("")
    return out[
        [
            "ticker",
            "delta_score_full",
            "delta_score_oos",
            "delta_score_recent_504d",
            "delta_score_recent_252d",
            "recent_gap_252_vs_oos",
            "recent_gap_504_vs_full",
            "recency_flag",
        ]
    ].drop_duplicates("ticker")


def enrich_event_quality(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    numeric_defaults = {
        "scan_gap_zscore_20": 0.0,
        "scan_gap_fill_share": 0.0,
        "scan_rel_volume_20": 0.0,
        "scan_close_in_range": 0.0,
        "scan_efficiency_ratio_20": 0.0,
        "scan_efficiency_ratio_60": 0.0,
        "scan_earnings_blackout_5d": 0.0,
        "scan_earnings_blackout_10d": 0.0,
        "scan_post_earnings_5d": 0.0,
    }
    for col, default in numeric_defaults.items():
        if col not in out.columns:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    if "scan_hot_archetype" not in out.columns:
        out["scan_hot_archetype"] = "not_hot"
    out["scan_hot_archetype"] = out["scan_hot_archetype"].fillna("not_hot").astype(str)

    gap_abs = out["scan_gap_zscore_20"].replace([float("inf"), float("-inf")], 0.0).abs().clip(upper=4.0)
    gap_fill_share = out["scan_gap_fill_share"].clip(lower=0.0, upper=1.0)
    rel_volume = out["scan_rel_volume_20"].clip(lower=0.0, upper=6.0)
    close_in_range = out["scan_close_in_range"].clip(lower=0.0, upper=1.0)
    efficiency20 = out["scan_efficiency_ratio_20"].clip(lower=0.0, upper=1.0)
    efficiency60 = out["scan_efficiency_ratio_60"].clip(lower=0.0, upper=1.0)
    earnings_blackout_5d = out["scan_earnings_blackout_5d"].astype(float)
    earnings_blackout_10d = out["scan_earnings_blackout_10d"].astype(float)
    post_earnings_5d = out["scan_post_earnings_5d"].astype(float)
    hot_archetype = out["scan_hot_archetype"]

    active_event = (
        (gap_abs > 0.0)
        | (gap_fill_share > 0.0)
        | (rel_volume > 0.0)
        | (close_in_range > 0.0)
        | (efficiency20 > 0.0)
        | (efficiency60 > 0.0)
        | (post_earnings_5d > 0.0)
        | (earnings_blackout_5d > 0.0)
        | (earnings_blackout_10d > 0.0)
    )

    base_session = (
        0.50 * close_in_range
        + 0.20 * (1.0 - gap_fill_share)
        + 0.15 * (1.0 - gap_abs / 4.0)
        + 0.15 * (0.40 * efficiency20 + 0.60 * efficiency60)
    ).clip(lower=0.0, upper=1.0)
    volume_quality = pd.Series(
        np.where(
            rel_volume <= 0.0,
            0.50,
            np.where(
                rel_volume < 0.8,
                0.40 + 0.10 * (rel_volume / 0.8),
                np.where(
                    rel_volume <= 2.5,
                    0.60 + 0.40 * ((rel_volume - 0.8) / 1.7),
                    np.where(
                        rel_volume <= 3.5,
                        0.88 - 0.18 * ((rel_volume - 2.5) / 1.0),
                        0.52 - 0.12 * np.minimum(1.0, (rel_volume - 3.5) / 2.5),
                    ),
                ),
            ),
        ),
        index=out.index,
        dtype=float,
    ).clip(lower=0.0, upper=1.0)
    blowoff_penalty = (
        0.45 * (gap_abs >= 2.0).astype(float)
        + 0.25 * (gap_fill_share >= 0.75).astype(float)
        + 0.30 * ((rel_volume >= 3.0) & (close_in_range <= 0.45)).astype(float)
    )
    constructive_event = (
        (close_in_range >= 0.60)
        & (gap_fill_share <= 0.70)
        & (gap_abs <= 2.25)
        & (rel_volume >= 0.80)
        & (rel_volume <= 3.20)
    ).astype(float)
    post_earnings_bonus = 0.35 * post_earnings_5d + 0.15 * constructive_event * post_earnings_5d
    hot_late_penalty = (
        0.45 * hot_archetype.eq("hot_late").astype(float)
        + 0.08 * hot_archetype.eq("hot_watch").astype(float)
    )
    event_score = pd.Series(
        np.where(
            active_event,
            0.55 * base_session
            + 0.25 * volume_quality
            + 0.10 * constructive_event
            + post_earnings_bonus
            - 0.40 * blowoff_penalty
            - 0.35 * earnings_blackout_5d
            - 0.15 * earnings_blackout_10d
            - hot_late_penalty
            - 0.38,
            0.0,
        ),
        index=out.index,
        dtype=float,
    )
    event_label = np.select(
        [
            ~active_event,
            event_score >= 0.15,
            event_score >= 0.05,
            event_score <= -0.35,
            event_score <= -0.15,
        ],
        [
            "inactive",
            "high",
            "supportive",
            "risky",
            "speculative",
        ],
        default="neutral",
    )
    out["scan_event_active"] = active_event.astype(int)
    out["scan_event_quality_score"] = event_score.round(4)
    out["scan_event_quality_label"] = event_label
    return out


def promotion_score(row: pd.Series | dict) -> float:
    score = safe_float(row.get("dynamic_conviction_score"))
    score += 0.04 * clip_small_float(row.get("full_delta_roi_pct"))
    score += 0.10 * clip_small_float(row.get("oos_delta_roi_pct"))
    score += 40.0 * clip_small_float(row.get("full_delta_sharpe"))
    score += 60.0 * clip_small_float(row.get("oos_delta_sharpe"))
    score += 0.04 * clip_small_float(row.get("add_mean_delta_roi_2017_2025"))
    score += 10.0 * clip_small_float(row.get("add_mean_delta_sharpe_2017_2025"))
    score += 0.5 * safe_int(row.get("add_roi_wins_2017_2025"))
    score += 0.35 * safe_int(row.get("add_sharpe_wins_2017_2025"))
    add_validation_status = str(row.get("add_validation_status") or "")
    if add_validation_status == "approved_add":
        score += 1.0
    elif add_validation_status == "watch_add":
        score += 0.25
    elif add_validation_status == "reject_add":
        score -= 2.0
    score -= 0.8 * safe_int(row.get("scan_entry_heat_flag"))
    score += 0.75 * min(4, safe_int(row.get("history_watch_scans")))
    score += 1.10 * min(3, safe_int(row.get("history_targeted_scans")))
    score += 0.90 * min(2, safe_int(row.get("history_probation_scans")))
    score += 0.60 * min(3, safe_int(row.get("history_consecutive_watch")))
    score += 0.90 * min(2, safe_int(row.get("history_consecutive_targeted")))
    score += 35.0 * safe_float(row.get("history_emergence_persistence_score"))
    score += 0.25 * min(4.0, safe_float(row.get("history_avg_support_count")))
    if str(row.get("history_best_promotion_stage") or "") == "probation_live":
        score += 1.25
    elif str(row.get("history_best_promotion_stage") or "") == "targeted_integration":
        score += 0.75
    if safe_int(row.get("metrics_carried_from_history")):
        score += 0.30
    score += 0.30 * safe_float(row.get("scan_early_leader_score"))
    score += 0.22 * safe_float(row.get("scan_quality_compounder_score"))
    score += 0.90 * safe_float(row.get("entry_timing_score"))
    score += 0.60 * safe_float(row.get("shadow_probation_score"))
    score += 0.015 * safe_float(row.get("candidate_memory_score"))
    score += 0.85 * min(1.25, safe_float(row.get("portfolio_diversification_score")))
    score -= 1.00 * min(1.50, safe_float(row.get("portfolio_crowding_score")))
    entry_bucket = str(row.get("entry_timing_bucket") or "")
    entry_zone = str(row.get("entry_timing_zone") or "")
    entry_proto = str(row.get("entry_hot_proto") or "")
    event_quality_score = safe_float(row.get("scan_event_quality_score"))
    event_quality_label = str(row.get("scan_event_quality_label") or "")
    shadow_readiness = str(row.get("shadow_readiness") or "")
    core_bridge_fit = str(row.get("core_rank_bridge_fit") or "")
    core_bridge_score = safe_float(row.get("core_rank_bridge_score"))
    if entry_bucket == "clean":
        score += 0.75
    elif entry_bucket == "constructive":
        score += 0.35
    elif entry_bucket == "hot":
        score -= 0.75
    elif entry_bucket == "event_risk":
        score -= 1.50
    if entry_zone == "hot_constructive":
        score += 0.45
    elif entry_zone == "hot_watch":
        score -= 0.20
    elif entry_zone == "hot_late":
        score -= 1.20
    if entry_zone == "hot_watch":
        if entry_proto == "hot_constructive_proto":
            score += 0.85
        elif entry_proto == "hot_late_proto":
            score -= 0.95
        elif entry_proto == "hot_watch_proto":
            score -= 0.20
    score += 0.80 * event_quality_score
    if event_quality_label == "high":
        score += 0.45
    elif event_quality_label == "supportive":
        score += 0.20
    elif event_quality_label == "speculative":
        score -= 0.35
    elif event_quality_label == "risky":
        score -= 0.85
    if shadow_readiness == "shadow_ready":
        score += 0.90
    elif shadow_readiness == "shadow_building":
        score += 0.35
    early_fit = str(row.get("scan_early_leader_fit") or "")
    quality_fit = str(row.get("scan_quality_compounder_fit") or "")
    if early_fit == "high":
        score += 0.75
    elif early_fit == "medium":
        score += 0.35
    if quality_fit == "high":
        score += 0.70
    elif quality_fit == "medium":
        score += 0.35
    score += 0.50 * min(core_bridge_score, 3.0)
    if core_bridge_fit == "high":
        score += 0.90
    elif core_bridge_fit == "medium":
        score += 0.45
    recent = safe_float(row.get("recent_score"))
    rel_recent = safe_float(row.get("scan_rel_recent_score"))
    if recent > 20.0:
        score -= 6.0
    if rel_recent > 15.0:
        score -= 3.0
    return round(score, 4)


def classify_promotion_stage(row: pd.Series | dict) -> str:
    if not bool(row.get("broker_tradeable", True)):
        return "blocked_broker"

    rec = str(row.get("recommendation") or "")
    add_validation_status = str(row.get("add_validation_status") or "")
    algo_fit = str(row.get("scan_algo_fit") or "")
    quality_fit = str(row.get("scan_quality_compounder_fit") or "")
    track = str(row.get("scan_candidate_track") or "")
    heat = safe_int(row.get("scan_entry_heat_flag")) > 0
    recent = safe_float(row.get("recent_score"))
    rel_recent = safe_float(row.get("scan_rel_recent_score"))
    rr_score = safe_float(row.get("scan_rr_score"))
    score = safe_float(row.get("dynamic_conviction_score"))
    support_count = max(safe_int(row.get("profile_count"), 1), safe_int(row.get("source_type_count"), 1))
    earnings_blackout = safe_int(row.get("scan_earnings_blackout_5d")) > 0
    early_fit = str(row.get("scan_early_leader_fit") or "")
    full_roi = clip_small_float(row.get("full_delta_roi_pct"))
    oos_roi = clip_small_float(row.get("oos_delta_roi_pct"))
    full_sharpe = clip_small_float(row.get("full_delta_sharpe"))
    oos_sharpe = clip_small_float(row.get("oos_delta_sharpe"))
    wf_available = safe_int(row.get("add_wf_available")) == 1
    wf_mean_roi = clip_small_float(row.get("add_mean_delta_roi_2017_2025"))
    wf_mean_sharpe = clip_small_float(row.get("add_mean_delta_sharpe_2017_2025"))
    wf_roi_wins = safe_int(row.get("add_roi_wins_2017_2025"))
    wf_sharpe_wins = safe_int(row.get("add_sharpe_wins_2017_2025"))
    history_watch = safe_int(row.get("history_watch_scans"))
    history_targeted = safe_int(row.get("history_targeted_scans"))
    history_probation = safe_int(row.get("history_probation_scans"))
    consecutive_watch = safe_int(row.get("history_consecutive_watch"))
    consecutive_targeted = safe_int(row.get("history_consecutive_targeted"))
    consecutive_probation = safe_int(row.get("history_consecutive_probation"))
    emergence_persistence_score = safe_float(row.get("history_emergence_persistence_score"))
    history_constructive_scans = safe_int(row.get("history_constructive_scans"))
    prior_stage = str(row.get("history_best_promotion_stage") or "")
    prior_stage_rank = promotion_stage_rank(prior_stage)
    carried = safe_int(row.get("metrics_carried_from_history")) > 0

    if add_validation_status == "reject_add" and full_roi <= 0 and oos_roi <= 0 and wf_mean_roi <= 0 and wf_mean_sharpe <= 0:
        if str(row.get("dynamic_status") or "") in {"watch", "prime_watch"} and score >= 14.0:
            return "watch_queue"
        return "review_queue"

    anomaly_like = recent >= 25.0 or rel_recent >= 15.0 or rr_score >= 10.0
    leader_like = track in {"persistent_leader", "emerging_leader", "quality_compounder"}
    high_like = algo_fit == "high" or quality_fit == "high"
    medium_like = algo_fit in {"high", "medium"} or quality_fit in {"high", "medium"}
    persistence_supported = (
        support_count >= 2
        or history_watch >= 2
        or consecutive_watch >= 2
        or prior_stage_rank >= promotion_stage_rank("targeted_integration")
        or emergence_persistence_score >= 3.0
    )
    strong_add = (
        rec in {"add", "watch"}
        and high_like
        and leader_like
        and persistence_supported
        and score >= 18.0
        and full_roi > 0.0
        and oos_roi >= 0.0
        and full_sharpe >= 0.0
        and oos_sharpe >= -0.01
    )
    wf_good = (
        wf_available
        and wf_mean_roi >= 0.0
        and wf_mean_sharpe >= 0.0
        and wf_roi_wins >= 2
        and wf_sharpe_wins >= 2
    )
    probation_supported = (
        history_targeted >= 2
        or history_probation >= 1
        or consecutive_targeted >= 1
        or prior_stage_rank >= promotion_stage_rank("probation_live")
        or (emergence_persistence_score >= 4.0 and history_constructive_scans >= 2)
    )
    entry_bucket = str(row.get("entry_timing_bucket") or "")
    entry_zone = str(row.get("entry_timing_zone") or entry_bucket)
    entry_proto = str(row.get("entry_hot_proto") or "")
    shadow_readiness = str(row.get("shadow_readiness") or "")
    shadow_snapshots = safe_int(row.get("shadow_snapshots"))
    shadow_positive_rate = safe_float(row.get("shadow_positive_rate"))
    shadow_probation_score = safe_float(row.get("shadow_probation_score"))
    shadow_hot_constructive_rate = safe_float(row.get("shadow_hot_constructive_rate"))
    shadow_hot_late_rate = safe_float(row.get("shadow_hot_late_rate"))
    event_active = safe_int(row.get("scan_event_active")) > 0
    event_quality_score = safe_float(row.get("scan_event_quality_score"))
    event_quality_label = str(row.get("scan_event_quality_label") or "")
    proto_constructive = entry_proto == "hot_constructive_proto"
    proto_late = entry_proto == "hot_late_proto"
    constructive_entry = (
        entry_bucket in {"clean", "constructive"}
        or entry_zone == "hot_constructive"
        or (entry_zone == "hot_watch" and proto_constructive)
    )
    hot_late = entry_zone == "hot_late" or (entry_zone == "hot_watch" and proto_late)
    early_constructive = early_fit in {"high", "medium"} and (
        entry_zone in {"clean", "constructive", "hot_constructive"}
        or (entry_zone == "hot_watch" and proto_constructive)
    )
    shadow_ready = shadow_readiness == "shadow_ready"
    shadow_building = shadow_readiness in {"shadow_ready", "shadow_building"}
    event_supportive = event_quality_label in {"high", "supportive"} or event_quality_score >= 0.05
    event_speculative = event_active and (event_quality_label in {"speculative", "risky"} or event_quality_score <= -0.15)
    event_risky = event_active and (event_quality_label == "risky" or event_quality_score <= -0.35)
    shadow_promotable = (
        shadow_snapshots >= 2
        and shadow_probation_score >= 1.25
        and shadow_positive_rate >= 0.5
        and shadow_hot_late_rate <= 0.25
        and not hot_late
        and not event_speculative
    )
    medium_fit_targeted = (
        rec in {"add", "watch"}
        and medium_like
        and (constructive_entry or early_constructive)
        and not anomaly_like
        and not hot_late
        and not event_speculative
        and not earnings_blackout
        and support_count >= 2
        and leader_like
        and full_roi > 0.0
        and oos_roi >= 0.0
        and wf_available
        and wf_mean_roi > 0.0
        and wf_mean_sharpe >= 0.0
        and score >= 13.0
    )
    approved_after_probation = (
        strong_add
        and constructive_entry
        and not anomaly_like
        and not hot_late
        and not event_risky
        and not earnings_blackout
        and (
            wf_good
            or (
                (history_probation >= 1 or consecutive_probation >= 1 or prior_stage_rank >= promotion_stage_rank("probation_live"))
                and oos_roi > 0.0
                and oos_sharpe >= 0.0
                and full_roi > 0.0
                and score >= 20.0
                and history_watch >= 1
            )
            or (
                shadow_ready
                and history_watch >= 1
                and oos_roi >= 0.0
                and full_roi > 0.0
                and score >= 19.0
            )
        )
    )

    if approved_after_probation:
        return "approved_live"
    if (
        strong_add
        and constructive_entry
        and not anomaly_like
        and not hot_late
        and not event_risky
        and not earnings_blackout
        and (probation_supported or carried or shadow_building or early_constructive)
    ):
        return "probation_live"
    if (
        rec in {"add", "watch"}
        and str(row.get("dynamic_status") or "") in {"watch", "prime_watch"}
        and support_count >= 2
        and (medium_like or early_fit in {"high", "medium"})
        and (constructive_entry or early_constructive or shadow_hot_constructive_rate > 0.0)
        and not anomaly_like
        and not event_speculative
        and not earnings_blackout
        and shadow_promotable
    ):
        return "targeted_integration"
    if medium_fit_targeted or (
        rec in {"add", "watch"}
        and high_like
        and (constructive_entry or early_constructive)
        and shadow_building
        and support_count >= 2
        and score >= 15.0
        and not hot_late
        and not event_speculative
    ):
        return "targeted_integration"
    if (
        (constructive_entry or early_constructive)
        and not anomaly_like
        and not hot_late
        and not event_speculative
        and (
            (rec in {"add", "watch"} and high_like and score >= 18.0 and leader_like)
            or (early_constructive and score >= 17.0 and support_count >= 2 and leader_like)
            or (event_supportive and score >= 17.0 and support_count >= 2 and leader_like)
            or (high_like and persistence_supported and full_roi >= 0.0 and score >= 18.0)
            or (prior_stage_rank >= promotion_stage_rank("targeted_integration") and score >= 16.0)
        )
    ):
        return "targeted_integration"
    if str(row.get("dynamic_status") or "") in {"prime_watch", "watch"} or score >= 12.0:
        return "watch_queue"
    if str(row.get("dynamic_status") or "") in {"review", "discovered"} or score >= 8.0:
        return "review_queue"
    return "reject_queue"


def classify_validation_lane(row: pd.Series | dict) -> str:
    full_roi = clip_small_float(row.get("full_delta_roi_pct"))
    oos_roi = clip_small_float(row.get("oos_delta_roi_pct"))
    wf_roi = clip_small_float(row.get("add_mean_delta_roi_2017_2025"))
    wf_sharpe = clip_small_float(row.get("add_mean_delta_sharpe_2017_2025"))
    recent_504d = clip_small_float(row.get("delta_score_recent_504d"))
    recent_252d = clip_small_float(row.get("delta_score_recent_252d"))
    heat = safe_int(row.get("scan_entry_heat_flag")) > 0
    fit = str(row.get("scan_algo_fit") or "")
    early_fit = str(row.get("scan_early_leader_fit") or "")
    entry_zone = str(row.get("entry_timing_zone") or "")
    shadow_readiness = str(row.get("shadow_readiness") or "")

    if (
        safe_int(row.get("add_wf_available")) == 1
        and (full_roi > 0 or oos_roi > 0)
        and (wf_roi > 0 or wf_sharpe > 0)
    ):
        return "robust_long"
    if (
        (fit in {"high", "medium"} or early_fit in {"high", "medium"})
        and entry_zone != "hot_late"
        and (not heat or entry_zone == "hot_constructive")
        and recent_252d > 0
        and recent_504d >= 0
        and shadow_readiness in {"shadow_ready", "shadow_building"}
    ):
        return "recent_accelerating"
    if recent_252d > 0 or recent_504d > 0:
        return "recent_candidate"
    if full_roi > 0 or oos_roi > 0:
        return "mixed_long"
    return "weak"


def classify_governance_role(row: pd.Series | dict) -> str:
    stage = str(row.get("promotion_stage") or "")
    lane = str(row.get("validation_lane") or "")
    status = str(row.get("dynamic_status") or "")
    shadow_readiness = str(row.get("shadow_readiness") or "")
    if stage == "approved_live":
        return "live_ready"
    if stage == "probation_live":
        return "probation"
    if stage == "targeted_integration" and shadow_readiness == "shadow_ready":
        return "shadow_ready"
    if stage == "targeted_integration":
        return "research_priority"
    if lane in {"recent_accelerating", "robust_long"} or status in {"watch", "prime_watch"}:
        return "watchlist"
    if status == "review":
        return "reserve"
    return "archive"


def enrich_broker_tradeability(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ("marketCap", "exchange", "fullExchangeName"):
        if col not in out.columns:
            out[col] = pd.NA
    if "broker_exchange_supported" not in out.columns:
        out["broker_exchange_supported"] = 1
    if "quoteType" not in out.columns:
        out["quoteType"] = "EQUITY"
    needs_lookup = (
        out["marketCap"].isna()
        & (
            out["candidate_status"].fillna("").eq("new")
            | out["recommendation"].fillna("").isin(["add", "watch"])
            | out["scan_algo_fit"].fillna("").isin(["high", "medium"])
        )
    )
    lookup_tickers = out.loc[needs_lookup, "ticker"].dropna().astype(str).unique().tolist()
    ctx_map = dud.bulk_ticker_context(lookup_tickers)
    for idx, row in out.loc[needs_lookup].iterrows():
        ctx = ctx_map.get(str(row.get("ticker") or ""), {})
        if pd.isna(out.at[idx, "marketCap"]):
            out.at[idx, "marketCap"] = ctx.get("marketCap")
        if pd.isna(out.at[idx, "exchange"]):
            out.at[idx, "exchange"] = ctx.get("exchange")
        if pd.isna(out.at[idx, "fullExchangeName"]):
            out.at[idx, "fullExchangeName"] = ctx.get("exchange")

    out["marketCap"] = pd.to_numeric(out["marketCap"], errors="coerce")
    exchange_name = out["fullExchangeName"].fillna("").astype(str).str.upper()
    exchange_code = out["exchange"].fillna("").astype(str).str.upper()
    unsupported_otc = exchange_name.str.contains("OTC", na=False) | exchange_code.isin(["PNK", "OQB", "OQX", "OTC", "GREY"])
    broker_hint = pd.to_numeric(out["broker_exchange_supported"], errors="coerce").fillna(1).astype(int)
    quote_type = out["quoteType"].fillna("EQUITY").astype(str).str.upper()
    if BROKER_PROFILE == "xtb":
        unknown_exchange_block = pd.Series(False, index=out.index)
    else:
        unknown_exchange_block = (
            (broker_hint == 0)
            & out["marketCap"].notna()
            & (out["marketCap"] < 5 * BROKER_MIN_MARKET_CAP)
        )
    out["broker_tradeable"] = ~(
        out["candidate_status"].fillna("").eq("new")
        & (
            (out["marketCap"].notna() & (out["marketCap"] < BROKER_MIN_MARKET_CAP))
            | unsupported_otc
            | unknown_exchange_block
            | (quote_type != "EQUITY")
        )
    )
    out["broker_min_market_cap"] = float(BROKER_MIN_MARKET_CAP)
    return out


def defragment_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # A fresh copy collapses fragmented internal blocks after large concat/merge stages.
    return df.copy()


def ensure_numeric_columns(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    default: float = 0.0,
    as_int: bool = False,
) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = default
        series = pd.to_numeric(df[col], errors="coerce").fillna(default)
        df[col] = series.astype(int) if as_int else series
    return df


def ensure_text_columns(df: pd.DataFrame, defaults: dict[str, str]) -> pd.DataFrame:
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    return df


def aggregate_database(profile_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in profile_frames.values() if not df.empty]
    if not frames:
        return pd.DataFrame()
    combined = defragment_frame(pd.concat(frames, ignore_index=True, sort=False))
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
        rows.append(best)

    out = pd.DataFrame(rows)
    pre_records = out.to_dict("records")
    pre_scores = []
    pre_statuses = []
    for row in pre_records:
        score = conviction_score(row)
        row["dynamic_conviction_score"] = score
        pre_scores.append(score)
        pre_statuses.append(classify_dynamic_status(row))
    out["dynamic_conviction_score"] = pre_scores
    out["dynamic_status"] = pre_statuses
    out["dynamic_status_rank"] = out["dynamic_status"].map(status_rank).fillna(-1)
    out = out.sort_values(
        ["dynamic_status_rank", "dynamic_conviction_score", "recommendation_rank", "scan_algo_compat_score_v2", "scan_algo_compat_score", "recent_score", "priority_score"],
        ascending=[False, False, False, False, False, False, False],
    )
    out = enrich_broker_tradeability(out)
    out = enrich_forward_data_blocks(out)
    core_bridge = load_core_rank_bridge_frame()
    if not core_bridge.empty:
        out = out.merge(core_bridge, on="ticker", how="left")
    add_summary = load_promotion_add_summary()
    if not add_summary.empty:
        out = out.merge(add_summary, on="ticker", how="left", suffixes=("", "_study"))
        for col in (
            "full_delta_roi_pct",
            "oos_delta_roi_pct",
            "full_delta_sharpe",
            "oos_delta_sharpe",
            "full_delta_maxdd_pct",
            "oos_delta_maxdd_pct",
            "backtestable_now",
        ):
            study_col = f"{col}_study"
            if study_col not in out.columns:
                continue
            current = pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(np.nan, index=out.index)
            study = pd.to_numeric(out[study_col], errors="coerce")
            if col == "backtestable_now":
                fillable = current.isna() | current.fillna(0).eq(0)
            else:
                fillable = current.isna() | current.fillna(0.0).abs().lt(FLOAT_ZERO_EPS)
            better = study.fillna(0.0).abs().ge(FLOAT_ZERO_EPS)
            out[col] = current.where(~(fillable & better), study)
            out = out.drop(columns=[study_col])
    add_wf_summary = load_promotion_add_wf_summary()
    if not add_wf_summary.empty:
        out = out.merge(add_wf_summary, on="ticker", how="left")
    recent_validation = load_recent_validation_summary()
    if not recent_validation.empty:
        out = out.merge(recent_validation, on="ticker", how="left")
    out = defragment_frame(out)
    out = merge_history_memory(out)
    out = dug.enrich_database_governance(out)
    out = enrich_event_quality(out)
    out = defragment_frame(out)
    out = ensure_text_columns(
        out,
        {
            "add_validation_status": "",
            "scan_early_leader_fit": "weak",
            "scan_quality_compounder_fit": "weak",
            "scan_hot_archetype": "not_hot",
            "core_rank_bridge_fit": "weak",
            "scan_event_quality_label": "inactive",
            "recency_flag": "",
            "entry_timing_bucket": "neutral",
            "entry_hot_proto": "not_hot",
            "shadow_readiness": "shadow_weak",
        },
    )
    out = ensure_numeric_columns(
        out,
        (
            "scan_early_leader_score",
            "scan_quality_compounder_score",
            "scan_hot_candidate_score",
            "scan_event_quality_score",
        ),
    )
    out = ensure_numeric_columns(
        out,
        (
            "scan_event_active",
            "core_top30_flag",
            "core_top15_flag",
        ),
        default=0,
        as_int=True,
    )
    out = ensure_numeric_columns(
        out,
        (
            "core_latest_rank",
            "core_latest_score",
            "core_latest_r63",
            "core_latest_r252",
            "core_days_top15_trend",
            "core_days_top5_trend",
            "core_rank_bridge_score",
        ),
    )
    out = ensure_numeric_columns(
        out,
        (
            "add_wf_available",
            "add_mean_delta_roi_2017_2025",
            "add_mean_delta_sharpe_2017_2025",
            "add_mean_delta_maxdd_2017_2025",
            "add_roi_wins_2017_2025",
            "add_sharpe_wins_2017_2025",
            "add_maxdd_wins_2017_2025",
            "add_delta_roi_2026_ytd",
            "add_delta_sharpe_2026_ytd",
            "add_delta_maxdd_2026_ytd",
            "delta_score_full",
            "delta_score_oos",
            "delta_score_recent_504d",
            "delta_score_recent_252d",
            "recent_gap_252_vs_oos",
            "recent_gap_504_vs_full",
            "entry_timing_score",
            "shadow_snapshots",
            "shadow_probation_score",
            "candidate_memory_score",
        ),
    )
    if "entry_timing_zone" not in out.columns:
        out["entry_timing_zone"] = out["entry_timing_bucket"]
    out["entry_timing_zone"] = out["entry_timing_zone"].fillna(out["entry_timing_bucket"]).fillna("neutral")
    records = out.to_dict("records")
    scores = []
    statuses = []
    promotion_scores = []
    promotion_stages = []
    for row in records:
        score = conviction_score(row)
        row["dynamic_conviction_score"] = score
        scores.append(score)
        dynamic_status = classify_dynamic_status(row)
        row["dynamic_status"] = dynamic_status
        statuses.append(dynamic_status)
        pscore = promotion_score(row)
        row["promotion_score"] = pscore
        promotion_scores.append(pscore)
        promotion_stages.append(classify_promotion_stage(row))
    out["dynamic_conviction_score"] = scores
    out["dynamic_status"] = statuses
    out["dynamic_status_rank"] = out["dynamic_status"].map(status_rank).fillna(-1)
    out["promotion_score"] = promotion_scores
    out["promotion_stage"] = promotion_stages
    out["promotion_stage_rank"] = out["promotion_stage"].map(promotion_stage_rank).fillna(-2)
    out["validation_lane"] = out.apply(classify_validation_lane, axis=1)
    out["governance_role"] = out.apply(classify_governance_role, axis=1)
    out = out.sort_values(
        [
            "promotion_stage_rank",
            "promotion_score",
            "dynamic_status_rank",
            "dynamic_conviction_score",
            "recommendation_rank",
            "scan_algo_compat_score_v2",
            "scan_algo_compat_score",
            "recent_score",
            "entry_timing_score",
            "priority_score",
        ],
        ascending=[False, False, False, False, False, False, False, False, False, False],
    )
    out["as_of"] = str(date.today())
    return out


def write_database_outputs(db: pd.DataFrame) -> dict[str, Path]:
    current_path = DATA_DIR / "dynamic_universe_current.csv"
    approved_path = DATA_DIR / "dynamic_universe_approved_additions.csv"
    promotion_path = DATA_DIR / "dynamic_universe_promotion_queue.csv"
    summary_path = DATA_DIR / "dynamic_universe_summary.md"
    snapshot_path = HISTORY_DIR / f"dynamic_universe_snapshot_{date.today().isoformat()}.csv"
    selected_adds_path = DATA_DIR / "dynamic_universe_selected_additions.csv"
    selected_dems_path = DATA_DIR / "dynamic_universe_selected_demotions.csv"

    db.to_csv(current_path, index=False)
    db.to_csv(snapshot_path, index=False)

    approved = db.loc[db["promotion_stage"] == "approved_live", ["ticker"]].drop_duplicates().sort_values("ticker")
    if approved.empty:
        approved = pd.DataFrame({"ticker": []})
    approved.to_csv(approved_path, index=False)
    promotion_queue_columns = [
        "ticker",
        "promotion_stage",
        "validation_lane",
        "governance_role",
        "recency_flag",
        "entry_timing_bucket",
        "entry_timing_zone",
        "entry_hot_proto",
        "entry_timing_score",
        "shadow_readiness",
        "shadow_probation_score",
        "promotion_score",
        "dynamic_status",
        "recommendation",
        "scan_algo_fit",
        "scan_early_leader_fit",
        "scan_early_leader_score",
        "scan_quality_compounder_fit",
        "scan_quality_compounder_score",
        "scan_event_active",
        "scan_event_quality_score",
        "scan_event_quality_label",
        "scan_hot_archetype",
        "scan_hot_candidate_score",
        "pit_cluster_key",
        "pit_data_context_score",
        "pit_low_float_flag",
        "pit_fx_headwind_flag",
        "core_latest_rank",
        "core_rank_bridge_fit",
        "core_rank_bridge_score",
        "pit_listing_age_days",
        "pit_earnings_snapshot_count",
        "profile_count",
        "source_type_count",
        "recent_score",
        "scan_entry_heat_flag",
        "history_scans_seen",
        "history_watch_scans",
        "history_targeted_scans",
        "history_probation_scans",
        "history_consecutive_watch",
        "history_consecutive_targeted",
        "history_consecutive_probation",
        "history_constructive_scans",
        "history_hot_late_scans",
        "history_avg_support_count",
        "history_stage_progression",
        "history_emergence_persistence_score",
        "history_best_promotion_stage",
        "metrics_carried_from_history",
        "candidate_memory_score",
        "full_delta_roi_pct",
        "oos_delta_roi_pct",
        "add_mean_delta_roi_2017_2025",
        "add_mean_delta_sharpe_2017_2025",
        "add_roi_wins_2017_2025",
        "add_sharpe_wins_2017_2025",
    ]
    promotion_queue = safe_columns(
        db.loc[db["promotion_stage"].isin(["approved_live", "probation_live", "targeted_integration"])].copy(),
        promotion_queue_columns,
    )
    promotion_queue.to_csv(promotion_path, index=False)
    quality_monitor = build_quality_compounder_forward_monitor(db)
    monitor_outputs = write_quality_compounder_forward_monitor(quality_monitor)

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
        f"- approved_live: `{int((db['promotion_stage'] == 'approved_live').sum())}`",
        f"- probation_live: `{int((db['promotion_stage'] == 'probation_live').sum())}`",
        f"- targeted_integration: `{int((db['promotion_stage'] == 'targeted_integration').sum())}`",
        f"- validation robust_long: `{int((db['validation_lane'] == 'robust_long').sum())}`",
        f"- validation recent_accelerating: `{int((db['validation_lane'] == 'recent_accelerating').sum())}`",
        f"- low_float_watch: `{int(pd.to_numeric(db.get('pit_low_float_flag'), errors='coerce').fillna(0).sum())}`",
        f"- fx_headwind_watch: `{int(pd.to_numeric(db.get('pit_fx_headwind_flag'), errors='coerce').fillna(0).sum())}`",
        f"- selected additions live: `{len(selected_adds) if not selected_adds.empty else 0}`",
        f"- selected demotions live: `{len(selected_dems) if not selected_dems.empty else 0}`",
        "",
        "## Shadow probation",
        "",
        read_optional_csv(dug.SHADOW_PATH).head(20).to_string(index=False) if dug.SHADOW_PATH.exists() else "(none)",
        "",
        "## Approved Live",
        "",
        approved.head(50).to_string(index=False) if not approved.empty else "(none)",
        "",
        "## Promotion Queue",
        "",
        promotion_queue.head(20).to_string(index=False) if not promotion_queue.empty else "(none)",
        "",
        "## Quality Compounder Forward Monitor",
        "",
        quality_monitor.head(20).to_string(index=False) if not quality_monitor.empty else "(none)",
        "",
        "## Governance roles",
        "",
        safe_columns(
            db.loc[
                db["governance_role"].isin(["live_ready", "probation", "shadow_ready", "research_priority", "watchlist"])
            ].copy(),
            [
                "ticker",
                "governance_role",
                "validation_lane",
                "promotion_stage",
                "dynamic_status",
                "entry_timing_bucket",
                "entry_timing_zone",
                "entry_hot_proto",
                "shadow_readiness",
                "scan_algo_fit",
                "scan_quality_compounder_fit",
                "scan_early_leader_fit",
                "pit_cluster_key",
                "pit_data_context_score",
                "pit_low_float_flag",
                "pit_fx_headwind_flag",
                "core_latest_rank",
                "core_rank_bridge_fit",
                "core_rank_bridge_score",
                "scan_hot_archetype",
                "recent_score",
                "promotion_score",
            ],
        ).head(30).to_string(index=False) if not db.empty else "(none)",
        "",
        "## Prime watch",
        "",
        safe_columns(
            db.loc[db["dynamic_status"] == "prime_watch"].copy(),
            [
                "ticker",
                "source_profiles",
                "profile_count",
                "scan_algo_fit",
                "scan_algo_compat_score_v2",
                "recent_score",
                "entry_timing_score",
                "dynamic_conviction_score",
                "recommendation",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
            ],
        ).head(20).to_string(index=False) if not db.empty else "(none)",
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
        safe_columns(
            db.loc[db["dynamic_status"].isin(["watch", "review"])].copy(),
            [
                "ticker",
                "dynamic_status",
                "promotion_stage",
                "source_profiles",
                "profile_count",
                "scan_algo_fit",
                "scan_algo_compat_score_v2",
                "recent_score",
                "dynamic_conviction_score",
                "recommendation",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
            ],
        ).head(30).to_string(index=False) if not db.empty else "(none)",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "current": current_path,
        "approved": approved_path,
        "promotion": promotion_path,
        "summary": summary_path,
        "snapshot": snapshot_path,
        **monitor_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the persisted dynamic universe database.")
    parser.add_argument("--profiles", default="targeted_current,broad_focus,broad_diversified,local_resilience", help="Comma-separated cycle profiles to aggregate.")
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
