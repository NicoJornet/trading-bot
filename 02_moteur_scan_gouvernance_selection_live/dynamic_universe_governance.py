from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "dynamic_universe"
HISTORY_DIR = DATA_DIR / "history"
EXPORTS_DIR = ROOT / "research" / "exports"
OHLCV_PATH = ROOT / "apex_ohlcv_full_2015_2026.csv"
SHADOW_PATH = DATA_DIR / "dynamic_universe_shadow_probation.csv"
SWAP_MEMORY_PATH = EXPORTS_DIR / "dynamic_universe_swap_memory.csv"
PORTFOLIO_PATH = ROOT / "portfolio.json"

SINGLE_SWAP_PATH = EXPORTS_DIR / "dynamic_universe_swap_single_summary.csv"
SINGLE_WALK_PATH = EXPORTS_DIR / "dynamic_universe_swap_walkforward_summary.csv"
LEGACY_SINGLE_WALK_PATH = EXPORTS_DIR / "dynamic_universe_swap_top12_walkforward_summary.csv"

FLOAT_ZERO_EPS = 1e-9


def read_optional_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(path, **kwargs)


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


def clip_small_float(value: object, eps: float = FLOAT_ZERO_EPS) -> float:
    out = safe_float(value)
    return 0.0 if abs(out) < eps else out


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


def status_rank(value: str) -> int:
    return {
        "approved": 5,
        "prime_watch": 4,
        "watch": 3,
        "review": 2,
        "discovered": 1,
        "reject": 0,
    }.get(str(value or ""), -1)


def _entry_bucket(score: float, blackout: int, heat: int) -> str:
    if blackout:
        return "event_risk"
    if heat:
        return "hot"
    if score >= 3.0:
        return "clean"
    if score >= 1.25:
        return "constructive"
    return "neutral"


def _entry_zone(score: float, blackout: int, heat: int) -> str:
    if blackout:
        return "event_risk"
    if heat:
        if score >= 1.75:
            return "hot_constructive"
        if score <= 0.25:
            return "hot_late"
        return "hot_watch"
    if score >= 3.0:
        return "clean"
    if score >= 1.25:
        return "constructive"
    return "neutral"


def _hot_proto(dist_sma220: float, r21: float, rr_score: float, rel_recent: float, heat: int) -> str:
    if not heat:
        return "not_hot"
    if (
        rel_recent >= 4.5
        and rr_score >= 6.5
        and r21 <= 0.30
        and dist_sma220 >= 0.25
        and dist_sma220 <= 1.00
    ):
        return "hot_constructive_proto"
    if dist_sma220 >= 1.20 or r21 >= 0.60:
        return "hot_late_proto"
    return "hot_watch_proto"


def add_entry_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    proto_input_cols = (
        "scan_dist_sma220",
        "scan_r21",
        "scan_rr_score",
        "scan_rel_recent_score",
    )
    existing_entry_hot_proto = out["entry_hot_proto"].copy() if "entry_hot_proto" in out.columns else pd.Series("not_hot", index=out.index)
    proto_inputs_present = pd.Series(True, index=out.index, dtype=bool)
    for col in proto_input_cols:
        if col in out.columns:
            proto_inputs_present &= out[col].notna()
        else:
            proto_inputs_present &= False
    for col in (
        "scan_pullback_quality",
        "scan_trend200_component",
        "scan_breakout252_component",
        "scan_early_leader_score",
        "scan_dist_sma220",
        "recent_score",
        "scan_rel_recent_score",
        "scan_rr_score",
        "scan_r21",
        "scan_entry_heat_flag",
        "scan_post_earnings_5d",
        "scan_earnings_blackout_5d",
        "scan_gap_zscore_20",
        "scan_gap_fill_share",
        "scan_rel_volume_20",
        "scan_close_in_range",
        "scan_efficiency_ratio_60",
        "scan_corr_63_spy",
        "scan_downside_beta_63_spy",
        "scan_sector_corr_63",
        "scan_sector_beta_63",
    ):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    pullback = out["scan_pullback_quality"].clip(lower=0.0, upper=2.0)
    trend = out["scan_trend200_component"].clip(lower=0.0, upper=2.0)
    breakout = out["scan_breakout252_component"].clip(lower=0.0, upper=1.5)
    early_leader = out["scan_early_leader_score"].clip(lower=0.0, upper=10.0)
    recent = out["recent_score"].clip(lower=0.0, upper=6.0)
    rel_recent = out["scan_rel_recent_score"].clip(lower=0.0, upper=6.0)
    rr_score = out["scan_rr_score"].clip(lower=0.0, upper=8.0)
    r21 = out["scan_r21"].clip(lower=-40.0, upper=40.0)
    heat = out["scan_entry_heat_flag"].clip(lower=0.0, upper=1.0)
    post_earnings = out["scan_post_earnings_5d"].clip(lower=0.0, upper=1.0)
    earnings_blackout = out["scan_earnings_blackout_5d"].clip(lower=0.0, upper=1.0)
    gap_abs = out["scan_gap_zscore_20"].abs().clip(lower=0.0, upper=4.0)
    gap_fill = out["scan_gap_fill_share"].clip(lower=0.0, upper=1.0)
    rel_volume = out["scan_rel_volume_20"].clip(lower=0.0, upper=6.0)
    close_in_range = out["scan_close_in_range"].clip(lower=0.0, upper=1.0)
    efficiency60 = out["scan_efficiency_ratio_60"].clip(lower=0.0, upper=1.0)
    crowding = (
        0.30 * (out["scan_corr_63_spy"].clip(lower=-1.0, upper=1.0) >= 0.85).astype(float)
        + 0.20 * (out["scan_downside_beta_63_spy"].clip(lower=-3.0, upper=4.0) >= 1.20).astype(float)
        + 0.30 * (out["scan_sector_corr_63"].clip(lower=-1.0, upper=1.0) >= 0.92).astype(float)
        + 0.20 * (out["scan_sector_beta_63"].clip(lower=-3.0, upper=4.0) >= 1.40).astype(float)
    )
    gap_discipline = (0.6 * (1.0 - gap_abs / 4.0) + 0.4 * (1.0 - gap_fill)).clip(lower=0.0, upper=1.0)
    volume_confirm = ((rel_volume - 0.8) / 2.0).clip(lower=0.0, upper=1.0)

    score = (
        1.75 * pullback
        + 1.10 * trend
        + 0.50 * breakout
        + 0.24 * rel_recent
        + 0.12 * recent
        + 0.16 * rr_score
        + 0.06 * early_leader
        + 0.55 * close_in_range
        + 0.75 * efficiency60
        + 0.30 * gap_discipline
        + 0.20 * volume_confirm
        + 0.50 * post_earnings
        - 1.85 * heat
        - 1.25 * earnings_blackout
        - 0.60 * crowding
        - 0.45 * (gap_abs >= 2.0).astype(float)
        - 0.35 * (gap_fill >= 0.75).astype(float)
        - 0.06 * (r21.clip(lower=10.0) - 10.0)
        - 0.15 * heat * breakout
    )
    out["entry_timing_score"] = score.round(4)
    out["entry_timing_bucket"] = [
        _entry_bucket(s, int(b), int(h))
        for s, b, h in zip(
            out["entry_timing_score"],
            earnings_blackout.astype(int),
            heat.astype(int),
        )
    ]
    out["entry_timing_zone"] = [
        _entry_zone(s, int(b), int(h))
        for s, b, h in zip(
            out["entry_timing_score"],
            earnings_blackout.astype(int),
            heat.astype(int),
        )
    ]
    out["entry_hot_proto"] = [
        _hot_proto(d, r, rr, rel, int(h))
        for d, r, rr, rel, h in zip(
            out["scan_dist_sma220"].fillna(0.0),
            out["scan_r21"].fillna(0.0),
            out["scan_rr_score"].fillna(0.0),
            out["scan_rel_recent_score"].fillna(0.0),
            heat.astype(int),
        )
    ]
    if "scan_hot_archetype" in out.columns:
        archetype_map = {
            "hot_constructive": "hot_constructive_proto",
            "hot_late": "hot_late_proto",
            "hot_watch": "hot_watch_proto",
            "not_hot": "not_hot",
        }
        scan_proto = out["scan_hot_archetype"].fillna("").astype(str).map(archetype_map)
        use_scan_proto = scan_proto.notna() & scan_proto.ne("")
        out.loc[use_scan_proto, "entry_hot_proto"] = scan_proto.loc[use_scan_proto]
    fallback_mask = (~proto_inputs_present) & existing_entry_hot_proto.fillna("").astype(str).ne("")
    if fallback_mask.any():
        out.loc[fallback_mask, "entry_hot_proto"] = existing_entry_hot_proto.loc[fallback_mask].astype(str)
    return out


def _load_close_history(tickers: list[str]) -> dict[str, pd.DataFrame]:
    if not tickers or not OHLCV_PATH.exists():
        return {}
    price = read_optional_csv(OHLCV_PATH, usecols=["date", "ticker", "adj_close", "close"])
    if price.empty:
        return {}
    price = price.loc[price["ticker"].isin(tickers)].copy()
    if price.empty:
        return {}
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price = price.dropna(subset=["date"])
    if "adj_close" in price.columns:
        price["px"] = pd.to_numeric(price["adj_close"], errors="coerce")
    else:
        price["px"] = pd.to_numeric(price["close"], errors="coerce")
    price["px"] = price["px"].fillna(pd.to_numeric(price["close"], errors="coerce"))
    out: dict[str, pd.DataFrame] = {}
    for ticker, grp in price.groupby("ticker", dropna=False):
        out[str(ticker)] = grp.sort_values("date")[["date", "px"]].reset_index(drop=True)
    return out


def _load_portfolio_positions() -> pd.DataFrame:
    if not PORTFOLIO_PATH.exists():
        return pd.DataFrame(columns=["ticker", "shares", "cost_eur"])
    try:
        payload = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return pd.DataFrame(columns=["ticker", "shares", "cost_eur"])
    positions = payload.get("positions", {}) if isinstance(payload, dict) else {}
    rows: list[dict[str, object]] = []
    for ticker, meta in positions.items():
        if not isinstance(meta, dict):
            continue
        rows.append(
            {
                "ticker": str(ticker),
                "shares": safe_float(meta.get("shares")),
                "cost_eur": safe_float(meta.get("cost_eur")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["ticker", "shares", "cost_eur"])
    out = pd.DataFrame(rows)
    out = out.loc[out["ticker"].ne("") & ((out["shares"] > 0) | (out["cost_eur"] > 0))].copy()
    return out.reset_index(drop=True)


def _latest_series_returns(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["date", "ret"])
    out = prices.sort_values("date")[["date", "px"]].copy()
    out["ret"] = pd.to_numeric(out["px"], errors="coerce").pct_change()
    out = out.dropna(subset=["date", "ret"])
    if window > 0:
        out = out.tail(window)
    return out.reset_index(drop=True)


def build_portfolio_crowding_state(df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if "sector" not in df.columns:
        df = df.copy()
        df["sector"] = ""
    if "industry" not in df.columns:
        df = df.copy()
        df["industry"] = ""
    positions = _load_portfolio_positions()
    if positions.empty:
        return pd.DataFrame()

    tickers = sorted(set(df["ticker"].astype(str)).union(set(positions["ticker"].astype(str))))
    prices = _load_close_history(tickers)
    if not prices:
        return pd.DataFrame()

    holdings = positions.loc[positions["ticker"].isin(prices.keys())].copy()
    if holdings.empty:
        return pd.DataFrame()

    market_values: list[float] = []
    valid_tickers: list[str] = []
    for row in holdings.itertuples(index=False):
        px_series = prices.get(str(row.ticker), pd.DataFrame())
        if px_series.empty:
            continue
        latest_px = safe_float(px_series["px"].iloc[-1])
        market_value = latest_px * max(0.0, safe_float(row.shares))
        if market_value <= 0:
            market_value = max(0.0, safe_float(row.cost_eur))
        if market_value <= 0:
            continue
        valid_tickers.append(str(row.ticker))
        market_values.append(market_value)
    if not valid_tickers:
        return pd.DataFrame()

    total_value = sum(market_values)
    if total_value <= 0:
        weights = {ticker: 1.0 / len(valid_tickers) for ticker in valid_tickers}
    else:
        weights = {ticker: value / total_value for ticker, value in zip(valid_tickers, market_values)}

    portfolio_frames: list[pd.DataFrame] = []
    for ticker in valid_tickers:
        ret_df = _latest_series_returns(prices.get(ticker, pd.DataFrame()), window=window)
        if ret_df.empty:
            continue
        weighted = ret_df[["date", "ret"]].rename(columns={"ret": ticker}).copy()
        weighted[ticker] = weighted[ticker] * weights.get(ticker, 0.0)
        portfolio_frames.append(weighted)
    if not portfolio_frames:
        return pd.DataFrame()

    portfolio_ret = portfolio_frames[0]
    for frame in portfolio_frames[1:]:
        portfolio_ret = portfolio_ret.merge(frame, on="date", how="outer")
    holding_cols = [col for col in portfolio_ret.columns if col != "date"]
    portfolio_ret[holding_cols] = portfolio_ret[holding_cols].apply(pd.to_numeric, errors="coerce")
    portfolio_ret["portfolio_ret"] = portfolio_ret[holding_cols].sum(axis=1, min_count=1)
    portfolio_ret = portfolio_ret.dropna(subset=["portfolio_ret"]).sort_values("date").tail(window)
    if portfolio_ret.empty:
        return pd.DataFrame()

    portfolio_sectors = set(
        df.loc[
            df["ticker"].astype(str).isin(valid_tickers),
            "sector",
        ].fillna("").astype(str).str.strip()
    )
    portfolio_sectors.discard("")
    portfolio_industries = set(
        df.loc[
            df["ticker"].astype(str).isin(valid_tickers),
            "industry",
        ].fillna("").astype(str).str.strip()
    )
    portfolio_industries.discard("")

    rows: list[dict[str, object]] = []
    portfolio_base = portfolio_ret[["date", "portfolio_ret"]].copy()
    for ticker in df["ticker"].fillna("").astype(str).unique():
        price_df = prices.get(ticker, pd.DataFrame())
        ret_df = _latest_series_returns(price_df, window=window)
        sector = ""
        industry = ""
        if "ticker" in df.columns:
            match = df.loc[df["ticker"].astype(str) == ticker]
            if not match.empty:
                sector = str(match.iloc[0].get("sector") or "").strip()
                industry = str(match.iloc[0].get("industry") or "").strip()
        if ret_df.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "portfolio_corr_63": 0.0,
                    "portfolio_beta_63": 0.0,
                    "portfolio_downside_beta_63": 0.0,
                    "portfolio_sector_overlap": 1.0 if sector and sector in portfolio_sectors else 0.0,
                    "portfolio_industry_overlap": 1.0 if industry and industry in portfolio_industries else 0.0,
                    "portfolio_crowding_score": 0.0,
                    "portfolio_diversification_score": 0.0,
                }
            )
            continue

        merged = portfolio_base.merge(ret_df[["date", "ret"]], on="date", how="inner")
        merged["portfolio_ret"] = pd.to_numeric(merged["portfolio_ret"], errors="coerce")
        merged["ret"] = pd.to_numeric(merged["ret"], errors="coerce")
        merged = merged.dropna(subset=["portfolio_ret", "ret"]).tail(window)

        corr = 0.0
        beta = 0.0
        downside_beta = 0.0
        if len(merged) >= 20:
            port_var = safe_float(merged["portfolio_ret"].var())
            corr = clip_small_float(merged["ret"].corr(merged["portfolio_ret"]))
            if port_var > FLOAT_ZERO_EPS:
                beta = clip_small_float(merged["ret"].cov(merged["portfolio_ret"]) / port_var)
            downside = merged.loc[merged["portfolio_ret"] < 0]
            if len(downside) >= 10:
                down_var = safe_float(downside["portfolio_ret"].var())
                if down_var > FLOAT_ZERO_EPS:
                    downside_beta = clip_small_float(downside["ret"].cov(downside["portfolio_ret"]) / down_var)

        sector_overlap = 1.0 if sector and sector in portfolio_sectors else 0.0
        industry_overlap = 1.0 if industry and industry in portfolio_industries else 0.0
        corr_norm = ((max(-1.0, min(1.0, corr)) + 1.0) / 2.0)
        beta_norm = max(0.0, min(2.0, beta)) / 2.0
        downside_beta_norm = max(0.0, min(2.0, downside_beta)) / 2.0
        crowding = (
            0.45 * max(0.0, (corr - 0.55) / 0.35)
            + 0.20 * max(0.0, (beta - 0.90) / 0.80)
            + 0.15 * max(0.0, (downside_beta - 0.85) / 0.80)
            + 0.12 * sector_overlap
            + 0.08 * industry_overlap
        )
        crowding = max(0.0, min(1.5, crowding))
        diversification = (
            0.55 * (1.0 - corr_norm)
            + 0.20 * (1.0 - beta_norm)
            + 0.10 * (1.0 - downside_beta_norm)
            + 0.10 * (1.0 - sector_overlap)
            + 0.05 * (1.0 - industry_overlap)
        )
        diversification = max(0.0, min(1.25, diversification))
        rows.append(
            {
                "ticker": ticker,
                "portfolio_corr_63": round(corr, 4),
                "portfolio_beta_63": round(beta, 4),
                "portfolio_downside_beta_63": round(downside_beta, 4),
                "portfolio_sector_overlap": sector_overlap,
                "portfolio_industry_overlap": industry_overlap,
                "portfolio_crowding_score": round(crowding, 4),
                "portfolio_diversification_score": round(diversification, 4),
            }
        )
    return pd.DataFrame(rows)


def _return_between(prices: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    if prices.empty:
        return 0.0
    start_match = prices.loc[prices["date"] >= start_date, "px"]
    end_match = prices.loc[prices["date"] >= end_date, "px"]
    if start_match.empty or end_match.empty:
        return 0.0
    start_px = safe_float(start_match.iloc[0])
    end_px = safe_float(end_match.iloc[0])
    if start_px <= 0:
        return 0.0
    return 100.0 * (end_px / start_px - 1.0)


def build_shadow_probation_state(limit: int = 8, save: bool = True) -> pd.DataFrame:
    snapshots = sorted(HISTORY_DIR.glob("dynamic_universe_snapshot_*.csv"))
    if not snapshots:
        return pd.DataFrame()
    snapshots = snapshots[-limit:]

    frames: list[pd.DataFrame] = []
    for path in snapshots:
        df = read_optional_csv(path)
        if df.empty or "ticker" not in df.columns:
            continue
        keep_cols = [
            "ticker",
            "promotion_stage",
            "dynamic_status",
            "validation_lane",
            "scan_entry_heat_flag",
            "entry_timing_score",
            "entry_timing_zone",
            "entry_hot_proto",
            "scan_dist_sma220",
            "scan_r21",
            "scan_rr_score",
            "scan_rel_recent_score",
        ]
        out = df[[col for col in keep_cols if col in df.columns]].copy()
        out["snapshot_date"] = pd.to_datetime(path.stem.replace("dynamic_universe_snapshot_", ""), errors="coerce")
        frames.append(out)
    if not frames:
        return pd.DataFrame()

    history = pd.concat(frames, ignore_index=True, sort=False)
    for col in ("promotion_stage", "dynamic_status", "validation_lane", "scan_entry_heat_flag"):
        if col not in history.columns:
            history[col] = ""
    history = add_entry_timing_features(history)
    history = history.dropna(subset=["snapshot_date"]).sort_values(["snapshot_date", "ticker"])
    history["stage_rank"] = history["promotion_stage"].fillna("").astype(str).map(promotion_stage_rank).fillna(-2)
    history["status_rank"] = history["dynamic_status"].fillna("").astype(str).map(status_rank).fillna(-1)

    snapshot_dates = sorted(history["snapshot_date"].dropna().unique())
    if len(snapshot_dates) < 2:
        return pd.DataFrame()
    next_dates = {snapshot_dates[i]: snapshot_dates[i + 1] for i in range(len(snapshot_dates) - 1)}

    keyed = {
        (str(row.ticker), row.snapshot_date): row
        for row in history.itertuples(index=False)
    }
    active_mask = (
        (history["stage_rank"] >= promotion_stage_rank("targeted_integration"))
        | (history["status_rank"] >= status_rank("watch"))
    )
    active = history.loc[active_mask].copy()
    if active.empty:
        return pd.DataFrame()

    prices = _load_close_history(sorted(active["ticker"].astype(str).unique().tolist()))
    rows: list[dict[str, object]] = []
    for row in active.itertuples(index=False):
        next_date = next_dates.get(row.snapshot_date)
        if next_date is None:
            continue
        next_row = keyed.get((str(row.ticker), next_date))
        next_stage_rank = promotion_stage_rank(getattr(next_row, "promotion_stage", "")) if next_row is not None else -2
        next_status_rank = status_rank(getattr(next_row, "dynamic_status", "")) if next_row is not None else -1
        still_tracked = int(
            next_stage_rank >= promotion_stage_rank("targeted_integration")
            or next_status_rank >= status_rank("watch")
        )
        rows.append(
            {
                "ticker": str(row.ticker),
                "snapshot_date": row.snapshot_date,
                "next_snapshot_date": next_date,
                "stage_rank": int(row.stage_rank),
                "next_stage_rank": int(next_stage_rank),
                "stage_upgrade": int(next_stage_rank > row.stage_rank),
                "still_tracked": still_tracked,
                "scan_entry_heat_flag": safe_int(getattr(row, "scan_entry_heat_flag", 0)),
                "entry_timing_score": safe_float(getattr(row, "entry_timing_score", 0.0)),
                "entry_hot_proto": str(getattr(row, "entry_hot_proto", "not_hot") or "not_hot"),
                "next_snapshot_return_pct": clip_small_float(
                    _return_between(prices.get(str(row.ticker), pd.DataFrame()), row.snapshot_date, next_date)
                ),
            }
        )
    if not rows:
        return pd.DataFrame()

    eval_df = pd.DataFrame(rows)
    summary = (
        eval_df.groupby("ticker", dropna=False)
        .agg(
            shadow_snapshots=("snapshot_date", "count"),
            shadow_mean_next_return_pct=("next_snapshot_return_pct", "mean"),
            shadow_positive_rate=("next_snapshot_return_pct", lambda s: float((s > 0).mean()) if len(s) else 0.0),
            shadow_stage_persist_rate=("still_tracked", "mean"),
            shadow_stage_upgrade_rate=("stage_upgrade", "mean"),
            shadow_heat_rate=("scan_entry_heat_flag", "mean"),
            shadow_entry_timing_mean=("entry_timing_score", "mean"),
            shadow_hot_constructive_rate=("entry_hot_proto", lambda s: float((s == "hot_constructive_proto").mean()) if len(s) else 0.0),
            shadow_hot_late_rate=("entry_hot_proto", lambda s: float((s == "hot_late_proto").mean()) if len(s) else 0.0),
        )
        .reset_index()
    )
    summary["shadow_mean_next_return_pct"] = summary["shadow_mean_next_return_pct"].fillna(0.0).round(4)
    summary["shadow_positive_rate"] = summary["shadow_positive_rate"].fillna(0.0).round(4)
    summary["shadow_stage_persist_rate"] = summary["shadow_stage_persist_rate"].fillna(0.0).round(4)
    summary["shadow_stage_upgrade_rate"] = summary["shadow_stage_upgrade_rate"].fillna(0.0).round(4)
    summary["shadow_heat_rate"] = summary["shadow_heat_rate"].fillna(0.0).round(4)
    summary["shadow_entry_timing_mean"] = summary["shadow_entry_timing_mean"].fillna(0.0).round(4)
    summary["shadow_hot_constructive_rate"] = summary["shadow_hot_constructive_rate"].fillna(0.0).round(4)
    summary["shadow_hot_late_rate"] = summary["shadow_hot_late_rate"].fillna(0.0).round(4)
    summary["shadow_probation_score"] = (
        0.25 * summary["shadow_snapshots"].clip(upper=4)
        + 0.60 * summary["shadow_stage_persist_rate"]
        + 0.40 * summary["shadow_stage_upgrade_rate"]
        + 0.35 * summary["shadow_positive_rate"]
        + 0.08 * summary["shadow_mean_next_return_pct"]
        + 0.15 * summary["shadow_entry_timing_mean"]
        + 0.35 * summary["shadow_hot_constructive_rate"]
        - 0.50 * summary["shadow_heat_rate"]
        - 0.45 * summary["shadow_hot_late_rate"]
    ).round(4)

    readiness = []
    for row in summary.itertuples(index=False):
        late_drag = row.shadow_hot_late_rate >= 0.5 and row.shadow_hot_constructive_rate <= 0.0 and row.shadow_positive_rate < 0.6
        if (
            row.shadow_snapshots >= 2
            and row.shadow_stage_persist_rate >= 0.5
            and row.shadow_positive_rate >= 0.5
            and row.shadow_probation_score >= 1.5
            and not late_drag
        ):
            readiness.append("shadow_ready")
        elif row.shadow_snapshots >= 1 and (row.shadow_stage_persist_rate >= 0.5 or row.shadow_positive_rate > 0):
            readiness.append("shadow_building")
        else:
            readiness.append("shadow_weak")
    summary["shadow_readiness"] = readiness

    if save:
        summary.sort_values(["shadow_readiness", "shadow_probation_score"], ascending=[False, False]).to_csv(SHADOW_PATH, index=False)
    return summary


def _parse_walk_pairs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "candidate" not in out.columns:
        out["candidate"] = pd.NA
    if "removed" not in out.columns:
        out["removed"] = pd.NA
    if "swap" in out.columns:
        needs_parse = out["candidate"].isna() | out["removed"].isna()
        if needs_parse.any():
            parsed = out.loc[needs_parse, "swap"].fillna("").astype(str).str.split("_for_", n=1, expand=True)
            if not parsed.empty and parsed.shape[1] == 2:
                out.loc[needs_parse, "candidate"] = parsed[0].values
                out.loc[needs_parse, "removed"] = parsed[1].values
    return out


def build_swap_memory(save: bool = True) -> pd.DataFrame:
    single = read_optional_csv(SINGLE_SWAP_PATH)
    if single.empty or "candidate" not in single.columns or "removed" not in single.columns:
        return pd.DataFrame()

    walk_frames = [read_optional_csv(SINGLE_WALK_PATH), read_optional_csv(LEGACY_SINGLE_WALK_PATH)]
    walk = pd.concat([frame for frame in walk_frames if not frame.empty], ignore_index=True, sort=False)
    walk = _parse_walk_pairs(walk) if not walk.empty else pd.DataFrame()
    if not walk.empty:
        walk = walk.drop_duplicates(["candidate", "removed"], keep="first")

    merged = single.copy()
    if not walk.empty:
        merged = merged.merge(
            walk[
                [
                    "candidate",
                    "removed",
                    "mean_delta_roi_2017_2025",
                    "mean_delta_sharpe_2017_2025",
                    "mean_delta_maxdd_2017_2025",
                    "roi_wins_2017_2025",
                    "sharpe_wins_2017_2025",
                ]
            ],
            on=["candidate", "removed"],
            how="left",
        )
    for col in (
        "full_delta_roi_pct",
        "oos_delta_roi_pct",
        "full_delta_sharpe",
        "oos_delta_sharpe",
        "mean_delta_roi_2017_2025",
        "mean_delta_sharpe_2017_2025",
        "mean_delta_maxdd_2017_2025",
        "roi_wins_2017_2025",
        "sharpe_wins_2017_2025",
    ):
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["pair_memory_score"] = (
        700.0 * merged["mean_delta_sharpe_2017_2025"]
        + 8.0 * merged["mean_delta_roi_2017_2025"]
        + 0.03 * merged["oos_delta_roi_pct"]
        + 12.0 * merged["oos_delta_sharpe"]
        + 0.015 * merged["full_delta_roi_pct"]
    ).round(4)

    labels: list[str] = []
    for row in merged.itertuples(index=False):
        if (
            row.mean_delta_sharpe_2017_2025 > 0
            and row.mean_delta_roi_2017_2025 > 0
            and row.oos_delta_roi_pct > 0
        ):
            labels.append("preferred")
        elif row.oos_delta_roi_pct > 0 or row.mean_delta_roi_2017_2025 > 0 or row.full_delta_roi_pct > 0:
            labels.append("viable")
        else:
            labels.append("weak")
    merged["pair_memory_label"] = labels

    candidate_summary = (
        merged.sort_values(["candidate", "pair_memory_score"], ascending=[True, False])
        .groupby("candidate", dropna=False)
        .agg(
            candidate_memory_score=("pair_memory_score", lambda s: float(s.head(3).mean()) if len(s) else 0.0),
            candidate_memory_preferred=("pair_memory_label", lambda s: int((s == "preferred").sum())),
            candidate_memory_pairs=("pair_memory_score", "count"),
        )
        .reset_index()
    )
    removed_summary = (
        merged.sort_values(["removed", "pair_memory_score"], ascending=[True, False])
        .groupby("removed", dropna=False)
        .agg(
            removed_memory_score=("pair_memory_score", lambda s: float(s.head(3).mean()) if len(s) else 0.0),
            removed_memory_preferred=("pair_memory_label", lambda s: int((s == "preferred").sum())),
            removed_memory_pairs=("pair_memory_score", "count"),
        )
        .reset_index()
    )

    out = merged.merge(candidate_summary, on="candidate", how="left").merge(removed_summary, on="removed", how="left")
    for col in (
        "candidate_memory_score",
        "removed_memory_score",
        "candidate_memory_preferred",
        "removed_memory_preferred",
        "candidate_memory_pairs",
        "removed_memory_pairs",
    ):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    if save:
        out.sort_values(["pair_memory_score", "oos_delta_roi_pct", "full_delta_roi_pct"], ascending=[False, False, False]).to_csv(SWAP_MEMORY_PATH, index=False)
    return out


def load_swap_memory() -> pd.DataFrame:
    memory = read_optional_csv(SWAP_MEMORY_PATH)
    if not memory.empty:
        return memory
    return build_swap_memory(save=True)


def enrich_database_governance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = add_entry_timing_features(df)

    shadow = build_shadow_probation_state(save=True)
    if not shadow.empty:
        out = out.merge(shadow, on="ticker", how="left")
    for col in (
        "shadow_snapshots",
        "shadow_mean_next_return_pct",
        "shadow_positive_rate",
        "shadow_stage_persist_rate",
        "shadow_stage_upgrade_rate",
        "shadow_heat_rate",
        "shadow_entry_timing_mean",
        "shadow_hot_constructive_rate",
        "shadow_hot_late_rate",
        "shadow_probation_score",
    ):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "shadow_readiness" not in out.columns:
        out["shadow_readiness"] = "shadow_weak"
    out["shadow_readiness"] = out["shadow_readiness"].fillna("shadow_weak")

    swap_memory = build_swap_memory(save=True)
    if not swap_memory.empty:
        candidate_cols = (
            swap_memory[["candidate", "candidate_memory_score", "candidate_memory_preferred", "candidate_memory_pairs"]]
            .drop_duplicates("candidate")
            .rename(columns={"candidate": "ticker"})
        )
        removed_cols = (
            swap_memory[["removed", "removed_memory_score", "removed_memory_preferred", "removed_memory_pairs"]]
            .drop_duplicates("removed")
            .rename(columns={"removed": "ticker"})
        )
        out = out.merge(candidate_cols, on="ticker", how="left").merge(removed_cols, on="ticker", how="left")
    for col in (
        "candidate_memory_score",
        "candidate_memory_preferred",
        "candidate_memory_pairs",
        "removed_memory_score",
        "removed_memory_preferred",
        "removed_memory_pairs",
    ):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    portfolio_state = build_portfolio_crowding_state(out)
    if not portfolio_state.empty:
        out = out.merge(portfolio_state, on="ticker", how="left")
    for col in (
        "portfolio_corr_63",
        "portfolio_beta_63",
        "portfolio_downside_beta_63",
        "portfolio_sector_overlap",
        "portfolio_industry_overlap",
        "portfolio_crowding_score",
        "portfolio_diversification_score",
    ):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out
