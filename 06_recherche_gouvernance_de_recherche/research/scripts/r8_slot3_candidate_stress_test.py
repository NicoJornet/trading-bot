from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\nicol\Downloads\214")
BEST_PATH = ROOT / "BEST_ALGO_184.json"
CSV_PATH = ROOT / "apex_ohlcv_full_2015_2026.csv"
ACTIVE_PATH = ROOT / "trading-bot" / "data" / "extracts" / "apex_tickers_active.csv"

OUT_SUMMARY = ROOT / "research" / "exports" / "r8_slot3_candidate_stress_summary.csv"
OUT_YEARLY = ROOT / "research" / "exports" / "r8_slot3_candidate_yearly_comparison.csv"
OUT_EPISODES = ROOT / "research" / "exports" / "r8_slot3_candidate_episode_comparison.csv"
OUT_WORST_DAYS = ROOT / "research" / "exports" / "r8_slot3_candidate_worst_days_comparison.csv"
OUT_MD = ROOT / "research" / "reports" / "R8_SLOT3_CANDIDATE_STRESS_TEST.md"


def load_module(path: Path, name: str):
    from importlib import util

    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dd_mod = load_module(ROOT / "research" / "scripts" / "r4_drawdown_deep_analysis.py", "r8_slot3_candidate_dd")
slot3_mod = load_module(ROOT / "research" / "scripts" / "r8_slot3_admission_sizing_study.py", "r8_slot3_candidate_slot3")


def load_cfg():
    obj = json.loads(BEST_PATH.read_text(encoding="utf-8"))
    return obj["cfg"], obj["pp"], obj


def year_returns_from_nav(nav: pd.Series) -> pd.DataFrame:
    nav = nav.dropna().copy()
    rows: list[dict[str, float | int]] = []
    for year, grp in nav.groupby(nav.index.year):
        if len(grp) < 2:
            continue
        roi = float(grp.iloc[-1] / grp.iloc[0] - 1.0) * 100.0
        daily = grp.pct_change().dropna()
        sharpe = float((daily.mean() / (daily.std() + 1e-12)) * np.sqrt(252.0)) if len(daily) > 1 else np.nan
        peak = grp.cummax()
        maxdd = float((grp / peak - 1.0).min()) * 100.0
        rows.append(
            {
                "year": int(year),
                "roi_twr_pct": roi,
                "sharpe": sharpe,
                "maxdd_pct": maxdd,
            }
        )
    return pd.DataFrame(rows)


def top_episode_frame(nav: pd.Series, label: str) -> pd.DataFrame:
    episodes = dd_mod.extract_drawdown_episodes(nav).copy()
    if episodes.empty:
        return episodes
    episodes["variant"] = label
    return episodes.loc[
        :,
        [
            "variant",
            "episode_rank",
            "peak_date",
            "trough_date",
            "recovery_date",
            "recovered",
            "drawdown_pct",
            "days_to_trough",
            "days_to_recovery",
            "recovery_days_after_trough",
        ],
    ]


def worst_day_frame(nav: pd.Series, label: str) -> pd.DataFrame:
    ret = nav.pct_change().fillna(0.0) * 100.0
    out = pd.DataFrame(
        {
            "variant": label,
            "date": nav.index,
            "nav": nav.values,
            "strategy_return_pct": ret.values,
        }
    )
    return out.sort_values("strategy_return_pct").reset_index(drop=True)


def markdown_table(df: pd.DataFrame) -> str:
    return dd_mod.markdown_table(df)


def render_report(summary: pd.DataFrame, yearly: pd.DataFrame, episodes: pd.DataFrame, worst_days: pd.DataFrame) -> str:
    summary_view = summary.copy()
    yearly_view = yearly.copy()
    episode_view = episodes.copy()
    worst_view = worst_days.copy()
    lines = [
        "# R8 Slot-3 Candidate Stress Test",
        "",
        "Stress-test of the `slot3_cap35_onlyrs05` candidate against the active `r8` baseline.",
        "",
        "Question:",
        "- does the apparent full-period gain come from a broad portfolio improvement,",
        "- or mostly from a narrow historical pocket that would be too fragile to promote?",
        "",
        "## Summary",
        "",
        markdown_table(summary_view),
        "",
        "## Year-By-Year Delta",
        "",
        markdown_table(yearly_view),
        "",
        "## Top Drawdown Episodes",
        "",
        markdown_table(episode_view),
        "",
        "## Worst Daily Returns",
        "",
        markdown_table(worst_view),
        "",
    ]
    return "\n".join(lines) + "\n"


def main():
    cfg, pp, doc = load_cfg()
    engine = slot3_mod.load_engine()
    prices = engine.load_prices_from_csv(str(CSV_PATH))
    replay_end = str(pd.to_datetime(prices.close.index.max()).date())

    base_tickers = slot3_mod.read_tickers(ACTIVE_PATH)
    activation_dates = {str(k): str(v) for k, v in doc.get("staged_activation_dates", {}).items() if str(k) in prices.close.columns}
    membership = slot3_mod.membership_frame(prices.close.index, prices.close.columns, base_tickers, activation_dates)
    run_fn = slot3_mod.build_dynamic_runner(engine, membership)

    variants = {
        "baseline_current": {},
        "slot3_cap35_onlyrs05": {
            "slot3_marginal_weight_cap_enable": 1,
            "slot3_marginal_rank_soft_max": 999,
            "slot3_marginal_rs63_min": 0.05,
            "slot3_marginal_require_both": 0,
            "slot3_marginal_weight_cap": 0.35,
        },
    }

    eq_map: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, float | int | str]] = []
    yearly_map: dict[str, pd.DataFrame] = {}
    episode_map: dict[str, pd.DataFrame] = {}
    worst_map: dict[str, pd.DataFrame] = {}

    for label, patch in variants.items():
        local_cfg = dict(cfg)
        local_cfg.update(patch)
        eq, tr, metrics = slot3_mod.run_variant(run_fn, prices, local_cfg, pp, "2015-01-02", replay_end)
        equity = eq["equity"].astype(float).copy()
        equity.index = pd.to_datetime(eq.index)
        nav, _flows = dd_mod.flow_adjusted_nav(equity, float(local_cfg.get("monthly_dca", 0.0)))
        eq_map[label] = nav
        summary_rows.append(
            {
                "variant": label,
                "roi_pct": float(metrics["ROI_%"]),
                "cagr_pct": float(metrics["CAGR_%"]),
                "maxdd_pct": float(metrics["MaxDD_%"]),
                "nav_maxdd_pct": float((nav / nav.cummax() - 1.0).min()) * 100.0,
                "sharpe": float(metrics["Sharpe"]),
                "orders": int(metrics["Orders"]),
            }
        )
        yearly_map[label] = year_returns_from_nav(nav)
        episode_map[label] = top_episode_frame(nav, label)
        worst_map[label] = worst_day_frame(nav, label)

    summary = pd.DataFrame(summary_rows)
    baseline = summary.loc[summary["variant"] == "baseline_current"].iloc[0]
    candidate = summary.loc[summary["variant"] == "slot3_cap35_onlyrs05"].iloc[0]
    delta_row = {
        "variant": "candidate_minus_baseline",
        "roi_pct": float(candidate["roi_pct"] - baseline["roi_pct"]),
        "cagr_pct": float(candidate["cagr_pct"] - baseline["cagr_pct"]),
        "maxdd_pct": float(candidate["maxdd_pct"] - baseline["maxdd_pct"]),
        "nav_maxdd_pct": float(candidate["nav_maxdd_pct"] - baseline["nav_maxdd_pct"]),
        "sharpe": float(candidate["sharpe"] - baseline["sharpe"]),
        "orders": int(candidate["orders"] - baseline["orders"]),
    }
    summary = pd.concat([summary, pd.DataFrame([delta_row])], ignore_index=True)

    yearly = yearly_map["baseline_current"].merge(
        yearly_map["slot3_cap35_onlyrs05"],
        on="year",
        how="outer",
        suffixes=("_baseline", "_candidate"),
    ).sort_values("year")
    yearly["delta_roi_twr_pct"] = yearly["roi_twr_pct_candidate"] - yearly["roi_twr_pct_baseline"]
    yearly["delta_sharpe"] = yearly["sharpe_candidate"] - yearly["sharpe_baseline"]
    yearly["delta_maxdd_pct"] = yearly["maxdd_pct_candidate"] - yearly["maxdd_pct_baseline"]

    episode_compare = episode_map["baseline_current"].merge(
        episode_map["slot3_cap35_onlyrs05"],
        on="episode_rank",
        how="outer",
        suffixes=("_baseline", "_candidate"),
    )
    episode_compare["delta_drawdown_pct"] = episode_compare["drawdown_pct_candidate"] - episode_compare["drawdown_pct_baseline"]
    episode_compare = episode_compare.loc[
        :,
        [
            "episode_rank",
            "peak_date_baseline",
            "trough_date_baseline",
            "drawdown_pct_baseline",
            "peak_date_candidate",
            "trough_date_candidate",
            "drawdown_pct_candidate",
            "delta_drawdown_pct",
        ],
    ].head(10)

    worst_compare = worst_map["baseline_current"].head(10).merge(
        worst_map["slot3_cap35_onlyrs05"].head(10),
        left_index=True,
        right_index=True,
        suffixes=("_baseline", "_candidate"),
    )
    worst_compare["delta_strategy_return_pct"] = (
        worst_compare["strategy_return_pct_candidate"] - worst_compare["strategy_return_pct_baseline"]
    )
    worst_compare = worst_compare.loc[
        :,
        [
            "date_baseline",
            "strategy_return_pct_baseline",
            "date_candidate",
            "strategy_return_pct_candidate",
            "delta_strategy_return_pct",
        ],
    ]

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_YEARLY.parent.mkdir(parents=True, exist_ok=True)
    OUT_EPISODES.parent.mkdir(parents=True, exist_ok=True)
    OUT_WORST_DAYS.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    yearly.to_csv(OUT_YEARLY, index=False)
    episode_compare.to_csv(OUT_EPISODES, index=False)
    worst_compare.to_csv(OUT_WORST_DAYS, index=False)
    OUT_MD.write_text(render_report(summary, yearly, episode_compare, worst_compare), encoding="utf-8")

    print(summary.to_string(index=False))
    print(yearly.to_string(index=False))
    print(episode_compare.to_string(index=False))
    print(worst_compare.to_string(index=False))
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_YEARLY}")
    print(f"Saved: {OUT_EPISODES}")
    print(f"Saved: {OUT_WORST_DAYS}")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()
