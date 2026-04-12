from __future__ import annotations

import inspect
import json
from importlib import util
from pathlib import Path

import pandas as pd

from report_governance import governance_section


ROOT = Path(r"C:\Users\nicol\Downloads\214")
ENGINE_PATH = ROOT / "trading-bot" / "engine_bundle" / "run_v11_slot_quality" / "v11_slot_quality" / "apex_engine_slot_quality.py"
BEST_PATH = ROOT / "BEST_ALGO_184.json"
CSV_PATH = ROOT / "apex_ohlcv_full_2015_2026.csv"
ACTIVE_PATH = ROOT / "trading-bot" / "data" / "extracts" / "apex_tickers_active.csv"
MISSING_SELECTION_PATH = ROOT / "research" / "exports" / "scan_algo_missing_selection_20260406.csv"

OUT_CSV = ROOT / "research" / "exports" / "r8_quality_compounder_ranking_study.csv"
OUT_MD = ROOT / "research" / "reports" / "R8_QUALITY_COMPOUNDER_RANKING_STUDY.md"


def load_module(path: Path, name: str):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_engine():
    return load_module(ENGINE_PATH, "apex_engine_slot_quality_quality_study")


def load_cfg():
    obj = json.loads(BEST_PATH.read_text(encoding="utf-8"))
    return obj["cfg"], obj["pp"], obj


def read_tickers(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return []
    return sorted({str(x).strip() for x in df["ticker"].dropna() if str(x).strip()})


def load_missing_selection_targets(topn: int = 20) -> list[str]:
    if not MISSING_SELECTION_PATH.exists():
        return []
    df = pd.read_csv(MISSING_SELECTION_PATH, low_memory=False)
    if df.empty or "ticker" not in df.columns:
        return []
    sort_cols = [c for c in ["priority_score", "total_return_pct", "max252_pct"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return df["ticker"].dropna().astype(str).head(topn).tolist()


def membership_frame(index: pd.Index, columns: pd.Index, base_tickers: list[str], staged_adds: dict[str, str]) -> pd.DataFrame:
    membership = pd.DataFrame(False, index=index, columns=columns)
    base_cols = [t for t in base_tickers if t in membership.columns]
    if base_cols:
        membership.loc[:, base_cols] = True
    for ticker, start in staged_adds.items():
        if ticker not in membership.columns:
            continue
        membership.loc[membership.index >= pd.Timestamp(start), ticker] = True
    return membership


def build_dynamic_runner(engine, universe_membership: pd.DataFrame):
    src = inspect.getsource(engine.run_backtest)
    needle = "        elig = elig & enough.reindex(c.columns, fill_value=False)\n"
    patch = needle + "        elig = elig & universe_membership.loc[d].reindex(c.columns, fill_value=False).astype(bool)\n"
    if needle not in src:
        raise RuntimeError("Unable to patch run_backtest for dynamic membership.")
    src = src.replace(needle, patch, 1)
    ns = dict(engine.__dict__)
    ns["universe_membership"] = universe_membership
    exec(src, ns)
    return ns["run_backtest"]


def run_variant(run_fn, prices, cfg: dict, pp: dict, start: str, end: str):
    local_cfg = dict(cfg)
    mie_enabled = bool(int(local_cfg.pop("mie_enabled", 1)))
    eq, trades, metrics = run_fn(
        prices=prices,
        start=start,
        end=end,
        pp_enabled=True,
        mie_enabled=mie_enabled,
        mie_rs_th=float(local_cfg.pop("mie_rs_th", pp["mie_rs_th"])),
        mie_min_hold=int(local_cfg.pop("mie_min_hold", pp["mie_min_hold"])),
        pp_mfe_trigger=float(pp["pp_mfe_trigger"]),
        pp_trail_dd=float(pp["pp_trail_dd"]),
        pp_min_days_after_arm=int(pp["pp_min_days_after_arm"]),
        **local_cfg,
    )
    return eq, trades, metrics


def year_returns(eq: pd.Series) -> dict[int, float]:
    out: dict[int, float] = {}
    if isinstance(eq, pd.DataFrame):
        if "Equity" in eq.columns:
            eq = eq["Equity"]
        else:
            eq = eq.iloc[:, 0]
    if eq.empty:
        return out
    years = sorted(set(eq.index.year))
    for year in years:
        part = eq.loc[eq.index.year == year].dropna()
        if part.empty:
            continue
        out[year] = float(part.iloc[-1] / part.iloc[0] - 1.0) * 100.0
    return out


def capture_stats(trades: pd.DataFrame, targets: list[str]) -> tuple[int, str]:
    if trades.empty:
        return 0, ""
    cols = {str(c).lower(): c for c in trades.columns}
    ticker_col = cols.get("ticker")
    if ticker_col is None:
        ticker_col = cols.get("symbol")
    if ticker_col is None:
        return 0, ""
    traded = sorted({str(x).strip() for x in trades[ticker_col].dropna().astype(str) if str(x).strip() in set(targets)})
    return len(traded), ",".join(traded)


def build_variants(cfg: dict) -> list[tuple[str, dict]]:
    variants: list[tuple[str, dict]] = [("baseline_current", dict(cfg))]

    def variant(name: str, **updates):
        local = dict(cfg)
        local.update(updates)
        variants.append((name, local))

    variant("quality063_005", score_quality_blend=0.05, score_quality_win=63)
    variant("quality063_010", score_quality_blend=0.10, score_quality_win=63)
    variant("quality126_010", score_quality_blend=0.10, score_quality_win=126)
    variant("quality126_015", score_quality_blend=0.15, score_quality_win=126)
    variant("rs003_quality010_063", score_rs_blend=0.03, score_quality_blend=0.10, score_quality_win=63)
    variant("rs005_quality010_063", score_rs_blend=0.05, score_quality_blend=0.10, score_quality_win=63)
    variant("resid003_quality010_063", score_residual_blend=0.03, score_quality_blend=0.10, score_quality_win=63)
    variant("resid005_quality010_126", score_residual_blend=0.05, score_quality_blend=0.10, score_quality_win=126)
    variant("rs003_breakout005_quality010", score_rs_blend=0.03, score_breakout_252_weight=0.05, score_quality_blend=0.10, score_quality_win=63)
    variant("rs003_resid003_quality010", score_rs_blend=0.03, score_residual_blend=0.03, score_quality_blend=0.10, score_quality_win=63)
    return variants


def render_report(summary: pd.DataFrame, targets: list[str]) -> str:
    baseline = summary.loc[summary["label"] == "baseline_current"].iloc[0]
    best = summary.iloc[0]
    lines = [
        "# R8 Quality Compounder Ranking Study",
        "",
        "Structural study on the core ranking lane above `r8`, using only existing engine score hooks (`RS`, `quality`, `residual`, `breakout`) on the reproducible staged replay.",
        "",
        "Target selection-miss basket used as diagnostic:",
        "",
        "- " + (", ".join(targets) if targets else "(none)"),
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Interpretation",
        "",
        f"- Best variant by study ordering: `{best['label']}`",
        f"- Full ROI delta vs baseline: `{best['full_roi_pct'] - baseline['full_roi_pct']:.4f}`",
        f"- OOS ROI delta vs baseline: `{best['oos_roi_pct'] - baseline['oos_roi_pct']:.4f}`",
        f"- Full Sharpe delta vs baseline: `{best['full_sharpe'] - baseline['full_sharpe']:.4f}`",
        f"- OOS Sharpe delta vs baseline: `{best['oos_sharpe'] - baseline['oos_sharpe']:.4f}`",
        f"- Full MaxDD delta vs baseline: `{best['full_maxdd_pct'] - baseline['full_maxdd_pct']:.4f}`",
        f"- OOS MaxDD delta vs baseline: `{best['oos_maxdd_pct'] - baseline['oos_maxdd_pct']:.4f}`",
        f"- Captured target names in full replay: `{int(best['target_capture_count_full'])}`",
        f"- Captured target names in OOS replay: `{int(best['target_capture_count_oos'])}`",
        "",
    ]
    lines.extend(
        governance_section(
            summary,
            "baseline_current",
            current_baseline="r8",
            robust_fallback="r7",
            study_verdict="reject" if best["label"] == "baseline_current" or not (best["full_roi_pct"] > baseline["full_roi_pct"] and best["oos_roi_pct"] > baseline["oos_roi_pct"] and best["full_sharpe"] > baseline["full_sharpe"] and best["oos_sharpe"] > baseline["oos_sharpe"]) else "monitor",
            recommendation="Do not promote unless the best variant improves both full and OOS with a clean drawdown profile and the gain is not concentrated in one year or one pocket of names.",
            economic_explanation="Quality / RS / residual overlays should only be promoted if they systematically help durable compounders without just reshuffling one narrow historical pocket.",
            concentration_note="Use the target-capture diagnostic only as a secondary clue; promotion still depends on full/OOS robustness rather than whether a few missed names become tradable.",
        )
    )
    return "\n".join(lines) + "\n"


def main():
    engine = load_engine()
    cfg, pp, doc = load_cfg()
    prices = engine.load_prices_from_csv(str(CSV_PATH))
    replay_end = str(pd.to_datetime(prices.close.index.max()).date())
    base_tickers = read_tickers(ACTIVE_PATH)
    staged_adds = {str(k): str(v) for k, v in doc.get("staged_activation_dates", {}).items()}
    membership = membership_frame(prices.close.index, prices.close.columns, base_tickers, staged_adds)
    run_fn = build_dynamic_runner(engine, membership)
    targets = load_missing_selection_targets(20)

    rows: list[dict[str, object]] = []
    for label, variant_cfg in build_variants(cfg):
        full_eq, full_tr, full = run_variant(run_fn, prices, variant_cfg, pp, "2015-01-02", replay_end)
        oos_eq, oos_tr, oos = run_variant(run_fn, prices, variant_cfg, pp, "2022-01-03", replay_end)
        years = year_returns(full_eq)
        full_capture_count, full_capture_names = capture_stats(full_tr, targets)
        oos_capture_count, oos_capture_names = capture_stats(oos_tr, targets)
        rows.append(
            {
                "label": label,
                "full_roi_pct": float(full["ROI_%"]),
                "full_cagr_pct": float(full["CAGR_%"]),
                "full_maxdd_pct": float(full["MaxDD_%"]),
                "full_sharpe": float(full["Sharpe"]),
                "full_orders": int(full["Orders"]),
                "oos_roi_pct": float(oos["ROI_%"]),
                "oos_cagr_pct": float(oos["CAGR_%"]),
                "oos_maxdd_pct": float(oos["MaxDD_%"]),
                "oos_sharpe": float(oos["Sharpe"]),
                "oos_orders": int(oos["Orders"]),
                "roi_2018_pct": years.get(2018, float("nan")),
                "roi_2019_pct": years.get(2019, float("nan")),
                "roi_2024_pct": years.get(2024, float("nan")),
                "roi_2025_pct": years.get(2025, float("nan")),
                "target_capture_count_full": full_capture_count,
                "target_capture_names_full": full_capture_names,
                "target_capture_count_oos": oos_capture_count,
                "target_capture_names_oos": oos_capture_names,
                "score_rs_blend": float(variant_cfg.get("score_rs_blend", 0.0)),
                "score_quality_blend": float(variant_cfg.get("score_quality_blend", 0.0)),
                "score_quality_win": int(variant_cfg.get("score_quality_win", 63)),
                "score_residual_blend": float(variant_cfg.get("score_residual_blend", 0.0)),
                "score_breakout_252_weight": float(variant_cfg.get("score_breakout_252_weight", 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    baseline = df.loc[df["label"] == "baseline_current"].iloc[0]
    for metric in ["full_roi_pct", "full_sharpe", "full_maxdd_pct", "oos_roi_pct", "oos_sharpe", "oos_maxdd_pct"]:
        df[f"delta_{metric}"] = df[metric] - float(baseline[metric])

    df["study_score"] = (
        1.0 * (df["delta_full_roi_pct"] > 0).astype(float)
        + 1.0 * (df["delta_oos_roi_pct"] > 0).astype(float)
        + 1.0 * (df["delta_full_sharpe"] > 0).astype(float)
        + 1.0 * (df["delta_oos_sharpe"] > 0).astype(float)
        + 0.50 * (df["delta_full_maxdd_pct"] >= -0.50).astype(float)
        + 0.50 * (df["delta_oos_maxdd_pct"] >= -0.50).astype(float)
        + 0.10 * df["target_capture_count_full"]
        + 0.05 * df["target_capture_count_oos"]
    )
    df = df.sort_values(
        ["study_score", "delta_oos_sharpe", "delta_full_sharpe", "delta_oos_roi_pct", "delta_full_roi_pct"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(render_report(df, targets), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()
