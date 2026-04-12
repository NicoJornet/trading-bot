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

OUT_CSV = ROOT / "research" / "exports" / "r8_state_quality_compounder_lane_study.csv"
OUT_MD = ROOT / "research" / "reports" / "R8_STATE_QUALITY_COMPOUNDER_LANE_STUDY.md"


def load_module(path: Path, name: str):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_engine():
    return load_module(ENGINE_PATH, "apex_engine_slot_quality_state_qcompound")


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
    membership_needle = "        elig = elig & enough.reindex(c.columns, fill_value=False)\n"
    membership_patch = membership_needle + "        elig = elig & universe_membership.loc[d].reindex(c.columns, fill_value=False).astype(bool)\n"
    if membership_needle not in src:
        raise RuntimeError("Unable to patch run_backtest for dynamic membership.")
    src = src.replace(membership_needle, membership_patch, 1)

    score_needle = "    ranks = score.rank(axis=1, ascending=False, method='min')\n"
    score_patch = """
    if int(p.get("qcompound_enable", 0)) == 1:
        q_win = max(5, int(p.get("qcompound_quality_win", 63)))
        q_path = c.diff().abs().rolling(q_win, min_periods=q_win).sum()
        q_eff = (c.diff(q_win).abs() / (q_path + 1e-12)).clip(lower=0.0, upper=1.0)
        q_lt_topn = int(p.get("qcompound_lt_topn", 20))
        q_mask = (rank126 <= q_lt_topn) & (rank252 <= q_lt_topn)
        q_breadth_min = float(p.get("qcompound_breadth_min", 0.0) or 0.0)
        if q_breadth_min > 0.0:
            q_mask = q_mask & (breadth.to_numpy()[:, None] >= q_breadth_min)
        q_rs63_min = p.get("qcompound_rs63_min", None)
        if q_rs63_min is not None:
            q_mask = q_mask & (rs63 >= float(q_rs63_min))
        q_r21_max = p.get("qcompound_r21_max", None)
        if q_r21_max is not None:
            q_mask = q_mask & (r21 <= float(q_r21_max))
        q_dd252_max = float(p.get("qcompound_dd252_max", 0.35))
        q_mask = q_mask & (dd252 >= -q_dd252_max)
        q_breakout = (1.0 + (dd252 / max(1e-9, q_dd252_max))).clip(lower=0.0, upper=1.0)
        q_component = 0.70 * q_eff + 0.30 * q_breakout
        score_eff = score_eff + float(p.get("qcompound_bonus", 0.0)) * q_component * q_mask.astype(float)
    ranks = score.rank(axis=1, ascending=False, method='min')
"""
    if score_needle not in src:
        raise RuntimeError("Unable to patch run_backtest for quality compounder lane.")
    src = src.replace(score_needle, score_patch, 1)

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


def year_returns(eq) -> dict[int, float]:
    if isinstance(eq, pd.DataFrame):
        eq = eq["Equity"] if "Equity" in eq.columns else eq.iloc[:, 0]
    out: dict[int, float] = {}
    if eq.empty:
        return out
    for year in sorted(set(eq.index.year)):
        part = eq.loc[eq.index.year == year].dropna()
        if part.empty:
            continue
        out[year] = float(part.iloc[-1] / part.iloc[0] - 1.0) * 100.0
    return out


def capture_stats(trades: pd.DataFrame, targets: list[str]) -> tuple[int, str]:
    if trades.empty:
        return 0, ""
    cols = {str(c).lower(): c for c in trades.columns}
    ticker_col = cols.get("ticker") or cols.get("symbol")
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

    common = {"qcompound_enable": 1, "qcompound_quality_win": 126}
    variant("qcomp_b45_lt20_bonus008", **common, qcompound_breadth_min=0.45, qcompound_lt_topn=20, qcompound_bonus=0.08, qcompound_rs63_min=0.00, qcompound_r21_max=0.30, qcompound_dd252_max=0.30)
    variant("qcomp_b50_lt20_bonus008", **common, qcompound_breadth_min=0.50, qcompound_lt_topn=20, qcompound_bonus=0.08, qcompound_rs63_min=0.00, qcompound_r21_max=0.30, qcompound_dd252_max=0.30)
    variant("qcomp_b50_lt20_bonus010", **common, qcompound_breadth_min=0.50, qcompound_lt_topn=20, qcompound_bonus=0.10, qcompound_rs63_min=0.00, qcompound_r21_max=0.30, qcompound_dd252_max=0.30)
    variant("qcomp_b50_lt25_bonus010_rs01", **common, qcompound_breadth_min=0.50, qcompound_lt_topn=25, qcompound_bonus=0.10, qcompound_rs63_min=0.01, qcompound_r21_max=0.30, qcompound_dd252_max=0.30)
    variant("qcomp_b50_lt25_bonus012_rs01", **common, qcompound_breadth_min=0.50, qcompound_lt_topn=25, qcompound_bonus=0.12, qcompound_rs63_min=0.01, qcompound_r21_max=0.28, qcompound_dd252_max=0.28)
    variant("qcomp_b45_lt25_bonus010_rs00", **common, qcompound_breadth_min=0.45, qcompound_lt_topn=25, qcompound_bonus=0.10, qcompound_rs63_min=0.00, qcompound_r21_max=0.32, qcompound_dd252_max=0.32)
    return variants


def render_report(summary: pd.DataFrame, targets: list[str]) -> str:
    baseline = summary.loc[summary["label"] == "baseline_current"].iloc[0]
    best = summary.iloc[0]
    lines = [
        "# R8 State Quality Compounder Lane Study",
        "",
        "State-dependent core ranking study above `r8`: a quality-compounder bonus is allowed only for long-term leaders that stay durable enough (`breadth + LT ranks + RS + extension guard`).",
        "",
        "Target selection-miss basket used as diagnostic:",
        "",
        "- " + (", ".join(targets) if targets else "(none)"),
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
    ]
    lines.extend(
        governance_section(
            summary,
            "baseline_current",
            current_baseline="r8",
            robust_fallback="r7",
            study_verdict="reject" if best["label"] == "baseline_current" or not (best["full_roi_pct"] > baseline["full_roi_pct"] and best["oos_roi_pct"] > baseline["oos_roi_pct"] and best["full_sharpe"] > baseline["full_sharpe"] and best["oos_sharpe"] > baseline["oos_sharpe"]) else "monitor",
            recommendation="Promote only if the state-dependent lane improves both full and OOS without paying the same broad historical cost as the static quality blend.",
            economic_explanation="A durable-compounder bonus is only credible if it helps long-term leaders in healthy contexts instead of flattening the whole momentum edge.",
            concentration_note="Target-capture counts help explain whether the lane is surfacing missed semicap / software compounders, but promotion still depends on the replay staying broad-based.",
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
                "qcompound_breadth_min": float(variant_cfg.get("qcompound_breadth_min", 0.0)),
                "qcompound_lt_topn": int(variant_cfg.get("qcompound_lt_topn", 0)),
                "qcompound_bonus": float(variant_cfg.get("qcompound_bonus", 0.0)),
                "qcompound_rs63_min": float(variant_cfg.get("qcompound_rs63_min", 0.0)),
                "qcompound_r21_max": float(variant_cfg.get("qcompound_r21_max", 0.0)),
                "qcompound_dd252_max": float(variant_cfg.get("qcompound_dd252_max", 0.0)),
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
