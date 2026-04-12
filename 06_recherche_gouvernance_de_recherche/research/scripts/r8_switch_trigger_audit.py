from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from report_governance import governance_section


ROOT = Path(r"C:\Users\nicol\Downloads\214")
BEST_PATH = ROOT / "BEST_ALGO_184.json"
CSV_PATH = ROOT / "apex_ohlcv_full_2015_2026.csv"
ACTIVE_PATH = ROOT / "trading-bot" / "data" / "extracts" / "apex_tickers_active.csv"

OUT_CSV = ROOT / "research" / "exports" / "r8_switch_trigger_audit.csv"
OUT_MD = ROOT / "research" / "reports" / "R8_SWITCH_TRIGGER_AUDIT.md"


def load_module(path: Path, name: str):
    from importlib import util

    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


slot3_mod = load_module(ROOT / "research" / "scripts" / "r8_slot3_admission_sizing_study.py", "r8_switch_audit_slot3")
dd_mod = load_module(ROOT / "research" / "scripts" / "r4_drawdown_deep_analysis.py", "r8_switch_audit_dd")


SWITCH_TRIGGER_MAP = [
    ("rebalance_td", "calendar cadence", "How often the portfolio is allowed to reconsider names at all."),
    ("delta_rebal", "delta threshold", "Suppresses small open-to-target differences; higher values reduce partial switch activity."),
    ("keep_rank", "persistence", "Lets current holdings survive rank slippage, reducing avoidable replacements."),
    ("corr_gate/corr_pick/corr_scan/corr_held_enable", "diversification switch", "Replaces crowded leaders with less correlated names when the rank pool becomes too similar."),
    ("held_release_*", "held release", "Lets newcomers displace weaker held names more aggressively when the top of the book changes."),
    ("quality_filter_enable + q_slot2/q_slot3", "quality admission", "Stops slot-2/slot-3 newcomers from entering when persistence/RS quality is weak."),
    ("slot3_gate_*", "third-slot admission", "Decides whether the marginal third line is allowed to exist at all."),
    ("slot3_weight_cap_*", "third-slot sizing", "Lets the third line exist but with smaller weight when it is weaker."),
    ("loss_cluster_guard_enable", "fragile cluster basket cap", "Prevents duplicate weak clusters from forcing one-for-one replacements."),
    ("state_cluster_guard_*", "state duplicate suppression", "Blocks same-cluster duplicates only when the cluster state is weak."),
    ("state_weight_cap_*", "state sizing cap", "Allows names to stay but shrinks them when a thematic basket is weak."),
    ("exit_smooth_*", "forced full-exit smoothing", "Defers some non-target sells, directly reducing switch churn."),
    ("convex_exit_guard_*", "convex non-PP guard", "Defers some convex exits that would otherwise create later re-entry switches."),
    ("reentry_*", "pullback re-entry", "Re-activates leaders after healthy pullbacks, creating return switches rather than fresh discoveries."),
]


def load_cfg():
    obj = json.loads(BEST_PATH.read_text(encoding="utf-8"))
    return obj["cfg"], obj["pp"], obj


def compute_switch_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "buy_orders": 0,
            "sell_orders": 0,
            "switch_days": 0,
            "switch_pairs_lb": 0,
            "avg_orders_per_switch_day": 0.0,
            "exit_not_target_sells": 0,
            "delta_sells": 0,
            "pp_trail_sells": 0,
        }

    tr = trades.copy()
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr["Side"] = tr["Side"].astype(str)
    tr["Reason"] = tr["Reason"].astype(str)

    buy_orders = int((tr["Side"] == "BUY").sum())
    sell_orders = int((tr["Side"] == "SELL").sum())
    exit_not_target_sells = int(((tr["Side"] == "SELL") & (tr["Reason"] == "EXIT_NOT_TARGET")).sum())
    delta_sells = int(((tr["Side"] == "SELL") & (tr["Reason"] == "DELTA_SELL")).sum())
    pp_trail_sells = int(((tr["Side"] == "SELL") & (tr["Reason"].str.startswith("PP_TRAIL"))).sum())

    grouped = []
    for dt, grp in tr.groupby("Date", sort=True):
        buys = grp.loc[grp["Side"] == "BUY", "Ticker"].astype(str).unique().tolist()
        sells = grp.loc[grp["Side"] == "SELL", "Ticker"].astype(str).unique().tolist()
        grouped.append(
            {
                "Date": dt,
                "buy_names": len(buys),
                "sell_names": len(sells),
                "orders": int(len(grp)),
            }
        )
    day_df = pd.DataFrame(grouped)
    if day_df.empty:
        switch_days = 0
        switch_pairs_lb = 0
        avg_orders_per_switch_day = 0.0
    else:
        switch_mask = (day_df["buy_names"] > 0) & (day_df["sell_names"] > 0)
        switch_days = int(switch_mask.sum())
        switch_pairs_lb = int(day_df.loc[switch_mask, ["buy_names", "sell_names"]].min(axis=1).sum()) if switch_days else 0
        avg_orders_per_switch_day = float(day_df.loc[switch_mask, "orders"].mean()) if switch_days else 0.0

    return {
        "buy_orders": buy_orders,
        "sell_orders": sell_orders,
        "switch_days": switch_days,
        "switch_pairs_lb": switch_pairs_lb,
        "avg_orders_per_switch_day": avg_orders_per_switch_day,
        "exit_not_target_sells": exit_not_target_sells,
        "delta_sells": delta_sells,
        "pp_trail_sells": pp_trail_sells,
    }


def summarize_variant(label: str, full_metrics: dict, oos_metrics: dict, recent_metrics: dict, switch_stats: dict[str, float]) -> dict[str, float | int | str]:
    return {
        "label": label,
        "full_roi_pct": float(full_metrics["ROI_%"]),
        "full_cagr_pct": float(full_metrics["CAGR_%"]),
        "full_maxdd_pct": float(full_metrics["MaxDD_%"]),
        "full_sharpe": float(full_metrics["Sharpe"]),
        "full_orders": int(full_metrics["Orders"]),
        "oos_roi_pct": float(oos_metrics["ROI_%"]),
        "oos_cagr_pct": float(oos_metrics["CAGR_%"]),
        "oos_maxdd_pct": float(oos_metrics["MaxDD_%"]),
        "oos_sharpe": float(oos_metrics["Sharpe"]),
        "oos_orders": int(oos_metrics["Orders"]),
        "since2025_roi_pct": float(recent_metrics["ROI_%"]),
        "since2025_cagr_pct": float(recent_metrics["CAGR_%"]),
        "since2025_maxdd_pct": float(recent_metrics["MaxDD_%"]),
        "since2025_sharpe": float(recent_metrics["Sharpe"]),
        "since2025_orders": int(recent_metrics["Orders"]),
        **switch_stats,
    }


def render_report(df: pd.DataFrame) -> str:
    baseline = df.loc[df["label"] == "baseline_current"].iloc[0]
    best = df.iloc[0]
    top = df.head(14).copy()
    trigger_map = pd.DataFrame(SWITCH_TRIGGER_MAP, columns=["parameter_family", "switch_stage", "mechanism"])

    lines = [
        "# R8 Switch Trigger Audit",
        "",
        "Targeted audit of the parameters that actually trigger portfolio switches in the current `r8` engine.",
        "",
        "## Switch Trigger Map",
        "",
        dd_mod.markdown_table(trigger_map),
        "",
        "## Variant Leaderboard",
        "",
        dd_mod.markdown_table(top),
        "",
        "## Best Variant Vs Baseline",
        "",
        f"- Best variant: `{best['label']}`",
        f"- Full ROI delta: `{best['full_roi_pct'] - baseline['full_roi_pct']:.2f}`",
        f"- OOS ROI delta: `{best['oos_roi_pct'] - baseline['oos_roi_pct']:.2f}`",
        f"- Full Sharpe delta: `{best['full_sharpe'] - baseline['full_sharpe']:.4f}`",
        f"- OOS Sharpe delta: `{best['oos_sharpe'] - baseline['oos_sharpe']:.4f}`",
        f"- Switch-day delta: `{int(best['switch_days'] - baseline['switch_days'])}`",
        f"- Switch-pairs lower-bound delta: `{int(best['switch_pairs_lb'] - baseline['switch_pairs_lb'])}`",
        "",
    ]
    lines.extend(
        governance_section(
            df,
            "baseline_current",
            current_baseline="r8",
            robust_fallback="r7",
            study_verdict="audit_only",
            recommendation="Treat only broad improvements that also keep switch churn sane as promotion candidates; everything else should remain an audit finding.",
            economic_explanation="A switch parameter is only worth touching if it improves the quality of replacements, not just the quantity of trades.",
            concentration_note="Switches interact with persistence, correlation, quality gates and state guards, so isolated churn reduction is not enough without full and OOS confirmation.",
        )
    )
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

    variants = [
        ("baseline_current", {}),
        ("delta_rebal_005", {"delta_rebal": 0.05}),
        ("delta_rebal_015", {"delta_rebal": 0.15}),
        ("delta_rebal_020", {"delta_rebal": 0.20}),
        ("keep_rank_6", {"keep_rank": 6}),
        ("corr_gate_090", {"corr_gate": 0.90}),
        ("corr_gate_095", {"corr_gate": 0.95}),
        ("corr_held_off", {"corr_held_enable": 0}),
        ("held_release_off", {"held_release_enable": 0}),
        ("held_release_tighter", {"held_release_weak_rank_th": 6, "held_release_new_topm": 4, "held_release_min_new": 1}),
        ("held_release_looser", {"held_release_weak_rank_th": 10, "held_release_new_topm": 6, "held_release_min_new": 2}),
        ("exit_smooth_2", {"exit_smooth_max_defers": 2}),
        ("exit_smooth_4", {"exit_smooth_max_defers": 4}),
        ("state_cluster_off", {"state_cluster_guard_enable": 0}),
        ("state_weight_off", {"state_weight_cap_enable": 0}),
        ("q_slot3_strict", {"q_slot3_rank_th": 4}),
        ("rank_pool_20", {"rank_pool": 20}),
    ]

    rows = []
    for label, patch in variants:
        local_cfg = dict(cfg)
        local_cfg.update(patch)
        _, tr_full, full_metrics = slot3_mod.run_variant(run_fn, prices, local_cfg, pp, "2015-01-02", replay_end)
        _, _, oos_metrics = slot3_mod.run_variant(run_fn, prices, local_cfg, pp, "2022-01-03", replay_end)
        _, _, recent_metrics = slot3_mod.run_variant(run_fn, prices, local_cfg, pp, "2025-01-02", replay_end)
        switch_stats = compute_switch_stats(tr_full)
        rows.append(summarize_variant(label, full_metrics, oos_metrics, recent_metrics, switch_stats))
        print(f"done {label}")

    df = pd.DataFrame(rows)
    baseline = df.loc[df["label"] == "baseline_current"].iloc[0]
    df["score"] = (
        (df["full_roi_pct"] - baseline["full_roi_pct"]) / 10000.0
        + (df["full_sharpe"] - baseline["full_sharpe"]) * 5.0
        + (df["oos_roi_pct"] - baseline["oos_roi_pct"]) / 5000.0
        + (df["oos_sharpe"] - baseline["oos_sharpe"]) * 10.0
        + (df["since2025_roi_pct"] - baseline["since2025_roi_pct"]) / 1000.0
        + (df["full_maxdd_pct"] - baseline["full_maxdd_pct"]) * 1.5
        + (df["oos_maxdd_pct"] - baseline["oos_maxdd_pct"]) * 2.0
        - (df["switch_pairs_lb"] - baseline["switch_pairs_lb"]) * 0.03
        - (df["switch_days"] - baseline["switch_days"]) * 0.02
    )
    df = df.sort_values(["score", "oos_sharpe", "full_sharpe"], ascending=False).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(render_report(df), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()
