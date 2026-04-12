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

OUT_CSV = ROOT / "research" / "exports" / "r8_slot3_admission_sizing_study.csv"
OUT_MD = ROOT / "research" / "reports" / "R8_SLOT3_ADMISSION_SIZING_STUDY.md"


def load_module(path: Path, name: str):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dd_mod = load_module(ROOT / "research" / "scripts" / "r4_drawdown_deep_analysis.py", "r8_slot3_dd")


def load_engine():
    return load_module(ENGINE_PATH, "apex_engine_slot_quality_r8_slot3")


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

    slot3_gate_needle = """        if int(p.get('slot3_gate_enable', 0)) == 1 and active_topk >= 3 and len(desired) >= 3:\n            t3 = desired[2]\n            rank3 = int(ranks.loc[d, t3]) if pd.notna(ranks.loc[d, t3]) else 10**9\n            is_lt_leader = bool((rank126.loc[d, t3] <= int(p.get('slot3_leader_exempt_topn', 12))) and (rank252.loc[d, t3] <= int(p.get('slot3_leader_exempt_topn', 12))))\n            slot3_rank_max = int(p.get('slot3_max_rank', 4))\n            weak_slot3_rank_max = int(p.get('slot3_weak_max_rank', 0))\n            if regime_weak and weak_slot3_rank_max > 0:\n                slot3_rank_max = min(slot3_rank_max, weak_slot3_rank_max)\n            if (rank3 > slot3_rank_max) and (not is_lt_leader):\n                desired = desired[:2]\n"""
    slot3_gate_patch = slot3_gate_needle + """
        if int(p.get("slot3_admission_guard_enable", 0)) == 1 and active_topk >= 3 and len(desired) >= 3:
            t3 = desired[2]
            rank3_now = int(ranks.loc[d, t3]) if pd.notna(ranks.loc[d, t3]) else 10**9
            rs3_now = float(rs63.loc[d, t3]) if pd.notna(rs63.loc[d, t3]) else -999.0
            is_lt_leader = bool((rank126.loc[d, t3] <= int(p.get("slot3_leader_exempt_topn", 12))) and (rank252.loc[d, t3] <= int(p.get("slot3_leader_exempt_topn", 12))))
            rank_soft = int(p.get("slot3_admission_rank_soft_max", 999))
            rs_min = float(p.get("slot3_admission_rs63_min", -999.0))
            require_both = bool(int(p.get("slot3_admission_require_both", 0)))
            rank_bad = rank3_now > rank_soft
            rs_bad = rs3_now < rs_min
            reject = (rank_bad and rs_bad) if require_both else (rank_bad or rs_bad)
            if reject and (not is_lt_leader):
                desired = desired[:2]
"""
    if slot3_gate_needle not in src:
        raise RuntimeError("Unable to patch run_backtest for slot3 admission guard.")
    src = src.replace(slot3_gate_needle, slot3_gate_patch, 1)

    slot3_weight_needle = """            if int(p.get("slot3_weight_cap_enable", 0)) == 1 and len(desired) >= 3:\n                if (int(p.get("slot3_weight_cap_weak_only", 0)) != 1) or regime_weak:\n                    slot3 = desired[2]\n                    if slot3 in w:\n                        w = cap_ticker_weights(\n                            weights=w,\n                            capped_names=[slot3],\n                            indiv_cap=float(p.get("slot3_weight_cap", 1.0)),\n                        )\n"""
    slot3_weight_patch = slot3_weight_needle + """
            if int(p.get("slot3_marginal_weight_cap_enable", 0)) == 1 and len(desired) >= 3:
                slot3 = desired[2]
                if slot3 in w:
                    rank3_now = int(ranks.loc[d, slot3]) if pd.notna(ranks.loc[d, slot3]) else 10**9
                    rs3_now = float(rs63.loc[d, slot3]) if pd.notna(rs63.loc[d, slot3]) else -999.0
                    is_lt_leader = bool((rank126.loc[d, slot3] <= int(p.get("slot3_leader_exempt_topn", 12))) and (rank252.loc[d, slot3] <= int(p.get("slot3_leader_exempt_topn", 12))))
                    rank_soft = int(p.get("slot3_marginal_rank_soft_max", 999))
                    rs_min = float(p.get("slot3_marginal_rs63_min", -999.0))
                    require_both = bool(int(p.get("slot3_marginal_require_both", 0)))
                    rank_bad = rank3_now > rank_soft
                    rs_bad = rs3_now < rs_min
                    cap_it = (rank_bad and rs_bad) if require_both else (rank_bad or rs_bad)
                    if cap_it and (not is_lt_leader):
                        w = cap_ticker_weights(
                            weights=w,
                            capped_names=[slot3],
                            indiv_cap=float(p.get("slot3_marginal_weight_cap", 1.0)),
                        )
"""
    if slot3_weight_needle not in src:
        raise RuntimeError("Unable to patch run_backtest for slot3 marginal weight cap.")
    src = src.replace(slot3_weight_needle, slot3_weight_patch, 1)

    ns = dict(engine.__dict__)
    ns["universe_membership"] = universe_membership
    exec(src, ns)
    return ns["run_backtest"]


def run_variant(run_fn, prices, cfg: dict, pp: dict, start: str, end: str):
    local_cfg = dict(cfg)
    mie_enabled = bool(int(local_cfg.pop("mie_enabled", 1)))
    eq, tr, metrics = run_fn(
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
    return eq, tr, metrics


def summarize_variant(label: str, full_metrics: dict, oos_metrics: dict, recent_metrics: dict) -> dict:
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
    }


def render_report(df: pd.DataFrame) -> str:
    baseline = df.loc[df["label"] == "baseline_current"].iloc[0]
    best = df.iloc[0]
    top = df.head(12).copy()
    lines = [
        "# R8 Slot-3 Admission And Sizing Study",
        "",
        "Targeted study of the remaining portfolio-logic candidate on top of `r8`: make the third line earn its place more clearly, or shrink it when it is only marginal.",
        "",
        "Idea:",
        "- the baseline already guards slot-3 by rank and quality;",
        "- this study tests whether `r8` still benefits from an extra marginal-slot admission rule or a softer marginal-slot weight cap;",
        "- the goal is to reduce carried concentration without falling back into blunt early exits.",
        "",
        "## Variant Leaderboard",
        "",
        dd_mod.markdown_table(top),
        "",
        "## Best Variant Vs Baseline",
        "",
        f"- Best variant: `{best['label']}`",
        f"- Full ROI delta: `{best['full_roi_pct'] - baseline['full_roi_pct']:.2f}`",
        f"- Full Sharpe delta: `{best['full_sharpe'] - baseline['full_sharpe']:.4f}`",
        f"- Full MaxDD delta: `{best['full_maxdd_pct'] - baseline['full_maxdd_pct']:.4f}`",
        f"- OOS ROI delta: `{best['oos_roi_pct'] - baseline['oos_roi_pct']:.2f}`",
        f"- OOS Sharpe delta: `{best['oos_sharpe'] - baseline['oos_sharpe']:.4f}`",
        f"- Since 2025 ROI delta: `{best['since2025_roi_pct'] - baseline['since2025_roi_pct']:.2f}`",
        "",
    ]
    lines.extend(
        governance_section(
            df,
            "baseline_current",
            current_baseline="r8",
            robust_fallback="r7",
            study_verdict="candidate_only" if best["label"] != "baseline_current" else "rejected_no_upgrade",
            recommendation="Promote only if a slot-3 rule improves both full and OOS without paying a material cost in recent performance or drawdown.",
            economic_explanation="Slot-3 is structurally weaker than slots 1 and 2, so a marginal admission or sizing rule is economically plausible even if it ultimately fails the replay.",
            concentration_note="This is a portfolio-architecture test, not a score tweak, which keeps it on the structural side of the anti-overfit line.",
        )
    )
    return "\n".join(lines) + "\n"


def main():
    engine = load_engine()
    cfg, pp, doc = load_cfg()
    prices = engine.load_prices_from_csv(str(CSV_PATH))
    replay_end = str(pd.to_datetime(prices.close.index.max()).date())

    base_tickers = read_tickers(ACTIVE_PATH)
    activation_dates = {str(k): str(v) for k, v in doc.get("staged_activation_dates", {}).items() if str(k) in prices.close.columns}
    membership = membership_frame(prices.close.index, prices.close.columns, base_tickers, activation_dates)
    run_fn = build_dynamic_runner(engine, membership)

    variants = [
        ("baseline_current", {}),
        ("slot3_tight_rank3", {
            "slot3_max_rank": 3,
            "q_slot3_rank_th": 4,
        }),
        ("slot3_admit_rs10", {
            "slot3_admission_guard_enable": 1,
            "slot3_admission_rank_soft_max": 4,
            "slot3_admission_rs63_min": 0.10,
            "slot3_admission_require_both": 0,
        }),
        ("slot3_admit_rank3_rs00", {
            "slot3_admission_guard_enable": 1,
            "slot3_admission_rank_soft_max": 3,
            "slot3_admission_rs63_min": 0.00,
            "slot3_admission_require_both": 1,
        }),
        ("slot3_cap35_rank3_rs10", {
            "slot3_marginal_weight_cap_enable": 1,
            "slot3_marginal_rank_soft_max": 3,
            "slot3_marginal_rs63_min": 0.10,
            "slot3_marginal_require_both": 0,
            "slot3_marginal_weight_cap": 0.35,
        }),
        ("slot3_cap30_rank3_rs10", {
            "slot3_marginal_weight_cap_enable": 1,
            "slot3_marginal_rank_soft_max": 3,
            "slot3_marginal_rs63_min": 0.10,
            "slot3_marginal_require_both": 0,
            "slot3_marginal_weight_cap": 0.30,
        }),
        ("slot3_cap30_rank4_rs15", {
            "slot3_marginal_weight_cap_enable": 1,
            "slot3_marginal_rank_soft_max": 4,
            "slot3_marginal_rs63_min": 0.15,
            "slot3_marginal_require_both": 0,
            "slot3_marginal_weight_cap": 0.30,
        }),
        ("slot3_combo_admitcap", {
            "slot3_admission_guard_enable": 1,
            "slot3_admission_rank_soft_max": 3,
            "slot3_admission_rs63_min": 0.00,
            "slot3_admission_require_both": 1,
            "slot3_marginal_weight_cap_enable": 1,
            "slot3_marginal_rank_soft_max": 3,
            "slot3_marginal_rs63_min": 0.10,
            "slot3_marginal_require_both": 0,
            "slot3_marginal_weight_cap": 0.35,
        }),
        ("slot3_cap35_onlyrs", {
            "slot3_marginal_weight_cap_enable": 1,
            "slot3_marginal_rank_soft_max": 999,
            "slot3_marginal_rs63_min": 0.10,
            "slot3_marginal_require_both": 0,
            "slot3_marginal_weight_cap": 0.35,
        }),
        ("slot3_cap40_onlyrs", {
            "slot3_marginal_weight_cap_enable": 1,
            "slot3_marginal_rank_soft_max": 999,
            "slot3_marginal_rs63_min": 0.10,
            "slot3_marginal_require_both": 0,
            "slot3_marginal_weight_cap": 0.40,
        }),
        ("slot3_cap45_onlyrs", {
            "slot3_marginal_weight_cap_enable": 1,
            "slot3_marginal_rank_soft_max": 999,
            "slot3_marginal_rs63_min": 0.10,
            "slot3_marginal_require_both": 0,
            "slot3_marginal_weight_cap": 0.45,
        }),
        ("slot3_cap35_onlyrs05", {
            "slot3_marginal_weight_cap_enable": 1,
            "slot3_marginal_rank_soft_max": 999,
            "slot3_marginal_rs63_min": 0.05,
            "slot3_marginal_require_both": 0,
            "slot3_marginal_weight_cap": 0.35,
        }),
    ]

    rows = []
    for label, patch_cfg in variants:
        local_cfg = dict(cfg)
        local_cfg.update(patch_cfg)
        _, _, full_metrics = run_variant(run_fn, prices, local_cfg, pp, "2015-01-02", replay_end)
        _, _, oos_metrics = run_variant(run_fn, prices, local_cfg, pp, "2022-01-03", replay_end)
        _, _, recent_metrics = run_variant(run_fn, prices, local_cfg, pp, "2025-01-02", replay_end)
        rows.append(summarize_variant(label, full_metrics, oos_metrics, recent_metrics))
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
