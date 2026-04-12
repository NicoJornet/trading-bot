from __future__ import annotations

import inspect
import json
from importlib import util
from pathlib import Path

import numpy as np
import pandas as pd

from report_governance import governance_section


ROOT = Path(r"C:\Users\nicol\Downloads\214")
ENGINE_PATH = ROOT / "trading-bot" / "engine_bundle" / "run_v11_slot_quality" / "v11_slot_quality" / "apex_engine_slot_quality.py"
BEST_PATH = ROOT / "BEST_ALGO_184.json"
CSV_PATH = ROOT / "apex_ohlcv_full_2015_2026.csv"
ACTIVE_PATH = ROOT / "trading-bot" / "data" / "extracts" / "apex_tickers_active.csv"

OUT_CSV = ROOT / "research" / "exports" / "r8_reason_aware_reentry_study.csv"
OUT_MD = ROOT / "research" / "reports" / "R8_REASON_AWARE_REENTRY_STUDY.md"


def load_module(path: Path, name: str):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dd_mod = load_module(ROOT / "research" / "scripts" / "r4_drawdown_deep_analysis.py", "r8_reason_reentry_dd")
reentry_mod = load_module(ROOT / "research" / "scripts" / "r7_reentry_exit_cluster_analysis.py", "r8_reason_reentry_analysis")


def load_engine():
    return load_module(ENGINE_PATH, "apex_engine_slot_quality_r8_reason_reentry")


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

    reentry_needle = "    score_eff = score.copy()\n    if int(p.get('reentry_enable', 0)) == 1:\n"
    reentry_patch = (
        "    reentry_base_mask = pd.DataFrame(False, index=c.index, columns=c.columns)\n"
        "    score_eff = score.copy()\n"
        "    if int(p.get('reentry_enable', 0)) == 1:\n"
    )
    if reentry_needle not in src:
        raise RuntimeError("Unable to patch run_backtest for reentry base mask.")
    src = src.replace(reentry_needle, reentry_patch, 1)

    score_eff_needle = "        score_eff = score_eff + float(p.get('reentry_bonus', 0.10)) * reentry_mask.astype(float)\n"
    score_eff_patch = score_eff_needle + "        reentry_base_mask = reentry_mask.copy()\n"
    if score_eff_needle not in src:
        raise RuntimeError("Unable to patch run_backtest for reentry mask capture.")
    src = src.replace(score_eff_needle, score_eff_patch, 1)

    init_needle = "    loss_last_exit_idx: Dict[str, int] = {}\n\n    # PP state\n"
    init_patch = (
        "    loss_last_exit_idx: Dict[str, int] = {}\n"
        "    last_full_exit_idx: Dict[str, int] = {}\n"
        "    last_full_exit_reason: Dict[str, str] = {}\n\n"
        "    # PP state\n"
    )
    if init_needle not in src:
        raise RuntimeError("Unable to patch run_backtest for exit history state.")
    src = src.replace(init_needle, init_patch, 1)

    score_rank_needle = "        score_rank = score_eff.loc[d].copy()\n"
    score_rank_patch = """        score_rank = score_eff.loc[d].copy()
        if int(p.get("reentry_reason_enable", 0)) == 1:
            held_names = set(positions.keys())
            candidate_mask = ~pd.Series(pd.Index(c.columns).isin(list(held_names)), index=c.columns)
            active_reentry = reentry_base_mask.loc[d].reindex(c.columns, fill_value=False).astype(bool) & candidate_mask
            lookback = max(1, int(p.get("reentry_reason_lookback_td", 20)))
            if active_reentry.any():
                reason_adjust = pd.Series(0.0, index=c.columns)
                block_recent_xnt = pd.Series(False, index=c.columns)
                xnt_mult = float(p.get("reentry_reason_exit_not_target_penalty_mult", 0.0))
                xnt_block = bool(int(p.get("reentry_reason_exit_not_target_block", 0)))
                pp_bonus = float(p.get("reentry_reason_pp_trail_bonus", 0.0))
                delta_bonus = float(p.get("reentry_reason_delta_sell_bonus", 0.0))
                recent_only = bool(int(p.get("reentry_reason_require_recent_exit", 1)))
                current_bonus = float(p.get("reentry_bonus", 0.0))
                for ticker, reason in last_full_exit_reason.items():
                    if ticker not in reason_adjust.index:
                        continue
                    exit_idx = last_full_exit_idx.get(ticker)
                    if exit_idx is None:
                        continue
                    age = i - int(exit_idx)
                    if recent_only and (age < 0 or age > lookback):
                        continue
                    if reason == "EXIT_NOT_TARGET":
                        if xnt_mult != 0.0:
                            reason_adjust.loc[ticker] -= current_bonus * xnt_mult
                        if xnt_block:
                            block_recent_xnt.loc[ticker] = True
                    elif reason == "PP_TRAIL" and pp_bonus != 0.0:
                        reason_adjust.loc[ticker] += pp_bonus
                    elif reason == "DELTA_SELL" and delta_bonus != 0.0:
                        reason_adjust.loc[ticker] += delta_bonus
                score_rank = score_rank + reason_adjust.where(active_reentry, 0.0)
                if block_recent_xnt.any():
                    elig = elig & (~(block_recent_xnt & active_reentry))
"""
    if score_rank_needle not in src:
        raise RuntimeError("Unable to patch run_backtest for reason-aware reentry.")
    src = src.replace(score_rank_needle, score_rank_patch, 1)

    pp_reason_needle = "                reason = \"PP_TRAIL\"\n"
    pp_reason_patch = (
        "                reason = \"PP_TRAIL\"\n"
        "                last_full_exit_idx[t] = i\n"
        "                last_full_exit_reason[t] = reason\n"
    )
    if pp_reason_needle not in src:
        raise RuntimeError("Unable to patch run_backtest for PP exit reason tracking.")
    src = src.replace(pp_reason_needle, pp_reason_patch, 1)

    exit_reason_needle = "                    exit_defer.pop(t, None)\n                    raw_price = float(px_open[t])\n"
    exit_reason_patch = (
        "                    exit_defer.pop(t, None)\n"
        "                    last_full_exit_idx[t] = i\n"
        "                    last_full_exit_reason[t] = \"EXIT_NOT_TARGET\"\n"
        "                    raw_price = float(px_open[t])\n"
    )
    if exit_reason_needle not in src:
        raise RuntimeError("Unable to patch run_backtest for EXIT_NOT_TARGET reason tracking.")
    src = src.replace(exit_reason_needle, exit_reason_patch, 1)

    delta_reason_needle = "                    if new_sh <= 1e-10:\n                        positions.pop(t, None)\n                        pp_state.pop(t, None)\n                        if t in loss_names:\n                            loss_last_exit_idx[t] = i\n"
    delta_reason_patch = (
        "                    if new_sh <= 1e-10:\n"
        "                        positions.pop(t, None)\n"
        "                        pp_state.pop(t, None)\n"
        "                        last_full_exit_idx[t] = i\n"
        "                        last_full_exit_reason[t] = \"DELTA_SELL\"\n"
        "                        if t in loss_names:\n"
        "                            loss_last_exit_idx[t] = i\n"
    )
    if delta_reason_needle not in src:
        raise RuntimeError("Unable to patch run_backtest for DELTA_SELL reason tracking.")
    src = src.replace(delta_reason_needle, delta_reason_patch, 1)

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


def render_report(df: pd.DataFrame, best_reason: pd.DataFrame) -> str:
    baseline = df.loc[df["label"] == "baseline_current"].iloc[0]
    best = df.iloc[0]
    top = df.head(12).copy()
    lines = [
        "# R8 Reason-Aware Reentry Study",
        "",
        "Targeted study of a remaining structural candidate on top of `r8`: let reentry permission depend on the last full exit reason.",
        "",
        "Idea:",
        "- keep the existing `r4` state-dependent reentry overlay;",
        "- but neutralize or penalize recent reentry after `EXIT_NOT_TARGET`, where past expectancy is much weaker;",
        "- and optionally reward recent reentry after `PP_TRAIL`, where past expectancy is strongest.",
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
        "## Best Variant Reentry-20 By Last Sell Reason",
        "",
        dd_mod.markdown_table(best_reason),
        "",
    ]
    lines.extend(
        governance_section(
            df,
            "baseline_current",
            current_baseline="r8",
            robust_fallback="r7",
            study_verdict="candidate_only" if best["label"] != "baseline_current" else "rejected_no_upgrade",
            recommendation="Promote only if the best variant improves full and OOS cleanly without paying a material drawdown cost; otherwise keep r8 unchanged.",
            economic_explanation="Recent reentries after PP_TRAIL are economically stronger than recent reentries after EXIT_NOT_TARGET, so conditioning the overlay on last exit reason is structurally plausible.",
            concentration_note="This remains a portfolio-logic change rather than a score tweak, which is why it is worth testing despite the anti-overfit freeze.",
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
        ("reason_xnt_neutral_20", {
            "reentry_reason_enable": 1,
            "reentry_reason_lookback_td": 20,
            "reentry_reason_exit_not_target_penalty_mult": 1.0,
            "reentry_reason_pp_trail_bonus": 0.0,
            "reentry_reason_delta_sell_bonus": 0.0,
        }),
        ("reason_xnt_penalty_20", {
            "reentry_reason_enable": 1,
            "reentry_reason_lookback_td": 20,
            "reentry_reason_exit_not_target_penalty_mult": 1.5,
            "reentry_reason_pp_trail_bonus": 0.0,
            "reentry_reason_delta_sell_bonus": 0.0,
        }),
        ("reason_xnt_block_20", {
            "reentry_reason_enable": 1,
            "reentry_reason_lookback_td": 20,
            "reentry_reason_exit_not_target_block": 1,
            "reentry_reason_exit_not_target_penalty_mult": 0.0,
            "reentry_reason_pp_trail_bonus": 0.0,
            "reentry_reason_delta_sell_bonus": 0.0,
        }),
        ("reason_xnt_neutral_40", {
            "reentry_reason_enable": 1,
            "reentry_reason_lookback_td": 40,
            "reentry_reason_exit_not_target_penalty_mult": 1.0,
            "reentry_reason_pp_trail_bonus": 0.0,
            "reentry_reason_delta_sell_bonus": 0.0,
        }),
        ("reason_ppbonus_20", {
            "reentry_reason_enable": 1,
            "reentry_reason_lookback_td": 20,
            "reentry_reason_exit_not_target_penalty_mult": 1.0,
            "reentry_reason_pp_trail_bonus": 0.05,
            "reentry_reason_delta_sell_bonus": 0.0,
        }),
        ("reason_ppdelta_20", {
            "reentry_reason_enable": 1,
            "reentry_reason_lookback_td": 20,
            "reentry_reason_exit_not_target_penalty_mult": 1.0,
            "reentry_reason_pp_trail_bonus": 0.05,
            "reentry_reason_delta_sell_bonus": 0.03,
        }),
        ("reason_ppdelta_40", {
            "reentry_reason_enable": 1,
            "reentry_reason_lookback_td": 40,
            "reentry_reason_exit_not_target_penalty_mult": 1.0,
            "reentry_reason_pp_trail_bonus": 0.05,
            "reentry_reason_delta_sell_bonus": 0.03,
        }),
        ("reason_ppdelta_20_xnt2", {
            "reentry_reason_enable": 1,
            "reentry_reason_lookback_td": 20,
            "reentry_reason_exit_not_target_penalty_mult": 2.0,
            "reentry_reason_pp_trail_bonus": 0.05,
            "reentry_reason_delta_sell_bonus": 0.03,
        }),
        ("reason_xnt_block20_pp05", {
            "reentry_reason_enable": 1,
            "reentry_reason_lookback_td": 20,
            "reentry_reason_exit_not_target_block": 1,
            "reentry_reason_exit_not_target_penalty_mult": 0.0,
            "reentry_reason_pp_trail_bonus": 0.05,
            "reentry_reason_delta_sell_bonus": 0.0,
        }),
    ]

    rows = []
    best_trades = None
    best_label = None
    for label, patch_cfg in variants:
        local_cfg = dict(cfg)
        local_cfg.update(patch_cfg)
        _, tr_full, full_metrics = run_variant(run_fn, prices, local_cfg, pp, "2015-01-02", replay_end)
        _, _, oos_metrics = run_variant(run_fn, prices, local_cfg, pp, "2022-01-03", replay_end)
        _, _, recent_metrics = run_variant(run_fn, prices, local_cfg, pp, "2025-01-02", replay_end)
        rows.append(summarize_variant(label, full_metrics, oos_metrics, recent_metrics))
        if best_trades is None:
            best_trades = tr_full
            best_label = label
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

    best_label = str(df.iloc[0]["label"])
    best_cfg = dict(cfg)
    for label, patch_cfg in variants:
        if label == best_label:
            best_cfg.update(patch_cfg)
            break
    _, best_trades, _ = run_variant(run_fn, prices, best_cfg, pp, "2015-01-02", replay_end)
    _, best_reason = reentry_mod.summarize_closed_trades(best_trades)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(render_report(df, best_reason), encoding="utf-8")
    print(df.to_string(index=False))
    print(best_reason.to_string(index=False))
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()
