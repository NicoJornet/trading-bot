from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import pandas as pd

import dynamic_universe_discovery as dud
import dynamic_universe_swap_study as swap_study


ROOT = Path(__file__).resolve().parent
EXPORTS_DIR = ROOT / "research" / "exports"
REPORTS_DIR = ROOT / "research" / "reports"
ACTIVE_PATH = ROOT / "data" / "extracts" / "apex_tickers_active.csv"

EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REMOVE_EXPORT = EXPORTS_DIR / "dynamic_universe_standalone_removals.csv"
PHASE1_EXPORT = EXPORTS_DIR / "dynamic_universe_standalone_removals_phase1.csv"
WALK_EXPORT = EXPORTS_DIR / "dynamic_universe_standalone_removal_walkforward.csv"
WALK_SUMMARY_EXPORT = EXPORTS_DIR / "dynamic_universe_standalone_removal_walkforward_summary.csv"
REPORT_PATH = REPORTS_DIR / "DYNAMIC_UNIVERSE_STANDALONE_REMOVAL_STUDY_184.md"
STATE_PATH = EXPORTS_DIR / "dynamic_universe_standalone_removal_state.json"
WALK_SHORTLIST_MAX = 4


def yearly_windows(full_end: str) -> list[tuple[str, str, str]]:
    return [(str(year), f"{year}-01-02", f"{year}-12-31") for year in range(2017, 2026)] + [("2026_ytd", "2026-01-02", full_end)]


def classify_removal(row: pd.Series) -> str:
    full_delta_roi = float(row.get("full_delta_roi_pct", 0.0) or 0.0)
    oos_delta_roi = float(row.get("oos_delta_roi_pct", 0.0) or 0.0)
    full_delta_sharpe = float(row.get("full_delta_sharpe", 0.0) or 0.0)
    oos_delta_sharpe = float(row.get("oos_delta_sharpe", 0.0) or 0.0)
    full_delta_maxdd = float(row.get("full_delta_maxdd_pct", 0.0) or 0.0)
    oos_delta_maxdd = float(row.get("oos_delta_maxdd_pct", 0.0) or 0.0)
    mean_delta_roi = float(row.get("mean_delta_roi_2017_2025", 0.0) or 0.0)
    mean_delta_sharpe = float(row.get("mean_delta_sharpe_2017_2025", 0.0) or 0.0)
    roi_wins = int(float(row.get("roi_wins_2017_2025", 0) or 0))
    sharpe_wins = int(float(row.get("sharpe_wins_2017_2025", 0) or 0))
    if (
        full_delta_roi > 0
        and oos_delta_roi >= 0
        and full_delta_sharpe >= 0
        and oos_delta_sharpe >= -0.01
        and full_delta_maxdd > -1.0
        and oos_delta_maxdd > -0.5
        and mean_delta_roi >= 0
        and mean_delta_sharpe >= 0
        and roi_wins >= 2
        and sharpe_wins >= 2
    ):
        return "approved_remove"
    if (
        full_delta_roi > 0
        or oos_delta_roi > 0
        or mean_delta_roi > 0
    ):
        return "watch_remove"
    return "reject_remove"


def removal_score(row: pd.Series) -> float:
    return (
        1000.0 * float(row.get("mean_delta_sharpe_2017_2025", 0.0) or 0.0)
        + 10.0 * float(row.get("mean_delta_roi_2017_2025", 0.0) or 0.0)
        + 0.05 * float(row.get("oos_delta_roi_pct", 0.0) or 0.0)
        + 20.0 * float(row.get("oos_delta_sharpe", 0.0) or 0.0)
        + 5.0 * float(row.get("mean_delta_maxdd_2017_2025", 0.0) or 0.0)
        + 10.0 * float(row.get("dead_score", 0.0) or 0.0)
    )


def dataframe_signature(df: pd.DataFrame, columns: list[str]) -> str:
    available = [col for col in columns if col in df.columns]
    if not available or df.empty:
        payload = ""
    else:
        payload = (
            df[available]
            .copy()
            .fillna("")
            .astype(str)
            .sort_values(available)
            .reset_index(drop=True)
            .to_csv(index=False)
        )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cached_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_cached_state(payload: dict) -> None:
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def removal_phase1_score(row: pd.Series) -> float:
    return (
        0.08 * float(row["oos_delta_roi_pct"])
        + 25.0 * float(row["oos_delta_sharpe"])
        + 0.02 * float(row["full_delta_roi_pct"])
        + 10.0 * float(row["full_delta_sharpe"])
        + 4.0 * float(row["full_delta_maxdd_pct"])
        + 2.0 * float(row["oos_delta_maxdd_pct"])
        + 5.0 * float(row["dead_score"])
    )


def add_phase1_removal_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["phase1_score"] = (
        0.08 * out["oos_delta_roi_pct"].astype(float)
        + 25.0 * out["oos_delta_sharpe"].astype(float)
        + 0.02 * out["full_delta_roi_pct"].astype(float)
        + 10.0 * out["full_delta_sharpe"].astype(float)
        + 4.0 * out["full_delta_maxdd_pct"].astype(float)
        + 2.0 * out["oos_delta_maxdd_pct"].astype(float)
        + 5.0 * out["dead_score"].astype(float)
    )
    out["phase1_pass"] = (
        (out["oos_delta_sharpe"] >= -0.02)
        & (out["oos_delta_maxdd_pct"] > -1.0)
        & (
            (out["oos_delta_roi_pct"] > 0)
            | (
                (out["full_delta_roi_pct"] > 0)
                & (out["full_delta_sharpe"] >= 0)
                & (out["full_delta_maxdd_pct"] > -1.5)
            )
        )
    )
    return out


def select_walkforward_removals(remove_df: pd.DataFrame, limit: int = WALK_SHORTLIST_MAX) -> pd.DataFrame:
    if remove_df.empty:
        return remove_df.copy()
    phase1 = add_phase1_removal_columns(remove_df)
    shortlist = phase1.loc[phase1["phase1_pass"]].sort_values(
        ["phase1_score", "oos_delta_roi_pct", "oos_delta_sharpe", "full_delta_roi_pct"],
        ascending=[False, False, False, False],
    )
    return shortlist.head(limit).copy()


def main() -> None:
    t0 = time.time()
    engine, _, cfg, pp, prices = dud.load_setup()
    print("[remove-study] setup loaded")
    active = pd.read_csv(ACTIVE_PATH)["ticker"].dropna().astype(str).tolist()
    active_cols = [col for col in prices.close.columns if col in active]
    prices = engine.Prices(open=prices.open[active_cols], close=prices.close[active_cols])

    full_start, full_end = "2015-01-02", "2026-03-21"
    oos_start = "2022-01-03"

    _, trades_full, baseline_full = dud.run_metrics(engine, prices, cfg, pp, full_start, full_end)
    _, _, baseline_oos = dud.run_metrics(engine, prices, cfg, pp, oos_start, full_end)
    print("[remove-study] baseline full/oos computed")

    diag = swap_study.baseline_diagnostics(prices, cfg, trades_full)
    demotions = swap_study.select_demotion_shortlist(diag, limit=8).copy()
    print(f"[remove-study] demotion shortlist ready count={len(demotions)}")

    state_payload = {
        "version": 2,
        "full_end": full_end,
        "demotion_sig": dataframe_signature(
            demotions,
            ["ticker", "dead_score", "retain_score", "latest_rank", "latest_score", "days_top15_trend"],
        ),
    }
    cached_state = load_cached_state()
    if (
        cached_state == state_payload
        and REMOVE_EXPORT.exists()
        and PHASE1_EXPORT.exists()
        and WALK_EXPORT.exists()
        and WALK_SUMMARY_EXPORT.exists()
    ):
        print("[remove-study] inputs unchanged; reusing cached exports")
        print(f"[remove-study] completed elapsed_sec={time.time() - t0:.1f}")
        return

    rows: list[dict] = []
    total_removals = int(len(demotions))
    rem_idx = 0
    for rem in demotions.itertuples(index=False):
        rem_idx += 1
        ticker = str(rem.ticker)
        print(f"[remove-study] evaluate removal={rem_idx}/{total_removals} ticker={ticker}")
        variant = engine.Prices(
            open=prices.open.drop(columns=[ticker], errors="ignore"),
            close=prices.close.drop(columns=[ticker], errors="ignore"),
        )
        _, _, full = dud.run_metrics(engine, variant, cfg, pp, full_start, full_end)
        _, _, oos = dud.run_metrics(engine, variant, cfg, pp, oos_start, full_end)
        rows.append(
            dud.row_from_run(
                ticker,
                full,
                oos,
                {
                    "ticker": ticker,
                    "dead_score": float(rem.dead_score),
                    "retain_score": float(rem.retain_score),
                    "latest_rank": float(rem.latest_rank) if pd.notna(rem.latest_rank) else float("nan"),
                    "latest_score": float(rem.latest_score) if pd.notna(rem.latest_score) else float("nan"),
                    "days_top15_trend": int(rem.days_top15_trend),
                    "realized_pnl_eur": float(rem.realized_pnl_eur),
                    "buy_count": int(rem.buy_count),
                    "full_delta_roi_pct": float(full["ROI_%"] - baseline_full["ROI_%"]),
                    "full_delta_sharpe": float(full["Sharpe"] - baseline_full["Sharpe"]),
                    "full_delta_maxdd_pct": float(full["MaxDD_%"] - baseline_full["MaxDD_%"]),
                    "oos_delta_roi_pct": float(oos["ROI_%"] - baseline_oos["ROI_%"]),
                    "oos_delta_sharpe": float(oos["Sharpe"] - baseline_oos["Sharpe"]),
                    "oos_delta_maxdd_pct": float(oos["MaxDD_%"] - baseline_oos["MaxDD_%"]),
                },
            )
        )

    remove_df = pd.DataFrame(rows)
    if remove_df.empty:
        REPORT_PATH.write_text("# Standalone Removal Study\n\nNo removal evaluated.\n", encoding="utf-8")
        return
    phase1_df = add_phase1_removal_columns(remove_df)
    phase1_df.to_csv(PHASE1_EXPORT, index=False)

    walk_rows: list[dict] = []
    top_walk = select_walkforward_removals(remove_df)
    print(f"[remove-study] walkforward shortlist count={len(top_walk)}")
    if top_walk.empty:
        pd.DataFrame(columns=["ticker", "window", "delta_roi_pct", "delta_sharpe", "delta_maxdd_pct"]).to_csv(WALK_EXPORT, index=False)
        pd.DataFrame(
            columns=[
                "ticker",
                "mean_delta_roi_2017_2025",
                "mean_delta_sharpe_2017_2025",
                "mean_delta_maxdd_2017_2025",
                "roi_wins_2017_2025",
                "sharpe_wins_2017_2025",
                "maxdd_wins_2017_2025",
                "delta_roi_2026_ytd",
                "delta_sharpe_2026_ytd",
                "delta_maxdd_2026_ytd",
            ]
        ).to_csv(WALK_SUMMARY_EXPORT, index=False)
        write_cached_state(state_payload)
        for col in (
            "mean_delta_roi_2017_2025",
            "mean_delta_sharpe_2017_2025",
            "mean_delta_maxdd_2017_2025",
            "roi_wins_2017_2025",
            "sharpe_wins_2017_2025",
            "maxdd_wins_2017_2025",
            "delta_roi_2026_ytd",
            "delta_sharpe_2026_ytd",
            "delta_maxdd_2026_ytd",
        ):
            if col not in remove_df.columns:
                remove_df[col] = 0.0
        remove_df["selection_status"] = remove_df.apply(classify_removal, axis=1)
        remove_df["selection_score"] = remove_df.apply(removal_score, axis=1)
        remove_df = remove_df.sort_values(
            ["selection_status", "selection_score", "oos_delta_roi_pct", "full_delta_roi_pct"],
            ascending=[False, False, False, False],
        )
        remove_df.to_csv(REMOVE_EXPORT, index=False)
        lines = [
            "# Dynamic Universe Standalone Removal Study",
            "",
            "## Demotion shortlist",
            "",
            demotions[
                [
                    "ticker",
                    "retain_score",
                    "dead_score",
                    "latest_rank",
                    "latest_score",
                    "days_top15_trend",
                    "realized_pnl_eur",
                    "buy_count",
                ]
            ].to_string(index=False),
            "",
            "## Phase 2 walk-forward shortlist",
            "",
            "(none)",
            "",
            "## Standalone removals",
            "",
            remove_df[
                [
                    "ticker",
                    "selection_status",
                    "selection_score",
                    "full_delta_roi_pct",
                    "oos_delta_roi_pct",
                    "full_delta_sharpe",
                    "oos_delta_sharpe",
                    "full_delta_maxdd_pct",
                    "oos_delta_maxdd_pct",
                ]
            ].to_string(index=False),
        ]
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved: {REMOVE_EXPORT}")
        print(f"Saved: {PHASE1_EXPORT}")
        print(f"Saved: {WALK_EXPORT}")
        print(f"Saved: {WALK_SUMMARY_EXPORT}")
        print(f"Saved: {REPORT_PATH}")
        print(f"[remove-study] completed elapsed_sec={time.time() - t0:.1f}")
        return

    base_window_metrics = {
        label: dud.run_metrics(engine, prices, cfg, pp, win_start, win_end)[2]
        for label, win_start, win_end in yearly_windows(full_end)
    }
    total_walk = int(len(top_walk) * len(base_window_metrics))
    walk_idx = 0
    for row in top_walk.itertuples(index=False):
        variant = engine.Prices(
            open=prices.open.drop(columns=[str(row.ticker)], errors="ignore"),
            close=prices.close.drop(columns=[str(row.ticker)], errors="ignore"),
        )
        for label, win_start, win_end in yearly_windows(full_end):
            walk_idx += 1
            print(f"[remove-study] walkforward {walk_idx}/{total_walk} ticker={row.ticker} window={label}")
            base_out = base_window_metrics[label]
            _, _, var_out = dud.run_metrics(engine, variant, cfg, pp, win_start, win_end)
            walk_rows.append(
                {
                    "ticker": str(row.ticker),
                    "window": label,
                    "delta_roi_pct": float(var_out["ROI_%"] - base_out["ROI_%"]),
                    "delta_sharpe": float(var_out["Sharpe"] - base_out["Sharpe"]),
                    "delta_maxdd_pct": float(var_out["MaxDD_%"] - base_out["MaxDD_%"]),
                }
            )

    walk_df = pd.DataFrame(walk_rows)
    walk_df.to_csv(WALK_EXPORT, index=False)

    summary_rows: list[dict] = []
    if not walk_df.empty:
        for ticker, group in walk_df.groupby("ticker"):
            yearly = group[group["window"] != "2026_ytd"]
            ytd = group[group["window"] == "2026_ytd"]
            summary_rows.append(
                {
                    "ticker": ticker,
                    "mean_delta_roi_2017_2025": float(yearly["delta_roi_pct"].mean()),
                    "mean_delta_sharpe_2017_2025": float(yearly["delta_sharpe"].mean()),
                    "mean_delta_maxdd_2017_2025": float(yearly["delta_maxdd_pct"].mean()),
                    "roi_wins_2017_2025": int((yearly["delta_roi_pct"] > 0).sum()),
                    "sharpe_wins_2017_2025": int((yearly["delta_sharpe"] > 0).sum()),
                    "maxdd_wins_2017_2025": int((yearly["delta_maxdd_pct"] >= 0).sum()),
                    "delta_roi_2026_ytd": float(ytd["delta_roi_pct"].mean()) if not ytd.empty else 0.0,
                    "delta_sharpe_2026_ytd": float(ytd["delta_sharpe"].mean()) if not ytd.empty else 0.0,
                    "delta_maxdd_2026_ytd": float(ytd["delta_maxdd_pct"].mean()) if not ytd.empty else 0.0,
                }
            )

    walk_summary = pd.DataFrame(summary_rows)
    if not walk_summary.empty:
        remove_df = remove_df.merge(walk_summary, on="ticker", how="left")
    for col in (
        "mean_delta_roi_2017_2025",
        "mean_delta_sharpe_2017_2025",
        "mean_delta_maxdd_2017_2025",
        "roi_wins_2017_2025",
        "sharpe_wins_2017_2025",
        "maxdd_wins_2017_2025",
        "delta_roi_2026_ytd",
        "delta_sharpe_2026_ytd",
        "delta_maxdd_2026_ytd",
    ):
        if col not in remove_df.columns:
            remove_df[col] = 0.0
        remove_df[col] = pd.to_numeric(remove_df[col], errors="coerce").fillna(0.0)

    remove_df["selection_status"] = remove_df.apply(classify_removal, axis=1)
    remove_df["selection_score"] = remove_df.apply(removal_score, axis=1)
    remove_df = remove_df.sort_values(
        ["selection_status", "selection_score", "oos_delta_roi_pct", "full_delta_roi_pct"],
        ascending=[False, False, False, False],
    )

    remove_df.to_csv(REMOVE_EXPORT, index=False)
    walk_summary.to_csv(WALK_SUMMARY_EXPORT, index=False)
    write_cached_state(state_payload)

    lines = [
        "# Dynamic Universe Standalone Removal Study",
        "",
        "## Demotion shortlist",
        "",
        demotions[
            [
                "ticker",
                "retain_score",
                "dead_score",
                "latest_rank",
                "latest_score",
                "days_top15_trend",
                "realized_pnl_eur",
                "buy_count",
            ]
        ].to_string(index=False),
        "",
        "## Phase 2 walk-forward shortlist",
        "",
        top_walk[
            [
                "ticker",
                "phase1_score",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
                "full_delta_sharpe",
                "oos_delta_sharpe",
                "dead_score",
            ]
        ].to_string(index=False) if not top_walk.empty else "(none)",
        "",
        "## Standalone removals",
        "",
        remove_df[
            [
                "ticker",
                "selection_status",
                "selection_score",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
                "full_delta_sharpe",
                "oos_delta_sharpe",
                "full_delta_maxdd_pct",
                "oos_delta_maxdd_pct",
                "mean_delta_roi_2017_2025",
                "mean_delta_sharpe_2017_2025",
            ]
        ].to_string(index=False),
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {REMOVE_EXPORT}")
    print(f"Saved: {PHASE1_EXPORT}")
    print(f"Saved: {WALK_EXPORT}")
    print(f"Saved: {WALK_SUMMARY_EXPORT}")
    print(f"Saved: {REPORT_PATH}")
    print(f"[remove-study] completed elapsed_sec={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
