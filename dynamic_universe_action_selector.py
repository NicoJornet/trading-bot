from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

import dynamic_universe_discovery as dud


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "dynamic_universe"
EXPORTS_DIR = ROOT / "research" / "exports"
REPORTS_DIR = ROOT / "research" / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_PATH = EXPORTS_DIR / "dynamic_universe_swap_single_summary.csv"
SINGLE_WALK_PATH = EXPORTS_DIR / "dynamic_universe_swap_walkforward_summary.csv"
LEGACY_SINGLE_WALK_PATH = EXPORTS_DIR / "dynamic_universe_swap_top12_walkforward_summary.csv"
COMBO_PATH = EXPORTS_DIR / "dynamic_universe_swap_combo_summary.csv"
COMBO_WALK_PATH = EXPORTS_DIR / "dynamic_universe_swap_combo_walkforward_summary.csv"
DB_PATH = DATA_DIR / "dynamic_universe_current.csv"
STANDALONE_REMOVE_PATH = EXPORTS_DIR / "dynamic_universe_standalone_removals.csv"

SELECTED_ADDS_PATH = DATA_DIR / "dynamic_universe_selected_additions.csv"
SELECTED_DEMS_PATH = DATA_DIR / "dynamic_universe_selected_demotions.csv"
SELECTED_MOVES_PATH = DATA_DIR / "dynamic_universe_selected_moves.csv"
SUMMARY_PATH = DATA_DIR / "dynamic_universe_actions_summary.md"

BROKER_MIN_MARKET_CAP = 1_000_000_000


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def resolve_broker_tradeability(tickers: set[str]) -> dict[str, bool]:
    db = read_optional_csv(DB_PATH)
    tradeable: dict[str, bool] = {}
    if not db.empty:
        for _, row in db.iterrows():
            ticker = str(row.get("ticker") or "")
            if not ticker:
                continue
            mcap = pd.to_numeric(pd.Series([row.get("marketCap")]), errors="coerce").iloc[0]
            if pd.notna(mcap):
                tradeable[ticker] = bool(mcap >= BROKER_MIN_MARKET_CAP)
    unresolved = [ticker for ticker in tickers if ticker and ticker not in tradeable]
    if unresolved:
        dud.setup_yf_cache(ROOT / ".yf_cache")
        for ticker in unresolved:
            ctx = dud.ticker_context(ticker)
            mcap = ctx.get("marketCap")
            try:
                tradeable[ticker] = bool(float(mcap) >= BROKER_MIN_MARKET_CAP)
            except (TypeError, ValueError):
                tradeable[ticker] = True
    return tradeable


def classify_single(row: pd.Series) -> str:
    if (
        row["full_delta_roi_pct"] > 0
        and row["oos_delta_roi_pct"] > 0
        and row["full_delta_sharpe"] >= 0
        and row["oos_delta_sharpe"] >= -0.01
        and row["full_delta_maxdd_pct"] > -3.0
        and row["oos_delta_maxdd_pct"] > -1.0
        and row["mean_delta_roi_2017_2025"] > 0
        and row["mean_delta_sharpe_2017_2025"] >= 0
        and row["roi_wins_2017_2025"] >= 2
        and row["sharpe_wins_2017_2025"] >= 2
    ):
        return "approved"
    if (
        row["oos_delta_roi_pct"] > 0
        or row["full_delta_roi_pct"] > 0
        or row["mean_delta_roi_2017_2025"] > 0
    ):
        return "watch"
    return "reject"


def classify_combo(row: pd.Series) -> str:
    if (
        bool(row.get("strict_reco", False))
        and row["mean_delta_roi_2017_2025"] > 0
        and row["mean_delta_sharpe_2017_2025"] > 0.01
        and row["mean_delta_maxdd_2017_2025"] >= 0
        and row["roi_wins_2017_2025"] >= 3
        and row["sharpe_wins_2017_2025"] >= 3
    ):
        return "approved"
    if row["oos_delta_roi_pct"] > 0 or row["mean_delta_roi_2017_2025"] > 0:
        return "watch"
    return "reject"


def single_score(row: pd.Series) -> float:
    return (
        1000.0 * float(row["mean_delta_sharpe_2017_2025"])
        + 10.0 * float(row["mean_delta_roi_2017_2025"])
        + 0.05 * float(row["oos_delta_roi_pct"])
        + 25.0 * float(row["oos_delta_sharpe"])
        + 5.0 * float(row["mean_delta_maxdd_2017_2025"])
    )


def combo_score(row: pd.Series) -> float:
    return (
        1200.0 * float(row["mean_delta_sharpe_2017_2025"])
        + 10.0 * float(row["mean_delta_roi_2017_2025"])
        + 0.05 * float(row["oos_delta_roi_pct"])
        + 20.0 * float(row["oos_delta_sharpe"])
        + 5.0 * float(row["mean_delta_maxdd_2017_2025"])
    )


def build_single_actions() -> pd.DataFrame:
    single = read_optional_csv(SINGLE_PATH)
    walk_current = read_optional_csv(SINGLE_WALK_PATH)
    walk_legacy = read_optional_csv(LEGACY_SINGLE_WALK_PATH)
    walk = pd.concat([walk_current, walk_legacy], ignore_index=True, sort=False)
    if not walk.empty:
        dedupe_cols = ["candidate", "removed"] if {"candidate", "removed"}.issubset(walk.columns) else ["swap"]
        walk = walk.drop_duplicates(dedupe_cols, keep="first")
    if single.empty or walk.empty:
        return pd.DataFrame()
    if "candidate" not in walk.columns or "removed" not in walk.columns:
        parsed = walk.get("swap", pd.Series(dtype=str)).fillna("").astype(str).str.split("_for_", n=1, expand=True)
        if not parsed.empty and parsed.shape[1] == 2:
            walk["candidate"] = parsed[0]
            walk["removed"] = parsed[1]
    if "maxdd_wins_2017_2025" not in walk.columns:
        walk["maxdd_wins_2017_2025"] = 0

    df = single.merge(
        walk[
            [
                "swap",
                "candidate",
                "removed",
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
        ],
        on=["candidate", "removed"],
        how="left",
    )
    numeric_cols = [c for c in df.columns if c.endswith(("_pct", "_sharpe", "_ytd")) or "_wins_" in c]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["selection_status"] = df.apply(classify_single, axis=1)
    df["selection_score"] = df.apply(single_score, axis=1)
    return df.sort_values(
        ["selection_status", "selection_score", "oos_delta_roi_pct", "full_delta_roi_pct"],
        ascending=[False, False, False, False],
    )


def build_combo_actions() -> pd.DataFrame:
    combo = read_optional_csv(COMBO_PATH)
    walk = read_optional_csv(COMBO_WALK_PATH)
    if combo.empty or walk.empty:
        return pd.DataFrame()

    df = combo.merge(
        walk[
            [
                "combo",
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
        ],
        on="combo",
        how="left",
    )
    numeric_cols = [c for c in df.columns if c.endswith(("_pct", "_sharpe", "_ytd")) or "_wins_" in c]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "strict_reco" not in df.columns:
        df["strict_reco"] = False
    df["selection_status"] = df.apply(classify_combo, axis=1)
    df["selection_score"] = df.apply(combo_score, axis=1)
    return df.sort_values(
        ["selection_status", "selection_score", "oos_delta_roi_pct", "full_delta_roi_pct"],
        ascending=[False, False, False, False],
    )


def build_dynamic_add_actions() -> pd.DataFrame:
    db = read_optional_csv(DB_PATH)
    if db.empty:
        return pd.DataFrame()
    if "scan_algo_compat_score_v2" not in db.columns:
        db["scan_algo_compat_score_v2"] = db.get("scan_algo_compat_score", 0.0)
    for col in ("dynamic_conviction_score", "recent_score", "scan_algo_compat_score", "scan_algo_compat_score_v2", "profile_count"):
        if col not in db.columns:
            db[col] = 0.0
        db[col] = pd.to_numeric(db[col], errors="coerce").fillna(0.0)
    out = db.copy()
    if "promotion_stage" in out.columns:
        out["selection_status"] = out["promotion_stage"].map(
            {
                "approved_live": "approved_add",
                "probation_live": "watch_add",
                "targeted_integration": "watch_add",
                "watch_queue": "watch_add",
            }
        ).fillna("reject_add")
    else:
        out["selection_status"] = out["dynamic_status"].map({"approved": "approved_add", "prime_watch": "watch_add", "watch": "watch_add"}).fillna("reject_add")
    out["selection_score"] = pd.to_numeric(out.get("promotion_score", out["dynamic_conviction_score"]), errors="coerce").fillna(out["dynamic_conviction_score"])
    return out.sort_values(
        ["selection_status", "selection_score", "scan_algo_compat_score_v2", "recent_score", "scan_algo_compat_score"],
        ascending=[False, False, False, False, False],
    )


def build_standalone_remove_actions() -> pd.DataFrame:
    df = read_optional_csv(STANDALONE_REMOVE_PATH)
    if df.empty:
        return pd.DataFrame()
    for col in (
        "selection_score",
        "full_delta_roi_pct",
        "oos_delta_roi_pct",
        "mean_delta_roi_2017_2025",
        "mean_delta_sharpe_2017_2025",
        "dead_score",
    ):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.sort_values(
        ["selection_status", "selection_score", "oos_delta_roi_pct", "full_delta_roi_pct"],
        ascending=[False, False, False, False],
    )


def parse_combo_moves(combo_label: str) -> list[tuple[str, str]]:
    parts = [x.strip() for x in str(combo_label).split("+") if x.strip()]
    out = []
    for part in parts:
        if "->" not in part:
            continue
        add, rem = [x.strip() for x in part.split("->", 1)]
        if add and rem:
            out.append((add, rem))
    return out


def choose_action_plan(
    single_df: pd.DataFrame,
    combo_df: pd.DataFrame,
    add_df: pd.DataFrame,
    remove_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    selected_rows: list[dict[str, object]] = []
    used_adds: set[str] = set()
    used_rems: set[str] = set()
    add_tickers = set(add_df.get("ticker", pd.Series(dtype=str)).dropna().astype(str))
    single_adds = set(single_df.get("candidate", pd.Series(dtype=str)).dropna().astype(str))
    combo_adds: set[str] = set()
    if "combo" in combo_df.columns:
        for combo in combo_df["combo"].dropna().astype(str):
            combo_adds.update(add for add, _ in parse_combo_moves(combo))
    broker_tradeable = resolve_broker_tradeability(add_tickers | single_adds | combo_adds)

    def push_row(action_group: str, action_type: str, ticker: str, paired_ticker: str, side: str, selection_status: str, selection_score: float, reason: str) -> None:
        selected_rows.append(
            {
                "action_group": action_group,
                "action_type": action_type,
                "ticker": ticker,
                "paired_ticker": paired_ticker,
                "side": side,
                "selection_status": selection_status,
                "selection_score": selection_score,
                "reason": reason,
            }
        )

    approved_combos = combo_df.loc[combo_df["selection_status"] == "approved"].copy()
    if not approved_combos.empty:
        top_combo = approved_combos.sort_values(
            ["selection_score", "mean_delta_sharpe_2017_2025", "mean_delta_roi_2017_2025"],
            ascending=[False, False, False],
        ).iloc[0]
        combo_moves = parse_combo_moves(str(top_combo["combo"]))
        if not all(broker_tradeable.get(add, True) for add, _ in combo_moves):
            combo_moves = []
        for add, rem in combo_moves:
            if add not in used_adds:
                used_adds.add(add)
                push_row(str(top_combo["combo"]), "combo", add, rem, "ADD", str(top_combo["selection_status"]), float(top_combo["selection_score"]), "approved_combo")
            if rem not in used_rems:
                used_rems.add(rem)
                push_row(str(top_combo["combo"]), "combo", rem, add, "REMOVE", str(top_combo["selection_status"]), float(top_combo["selection_score"]), "approved_combo")

    approved_singles = single_df.loc[single_df["selection_status"] == "approved"].copy()
    for row in approved_singles.sort_values(
        ["selection_score", "mean_delta_sharpe_2017_2025", "mean_delta_roi_2017_2025"],
        ascending=[False, False, False],
    ).itertuples(index=False):
        if not broker_tradeable.get(str(row.candidate), True):
            continue
        if row.candidate in used_adds or row.removed in used_rems:
            continue
        used_adds.add(str(row.candidate))
        used_rems.add(str(row.removed))
        group = f"{row.candidate}->{row.removed}"
        push_row(group, "single", str(row.candidate), str(row.removed), "ADD", str(row.selection_status), float(row.selection_score), "approved_single")
        push_row(group, "single", str(row.removed), str(row.candidate), "REMOVE", str(row.selection_status), float(row.selection_score), "approved_single")

    approved_adds = add_df.loc[add_df["selection_status"] == "approved_add"].copy()
    for row in approved_adds.sort_values(
        ["selection_score", "scan_algo_compat_score_v2", "recent_score", "scan_algo_compat_score"],
        ascending=[False, False, False, False],
    ).itertuples(index=False):
        ticker = str(row.ticker)
        if not broker_tradeable.get(ticker, True):
            continue
        if ticker in used_adds:
            continue
        used_adds.add(ticker)
        push_row(f"ADD:{ticker}", "standalone_add", ticker, "", "ADD", str(row.selection_status), float(row.selection_score), "approved_add")

    approved_removes = remove_df.loc[remove_df["selection_status"] == "approved_remove"].copy()
    for row in approved_removes.sort_values(
        ["selection_score", "mean_delta_sharpe_2017_2025", "mean_delta_roi_2017_2025"],
        ascending=[False, False, False],
    ).itertuples(index=False):
        ticker = str(row.ticker)
        if ticker in used_rems:
            continue
        used_rems.add(ticker)
        push_row(f"REMOVE:{ticker}", "standalone_remove", ticker, "", "REMOVE", str(row.selection_status), float(row.selection_score), "approved_remove")

    selected = pd.DataFrame(selected_rows)
    return selected, {
        "approved_combos": approved_combos,
        "approved_singles": approved_singles,
        "approved_adds": approved_adds,
        "approved_removes": approved_removes,
    }


def write_outputs(single_df: pd.DataFrame, combo_df: pd.DataFrame, add_df: pd.DataFrame, remove_df: pd.DataFrame, selected: pd.DataFrame) -> None:
    adds = (
        selected.loc[selected["side"] == "ADD", ["ticker"]]
        .drop_duplicates()
        .sort_values("ticker")
        if not selected.empty
        else pd.DataFrame({"ticker": []})
    )
    dems = (
        selected.loc[selected["side"] == "REMOVE", ["ticker"]]
        .drop_duplicates()
        .sort_values("ticker")
        if not selected.empty
        else pd.DataFrame({"ticker": []})
    )

    adds.to_csv(SELECTED_ADDS_PATH, index=False)
    dems.to_csv(SELECTED_DEMS_PATH, index=False)
    selected.to_csv(SELECTED_MOVES_PATH, index=False)

    approved_single = single_df.loc[single_df["selection_status"] == "approved"].copy()
    watch_single = single_df.loc[single_df["selection_status"] == "watch"].copy()
    approved_combo = combo_df.loc[combo_df["selection_status"] == "approved"].copy()
    watch_combo = combo_df.loc[combo_df["selection_status"] == "watch"].copy()
    approved_add = add_df.loc[add_df["selection_status"] == "approved_add"].copy()
    watch_add = add_df.loc[add_df["selection_status"] == "watch_add"].copy()
    approved_remove = remove_df.loc[remove_df["selection_status"] == "approved_remove"].copy()
    watch_remove = remove_df.loc[remove_df["selection_status"] == "watch_remove"].copy()

    lines = [
        "# Dynamic Universe Actions Summary",
        "",
        f"- as_of: `{date.today().isoformat()}`",
        f"- approved single swaps: `{len(approved_single)}`",
        f"- approved combo swaps: `{len(approved_combo)}`",
        f"- approved standalone adds: `{len(approved_add)}`",
        f"- approved standalone removes: `{len(approved_remove)}`",
        f"- selected additions: `{len(adds)}`",
        f"- selected demotions: `{len(dems)}`",
        "",
        "## Selected moves",
        "",
        selected.to_string(index=False) if not selected.empty else "(none)",
        "",
        "## Approved standalone adds",
        "",
        approved_add[
            [
                "ticker",
                "selection_score",
                "promotion_stage",
                "dynamic_status",
                "profile_count",
                "scan_algo_fit",
                "scan_algo_compat_score_v2",
                "recent_score",
            ]
        ].to_string(index=False)
        if not approved_add.empty
        else "(none)",
        "",
        "## Approved standalone removes",
        "",
        approved_remove[
            [
                "ticker",
                "selection_score",
                "dead_score",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
                "mean_delta_roi_2017_2025",
                "mean_delta_sharpe_2017_2025",
            ]
        ].to_string(index=False)
        if not approved_remove.empty
        else "(none)",
        "",
        "## Approved single swaps",
        "",
        approved_single[
            [
                "candidate",
                "removed",
                "selection_score",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
                "mean_delta_roi_2017_2025",
                "mean_delta_sharpe_2017_2025",
            ]
        ].to_string(index=False)
        if not approved_single.empty
        else "(none)",
        "",
        "## Approved combo swaps",
        "",
        approved_combo[
            [
                "combo",
                "selection_score",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
                "mean_delta_roi_2017_2025",
                "mean_delta_sharpe_2017_2025",
            ]
        ].to_string(index=False)
        if not approved_combo.empty
        else "(none)",
        "",
        "## Watch single swaps",
        "",
        watch_single[
            [
                "candidate",
                "removed",
                "selection_score",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
                "mean_delta_roi_2017_2025",
                "mean_delta_sharpe_2017_2025",
            ]
        ].head(20).to_string(index=False)
        if not watch_single.empty
        else "(none)",
        "",
        "## Watch combo swaps",
        "",
        watch_combo[
            [
                "combo",
                "selection_score",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
                "mean_delta_roi_2017_2025",
                "mean_delta_sharpe_2017_2025",
            ]
        ].head(20).to_string(index=False)
        if not watch_combo.empty
        else "(none)",
        "",
        "## Watch standalone adds",
        "",
        watch_add[
            [
                "ticker",
                "selection_score",
                "promotion_stage",
                "dynamic_status",
                "profile_count",
                "scan_algo_fit",
                "scan_algo_compat_score_v2",
                "recent_score",
            ]
        ].head(20).to_string(index=False)
        if not watch_add.empty
        else "(none)",
        "",
        "## Watch standalone removes",
        "",
        watch_remove[
            [
                "ticker",
                "selection_score",
                "dead_score",
                "full_delta_roi_pct",
                "oos_delta_roi_pct",
                "mean_delta_roi_2017_2025",
                "mean_delta_sharpe_2017_2025",
            ]
        ].head(20).to_string(index=False)
        if not watch_remove.empty
        else "(none)",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Select dynamic universe additions/demotions from swap studies.")
    parser.parse_args(argv)

    single_df = build_single_actions()
    combo_df = build_combo_actions()
    add_df = build_dynamic_add_actions()
    remove_df = build_standalone_remove_actions()
    selected, _ = choose_action_plan(single_df, combo_df, add_df, remove_df)
    write_outputs(single_df, combo_df, add_df, remove_df, selected)

    print(f"selected_additions: {SELECTED_ADDS_PATH}")
    print(f"selected_demotions: {SELECTED_DEMS_PATH}")
    print(f"selected_moves: {SELECTED_MOVES_PATH}")
    print(f"summary: {SUMMARY_PATH}")
    if not selected.empty:
        print(selected.to_string(index=False))
    else:
        print("No selected moves.")


if __name__ == "__main__":
    main()
