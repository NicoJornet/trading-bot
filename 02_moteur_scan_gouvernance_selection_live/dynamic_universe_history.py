from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "dynamic_universe"
HISTORY_DIR = DATA_DIR / "history"
ALT_DATA_DIR = ROOT.parent / "data" / "dynamic_universe"
ALT_HISTORY_DIR = ALT_DATA_DIR / "history"

SELECTED_ADDS_PATH = DATA_DIR / "dynamic_universe_selected_additions.csv"
SELECTED_DEMS_PATH = DATA_DIR / "dynamic_universe_selected_demotions.csv"
SELECTED_MOVES_PATH = DATA_DIR / "dynamic_universe_selected_moves.csv"

EVENT_TIMELINE_PATH = DATA_DIR / "dynamic_universe_governance_event_timeline.csv"
HISTORY_READINESS_PATH = DATA_DIR / "dynamic_universe_history_readiness.csv"
HISTORY_READINESS_MD_PATH = DATA_DIR / "dynamic_universe_history_readiness.md"


def read_optional_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    kwargs.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, **kwargs)
    except EmptyDataError:
        return pd.DataFrame()


def history_dirs() -> list[Path]:
    dirs: list[Path] = []
    for path in (HISTORY_DIR, ALT_HISTORY_DIR):
        if path.exists() and path not in dirs:
            dirs.append(path)
    return dirs


def _dedupe_snapshots(paths: list[Path]) -> list[Path]:
    best_by_name: dict[str, Path] = {}
    for path in paths:
        current = best_by_name.get(path.name)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            best_by_name[path.name] = path
    return sorted(best_by_name.values(), key=lambda p: p.name)


def snapshot_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in history_dirs():
        paths.extend(directory.glob("dynamic_universe_snapshot_*.csv"))
    return _dedupe_snapshots(paths)


def _date_from_stem(path: Path, prefix: str) -> pd.Timestamp | pd.NaT:
    raw = path.stem.replace(prefix, "", 1)
    return pd.to_datetime(raw, errors="coerce")


def next_trading_date(decision_date: object, trading_index: pd.Index | None) -> pd.Timestamp:
    ts = pd.Timestamp(decision_date)
    if trading_index is None or len(trading_index) == 0:
        return ts
    idx = pd.Index(pd.to_datetime(trading_index))
    future = idx[idx > ts]
    if len(future) == 0:
        return ts
    return pd.Timestamp(future[0])


def save_action_history(
    adds: pd.DataFrame,
    dems: pd.DataFrame,
    selected: pd.DataFrame,
    decision_date: date | str | None = None,
) -> dict[str, Path]:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    if decision_date is None:
        decision_date = date.today().isoformat()
    decision_label = str(decision_date)
    outputs = {
        "adds": HISTORY_DIR / f"dynamic_universe_selected_additions_{decision_label}.csv",
        "dems": HISTORY_DIR / f"dynamic_universe_selected_demotions_{decision_label}.csv",
        "moves": HISTORY_DIR / f"dynamic_universe_selected_moves_{decision_label}.csv",
    }
    (adds if not adds.empty else pd.DataFrame({"ticker": []})).to_csv(outputs["adds"], index=False)
    (dems if not dems.empty else pd.DataFrame({"ticker": []})).to_csv(outputs["dems"], index=False)
    (selected if not selected.empty else pd.DataFrame(columns=["ticker", "side"])).to_csv(outputs["moves"], index=False)
    return outputs


def load_snapshot_stage_history() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in snapshot_paths():
        df = read_optional_csv(path)
        if df.empty or "ticker" not in df.columns:
            continue
        out = df.copy()
        out["snapshot_date"] = _date_from_stem(path, "dynamic_universe_snapshot_")
        out["snapshot_name"] = path.name
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True, sort=False)
    history = history.dropna(subset=["snapshot_date"]).sort_values(["snapshot_date", "ticker"])
    return history


def _load_action_history_files(kind: str) -> list[Path]:
    paths: list[Path] = []
    suffix = f"dynamic_universe_selected_{kind}_*.csv"
    for directory in history_dirs():
        paths.extend(directory.glob(suffix))
    best_by_name: dict[str, Path] = {}
    for path in paths:
        current = best_by_name.get(path.name)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            best_by_name[path.name] = path
    return sorted(best_by_name.values(), key=lambda p: p.name)


def load_action_history(kind: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _load_action_history_files(kind):
        df = read_optional_csv(path)
        if df.empty:
            continue
        out = df.copy()
        out["decision_date"] = _date_from_stem(path, f"dynamic_universe_selected_{kind}_")
        out["history_file"] = path.name
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.dropna(subset=["decision_date"]).sort_values(["decision_date"])
    return out


def _promotion_stage_rank(value: object) -> int:
    return {
        "approved_live": 5,
        "probation_live": 4,
        "targeted_integration": 3,
        "watch_queue": 2,
        "review_queue": 1,
        "reject_queue": 0,
        "blocked_broker": -1,
    }.get(str(value or ""), -2)


def build_governance_event_timeline(
    trading_index: pd.Index | None = None,
    include_current_files: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    move_history = load_action_history("moves")
    if move_history.empty and include_current_files:
        current = read_optional_csv(SELECTED_MOVES_PATH)
        if not current.empty:
            move_history = current.copy()
            move_history["decision_date"] = pd.Timestamp(date.today())
            move_history["history_file"] = SELECTED_MOVES_PATH.name
    if not move_history.empty:
        for row in move_history.itertuples(index=False):
            ticker = str(getattr(row, "ticker", "") or "")
            side = str(getattr(row, "side", "") or "").upper()
            if not ticker or side not in {"ADD", "REMOVE"}:
                continue
            decision_date = pd.Timestamp(getattr(row, "decision_date"))
            rows.append(
                {
                    "decision_date": decision_date,
                    "effective_date": next_trading_date(decision_date, trading_index),
                    "ticker": ticker,
                    "side": side,
                    "event_type": "selected_move",
                    "action_group": str(getattr(row, "action_group", "") or ""),
                    "paired_ticker": str(getattr(row, "paired_ticker", "") or ""),
                    "from_stage": "",
                    "to_stage": "",
                    "source": str(getattr(row, "history_file", "") or "current_selected_moves"),
                }
            )

    history = load_snapshot_stage_history()
    if not history.empty:
        keep_cols = ["ticker", "snapshot_date", "promotion_stage", "dynamic_status"]
        for col in keep_cols:
            if col not in history.columns:
                history[col] = ""
        history = history[keep_cols].copy()
        history["stage_rank"] = history["promotion_stage"].map(_promotion_stage_rank).fillna(-2)
        history = history.sort_values(["ticker", "snapshot_date"])
        for ticker, grp in history.groupby("ticker", dropna=False):
            prev_row = None
            for row in grp.itertuples(index=False):
                if prev_row is None:
                    prev_row = row
                    continue
                from_stage = str(getattr(prev_row, "promotion_stage", "") or "")
                to_stage = str(getattr(row, "promotion_stage", "") or "")
                from_rank = _promotion_stage_rank(from_stage)
                to_rank = _promotion_stage_rank(to_stage)
                if from_rank == to_rank:
                    prev_row = row
                    continue
                event_type = "stage_upgrade" if to_rank > from_rank else "stage_downgrade"
                inferred_side = ""
                if to_rank >= _promotion_stage_rank("approved_live") and from_rank < _promotion_stage_rank("approved_live"):
                    inferred_side = "ADD"
                elif from_rank >= _promotion_stage_rank("approved_live") and to_rank < _promotion_stage_rank("approved_live"):
                    inferred_side = "REMOVE"
                rows.append(
                    {
                        "decision_date": pd.Timestamp(getattr(row, "snapshot_date")),
                        "effective_date": next_trading_date(getattr(row, "snapshot_date"), trading_index),
                        "ticker": str(ticker),
                        "side": inferred_side,
                        "event_type": event_type,
                        "action_group": "",
                        "paired_ticker": "",
                        "from_stage": from_stage,
                        "to_stage": to_stage,
                        "source": "snapshot_transition",
                    }
                )
                prev_row = row

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["decision_date"] = pd.to_datetime(out["decision_date"], errors="coerce")
    out["effective_date"] = pd.to_datetime(out["effective_date"], errors="coerce")
    out = out.dropna(subset=["decision_date", "effective_date", "ticker"])
    out = out.sort_values(["effective_date", "decision_date", "ticker", "side", "event_type"])
    out = out.drop_duplicates(
        ["decision_date", "effective_date", "ticker", "side", "event_type", "from_stage", "to_stage", "source"],
        keep="first",
    )
    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(EVENT_TIMELINE_PATH, index=False)
    return out


def build_membership_from_events(
    trading_index: pd.Index,
    columns: pd.Index,
    base_tickers: list[str],
    staged_adds: dict[str, str] | None = None,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    membership = pd.DataFrame(False, index=trading_index, columns=columns)
    base_cols = [ticker for ticker in base_tickers if ticker in membership.columns]
    if base_cols:
        membership.loc[:, base_cols] = True
    for ticker, start in (staged_adds or {}).items():
        if ticker not in membership.columns:
            continue
        membership.loc[membership.index >= pd.Timestamp(start), ticker] = True
    if events is None or events.empty:
        return membership

    timeline = events.copy()
    timeline["effective_date"] = pd.to_datetime(timeline["effective_date"], errors="coerce")
    timeline = timeline.dropna(subset=["effective_date"]).sort_values(["effective_date", "decision_date", "ticker"])
    for row in timeline.itertuples(index=False):
        ticker = str(getattr(row, "ticker", "") or "")
        side = str(getattr(row, "side", "") or "").upper()
        if ticker not in membership.columns or side not in {"ADD", "REMOVE"}:
            continue
        ts = pd.Timestamp(getattr(row, "effective_date"))
        if side == "ADD":
            membership.loc[membership.index >= ts, ticker] = True
        else:
            membership.loc[membership.index >= ts, ticker] = False
    return membership


def build_history_readiness_snapshot() -> pd.DataFrame:
    snapshots = snapshot_paths()
    snapshot_dates = [_date_from_stem(path, "dynamic_universe_snapshot_") for path in snapshots]
    snapshot_dates = [ts for ts in snapshot_dates if pd.notna(ts)]

    move_files = _load_action_history_files("moves")
    add_files = _load_action_history_files("additions")
    dem_files = _load_action_history_files("demotions")
    move_history = load_action_history("moves")
    timeline = build_governance_event_timeline(save=False)

    rows: list[dict[str, object]] = []
    span_days = 0
    if snapshot_dates:
        span_days = int((max(snapshot_dates) - min(snapshot_dates)).days)
    move_days = len(move_files)
    readiness_score = 0.0
    readiness_score += min(len(snapshot_dates), 20) / 20.0
    readiness_score += min(move_days, 10) / 10.0
    readiness_score += min(span_days, 60) / 60.0
    readiness_score = round(readiness_score / 3.0, 4)
    if readiness_score >= 0.8:
        readiness_label = "ready_for_governance_replay"
    elif readiness_score >= 0.45:
        readiness_label = "building_history"
    else:
        readiness_label = "too_early"

    rows.append(
        {
            "as_of": date.today().isoformat(),
            "snapshot_files": int(len(snapshot_dates)),
            "first_snapshot_date": str(min(snapshot_dates).date()) if snapshot_dates else "",
            "last_snapshot_date": str(max(snapshot_dates).date()) if snapshot_dates else "",
            "snapshot_span_days": int(span_days),
            "selected_move_days": int(move_days),
            "selected_add_history_files": int(len(add_files)),
            "selected_demotion_history_files": int(len(dem_files)),
            "governance_event_rows": int(len(timeline)),
            "history_readiness_score": readiness_score,
            "history_readiness_label": readiness_label,
        }
    )
    return pd.DataFrame(rows)


def write_history_readiness_outputs() -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    readiness = build_history_readiness_snapshot()
    readiness.to_csv(HISTORY_READINESS_PATH, index=False)

    snapshot_history = load_snapshot_stage_history()
    lines = [
        "# Dynamic Universe History Readiness",
        "",
        readiness.to_string(index=False) if not readiness.empty else "(none)",
        "",
        "## Recent snapshot files",
        "",
        "\n".join(f"- `{path.name}`" for path in snapshot_paths()[-20:]) if snapshot_paths() else "(none)",
        "",
        "## Recent move history files",
        "",
        "\n".join(f"- `{path.name}`" for path in _load_action_history_files("moves")[-20:]) if _load_action_history_files("moves") else "(none)",
        "",
        "## Stage coverage",
        "",
    ]
    if not snapshot_history.empty and "promotion_stage" in snapshot_history.columns:
        stage_counts = (
            snapshot_history.groupby("promotion_stage", dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
        )
        lines.append(stage_counts.to_string(index=False))
    else:
        lines.append("(none)")
    HISTORY_READINESS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "csv": HISTORY_READINESS_PATH,
        "md": HISTORY_READINESS_MD_PATH,
    }
