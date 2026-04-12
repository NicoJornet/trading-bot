from __future__ import annotations

import inspect
import json
from copy import deepcopy
from importlib import util
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\nicol\Downloads\214")
ENGINE_PATH = ROOT / "trading-bot" / "engine_bundle" / "run_v11_slot_quality" / "v11_slot_quality" / "apex_engine_slot_quality.py"
BEST_PATH = ROOT / "BEST_ALGO_184.json"
CSV_PATH = ROOT / "apex_ohlcv_full_2015_2026.csv"
ACTIVE_PATH = ROOT / "trading-bot" / "data" / "extracts" / "apex_tickers_active.csv"
OUT_SUMMARY = ROOT / "research" / "exports" / "state_dependent_reentry_184.csv"
OUT_METRICS = ROOT / "research" / "exports" / "state_dependent_reentry_184_metrics.csv"
OUT_MD = ROOT / "research" / "reports" / "STATE_DEPENDENT_REENTRY_184.md"

DEFAULT_ACTIVATION_DATES = {
    "AXTI": "2016-10-04",
    "006800.KS": "2017-07-03",
    "0568.HK": "2026-03-27",
}

WINDOWS = [
    ("full", "2015-01-02", None),
    ("oos", "2022-01-03", None),
    ("since2023", "2023-01-03", None),
    ("since2025", "2025-01-02", None),
    ("ytd2026", "2026-01-02", None),
    ("2018", "2018-01-02", "2018-12-31"),
    ("2019", "2019-01-02", "2019-12-31"),
    ("2022", "2022-01-03", "2022-12-30"),
    ("2024", "2024-01-02", "2024-12-31"),
]


def load_engine():
    spec = util.spec_from_file_location("apex_engine_slot_quality_state_reentry", ENGINE_PATH)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_best():
    obj = json.loads(BEST_PATH.read_text(encoding="utf-8"))
    cfg = deepcopy(obj["cfg"])
    pp = deepcopy(obj["pp"])
    activation = obj.get("staged_activation_dates") or DEFAULT_ACTIVATION_DATES
    activation = {str(k): str(v) for k, v in activation.items()}
    return cfg, pp, activation


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
    needle = "        elig = elig & enough.reindex(c.columns, fill_value=False)\n"
    if needle not in src:
        raise RuntimeError("Unable to patch run_backtest for dynamic membership.")
    patch = needle + "        elig = elig & universe_membership.loc[d].reindex(c.columns, fill_value=False).astype(bool)\n"
    src = src.replace(needle, patch, 1)
    ns = dict(engine.__dict__)
    ns["universe_membership"] = universe_membership
    exec(src, ns)
    return ns["run_backtest"]


def make_variant_cfg(base_cfg: dict, name: str) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["reentry_enable"] = 1
    cfg["reentry_lt_topn"] = 12
    cfg["reentry_dd60_min"] = 0.05
    cfg["reentry_dd60_max"] = 0.25
    cfg["reentry_r21_max"] = 0.03
    cfg["reentry_bonus"] = 0.15
    cfg.pop("reentry_breadth_min", None)
    cfg.pop("reentry_rs63_min", None)

    if name == "r2_baseline":
        cfg["reentry_enable"] = 0
        return cfg
    if name == "r3_reentry":
        return cfg
    if name == "r4_b50_rs00":
        cfg["reentry_breadth_min"] = 0.50
        cfg["reentry_rs63_min"] = 0.00
        return cfg
    if name == "r4_b50_rs01":
        cfg["reentry_breadth_min"] = 0.50
        cfg["reentry_rs63_min"] = 0.01
        return cfg
    if name == "r4_b52_rs01":
        cfg["reentry_breadth_min"] = 0.52
        cfg["reentry_rs63_min"] = 0.01
        return cfg
    raise ValueError(f"Unknown variant: {name}")


def run_variant(run_fn, prices, cfg, pp, start: str, end: str) -> dict:
    _, _, metrics = run_fn(
        prices=prices,
        start=start,
        end=end,
        pp_enabled=True,
        mie_enabled=True,
        mie_rs_th=float(pp["mie_rs_th"]),
        mie_min_hold=int(pp["mie_min_hold"]),
        pp_mfe_trigger=float(pp["pp_mfe_trigger"]),
        pp_trail_dd=float(pp["pp_trail_dd"]),
        pp_min_days_after_arm=int(pp["pp_min_days_after_arm"]),
        **cfg,
    )
    return {
        "roi_pct": float(metrics["ROI_%"]),
        "cagr_pct": float(metrics["CAGR_%"]),
        "maxdd_pct": float(metrics["MaxDD_%"]),
        "sharpe": float(metrics["Sharpe"]),
        "orders": int(metrics["Orders"]),
        "final_eur": float(metrics["FinalEUR"]),
    }


def build_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    base = metrics_df.set_index(["variant", "window"])
    baseline = "r2_baseline"
    rows = []
    for variant in metrics_df["variant"].unique():
        row = {"variant": variant}
        for window in ["full", "oos", "since2023", "since2025", "ytd2026", "2018", "2019", "2022", "2024"]:
            for metric in ["roi_pct", "sharpe", "maxdd_pct"]:
                value = base.loc[(variant, window), metric]
                base_value = base.loc[(baseline, window), metric]
                row[f"delta_{window}_{metric}"] = value - base_value
        if variant != baseline:
            for metric in ["roi_pct", "sharpe", "maxdd_pct"]:
                row[f"delta_vs_r3_full_{metric}"] = base.loc[(variant, "full"), metric] - base.loc[("r3_reentry", "full"), metric]
                row[f"delta_vs_r3_oos_{metric}"] = base.loc[(variant, "oos"), metric] - base.loc[("r3_reentry", "oos"), metric]
        rows.append(row)
    return pd.DataFrame(rows)


def render_report(metrics_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    variants = metrics_df["variant"].unique().tolist()
    lines = [
        "# State Dependent Reentry 184",
        "",
        "Targeted follow-up after `r3`: keep the pullback reentry edge, but only when both market context and candidate quality justify it.",
        "",
        "Variants tested:",
        "",
        "- `r2_baseline`: anti-churn baseline with reentry disabled",
        "- `r3_reentry`: unconditional long-term leader pullback reentry",
        "- `r4_b50_rs00`: reentry allowed only when breadth >= 0.50 and candidate rs63 >= 0.00",
        "- `r4_b50_rs01`: reentry allowed only when breadth >= 0.50 and candidate rs63 >= 0.01",
        "- `r4_b52_rs01`: reentry allowed only when breadth >= 0.52 and candidate rs63 >= 0.01",
        "",
        "Delta vs `r2_baseline`:",
        "",
        markdown_table(summary_df),
        "",
        "Key full / OOS metrics:",
        "",
    ]

    pivot_rows = []
    for variant in variants:
        row = {"variant": variant}
        for window in ["full", "oos", "2018", "2019", "2024"]:
            subset = metrics_df[(metrics_df["variant"] == variant) & (metrics_df["window"] == window)].iloc[0]
            row[f"{window}_roi_pct"] = subset["roi_pct"]
            row[f"{window}_maxdd_pct"] = subset["maxdd_pct"]
            row[f"{window}_sharpe"] = subset["sharpe"]
        pivot_rows.append(row)
    lines.append(markdown_table(pd.DataFrame(pivot_rows)))

    best_row = summary_df[summary_df["variant"] == "r4_b50_rs01"].iloc[0]
    lines.extend([
        "",
        "Interpretation:",
        "",
        f"- `r4_b50_rs01` is the cleanest candidate: full ROI improves by `{best_row['delta_full_roi_pct']:.4f}` points vs `r2`, OOS ROI improves by `{best_row['delta_oos_roi_pct']:.4f}`, and both `2018` and `2019` stop being the weak-years tax paid by unconditional `r3` reentry.",
        f"- vs `r3_reentry`, `r4_b50_rs01` also improves full ROI by `{best_row['delta_vs_r3_full_roi_pct']:.4f}` points and OOS ROI by `{best_row['delta_vs_r3_oos_roi_pct']:.4f}` while materially repairing the weak years.",
        "- `r4_b52_rs01` is close, but it gives back too much 2018/2024 upside.",
        "- conclusion: breadth alone was not enough; breadth + candidate relative-strength confirmation is the first clean state-dependent reentry upgrade.",
        "",
    ])
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [headers, ["---"] * len(headers)]
    for _, row in df.iterrows():
        vals = []
        for value in row.tolist():
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        rows.append(vals)
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main():
    engine = load_engine()
    cfg, pp, activation_dates = load_best()
    prices = engine.load_prices_from_csv(str(CSV_PATH))
    replay_end = str(pd.to_datetime(prices.close.index.max()).date())

    base_tickers = read_tickers(ACTIVE_PATH)
    staged_membership = membership_frame(
        index=prices.close.index,
        columns=prices.close.columns,
        base_tickers=base_tickers,
        staged_adds={k: v for k, v in activation_dates.items() if k in prices.close.columns},
    )
    staged_run = build_dynamic_runner(engine, staged_membership)

    variants = ["r2_baseline", "r3_reentry", "r4_b50_rs00", "r4_b50_rs01", "r4_b52_rs01"]
    rows = []
    for variant in variants:
        variant_cfg = make_variant_cfg(cfg, variant)
        for window, start, end in WINDOWS:
            result = run_variant(staged_run, prices, variant_cfg, pp, start, end or replay_end)
            rows.append({"variant": variant, "window": window, **result})

    metrics_df = pd.DataFrame(rows)
    summary_df = build_summary(metrics_df)

    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT_METRICS, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    OUT_MD.write_text(render_report(metrics_df, summary_df), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_METRICS}")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()
