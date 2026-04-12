from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\nicol\Downloads\214")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dynamic_universe_discovery as dud

EXPORTS_DIR = ROOT / "research" / "exports"
REPORTS_DIR = ROOT / "research" / "reports"
MISSING_UNIVERSE_PATH = EXPORTS_DIR / "scan_algo_missing_universe_20260406.csv"
MISSING_SELECTION_PATH = EXPORTS_DIR / "scan_algo_missing_selection_20260406.csv"

OUT_SUMMARY = EXPORTS_DIR / "scan_engine_hardening_summary_20260411.csv"
OUT_UNIQUE = EXPORTS_DIR / "scan_engine_hardening_unique_hits_20260411.csv"
OUT_FAMILY = EXPORTS_DIR / "scan_engine_hardening_family_coverage_20260411.csv"
OUT_REPORT = REPORTS_DIR / "SCAN_ENGINE_HARDENING_2026-04-11.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def markdown_table(df: pd.DataFrame, rows: int = 20) -> str:
    if df.empty:
        return "(none)"
    view = df.head(rows).copy()
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals: list[str] = []
        for value in row.tolist():
            if isinstance(value, float):
                vals.append(f"{value:.2f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def infer_sector_names(seed_tickers: list[str]) -> list[str]:
    sector_names: list[str] = []
    seen: set[str] = set()
    for ticker in seed_tickers:
        ctx = dud.ticker_context(ticker)
        sector_name = str(ctx.get("sector") or "").strip()
        if sector_name and sector_name not in seen:
            seen.add(sector_name)
            sector_names.append(sector_name)
    return sector_names or list(dud.DEFAULT_SECTOR_NAMES)


def run_profile(
    *,
    name: str,
    source_profile: str,
    seeds: list[str],
    keywords: list[str],
    sector_names: list[str],
) -> pd.DataFrame:
    outputs = dud.run_discovery(
        mode="broad",
        label=f"audit_{name}_20260411",
        source_profile=source_profile,
        seed_tickers=seeds,
        keywords=keywords,
        regions=list(dud.DEFAULT_REGIONS),
        sector_names=sector_names,
        min_market_cap=dud.DEFAULT_MIN_MARKET_CAP,
        min_price=dud.DEFAULT_MIN_PRICE,
        min_adv=dud.DEFAULT_MIN_ADV,
        screen_count=20,
        screen_pages=1,
        top_etfs_per_sector=1,
        etf_holding_count=4,
        max_single_backtests=0,
        max_combo_backtests=0,
        skip_backtest=True,
    )
    discovery = read_csv(outputs["discovery"])
    if discovery.empty:
        return discovery
    discovery = discovery.copy()
    discovery["ticker"] = discovery["ticker"].astype(str)
    return discovery.assign(profile=name, source_profile=source_profile)


def prepare_gap_bucket(path: Path, bucket: str) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame(columns=["ticker", "bucket", "theme_cluster"])
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str)
    out["bucket"] = bucket
    if "theme_cluster" not in out.columns:
        out["theme_cluster"] = "unknown"
    return out


def coverage_rows(profile_df: pd.DataFrame, gaps: pd.DataFrame) -> list[dict[str, object]]:
    if profile_df.empty:
        return []
    ranked = profile_df.sort_values(["priority_score", "source_count", "ticker"], ascending=[False, False, True])
    discovered = set(ranked["ticker"].astype(str))
    top50 = set(ranked["ticker"].astype(str).head(50))
    top100 = set(ranked["ticker"].astype(str).head(100))
    new_discovered = set(ranked.loc[ranked["candidate_status"].astype(str) == "new", "ticker"].astype(str))
    local_sourced = set(
        ranked.loc[
            ranked["source_types"].astype(str).str.contains("local_leader_sentinel|local_quality_sentinel|local_cluster_leader", na=False),
            "ticker",
        ].astype(str)
    )
    rows: list[dict[str, object]] = []
    for bucket, grp in gaps.groupby("bucket", dropna=False):
        gap_tickers = set(grp["ticker"].astype(str))
        rows.append(
            {
                "profile": str(profile_df["profile"].iloc[0]),
                "source_profile": str(profile_df["source_profile"].iloc[0]),
                "bucket": str(bucket),
                "gap_count": int(len(grp)),
                "discovered_hits": int(len(gap_tickers & discovered)),
                "discovered_hit_pct": 100.0 * len(gap_tickers & discovered) / max(1, len(grp)),
                "top50_hits": int(len(gap_tickers & top50)),
                "top100_hits": int(len(gap_tickers & top100)),
                "new_only_hits": int(len(gap_tickers & new_discovered)),
                "hybrid_local_source_hits": int(len(gap_tickers & local_sourced)),
            }
        )
    return rows


def family_coverage(profile_df: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    if profile_df.empty or gaps.empty:
        return pd.DataFrame()
    discovered = set(profile_df["ticker"].astype(str))
    rows: list[dict[str, object]] = []
    for (bucket, family), grp in gaps.groupby(["bucket", "theme_cluster"], dropna=False):
        tickers = set(grp["ticker"].astype(str))
        rows.append(
            {
                "profile": str(profile_df["profile"].iloc[0]),
                "bucket": str(bucket),
                "family": str(family),
                "count": int(len(grp)),
                "hits": int(len(tickers & discovered)),
                "hit_pct": 100.0 * len(tickers & discovered) / max(1, len(grp)),
            }
        )
    return pd.DataFrame(rows)


def build_unique_hits(profiles: dict[str, pd.DataFrame], gaps: pd.DataFrame) -> pd.DataFrame:
    gap_tickers = set(gaps["ticker"].astype(str))
    discovered_by_profile = {
        name: set(df["ticker"].astype(str)) & gap_tickers
        for name, df in profiles.items()
        if not df.empty
    }
    rows: list[dict[str, object]] = []
    all_names = sorted(discovered_by_profile)
    for name in all_names:
        others = set().union(*(discovered_by_profile[other] for other in all_names if other != name))
        unique_hits = sorted(discovered_by_profile[name] - others)
        for ticker in unique_hits:
            bucket = gaps.loc[gaps["ticker"].astype(str) == ticker, "bucket"].iloc[0]
            family = gaps.loc[gaps["ticker"].astype(str) == ticker, "theme_cluster"].iloc[0]
            rows.append(
                {
                    "profile": name,
                    "ticker": ticker,
                    "bucket": bucket,
                    "family": family,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dud.setup_yf_cache(dud.ROOT / ".yf_cache")
    _, _, cfg, _, prices = dud.load_setup()
    seeds = dud.latest_raw_leaders(prices, cfg, topn=6)
    sector_names = infer_sector_names(seeds)
    keywords = [
        "optical",
        "networking",
        "datacenter",
        "power equipment",
        "electrification",
        "aerospace defense",
        "industrial automation",
        "oil services",
    ]

    profiles = {
        "hybrid": run_profile(
            name="hybrid",
            source_profile="hybrid",
            seeds=seeds,
            keywords=keywords,
            sector_names=sector_names,
        ),
        "local_structural": run_profile(
            name="local_structural",
            source_profile="local_structural",
            seeds=seeds,
            keywords=keywords,
            sector_names=sector_names,
        ),
        "yahoo_external": run_profile(
            name="yahoo_external",
            source_profile="yahoo_external",
            seeds=seeds,
            keywords=keywords,
            sector_names=sector_names,
        ),
    }

    gaps = pd.concat(
        [
            prepare_gap_bucket(MISSING_UNIVERSE_PATH, "missing_universe"),
            prepare_gap_bucket(MISSING_SELECTION_PATH, "missing_selection"),
        ],
        ignore_index=True,
        sort=False,
    )

    summary_rows: list[dict[str, object]] = []
    family_frames: list[pd.DataFrame] = []
    for df in profiles.values():
        summary_rows.extend(coverage_rows(df, gaps))
        family_frames.append(family_coverage(df, gaps))

    summary = pd.DataFrame(summary_rows)
    family = pd.concat([df for df in family_frames if not df.empty], ignore_index=True, sort=False) if family_frames else pd.DataFrame()
    unique_hits = build_unique_hits(profiles, gaps)

    summary.to_csv(OUT_SUMMARY, index=False)
    unique_hits.to_csv(OUT_UNIQUE, index=False)
    family.to_csv(OUT_FAMILY, index=False)

    key_summary = summary[["profile", "bucket", "discovered_hits", "discovered_hit_pct", "top50_hits", "top100_hits", "new_only_hits"]].copy()
    if unique_hits.empty:
        unique_counts = pd.DataFrame(columns=["profile", "bucket", "unique_hits"])
    else:
        unique_counts = (
            unique_hits.groupby(["profile", "bucket"], dropna=False)["ticker"]
            .size()
            .rename("unique_hits")
            .reset_index()
            .sort_values(["profile", "bucket"])
        )
    family_focus = family.loc[
        family["family"].astype(str).isin(["semiconductors", "optical-networking", "industrial-compounders", "cloud-software"])
    ].sort_values(["bucket", "family", "profile"])
    hybrid_local = summary.loc[summary["profile"] == "hybrid", ["bucket", "hybrid_local_source_hits"]].copy()

    lines = [
        "# Scan Engine Hardening Audit",
        "",
        "Goal: compare three discovery stacks against our known historical compatible misses.",
        "",
        "Profiles tested:",
        "- `hybrid`: Yahoo expansion + local structural lanes;",
        "- `local_structural`: local OHLCV/context/taxonomy/market-structure lanes only;",
        "- `yahoo_external`: Yahoo discovery only, without local structural lanes.",
        "",
        f"Seeds: `{', '.join(seeds)}`",
        f"Sectors inferred: `{', '.join(sector_names)}`",
        "",
        "## Coverage Summary",
        "",
        markdown_table(key_summary, rows=10),
        "",
        "## Unique Gap Hits By Profile",
        "",
        markdown_table(unique_counts, rows=12),
        "",
        "## Family Coverage Focus",
        "",
        markdown_table(family_focus, rows=24),
        "",
        "## Hybrid Local-Sentinel Contribution",
        "",
        markdown_table(hybrid_local, rows=4),
        "",
        "## Conclusions",
        "",
        "- `hybrid` should remain the default scan architecture: it keeps the broadest recall.",
        "- `local_structural` is now a credible resilience layer instead of a weak fallback.",
        "- `yahoo_external` is not enough on its own if we want to reduce historical family misses with discipline.",
        "- The new local sentinel lanes matter only if they add coverage on historical compatible misses without forcing ticker-specific hindsight.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_UNIQUE}")
    print(f"Saved: {OUT_FAMILY}")
    print(f"Saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
