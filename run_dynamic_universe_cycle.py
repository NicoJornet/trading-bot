from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import dynamic_universe_discovery as dud


@dataclass(frozen=True)
class CycleProfile:
    name: str
    mode: str
    label: str
    seed_count: int
    keywords: tuple[str, ...]
    regions: tuple[str, ...]
    screen_count: int
    screen_pages: int
    top_etfs_per_sector: int
    etf_holding_count: int
    max_single_backtests: int
    max_combo_backtests: int
    sector_names: tuple[str, ...] = ()
    min_market_cap: float = dud.DEFAULT_MIN_MARKET_CAP
    min_price: float = dud.DEFAULT_MIN_PRICE
    min_adv: float = dud.DEFAULT_MIN_ADV


PROFILES: dict[str, CycleProfile] = {
    "targeted_current": CycleProfile(
        name="targeted_current",
        mode="targeted",
        label="cycle_targeted_current",
        seed_count=4,
        keywords=(),
        regions=(),
        screen_count=0,
        screen_pages=0,
        top_etfs_per_sector=1,
        etf_holding_count=4,
        max_single_backtests=4,
        max_combo_backtests=2,
    ),
    "broad_focus": CycleProfile(
        name="broad_focus",
        mode="broad",
        label="cycle_broad_focus",
        seed_count=6,
        keywords=(),
        regions=("us", "de", "fr", "gb"),
        screen_count=0,
        screen_pages=0,
        top_etfs_per_sector=1,
        etf_holding_count=4,
        max_single_backtests=6,
        max_combo_backtests=3,
    ),
    "broad_diversified": CycleProfile(
        name="broad_diversified",
        mode="broad",
        label="cycle_broad_diversified",
        seed_count=8,
        keywords=(
            "optical",
            "networking",
            "datacenter",
            "power equipment",
            "electrification",
            "aerospace defense",
            "uranium",
            "industrial automation",
        ),
        regions=dud.DEFAULT_REGIONS,
        screen_count=25,
        screen_pages=2,
        top_etfs_per_sector=2,
        etf_holding_count=6,
        max_single_backtests=8,
        max_combo_backtests=4,
        sector_names=(
            "Technology",
            "Industrials",
            "Communication Services",
            "Utilities",
            "Energy",
            "Basic Materials",
            "Healthcare",
            "Financial Services",
        ),
    ),
}


def infer_sector_names(seed_tickers: Sequence[str]) -> list[str]:
    sector_names: list[str] = []
    seen: set[str] = set()
    for ticker in seed_tickers:
        ctx = dud.ticker_context(ticker)
        sector_name = str(ctx.get("sector") or "").strip()
        if sector_name and sector_name not in seen:
            seen.add(sector_name)
            sector_names.append(sector_name)
    return sector_names or list(dud.DEFAULT_SECTOR_NAMES)


def run_profile(profile: CycleProfile, extra_keywords: Sequence[str]) -> dict[str, str]:
    dud.setup_yf_cache(dud.ROOT / ".yf_cache")
    _, _, cfg, _, prices = dud.load_setup()
    seeds = dud.latest_raw_leaders(prices, cfg, topn=profile.seed_count)
    if profile.sector_names:
        sector_names = list(profile.sector_names)
    elif profile.mode == "broad":
        sector_names = infer_sector_names(seeds)
    else:
        sector_names = list(dud.DEFAULT_SECTOR_NAMES)
    outputs = dud.run_discovery(
        mode=profile.mode,
        label=profile.label,
        seed_tickers=seeds,
        keywords=tuple(profile.keywords) + tuple(extra_keywords),
        regions=profile.regions,
        sector_names=sector_names,
        min_market_cap=profile.min_market_cap,
        min_price=profile.min_price,
        min_adv=profile.min_adv,
        screen_count=profile.screen_count,
        screen_pages=profile.screen_pages,
        top_etfs_per_sector=profile.top_etfs_per_sector,
        etf_holding_count=profile.etf_holding_count,
        max_single_backtests=profile.max_single_backtests,
        max_combo_backtests=profile.max_combo_backtests,
        skip_backtest=False,
    )
    print(f"profile: {profile.name}")
    print("seeds:", ", ".join(seeds))
    if profile.mode == "broad":
        print("sector_names:", ", ".join(sector_names))
    return {k: str(v) for k, v in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ready-made dynamic universe research cycle.")
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="targeted_current")
    parser.add_argument("--keywords", default="", help="Optional comma-separated lookup keywords to append to the profile.")
    args = parser.parse_args()

    extra_keywords = [x.strip() for x in args.keywords.split(",") if x.strip()]
    outputs = run_profile(PROFILES[args.profile], extra_keywords)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
