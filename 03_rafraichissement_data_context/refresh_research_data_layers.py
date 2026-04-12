from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(args: list[str]) -> None:
    print(f"[data-layers] running: {' '.join(args)}")
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh all research data layers used by the dynamic universe stack.")
    parser.add_argument("--through", default=date.today().isoformat())
    parser.add_argument("--profiles", default="targeted_current,broad_focus,broad_diversified,local_resilience")
    parser.add_argument("--skip-dynamic-db", action="store_true")
    args = parser.parse_args()

    run_step(["refresh_ohlcv_to_today.py", "--csv", "apex_ohlcv_full_2015_2026.csv", "--through", args.through, "--lookback-days", "500"])
    run_step(["refresh_sector_benchmarks.py", "--through", args.through])
    run_step(["refresh_context_earnings_snapshots.py", "--max-refresh", "1200", "--passes", "2"])
    run_step(["refresh_market_structure_history.py"])
    run_step(["refresh_earnings_event_history.py", "--limit", "12"])
    run_step(["build_taxonomy_point_in_time.py"])
    run_step(["refresh_fx_reference_layers.py", "--through", args.through])
    run_step(["refresh_listing_corporate_metadata.py", "--max-refresh", "120"])
    run_step(["extend_ohlcv_with_dynamic_candidates.py", "--csv", "apex_ohlcv_full_2015_2026.csv", "--max-count", "60"])
    run_step(["enrich_ohlcv_feature_layers.py", "--csv", "apex_ohlcv_full_2015_2026.csv"])
    if not args.skip_dynamic_db:
        run_step(["refresh_dynamic_universe_database.py", "--profiles", args.profiles])
    run_step(["research/scripts/research_scoreboard_latest.py"])
    run_step(["research/scripts/winner_recall_dashboard_latest.py"])
    run_step(["research/scripts/engine_research_scoreboard_latest.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
