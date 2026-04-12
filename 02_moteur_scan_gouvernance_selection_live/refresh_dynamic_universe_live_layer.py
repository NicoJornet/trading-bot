from __future__ import annotations

import argparse
import time

import dynamic_universe_action_selector as selector
import dynamic_universe_history as duh
import dynamic_universe_standalone_removal_study as standalone_remove
import dynamic_universe_swap_study as swap_study
import refresh_dynamic_universe_database as db_refresh


def main() -> None:
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Refresh the full dynamic live layer: discovery DB, swap study, action selection.")
    parser.add_argument("--profiles", default="targeted_current,broad_focus,broad_diversified", help="Comma-separated cycle profiles to aggregate.")
    parser.add_argument("--keywords", default="", help="Optional comma-separated lookup keywords.")
    parser.add_argument("--skip-cycle", action="store_true", help="Reuse existing cycle outputs.")
    parser.add_argument("--skip-swap", action="store_true", help="Reuse existing swap study outputs.")
    args = parser.parse_args()

    profile_names = [x.strip() for x in args.profiles.split(",") if x.strip()]
    keywords = [x.strip() for x in args.keywords.split(",") if x.strip()]

    if args.skip_cycle:
        print("[live-layer] loading existing cycle outputs")
        paths = db_refresh.load_profiles(profile_names)
    else:
        print(f"[live-layer] refreshing profiles={profile_names}")
        paths = db_refresh.run_profiles(profile_names, keywords)

    print("[live-layer] merging profile frames")
    profile_frames = {
        profile_name: db_refresh.merge_profile_data(profile_name, profile_paths)
        for profile_name, profile_paths in paths.items()
    }
    db = db_refresh.aggregate_database(profile_frames)
    db_refresh.write_database_outputs(db)
    print(f"[live-layer] database refreshed rows={len(db)}")

    if not args.skip_swap:
        print("[live-layer] start swap study")
        swap_study.main()
        print("[live-layer] start standalone removal study")
        standalone_remove.main()

    print("[live-layer] start final action selector")
    selector.main([])
    history_outputs = duh.write_history_readiness_outputs()
    for name, path in history_outputs.items():
        print(f"[live-layer] history_{name}: {path}")
    print(f"[live-layer] completed elapsed_sec={time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
