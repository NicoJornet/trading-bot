from __future__ import annotations

import argparse

import dynamic_universe_action_selector as selector
import dynamic_universe_standalone_removal_study as standalone_remove
import dynamic_universe_swap_study as swap_study
import refresh_dynamic_universe_database as db_refresh


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the full dynamic live layer: discovery DB, swap study, action selection.")
    parser.add_argument("--profiles", default="targeted_current,broad_focus,broad_diversified", help="Comma-separated cycle profiles to aggregate.")
    parser.add_argument("--keywords", default="", help="Optional comma-separated lookup keywords.")
    parser.add_argument("--skip-cycle", action="store_true", help="Reuse existing cycle outputs.")
    parser.add_argument("--skip-swap", action="store_true", help="Reuse existing swap study outputs.")
    args = parser.parse_args()

    profile_names = [x.strip() for x in args.profiles.split(",") if x.strip()]
    keywords = [x.strip() for x in args.keywords.split(",") if x.strip()]

    if args.skip_cycle:
        paths = db_refresh.load_profiles(profile_names)
    else:
        paths = db_refresh.run_profiles(profile_names, keywords)

    profile_frames = {
        profile_name: db_refresh.merge_profile_data(profile_name, profile_paths)
        for profile_name, profile_paths in paths.items()
    }
    db = db_refresh.aggregate_database(profile_frames)
    db_refresh.write_database_outputs(db)

    if not args.skip_swap:
        swap_study.main()
        standalone_remove.main()

    selector.main([])


if __name__ == "__main__":
    main()
