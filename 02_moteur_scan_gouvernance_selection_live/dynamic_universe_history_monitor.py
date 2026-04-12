from __future__ import annotations

import dynamic_universe_history as duh


def main() -> None:
    outputs = duh.write_history_readiness_outputs()
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
