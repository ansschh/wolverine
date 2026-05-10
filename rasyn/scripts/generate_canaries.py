"""Generate canaries for the current sealed-case registry and write to YAML.

Run from the rasyn/ project root:
    python scripts/generate_canaries.py [--per-layer 4]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rasyn.data.registry.canary_generator import generate_canaries_for_registry, write_canaries_yaml
from rasyn.data.registry.loader import load_sealed_case_registry

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "rasyn/data/registry/canaries.yaml"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-layer", type=int, default=4)
    p.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = p.parse_args(argv)
    reg = load_sealed_case_registry()
    canaries = generate_canaries_for_registry(reg, per_layer=args.per_layer)
    write_canaries_yaml(canaries, args.output)
    print(f"Wrote {len(canaries)} canaries to {args.output}")


if __name__ == "__main__":
    main()
