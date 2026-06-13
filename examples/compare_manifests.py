from __future__ import annotations

import argparse
from pathlib import Path

from audio_super_resolution import compare_manifests, format_manifest_comparison, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two audio super-resolution run manifests.")
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--duration-tolerance-seconds", type=float, default=0.05)
    parser.add_argument("--check-output-files", action="store_true")
    args = parser.parse_args()

    comparison = compare_manifests(
        load_manifest(args.expected),
        load_manifest(args.actual),
        duration_tolerance_seconds=args.duration_tolerance_seconds,
        check_output_files=args.check_output_files,
    )
    print(format_manifest_comparison(comparison))
    return 0 if comparison.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
