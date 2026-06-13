from __future__ import annotations

import argparse
from pathlib import Path

from audio_super_resolution import format_quality_report, inspect_audio_quality


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect basic audio quality properties.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--expected-sr", type=int)
    parser.add_argument("--expected-duration", type=float)
    args = parser.parse_args()

    report = inspect_audio_quality(
        args.audio,
        expected_sample_rate=args.expected_sr,
        expected_duration_seconds=args.expected_duration,
    )
    print(format_quality_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
