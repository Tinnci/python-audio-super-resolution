from __future__ import annotations

import argparse
from pathlib import Path

from audio_super_resolution import AudioSuperResolver


def main() -> int:
    parser = argparse.ArgumentParser(description="Enhance a directory of audio files.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-sr", type=int, default=48000)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    resolver = AudioSuperResolver(target_sr=args.target_sr)
    results = resolver.enhance_many(args.input_dir, args.output_dir, recursive=args.recursive)

    for result in results:
        print(f"{result.input_path} -> {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
