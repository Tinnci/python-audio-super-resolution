from __future__ import annotations

import argparse
from pathlib import Path

from audio_super_resolution import AudioSuperResolver, InferenceConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Enhance one audio file.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-sr", type=int, default=48000)
    args = parser.parse_args()

    config = InferenceConfig(device="cpu", precision="float32")
    resolver = AudioSuperResolver(target_sr=args.target_sr, config=config)
    result = resolver.enhance(args.input, args.output)

    print(f"{result.input_path} -> {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
