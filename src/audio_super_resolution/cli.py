from __future__ import annotations

import argparse
import platform
from pathlib import Path

from . import __version__
from .resolver import AudioSuperResolver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-super-res",
        description="Enhance audio to a target sample rate from the command line.",
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input audio file.")
    parser.add_argument("output", nargs="?", type=Path, help="Output audio file.")
    parser.add_argument("--target-sr", type=int, default=48000, help="Target sample rate. Defaults to 48000.")
    parser.add_argument("--env-info", action="store_true", help="Print environment information and exit.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def print_env_info() -> None:
    print(f"audio-super-resolution: {__version__}")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.env_info:
        print_env_info()
        return 0

    if args.input is None or args.output is None:
        parser.error("input and output are required unless --env-info is used")

    resolver = AudioSuperResolver(target_sr=args.target_sr)
    result = resolver.enhance(args.input, args.output)
    print(f"Wrote {result.output_path} at {result.sample_rate} Hz using {result.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
