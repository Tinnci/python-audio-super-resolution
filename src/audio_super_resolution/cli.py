from __future__ import annotations

import argparse
import platform
from pathlib import Path

from . import __version__
from .resolver import AudioSuperResolver, available_backends, plan_enhancements


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-super-res",
        description="Enhance audio to a target sample rate from the command line.",
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input audio file or directory.")
    parser.add_argument("output", nargs="?", type=Path, help="Output file or output directory.")
    parser.add_argument("--target-sr", type=int, default=48000, help="Target sample rate. Defaults to 48000.")
    parser.add_argument(
        "--backend",
        default="sinc-resample",
        choices=_backend_names(),
        help="Enhancement backend to use.",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan input directories recursively.")
    parser.add_argument(
        "--extension",
        action="append",
        dest="extensions",
        help="Audio extension to include when scanning directories. Can be repeated.",
    )
    parser.add_argument("--suffix", default="-sr", help="Suffix used for generated output file names.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned input/output paths without writing files.",
    )
    parser.add_argument("--list-backends", action="store_true", help="List available enhancement backends and exit.")
    parser.add_argument("--env-info", action="store_true", help="Print environment information and exit.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def print_env_info() -> None:
    print(f"audio-super-resolution: {__version__}")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")


def print_backends() -> None:
    for backend in available_backends():
        print(f"{backend.name}: {backend.description}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_backends:
        print_backends()
        return 0

    if args.env_info:
        print_env_info()
        return 0

    if args.input is None:
        parser.error("input is required unless --env-info or --list-backends is used")

    if args.target_sr <= 0:
        parser.error("--target-sr must be greater than zero")

    extensions = tuple(args.extensions) if args.extensions else None
    jobs = plan_enhancements(
        input_path=args.input,
        output_path=args.output,
        target_sr=args.target_sr,
        recursive=args.recursive,
        extensions=extensions,
        suffix=args.suffix,
    )

    if not jobs:
        parser.error("no supported audio files found")

    if args.dry_run:
        for job in jobs:
            print(f"{job.input_path} -> {job.output_path}")
        return 0

    resolver = AudioSuperResolver(target_sr=args.target_sr, backend=args.backend)
    results = [
        resolver.enhance(
            input_path=job.input_path,
            output_path=job.output_path,
            target_sr=args.target_sr,
        )
        for job in jobs
    ]

    for result in results:
        print(
            f"Wrote {result.output_path} at {result.sample_rate} Hz "
            f"from {result.input_sample_rate} Hz using {result.backend}"
        )
    return 0


def _backend_names() -> list[str]:
    return [backend.name for backend in available_backends()]


if __name__ == "__main__":
    raise SystemExit(main())
