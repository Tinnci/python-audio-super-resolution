from __future__ import annotations

import argparse
import platform
from pathlib import Path

from . import __version__
from .audiosr_backend import AUDIOSR_MODEL_NAMES
from .config import VALID_DEVICES, VALID_PRECISIONS, InferenceConfig, default_model_cache_dir
from .quality import format_quality_report, inspect_audio_quality
from .resolver import AudioSuperResolver, available_backends, plan_enhancements


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-super-res",
        description="Enhance audio to a target sample rate from the command line.",
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input audio file or directory.")
    parser.add_argument("output", nargs="?", type=Path, help="Output file or output directory.")
    parser.add_argument("--target-sr", type=int, default=48000, help="Target sample rate. Defaults to 48000.")
    parser.add_argument("--device", choices=VALID_DEVICES, default="cpu", help="Inference device. Defaults to cpu.")
    parser.add_argument(
        "--precision",
        choices=VALID_PRECISIONS,
        default="float32",
        help="Inference precision. Defaults to float32.",
    )
    parser.add_argument("--chunk-seconds", type=float, default=30.0, help="Chunk size for model backends.")
    parser.add_argument("--overlap-seconds", type=float, default=1.0, help="Chunk overlap for model backends.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic model backends.")
    parser.add_argument("--model-cache-dir", type=Path, help="Directory for model weights and metadata.")
    parser.add_argument(
        "--model-name",
        choices=AUDIOSR_MODEL_NAMES,
        default="basic",
        help="Model name for model-backed backends. Defaults to basic.",
    )
    parser.add_argument("--ddim-steps", type=int, default=50, help="DDIM sampling steps for diffusion backends.")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale for diffusion backends.")
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
    parser.add_argument("--config-info", action="store_true", help="Print resolved inference configuration and exit.")
    parser.add_argument("--prepare-model-cache", action="store_true", help="Create the model cache directory and exit.")
    parser.add_argument(
        "--quality-report", action="store_true", help="Print quality checks for each written output file."
    )
    parser.add_argument(
        "--fail-on-quality-issue",
        action="store_true",
        help="Return a non-zero exit code if quality checks find issues.",
    )
    parser.add_argument("--env-info", action="store_true", help="Print environment information and exit.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def print_env_info(config: InferenceConfig) -> None:
    print(f"audio-super-resolution: {__version__}")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"model_cache_dir: {config.model_cache_dir}")


def print_backends() -> None:
    for backend in available_backends():
        print(f"{backend.name}: {backend.description}")


def print_config(config: InferenceConfig) -> None:
    for key, value in config.as_dict().items():
        print(f"{key}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = _build_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.list_backends:
        print_backends()
        return 0

    if args.config_info:
        print_config(config)
        return 0

    if args.prepare_model_cache:
        print(config.ensure_model_cache_dir())
        return 0

    if args.env_info:
        print_env_info(config)
        return 0

    if args.input is None:
        parser.error("input is required unless an informational flag is used")

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

    resolver = AudioSuperResolver(target_sr=args.target_sr, backend=args.backend, config=config)
    try:
        results = [
            resolver.enhance(
                input_path=job.input_path,
                output_path=job.output_path,
                target_sr=args.target_sr,
            )
            for job in jobs
        ]
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    for result in results:
        print(
            f"Wrote {result.output_path} at {result.sample_rate} Hz "
            f"from {result.input_sample_rate} Hz using {result.backend}"
        )

    if args.quality_report or args.fail_on_quality_issue:
        reports = [
            inspect_audio_quality(
                result.output_path,
                expected_sample_rate=result.sample_rate,
                expected_duration_seconds=result.input_duration_seconds,
            )
            for result in results
        ]
        for report in reports:
            print(format_quality_report(report))

        if args.fail_on_quality_issue and any(not report.passed for report in reports):
            return 1

    return 0


def _backend_names() -> list[str]:
    return [backend.name for backend in available_backends()]


def _build_config(args: argparse.Namespace) -> InferenceConfig:
    model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else default_model_cache_dir()
    return InferenceConfig(
        device=args.device,
        precision=args.precision,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        seed=args.seed,
        model_cache_dir=model_cache_dir,
        model_name=args.model_name,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    raise SystemExit(main())
