from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .config import (
    VALID_DEVICES,
    VALID_PRECISIONS,
    VALID_PREPROCESSING_MODES,
    InferenceConfig,
    default_model_cache_dir,
)
from .manifest import (
    build_manifest,
    compare_manifests,
    format_manifest_comparison,
    load_manifest,
    manifest_comparison_to_dict,
    write_manifest,
)
from .models import find_model_spec, list_models
from .quality import format_quality_report, inspect_audio_quality, write_quality_report_bundle
from .resolver import AudioSuperResolver, available_backends, plan_enhancements
from .weight_store import download_model_weights, verify_model_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-super-res",
        description="Enhance audio to a target sample rate from the command line.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=42),
    )

    parser.add_argument("input", nargs="?", type=Path, help="Input audio file or directory.")
    parser.add_argument("output", nargs="?", type=Path, help="Output file or output directory.")

    info_params = parser.add_argument_group("Info and Debugging")
    info_params.add_argument(
        "--list-backends", action="store_true", help="List available enhancement backends and exit."
    )
    info_params.add_argument("--list-models", action="store_true", help="List known enhancement models and exit.")
    info_params.add_argument("--list-filter", help="Filter model listings by id, backend, name, or description.")
    info_params.add_argument(
        "--list-format",
        choices=["pretty", "json"],
        default="pretty",
        help="Format for backend and model listings. Defaults to pretty.",
    )
    info_params.add_argument(
        "--config-info", action="store_true", help="Print resolved inference configuration and exit."
    )
    info_params.add_argument("--env-info", action="store_true", help="Print environment information and exit.")
    info_params.add_argument(
        "--compare-manifests",
        nargs=2,
        metavar=("EXPECTED", "ACTUAL"),
        type=Path,
        help="Compare two JSON run manifests and exit with 1 if differences are found.",
    )
    info_params.add_argument(
        "--compare-format",
        choices=["pretty", "json"],
        default="pretty",
        help="Format for manifest comparison output. Defaults to pretty.",
    )
    info_params.add_argument(
        "--duration-tolerance-seconds",
        type=float,
        default=0.05,
        help="Duration tolerance for manifest comparison. Defaults to 0.05.",
    )
    info_params.add_argument(
        "--check-output-files",
        action="store_true",
        help="During manifest comparison, also verify that actual output files exist.",
    )
    info_params.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    io_params = parser.add_argument_group("Enhancement I/O")
    io_params.add_argument("--target-sr", type=int, default=48000, help="Target sample rate. Defaults to 48000.")
    io_params.add_argument("--recursive", action="store_true", help="Scan input directories recursively.")
    io_params.add_argument(
        "--extension",
        action="append",
        dest="extensions",
        help="Audio extension to include when scanning directories. Can be repeated.",
    )
    io_params.add_argument("--suffix", default="-sr", help="Suffix used for generated output file names.")
    io_params.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned input/output paths without writing files.",
    )
    io_params.add_argument("--manifest", type=Path, help="Write a JSON manifest for planned or completed jobs.")

    backend_params = parser.add_argument_group("Backend and Inference")
    backend_params.add_argument(
        "--backend",
        default="sinc-resample",
        choices=_backend_names(),
        help="Enhancement backend to use.",
    )
    backend_params.add_argument(
        "--device", choices=VALID_DEVICES, default="cpu", help="Inference device. Defaults to cpu."
    )
    backend_params.add_argument(
        "--precision",
        choices=VALID_PRECISIONS,
        default="float32",
        help="Inference precision. Defaults to float32.",
    )
    backend_params.add_argument("--chunked", action="store_true", help="Process inputs in overlapping chunks.")
    backend_params.add_argument("--chunk-seconds", type=float, default=30.0, help="Chunk size for model backends.")
    backend_params.add_argument("--overlap-seconds", type=float, default=1.0, help="Chunk overlap for model backends.")
    backend_params.add_argument("--seed", type=int, default=0, help="Seed for deterministic model backends.")
    backend_params.add_argument("--model-cache-dir", type=Path, help="Directory for model weights and metadata.")
    backend_params.add_argument(
        "--prepare-model-cache", action="store_true", help="Create the model cache directory and exit."
    )
    backend_params.add_argument("--weights-manifest", type=Path, help="Path to a local weight manifest JSON file.")
    backend_params.add_argument(
        "--download-weights",
        action="store_true",
        help="Explicitly download model weights for the selected backend/model.",
    )
    backend_params.add_argument(
        "--force-download",
        action="store_true",
        help="Replace an existing verified model weight cache during explicit download.",
    )
    backend_params.add_argument("--weight-revision", help="Model weight revision for download providers.")
    backend_params.add_argument(
        "--verify-weights",
        action="store_true",
        help="Verify local weights for the selected backend/model and exit.",
    )
    backend_params.add_argument(
        "--model-name",
        default="basic",
        help="Model name for model-backed backends. Backend-specific validation is applied at runtime.",
    )
    backend_params.add_argument(
        "--ddim-steps", type=int, default=50, help="DDIM sampling steps for diffusion backends."
    )
    backend_params.add_argument(
        "--guidance-scale", type=float, default=3.5, help="Guidance scale for diffusion backends."
    )
    backend_params.add_argument(
        "--preprocess",
        choices=VALID_PREPROCESSING_MODES,
        default="none",
        help="Optional input preprocessing before enhancement. Defaults to none.",
    )
    backend_params.add_argument(
        "--lowpass-cutoff-hz",
        type=float,
        help="Low-pass cutoff for --preprocess lowpass. Defaults to min(16000, 45%% of input sample rate).",
    )
    backend_params.add_argument(
        "--lowpass-order",
        type=int,
        default=8,
        help="Butterworth filter order for --preprocess lowpass. Defaults to 8.",
    )
    backend_params.add_argument(
        "--denoise",
        action="store_true",
        help="Enable backend-specific denoising when supported.",
    )

    quality_params = parser.add_argument_group("Quality Checks")
    quality_params.add_argument(
        "--quality-report", action="store_true", help="Print quality checks for each written output file."
    )
    quality_params.add_argument(
        "--quality-report-json",
        type=Path,
        help="Write combined quality checks to a JSON file.",
    )
    quality_params.add_argument(
        "--fail-on-quality-issue",
        action="store_true",
        help="Return a non-zero exit code if quality checks find issues.",
    )
    return parser


def print_env_info(config: InferenceConfig) -> None:
    print(f"audio-super-resolution: {__version__}")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"model_cache_dir: {config.model_cache_dir}")


def print_backends(list_format: str = "pretty") -> None:
    backends = available_backends()
    if list_format == "json":
        print(json.dumps([asdict(backend) for backend in backends], indent=2))
        return

    name_width = max(len("Backend"), *(len(backend.name) for backend in backends))
    status_width = len("Installed")
    extra_width = max(len("Extra"), *(len(backend.package_extra or "-") for backend in backends))

    print(f"{'Backend':<{name_width}}  {'Installed':<{status_width}}  {'Extra':<{extra_width}}  Description")
    print("-" * (name_width + status_width + extra_width + 15 + 11))
    for backend in backends:
        installed = "yes" if backend.installed else "no"
        extra = backend.package_extra or "-"
        print(
            f"{backend.name:<{name_width}}  {installed:<{status_width}}  {extra:<{extra_width}}  {backend.description}"
        )


def print_models(filter_text: str | None = None, list_format: str = "pretty") -> None:
    models = list_models(filter_text=filter_text)
    if list_format == "json":
        print(json.dumps([asdict(model) for model in models], indent=2))
        return

    if not models:
        print("No models found.")
        return

    id_width = max(len("Model ID"), *(len(model.id) for model in models))
    backend_width = max(len("Backend"), *(len(model.backend) for model in models))
    implementation_width = max(len("Implementation"), *(len(model.implementation) for model in models))
    domain_width = max(len("Domain"), *(len(",".join(model.domain) or "-") for model in models))
    installed_width = len("Installed")
    target_width = len("Target SR")
    maturity_width = max(len("Maturity"), *(len(model.maturity) for model in models))

    print(
        f"{'Model ID':<{id_width}}  {'Backend':<{backend_width}}  "
        f"{'Implementation':<{implementation_width}}  {'Domain':<{domain_width}}  "
        f"{'Installed':<{installed_width}}  {'Target SR':<{target_width}}  "
        f"{'Maturity':<{maturity_width}}  Description"
    )
    print(
        "-"
        * (
            id_width
            + backend_width
            + implementation_width
            + domain_width
            + installed_width
            + target_width
            + maturity_width
            + 30
            + 11
        )
    )
    for model in models:
        installed = "yes" if model.installed else "no"
        target = "any" if model.target_sample_rate is None else str(model.target_sample_rate)
        domain = ",".join(model.domain) or "-"
        print(
            f"{model.id:<{id_width}}  {model.backend:<{backend_width}}  "
            f"{model.implementation:<{implementation_width}}  {domain:<{domain_width}}  "
            f"{installed:<{installed_width}}  {target:<{target_width}}  "
            f"{model.maturity:<{maturity_width}}  {model.description}"
        )


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
        print_backends(args.list_format)
        return 0

    if args.list_models:
        print_models(filter_text=args.list_filter, list_format=args.list_format)
        return 0

    if args.config_info:
        print_config(config)
        return 0

    if args.prepare_model_cache:
        cache_dir = config.ensure_model_cache_dir()
        if args.download_weights:
            try:
                model_spec = find_model_spec(args.backend, args.model_name)
                weight_dir = download_model_weights(
                    model_spec,
                    cache_dir=cache_dir,
                    revision=args.weight_revision,
                    force=args.force_download,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                parser.error(str(exc))
            print(weight_dir)
            return 0
        print(cache_dir)
        return 0

    if args.verify_weights:
        try:
            model_spec = find_model_spec(args.backend, args.model_name)
            resolved_weights = verify_model_weights(
                model_spec,
                cache_dir=config.model_cache_dir,
                manifest_path=config.weights_manifest,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Weight verification failed: {exc}", file=sys.stderr)
            return 1
        print(f"Verified weights {resolved_weights.manifest_path}")
        return 0

    if args.env_info:
        print_env_info(config)
        return 0

    if args.compare_manifests:
        try:
            expected_manifest = load_manifest(args.compare_manifests[0])
            actual_manifest = load_manifest(args.compare_manifests[1])
            comparison = compare_manifests(
                expected_manifest,
                actual_manifest,
                duration_tolerance_seconds=args.duration_tolerance_seconds,
                check_output_files=args.check_output_files,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))

        if args.compare_format == "json":
            print(json.dumps(manifest_comparison_to_dict(comparison), indent=2))
        else:
            print(format_manifest_comparison(comparison))
        return 0 if comparison.passed else 1

    if args.input is None:
        parser.error("input is required unless an informational flag is used")

    if args.target_sr <= 0:
        parser.error("--target-sr must be greater than zero")

    extensions = tuple(args.extensions) if args.extensions else None
    try:
        jobs = plan_enhancements(
            input_path=args.input,
            output_path=args.output,
            target_sr=args.target_sr,
            recursive=args.recursive,
            extensions=extensions,
            suffix=args.suffix,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if not jobs:
        parser.error("no supported audio files found")

    if args.dry_run:
        for job in jobs:
            print(f"{job.input_path} -> {job.output_path}")
        if args.manifest:
            manifest_path = write_manifest(
                args.manifest,
                build_manifest(
                    mode="dry-run",
                    jobs=jobs,
                    config=config,
                    backend=args.backend,
                    target_sample_rate=args.target_sr,
                ),
            )
            print(f"Wrote manifest {manifest_path}")
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

    reports = []
    if args.quality_report or args.quality_report_json or args.fail_on_quality_issue:
        reports = [
            inspect_audio_quality(
                result.output_path,
                expected_sample_rate=result.sample_rate,
                expected_duration_seconds=result.input_duration_seconds,
            )
            for result in results
        ]
        if args.quality_report:
            for report in reports:
                print(format_quality_report(report))

        if args.quality_report_json:
            quality_report_path = write_quality_report_bundle(args.quality_report_json, reports)
            print(f"Wrote quality report {quality_report_path}")

    if args.manifest:
        manifest_path = write_manifest(
            args.manifest,
            build_manifest(
                mode="completed",
                jobs=jobs,
                config=config,
                backend=args.backend,
                target_sample_rate=args.target_sr,
                results=results,
                quality_reports=reports,
            ),
        )
        print(f"Wrote manifest {manifest_path}")

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
        chunked=args.chunked,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        seed=args.seed,
        model_cache_dir=model_cache_dir,
        model_name=args.model_name,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        preprocess=args.preprocess,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        lowpass_order=args.lowpass_order,
        weights_manifest=args.weights_manifest,
        download_weights=args.download_weights,
        force_download=args.force_download,
        weight_revision=args.weight_revision,
        denoise=args.denoise,
    )


if __name__ == "__main__":
    raise SystemExit(main())
