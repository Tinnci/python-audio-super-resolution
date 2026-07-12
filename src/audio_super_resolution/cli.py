from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .benchmark import build_benchmark_report, write_benchmark_report
from .config import (
    VALID_DEVICES,
    VALID_PRECISIONS,
    VALID_PREPROCESSING_MODES,
    VALID_RUNTIME_PROVIDERS,
    InferenceConfig,
    default_model_cache_dir,
)
from .evaluation import (
    OPTIONAL_FULL_REFERENCE_METRICS,
    SUPPORTED_DEGRADERS,
    SUPPORTED_DOWNSTREAM_EVALUATORS,
    SUPPORTED_LISTENING_PROTOCOLS,
    SUPPORTED_NO_REFERENCE_EVALUATORS,
    bundle_eval_artifacts,
    compare_eval_manifests,
    compare_eval_matrices,
    init_failure_case_evalset,
    init_speech_bwe_evalset,
    inspect_torch_checkpoint,
    load_eval_manifest,
    load_threshold_policy,
    run_downstream_eval,
    run_eval_dataset,
    run_eval_matrix,
    run_listening_export,
    run_no_reference_eval,
    validate_eval_dataset_manifest,
    write_eval_manifest,
    write_eval_report,
)
from .manifest import (
    build_manifest,
    compare_manifests,
    format_manifest_comparison,
    load_manifest,
    manifest_comparison_to_dict,
    write_manifest,
)
from .model_weights import download_model_weights, verify_model_weights
from .models import find_model_spec, list_models
from .quality import format_quality_report, inspect_audio_quality, write_quality_report_bundle
from .resolver import AudioSuperResolver, EnhancementResult, PlannedEnhancement, available_backends, plan_enhancements


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
        "--runtime-provider",
        choices=VALID_RUNTIME_PROVIDERS,
        default="auto",
        help="Runtime provider selection. Defaults to auto.",
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
        help="Model name for model-backed backends. Omit for backend defaults or single-model backends.",
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
        help="Low-pass transition steepness for --preprocess lowpass. Defaults to 8.",
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
        "--benchmark-json",
        type=Path,
        help="Write a machine-readable runtime benchmark summary for the enhancement run.",
    )
    quality_params.add_argument(
        "--fail-on-quality-issue",
        action="store_true",
        help="Return a non-zero exit code if quality checks find issues.",
    )
    return parser


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-super-res eval",
        description="Run and compare backend evaluation manifests.",
    )
    subparsers = parser.add_subparsers(dest="eval_command", required=True)

    init_parser = subparsers.add_parser("init-speech-bwe", help="Create a tiny deterministic speech BWE evalset.")
    init_parser.add_argument("--output-dir", type=Path, required=True, help="Output evalset directory.")
    init_parser.add_argument("--count", type=int, default=20, help="Number of clean reference WAV files.")
    init_parser.add_argument("--sample-rate", type=int, default=48000, help="Reference sample rate. Defaults to 48000.")
    init_parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.35,
        help="Duration of each generated fixture. Defaults to 0.35.",
    )
    init_parser.add_argument("--force", action="store_true", help="Allow writing into a non-empty output directory.")

    failure_parser = subparsers.add_parser(
        "init-failure-cases",
        help="Create deterministic stability/failure-case eval fixtures.",
    )
    failure_parser.add_argument("--output-dir", type=Path, required=True, help="Output evalset directory.")
    failure_parser.add_argument("--sample-rate", type=int, default=48000, help="Reference sample rate.")
    failure_parser.add_argument("--force", action="store_true", help="Allow writing into a non-empty output directory.")

    validate_dataset_parser = subparsers.add_parser("validate-dataset", help="Validate an eval dataset manifest.")
    validate_dataset_parser.add_argument("--manifest", type=Path, required=True, help="Dataset manifest JSON path.")
    validate_dataset_parser.add_argument("--output", type=Path, help="Optional validation JSON output path.")

    matrix_parser = subparsers.add_parser("matrix", help="Run a backend/degrader evaluation matrix.")
    matrix_parser.add_argument("--dataset", type=Path, required=True, help="Directory of clean reference WAV files.")
    matrix_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for matrix artifacts.")
    matrix_parser.add_argument(
        "--backend",
        action="append",
        choices=_backend_names(),
        default=[],
        help="Backend to evaluate. Can be repeated. Defaults to sinc-resample.",
    )
    matrix_parser.add_argument(
        "--degrader",
        action="append",
        choices=SUPPORTED_DEGRADERS,
        default=[],
        help="Controlled degradation recipe. Can be repeated. Defaults to wideband_16k.",
    )
    matrix_parser.add_argument("--target-sr", type=int, default=48000, help="Target sample rate. Defaults to 48000.")
    matrix_parser.add_argument("--limit", type=int, help="Limit the number of reference files for smoke runs.")
    matrix_parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing run manifests in the output directory when present.",
    )
    matrix_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the matrix on the first combination-level failure.",
    )
    matrix_parser.add_argument(
        "--optional-metric",
        action="append",
        choices=OPTIONAL_FULL_REFERENCE_METRICS,
        default=[],
        help="Explicit optional full-reference metric to attempt for each matrix run.",
    )
    matrix_parser.add_argument(
        "--device", choices=VALID_DEVICES, default="cpu", help="Inference device. Defaults to cpu."
    )
    matrix_parser.add_argument(
        "--runtime-provider",
        choices=VALID_RUNTIME_PROVIDERS,
        default="auto",
        help="Runtime provider selection. Defaults to auto.",
    )
    matrix_parser.add_argument("--model-cache-dir", type=Path, help="Directory for model weights and metadata.")

    run_parser = subparsers.add_parser("run", help="Run an evaluation dataset.")
    run_parser.add_argument("--dataset", type=Path, required=True, help="Directory of clean reference WAV files.")
    run_parser.add_argument("--backend", default="sinc-resample", choices=_backend_names(), help="Backend to evaluate.")
    run_parser.add_argument("--output", type=Path, required=True, help="Output eval manifest JSON path.")
    run_parser.add_argument("--work-dir", type=Path, required=True, help="Directory for degraded and enhanced files.")
    run_parser.add_argument("--target-sr", type=int, default=48000, help="Target sample rate. Defaults to 48000.")
    run_parser.add_argument(
        "--degrader",
        choices=SUPPORTED_DEGRADERS,
        default="wideband_16k",
        help="Controlled degradation recipe. Defaults to wideband_16k.",
    )
    run_parser.add_argument("--limit", type=int, help="Limit the number of reference files for smoke runs.")
    run_parser.add_argument(
        "--optional-metric",
        action="append",
        choices=OPTIONAL_FULL_REFERENCE_METRICS,
        default=[],
        help="Explicit optional full-reference metric to attempt. Missing dependencies are recorded as skipped.",
    )
    run_parser.add_argument("--device", choices=VALID_DEVICES, default="cpu", help="Inference device. Defaults to cpu.")
    run_parser.add_argument(
        "--runtime-provider",
        choices=VALID_RUNTIME_PROVIDERS,
        default="auto",
        help="Runtime provider selection. Defaults to auto.",
    )
    run_parser.add_argument(
        "--model-cache-dir",
        type=Path,
        help="Directory for model weights and metadata.",
    )

    no_reference_parser = subparsers.add_parser("no-reference", help="Run no-reference screening metrics.")
    no_reference_parser.add_argument("--input", type=Path, required=True, help="Audio file or directory to evaluate.")
    no_reference_parser.add_argument("--output", type=Path, required=True, help="Output eval manifest JSON path.")
    no_reference_parser.add_argument("--recursive", action="store_true", help="Scan input directories recursively.")
    no_reference_parser.add_argument("--limit", type=int, help="Limit the number of audio files for smoke runs.")
    no_reference_parser.add_argument(
        "--evaluator",
        choices=SUPPORTED_NO_REFERENCE_EVALUATORS,
        default="signal-stats",
        help="No-reference evaluator. Heavy evaluators are documented but gated. Defaults to signal-stats.",
    )

    downstream_parser = subparsers.add_parser("downstream", help="Run downstream task evaluation.")
    downstream_parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSON transcript dataset for downstream evaluation.",
    )
    downstream_parser.add_argument("--output", type=Path, required=True, help="Output eval manifest JSON path.")
    downstream_parser.add_argument("--dataset-id", help="Stable dataset identity to write into records.")
    downstream_parser.add_argument("--limit", type=int, help="Limit the number of records for smoke runs.")
    downstream_parser.add_argument(
        "--evaluator",
        choices=SUPPORTED_DOWNSTREAM_EVALUATORS,
        default="transcript-error-rate",
        help="Downstream evaluator. Heavy evaluators are documented but gated. Defaults to transcript-error-rate.",
    )

    listening_parser = subparsers.add_parser("listening-export", help="Export a blind listening-test bundle.")
    listening_parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        required=True,
        help="Eval manifest JSON to include. Repeat for multiple backend runs.",
    )
    listening_parser.add_argument("--output-dir", type=Path, required=True, help="Output bundle directory.")
    listening_parser.add_argument(
        "--protocol",
        choices=SUPPORTED_LISTENING_PROTOCOLS,
        default="mushra",
        help="Listening protocol metadata to write. Defaults to mushra.",
    )
    listening_parser.add_argument("--seed", type=int, default=0, help="Deterministic blind ordering seed.")

    compare_parser = subparsers.add_parser("compare", help="Compare two evaluation manifests.")
    compare_parser.add_argument("baseline", type=Path, help="Baseline eval manifest JSON path.")
    compare_parser.add_argument("candidate", type=Path, help="Candidate eval manifest JSON path.")
    compare_parser.add_argument("--output", type=Path, help="Write comparison JSON to this path.")
    compare_parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="FIELD=MAX_REGRESSION",
        help=(
            "Fail if FIELD regresses by more than MAX_REGRESSION. Can be repeated. "
            "Metric direction is inferred, so SI-SDR drops fail while LSD/RTF increases fail."
        ),
    )
    compare_parser.add_argument("--threshold-policy", type=Path, help="JSON threshold policy file.")

    matrix_compare_parser = subparsers.add_parser("matrix-compare", help="Compare two evaluation matrix manifests.")
    matrix_compare_parser.add_argument("baseline", type=Path, help="Baseline matrix JSON path.")
    matrix_compare_parser.add_argument("candidate", type=Path, help="Candidate matrix JSON path.")
    matrix_compare_parser.add_argument("--output", type=Path, help="Write matrix comparison JSON to this path.")
    matrix_compare_parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="FIELD=MAX_REGRESSION",
        help="Fail if FIELD regresses by more than MAX_REGRESSION. Can be repeated.",
    )
    matrix_compare_parser.add_argument("--threshold-policy", type=Path, help="JSON threshold policy file.")

    report_parser = subparsers.add_parser("report", help="Render a Markdown report for an eval artifact.")
    report_parser.add_argument("--manifest", type=Path, required=True, help="Eval artifact JSON path.")
    report_parser.add_argument("--output", type=Path, required=True, help="Output Markdown report path.")

    bundle_parser = subparsers.add_parser("bundle", help="Bundle eval manifests and referenced run manifests.")
    bundle_parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        required=True,
        help="Eval manifest JSON to include. Repeat for multiple artifacts.",
    )
    bundle_parser.add_argument("--output-dir", type=Path, required=True, help="Output bundle directory.")
    bundle_parser.add_argument("--archive", type=Path, help="Optional .tar.gz archive path.")

    checkpoint_parser = subparsers.add_parser(
        "inspect-checkpoint",
        help="Inspect tensor keys/shapes from an explicit local torch checkpoint.",
    )
    checkpoint_parser.add_argument("--checkpoint", type=Path, required=True, help="Local torch checkpoint path.")
    checkpoint_parser.add_argument("--output", type=Path, required=True, help="Output checkpoint inspection JSON path.")
    checkpoint_parser.add_argument("--limit", type=int, help="Limit reported tensor records.")
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
    accelerator_width = max(len("Accelerators"), *(len(",".join(backend.accelerators) or "-") for backend in backends))
    provider_width = max(
        len("Runtime Providers"), *(len(",".join(backend.runtime_providers) or "-") for backend in backends)
    )

    print(
        f"{'Backend':<{name_width}}  {'Installed':<{status_width}}  {'Extra':<{extra_width}}  "
        f"{'Accelerators':<{accelerator_width}}  {'Runtime Providers':<{provider_width}}  Description"
    )
    print("-" * (name_width + status_width + extra_width + accelerator_width + provider_width + 23 + 11))
    for backend in backends:
        installed = "yes" if backend.installed else "no"
        extra = backend.package_extra or "-"
        accelerators = ",".join(backend.accelerators) or "-"
        providers = ",".join(backend.runtime_providers) or "-"
        print(
            f"{backend.name:<{name_width}}  {installed:<{status_width}}  {extra:<{extra_width}}  "
            f"{accelerators:<{accelerator_width}}  {providers:<{provider_width}}  {backend.description}"
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
    accelerator_width = max(len("Accelerators"), *(len(",".join(model.accelerators) or "-") for model in models))
    provider_width = max(len("Runtime Providers"), *(len(",".join(model.runtime_providers) or "-") for model in models))
    maturity_width = max(len("Maturity"), *(len(model.maturity) for model in models))

    print(
        f"{'Model ID':<{id_width}}  {'Backend':<{backend_width}}  "
        f"{'Implementation':<{implementation_width}}  {'Domain':<{domain_width}}  "
        f"{'Installed':<{installed_width}}  {'Target SR':<{target_width}}  "
        f"{'Accelerators':<{accelerator_width}}  {'Runtime Providers':<{provider_width}}  "
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
            + accelerator_width
            + provider_width
            + maturity_width
            + 34
            + 11
        )
    )
    for model in models:
        installed = "yes" if model.installed else "no"
        target = "any" if model.target_sample_rate is None else str(model.target_sample_rate)
        domain = ",".join(model.domain) or "-"
        accelerators = ",".join(model.accelerators) or "-"
        providers = ",".join(model.runtime_providers) or "-"
        print(
            f"{model.id:<{id_width}}  {model.backend:<{backend_width}}  "
            f"{model.implementation:<{implementation_width}}  {domain:<{domain_width}}  "
            f"{installed:<{installed_width}}  {target:<{target_width}}  "
            f"{accelerators:<{accelerator_width}}  {providers:<{provider_width}}  "
            f"{model.maturity:<{maturity_width}}  {model.description}"
        )


def print_config(config: InferenceConfig) -> None:
    for key, value in config.as_dict().items():
        print(f"{key}: {value}")


def main(argv: list[str] | None = None) -> int:
    args_list = sys.argv[1:] if argv is None else argv
    if args_list and args_list[0] == "eval":
        return _run_eval_command(args_list[1:])

    parser = build_parser()
    args = parser.parse_args(args_list)

    try:
        config = _build_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    return _run_command(args, parser, config)


def _run_eval_command(argv: list[str]) -> int:
    parser = build_eval_parser()
    args = parser.parse_args(argv)
    try:
        if args.eval_command == "init-speech-bwe":
            manifest = init_speech_bwe_evalset(
                output_dir=args.output_dir,
                count=args.count,
                sample_rate=args.sample_rate,
                duration_seconds=args.duration_seconds,
                force=args.force,
            )
            print(f"Wrote speech BWE evalset {args.output_dir} ({manifest['record_count']} reference file(s))")
            return 0
        if args.eval_command == "init-failure-cases":
            manifest = init_failure_case_evalset(
                output_dir=args.output_dir,
                sample_rate=args.sample_rate,
                force=args.force,
            )
            print(f"Wrote failure-case evalset {args.output_dir} ({manifest['record_count']} reference file(s))")
            return 0
        if args.eval_command == "validate-dataset":
            validation = validate_eval_dataset_manifest(args.manifest)
            if args.output:
                write_eval_manifest(args.output, validation)
                print(f"Wrote dataset validation {args.output}")
            else:
                print(json.dumps(validation, indent=2))
            return 0 if validation["passed"] else 1
        if args.eval_command == "matrix":
            model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else default_model_cache_dir()
            manifest = run_eval_matrix(
                dataset_dir=args.dataset,
                output_dir=args.output_dir,
                backends=tuple(args.backend or ["sinc-resample"]),
                degraders=tuple(args.degrader or ["wideband_16k"]),
                target_sample_rate=args.target_sr,
                limit=args.limit,
                optional_metrics=tuple(args.optional_metric),
                reuse_existing=args.reuse_existing,
                fail_fast=args.fail_fast,
                config=InferenceConfig(
                    device=args.device,
                    runtime_provider=args.runtime_provider,
                    model_cache_dir=model_cache_dir,
                ),
            )
            print(f"Wrote eval matrix {args.output_dir / 'matrix.json'} ({manifest['run_count']} run(s))")
            return 0 if manifest["passed"] else 1
        if args.eval_command == "run":
            model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else default_model_cache_dir()
            manifest = run_eval_dataset(
                dataset_dir=args.dataset,
                backend=args.backend,
                output_path=args.output,
                work_dir=args.work_dir,
                target_sample_rate=args.target_sr,
                degrader=args.degrader,
                limit=args.limit,
                optional_metrics=tuple(args.optional_metric),
                config=InferenceConfig(
                    device=args.device,
                    runtime_provider=args.runtime_provider,
                    model_cache_dir=model_cache_dir,
                ),
            )
            results = manifest.get("results")
            result_count = len(results) if isinstance(results, list) else 0
            print(f"Wrote eval manifest {args.output} ({result_count} result(s))")
            return 0
        if args.eval_command == "no-reference":
            manifest = run_no_reference_eval(
                input_path=args.input,
                output_path=args.output,
                recursive=args.recursive,
                evaluator=args.evaluator,
                limit=args.limit,
            )
            results = manifest.get("results")
            result_count = len(results) if isinstance(results, list) else 0
            print(f"Wrote no-reference eval manifest {args.output} ({result_count} result(s))")
            return 0
        if args.eval_command == "downstream":
            manifest = run_downstream_eval(
                dataset_path=args.dataset,
                output_path=args.output,
                evaluator=args.evaluator,
                dataset_id=args.dataset_id,
                limit=args.limit,
            )
            results = manifest.get("results")
            result_count = len(results) if isinstance(results, list) else 0
            print(f"Wrote downstream eval manifest {args.output} ({result_count} result(s))")
            return 0
        if args.eval_command == "listening-export":
            bundle = run_listening_export(
                manifest_paths=args.manifest,
                output_dir=args.output_dir,
                protocol=args.protocol,
                seed=args.seed,
            )
            print(f"Wrote listening manifest {bundle['manifest_path']}")
            print(f"Wrote listening answer key {bundle['answer_key_path']}")
            return 0
        if args.eval_command == "compare":
            comparison = compare_eval_manifests(
                load_eval_manifest(args.baseline),
                load_eval_manifest(args.candidate),
                thresholds=_parse_eval_thresholds(args.threshold, policy_path=args.threshold_policy),
            )
            if args.output:
                write_eval_manifest(args.output, comparison)
                print(f"Wrote eval comparison {args.output}")
            else:
                print(json.dumps(comparison, indent=2))
            return 0 if comparison["passed"] else 1
        if args.eval_command == "matrix-compare":
            comparison = compare_eval_matrices(
                args.baseline,
                args.candidate,
                thresholds=_parse_eval_thresholds(args.threshold, policy_path=args.threshold_policy),
            )
            if args.output:
                write_eval_manifest(args.output, comparison)
                print(f"Wrote eval matrix comparison {args.output}")
            else:
                print(json.dumps(comparison, indent=2))
            return 0 if comparison["passed"] else 1
        if args.eval_command == "report":
            write_eval_report(args.manifest, args.output)
            print(f"Wrote eval report {args.output}")
            return 0
        if args.eval_command == "bundle":
            bundle = bundle_eval_artifacts(
                manifest_paths=args.manifest,
                output_dir=args.output_dir,
                archive_path=args.archive,
            )
            print(f"Wrote eval bundle {args.output_dir} ({bundle['artifact_count']} artifact(s))")
            if bundle.get("archive_path"):
                print(f"Wrote eval bundle archive {bundle['archive_path']}")
            return 0
        if args.eval_command == "inspect-checkpoint":
            inspection = inspect_torch_checkpoint(args.checkpoint, limit=args.limit)
            write_eval_manifest(args.output, inspection)
            print(f"Wrote checkpoint inspection {args.output}")
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    parser.error(f"Unknown eval command: {args.eval_command}")


def _parse_eval_thresholds(raw_thresholds: list[str], *, policy_path: Path | None = None) -> dict[str, float]:
    thresholds: dict[str, float] = load_threshold_policy(policy_path) if policy_path is not None else {}
    for raw_threshold in raw_thresholds:
        if "=" not in raw_threshold:
            raise ValueError(f"eval threshold must use FIELD=MAX_REGRESSION syntax: {raw_threshold!r}")
        field, raw_value = raw_threshold.split("=", 1)
        field = field.strip()
        if not field:
            raise ValueError(f"eval threshold field is empty: {raw_threshold!r}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"eval threshold value must be numeric: {raw_threshold!r}") from exc
        if value < 0:
            raise ValueError(f"eval threshold value must be non-negative: {raw_threshold!r}")
        thresholds[field] = value
    return thresholds


def _run_command(args: argparse.Namespace, parser: argparse.ArgumentParser, config: InferenceConfig) -> int:
    info_result = _handle_info_command(args, parser, config)
    if info_result is not None:
        return info_result

    if args.input is None:
        parser.error("input is required unless an informational flag is used")

    if args.target_sr <= 0:
        parser.error("--target-sr must be greater than zero")

    jobs = _plan_cli_jobs(args, parser)

    if args.dry_run:
        return _handle_dry_run(args, config, jobs)

    return _handle_enhancement(args, parser, config, jobs)


def _handle_info_command(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    config: InferenceConfig,
) -> int | None:
    result = None
    if args.list_backends:
        print_backends(args.list_format)
        result = 0
    elif args.list_models:
        print_models(filter_text=args.list_filter, list_format=args.list_format)
        result = 0
    elif args.config_info:
        print_config(config)
        result = 0
    elif args.prepare_model_cache:
        result = _handle_prepare_model_cache(args, parser, config)
    elif args.verify_weights:
        result = _handle_verify_weights(args, config)
    elif args.env_info:
        print_env_info(config)
        result = 0
    elif args.compare_manifests:
        result = _handle_compare_manifests(args, parser)
    return result


def _handle_prepare_model_cache(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    config: InferenceConfig,
) -> int:
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


def _handle_verify_weights(args: argparse.Namespace, config: InferenceConfig) -> int:
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


def _handle_compare_manifests(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
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


def _plan_cli_jobs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[PlannedEnhancement]:
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

    return jobs


def _handle_dry_run(args: argparse.Namespace, config: InferenceConfig, jobs: list[PlannedEnhancement]) -> int:
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


def _handle_enhancement(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    config: InferenceConfig,
    jobs: list[PlannedEnhancement],
) -> int:
    init_started_at = time.perf_counter()
    resolver = AudioSuperResolver(target_sr=args.target_sr, backend=args.backend, config=config)
    backend_init_seconds = time.perf_counter() - init_started_at
    started_at = time.perf_counter()
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
    elapsed_seconds = time.perf_counter() - started_at

    for result in results:
        print(
            f"Wrote {result.output_path} at {result.sample_rate} Hz "
            f"from {result.input_sample_rate} Hz using {result.backend}"
        )

    reports = _handle_quality_reports(args, results)

    if args.benchmark_json:
        benchmark_path = write_benchmark_report(
            args.benchmark_json,
            build_benchmark_report(
                backend=args.backend,
                target_sample_rate=args.target_sr,
                config=config,
                results=results,
                quality_reports=reports,
                backend_init_seconds=backend_init_seconds,
                elapsed_seconds=elapsed_seconds,
            ),
        )
        print(f"Wrote benchmark report {benchmark_path}")

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


def _handle_quality_reports(args: argparse.Namespace, results: list[EnhancementResult]):
    if not (args.quality_report or args.quality_report_json or args.fail_on_quality_issue or args.benchmark_json):
        return []

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
    return reports


def _backend_names() -> list[str]:
    return [backend.name for backend in available_backends()]


def _build_config(args: argparse.Namespace) -> InferenceConfig:
    model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else default_model_cache_dir()
    return InferenceConfig(
        device=args.device,
        runtime_provider=args.runtime_provider,
        precision=args.precision,
        chunked=args.chunked,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        seed=args.seed,
        model_cache_dir=model_cache_dir,
        model_name=args.model_name or "basic",
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
