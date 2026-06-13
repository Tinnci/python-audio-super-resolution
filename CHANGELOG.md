# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning once published to PyPI.

## 0.1.0

- Added the initial Pixi-managed Python package.
- Added `audio-super-res` and `audiosr` CLI entry points.
- Added single-file and recursive directory batch processing.
- Added backend selection with the baseline `sinc-resample` backend.
- Added inference configuration for device, precision, chunking, seed, and model cache path.
- Added audio quality checks for sample rate, duration drift, clipping, and peak level.
- Added examples for single-file enhancement, batch processing, and quality checks.
- Added the optional `audiosr` backend wrapper for AudioSR latent diffusion inference.
