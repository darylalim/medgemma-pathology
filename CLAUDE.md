# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Streamlit web app that analyzes whole slide digital pathology images using Google's MedGemma 1.5 4B model. Requires CUDA. Authenticates via `HF_TOKEN` in `.env`.

## Commands

```bash
uv sync                                # Install dependencies
uv run streamlit run streamlit_app.py   # Run the app
uv run pytest                           # Run all tests
uv run pytest tests/test_streamlit_app.py::TestExtractPatches::test_filters_white_background  # Single test
uv run ruff check .                     # Lint
uv run ruff format .                    # Format
uv run ty check                         # Type check
```

## Architecture

Single-module app (`streamlit_app.py`) with a linear pipeline:

- `extract_patches` — Split image into non-overlapping patches, filter background (mean >= 240).
- `encode_patch` — Convert numpy patch to base64 JPEG data URI.
- `build_messages` — Assemble patches and prompt into Transformers chat message format.
- `load_model` — Load processor and model onto CUDA with bfloat16. Cached via `@st.cache_resource`.
- `run_inference` — Generate response, strip input tokens via tensor slicing.
- `main` — Streamlit UI: sidebar settings, file upload, patch preview, inference trigger.

Tests (`tests/test_streamlit_app.py`) cover pure functions only (`extract_patches`, `encode_patch`, `build_messages`) with synthetic numpy arrays.

## Conventions

- `from __future__ import annotations` in every module.
- Type annotations on all function signatures.
- `DTYPE` module constant for `torch.bfloat16`.
- Early returns to keep `main()` flat.
