# MedGemma Pathology

Analyze whole slide digital pathology images with the [MedGemma 1.5](https://huggingface.co/google/medgemma-1.5-4b-it) model via a Streamlit web app.

> **Disclaimer:** Educational tool only. Not intended for medical diagnosis or treatment.

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA support
- [Hugging Face](https://huggingface.co/) account with access to `google/medgemma-1.5-4b-it`

## Setup

Install dependencies:

```bash
uv sync
```

Create a `.env` file with your Hugging Face token:

```
HF_TOKEN=hf_your_token_here
```

## Usage

Start the app:

```bash
uv run streamlit run streamlit_app.py
```

1. Upload a pathology image (PNG, JPEG, or TIFF).
2. Adjust patch size and max patches in the sidebar.
3. Click **Analyze with MedGemma** to run inference.

## How It Works

1. The uploaded image is split into non-overlapping patches (default 896px).
2. Background patches are filtered out.
3. Tissue patches are encoded and sent to MedGemma for analysis.
4. The model returns a descriptive summary of the tissue.

## Tests

Run all tests from the `tests/` directory:

```bash
uv run pytest
```
