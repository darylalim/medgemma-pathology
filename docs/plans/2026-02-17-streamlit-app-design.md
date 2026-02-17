# MedGemma Pathology Streamlit App Design

## Overview

Single-page Streamlit app that wraps the MedGemma 1.5 4B pathology notebook into an interactive UI. Users upload pathology images, the app extracts patches, and MedGemma generates educational descriptions.

**Key decisions:**
- Local image upload (no DICOM/Google Cloud dependency)
- Runs on Mac with MPS (Apple Silicon), with CUDA/CPU fallback
- Customizable analysis prompt
- Simple single-page layout

## Architecture

Single `streamlit_app.py` with four logical sections:

1. **Model loading** - `@st.cache_resource`, auto-detects MPS/CUDA/CPU
2. **Image upload & patch extraction** - PIL-based tiling with tissue detection
3. **Prompt construction** - multi-image chat format matching notebook
4. **Inference & display** - MedGemma generation with markdown output

### Dependencies

- `streamlit` - UI framework
- `torch` - model inference
- `transformers` - MedGemma model/processor loading
- `Pillow` - image processing
- `numpy` - array operations

No `ez-wsi-dicomweb` dependency (replaced by PIL-based patch extraction).

## UI Layout

### Sidebar

- HF token input (password field, env var fallback)
- Patch size slider (default 896)
- Max patches slider (default 125)
- Prompt text area (pre-filled with notebook default)

### Main Area

1. Title + educational disclaimer
2. File uploader (PNG, JPEG, TIFF)
3. Uploaded image thumbnail preview
4. Extracted patch grid preview
5. "Analyze" button
6. Model response (markdown)

## Patch Extraction

1. Open image with PIL
2. Tile into non-overlapping `patch_size x patch_size` regions
3. Filter background patches (mean pixel value > 240 = likely white/empty)
4. Randomly sample up to `max_patches` tissue-containing patches
5. Encode as base64 JPEG (matching notebook's `_encode`)

## Device Handling

- MPS: `float16` (bfloat16 not supported)
- CUDA: `bfloat16`
- CPU: `float32` (fallback, will be slow)

## Testing

Unit tests for:
- `extract_patches()` - patch extraction from synthetic images
- `filter_tissue_patches()` - background filtering
- `encode_patch()` - base64 encoding
- `detect_device()` - device/dtype selection
- `build_prompt()` - chat message construction
