# MedGemma Pathology Streamlit App Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Streamlit app that lets users upload pathology images, extract patches, and analyze them with MedGemma 1.5 4B locally on Mac (MPS).

**Architecture:** Single `streamlit_app.py` with helper functions for patch extraction, encoding, device detection, and prompt building. Model loaded once via `@st.cache_resource`. PIL-based patch extraction replaces the notebook's DICOM pipeline.

**Tech Stack:** Streamlit, PyTorch, Transformers, Pillow, NumPy, uv (package management), ruff/ty (linting), pytest (testing)

---

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`

**Step 1: Initialize uv project and add dependencies**

```bash
cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/medgemma-pathology"
uv init --name medgemma-pathology
uv add streamlit torch transformers Pillow numpy accelerate
uv add --dev ruff ty pytest
```

Note: `accelerate` is needed for `device_map="auto"` in transformers.

**Step 2: Verify setup**

Run: `uv run python -c "import streamlit; import torch; import transformers; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: initialize project with uv and dependencies"
```

---

### Task 2: Patch Extraction and Encoding Functions

**Files:**
- Create: `streamlit_app.py`
- Create: `test_streamlit_app.py`

**Step 1: Write failing tests for patch extraction and encoding**

```python
# test_streamlit_app.py
import base64

import numpy as np
import pytest
from PIL import Image

from streamlit_app import detect_device, encode_patch, extract_patches


class TestExtractPatches:
    def test_extracts_correct_number_of_patches(self) -> None:
        # 2x2 grid of 100x100 patches from a 200x200 image
        img = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))
        patches = extract_patches(img, patch_size=100, max_patches=10)
        assert len(patches) == 4

    def test_filters_white_background(self) -> None:
        # Create image: left half is tissue (dark), right half is white
        arr = np.ones((100, 200, 3), dtype=np.uint8) * 255
        arr[:, :100, :] = 50  # dark tissue on left
        img = Image.fromarray(arr)
        patches = extract_patches(img, patch_size=100, max_patches=10)
        assert len(patches) == 1
        # The tissue patch should be dark
        assert np.array(patches[0]).mean() < 100

    def test_respects_max_patches(self) -> None:
        # Large image with many patches, but max_patches limits output
        img = Image.fromarray(np.zeros((500, 500, 3), dtype=np.uint8))
        patches = extract_patches(img, patch_size=100, max_patches=3)
        assert len(patches) <= 3

    def test_returns_empty_for_all_white_image(self) -> None:
        img = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 255)
        patches = extract_patches(img, patch_size=100, max_patches=10)
        assert len(patches) == 0


class TestEncodePatch:
    def test_returns_base64_data_uri(self) -> None:
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = encode_patch(arr)
        assert result.startswith("data:image/jpeg;base64,")

    def test_is_valid_base64(self) -> None:
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = encode_patch(arr)
        b64_data = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_data)
        assert len(decoded) > 0


class TestDetectDevice:
    def test_returns_tuple(self) -> None:
        device, dtype = detect_device()
        assert isinstance(device, str)
        assert device in ("mps", "cuda", "cpu")
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: FAIL (ModuleNotFoundError for streamlit_app)

**Step 3: Implement the functions**

```python
# streamlit_app.py (initial version - just the helper functions)
import base64
import io
import random

import numpy as np
import PIL.Image
import torch


def detect_device() -> tuple[str, torch.dtype]:
    """Auto-detect the best available device and dtype."""
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    elif torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def extract_patches(
    image: PIL.Image.Image,
    patch_size: int = 896,
    max_patches: int = 125,
) -> list[np.ndarray]:
    """Extract non-overlapping patches from an image, filtering background."""
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = arr[y : y + patch_size, x : x + patch_size]
            # Filter out mostly-white/background patches
            if patch.mean() < 240:
                patches.append(patch)
    if len(patches) > max_patches:
        patches = random.sample(patches, max_patches)
    return patches


def encode_patch(data: np.ndarray) -> str:
    """Encode a patch as a base64 JPEG data URI."""
    with PIL.Image.fromarray(data) as img:
        with io.BytesIO() as buf:
            img.save(buf, format="jpeg")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All tests PASS

**Step 5: Lint and format**

```bash
uv run ruff check --fix streamlit_app.py test_streamlit_app.py
uv run ruff format streamlit_app.py test_streamlit_app.py
```

**Step 6: Commit**

```bash
git add streamlit_app.py test_streamlit_app.py
git commit -m "feat: add patch extraction and encoding functions with tests"
```

---

### Task 3: Prompt Building Function

**Files:**
- Modify: `streamlit_app.py`
- Modify: `test_streamlit_app.py`

**Step 1: Write failing test for prompt building**

Append to `test_streamlit_app.py`:

```python
from streamlit_app import build_messages

class TestBuildMessages:
    def test_returns_correct_structure(self) -> None:
        patches = [np.zeros((100, 100, 3), dtype=np.uint8)]
        prompt = "Describe this tissue."
        messages = build_messages(patches, prompt)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        # First item is text, rest are images
        assert content[0]["type"] == "text"
        assert content[0]["text"] == prompt
        assert content[1]["type"] == "image"
        assert content[1]["image"].startswith("data:image/jpeg;base64,")

    def test_includes_all_patches(self) -> None:
        patches = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(5)]
        messages = build_messages(patches, "test")
        content = messages[0]["content"]
        # 1 text + 5 images
        assert len(content) == 6
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest test_streamlit_app.py::TestBuildMessages -v`
Expected: FAIL

**Step 3: Implement build_messages**

Add to `streamlit_app.py`:

```python
def build_messages(
    patches: list[np.ndarray], prompt: str
) -> list[dict]:
    """Build chat-completion messages with images from patches."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for patch in patches:
        content.append({"type": "image", "image": encode_patch(patch)})
    return [{"role": "user", "content": content}]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest test_streamlit_app.py -v`
Expected: All PASS

**Step 5: Lint, format, commit**

```bash
uv run ruff check --fix streamlit_app.py test_streamlit_app.py
uv run ruff format streamlit_app.py test_streamlit_app.py
git add streamlit_app.py test_streamlit_app.py
git commit -m "feat: add prompt building function with tests"
```

---

### Task 4: Streamlit UI and Model Integration

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Add model loading and Streamlit UI**

Add to `streamlit_app.py`:

```python
import streamlit as st
import transformers


DEFAULT_PROMPT = (
    "You are an instructor teaching medical students. For education "
    "purposes, provide a brief descriptive text for the set of pathology "
    "patches extracted from a pathology slide. Consider the tissue type "
    "and procedure (below) when deciding what to include in the "
    "descriptive text.\ncolon, biopsy:"
)

MODEL_ID = "google/medgemma-1.5-4b-it"


@st.cache_resource
def load_model(
    hf_token: str,
) -> tuple[transformers.AutoProcessor, transformers.AutoModelForImageTextToText]:
    """Load MedGemma model and processor, cached across reruns."""
    device, dtype = detect_device()
    model_kwargs = dict(
        dtype=dtype,
        device_map=device,
        offload_buffers=True,
    )
    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_ID, use_fast=True, token=hf_token, **model_kwargs
    )
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, token=hf_token, **model_kwargs
    )
    return processor, model


def run_inference(
    processor: transformers.AutoProcessor,
    model: transformers.AutoModelForImageTextToText,
    messages: list[dict],
) -> str:
    """Run MedGemma inference and return the response text."""
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        continue_final_message=False,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    )
    device, dtype = detect_device()
    with torch.inference_mode():
        inputs = inputs.to(model.device, dtype=dtype)
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=2000)
    response = processor.post_process_image_text_to_text(
        generated, skip_special_tokens=True
    )
    decoded_input = processor.post_process_image_text_to_text(
        inputs["input_ids"], skip_special_tokens=True
    )
    result = response[0]
    idx = result.find(decoded_input[0])
    if 0 <= idx <= 2:
        result = result[idx + len(decoded_input[0]) :]
    return result


def main() -> None:
    st.set_page_config(page_title="MedGemma Pathology", layout="wide")
    st.title("MedGemma Pathology Analyzer")
    st.caption(
        "Educational tool only. Not intended for medical diagnosis or treatment. "
        "See HAI-DEF Terms of Use."
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        hf_token = st.text_input(
            "Hugging Face Token",
            type="password",
            value=os.environ.get("HF_TOKEN", ""),
            help="Required to download MedGemma. Get one at huggingface.co/settings/tokens",
        )
        patch_size = st.slider("Patch size (px)", 224, 1024, 896, step=32)
        max_patches = st.slider("Max patches", 1, 200, 125)
        prompt = st.text_area("Analysis prompt", value=DEFAULT_PROMPT, height=150)

    # --- Main area ---
    uploaded = st.file_uploader(
        "Upload a pathology image", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    if uploaded is not None:
        image = PIL.Image.open(uploaded)
        st.subheader("Uploaded Image")
        # Show a resized preview
        preview = image.copy()
        preview.thumbnail((800, 800))
        st.image(preview, use_container_width=True)

        # Extract patches
        patches = extract_patches(image, patch_size=patch_size, max_patches=max_patches)
        st.subheader(f"Extracted Patches ({len(patches)})")

        if len(patches) == 0:
            st.warning("No tissue-containing patches found. Try a smaller patch size.")
        else:
            # Show up to 6 sample patches in a grid
            cols = st.columns(min(len(patches), 3))
            for i, patch in enumerate(patches[: min(6, len(patches))]):
                cols[i % 3].image(patch, caption=f"Patch {i + 1}", width=200)

            if st.button("Analyze with MedGemma", type="primary"):
                if not hf_token:
                    st.error("Please enter your Hugging Face token in the sidebar.")
                else:
                    with st.spinner("Loading model (first run may take several minutes)..."):
                        processor, model = load_model(hf_token)
                    messages = build_messages(patches, prompt)
                    with st.spinner(f"Analyzing {len(patches)} patches..."):
                        response = run_inference(processor, model, messages)
                    st.subheader("MedGemma Analysis")
                    st.markdown(response)


if __name__ == "__main__":
    main()
```

Note: The `import os` is already at the top of the file from Task 2.

**Step 2: Add missing import (`os`) to the top of the file if not present**

Ensure `import os` is in the imports section.

**Step 3: Lint and format**

```bash
uv run ruff check --fix streamlit_app.py
uv run ruff format streamlit_app.py
```

**Step 4: Verify app starts without error**

```bash
uv run streamlit run streamlit_app.py --server.headless true &
sleep 5
curl -s http://localhost:8501/_stcore/health | head -1
# Expected: "ok" or similar health response
kill %1
```

**Step 5: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add Streamlit UI with model loading and inference"
```

---

### Task 5: Final Lint, Type Check, and Test Pass

**Files:**
- Modify: `streamlit_app.py` (if lint/type fixes needed)
- Modify: `test_streamlit_app.py` (if lint/type fixes needed)

**Step 1: Run full lint and format**

```bash
uv run ruff check --fix streamlit_app.py test_streamlit_app.py
uv run ruff format streamlit_app.py test_streamlit_app.py
```

**Step 2: Run type check**

```bash
uv run ty check streamlit_app.py
```

Fix any type errors reported.

**Step 3: Run all tests**

```bash
uv run pytest test_streamlit_app.py -v
```

Expected: All PASS

**Step 4: Commit any fixes**

```bash
git add streamlit_app.py test_streamlit_app.py
git commit -m "chore: lint, format, and type check cleanup"
```

---

### Task 6: Clean Up and Final Commit

**Files:**
- Delete: `notebook.ipynb` (downloaded copy, not needed)

**Step 1: Remove downloaded notebook**

```bash
rm notebook.ipynb
```

**Step 2: Final git status check and commit if needed**

```bash
git status
```
