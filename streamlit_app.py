from __future__ import annotations

import base64
import io
import os
import random
from typing import Any

from dotenv import load_dotenv
import numpy as np
import PIL.Image
import streamlit as st
import torch
import transformers

load_dotenv()

DTYPE = torch.bfloat16
MODEL_ID = "google/medgemma-1.5-4b-it"
DEFAULT_PROMPT = (
    "You are an instructor teaching medical students. For education "
    "purposes, provide a brief descriptive text for the set of pathology "
    "patches extracted from a pathology slide. Consider the tissue type "
    "and procedure (below) when deciding what to include in the "
    "descriptive text.\ncolon, biopsy:"
)


def extract_patches(
    image: PIL.Image.Image,
    patch_size: int = 896,
    max_patches: int = 125,
) -> list[np.ndarray]:
    """Extract non-overlapping tissue patches from an image."""
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    patches = [
        patch
        for y in range(0, h - patch_size + 1, patch_size)
        for x in range(0, w - patch_size + 1, patch_size)
        if (patch := arr[y : y + patch_size, x : x + patch_size]).mean() < 240
    ]
    if len(patches) > max_patches:
        patches = random.sample(patches, max_patches)
    return patches


def encode_patch(data: np.ndarray) -> str:
    """Encode a patch as a base64 JPEG data URI."""
    buf = io.BytesIO()
    PIL.Image.fromarray(data).save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def build_messages(patches: list[np.ndarray], prompt: str) -> list[dict]:
    """Build chat-completion messages with images from patches."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image", "image": encode_patch(p)} for p in patches)
    return [{"role": "user", "content": content}]


@st.cache_resource
def load_model(hf_token: str) -> tuple[Any, Any]:
    """Load MedGemma model and processor, cached across reruns."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available on this system.")
    kwargs: dict[str, Any] = dict(
        dtype=DTYPE, device_map="cuda", offload_buffers=True, token=hf_token
    )
    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_ID, use_fast=True, **kwargs
    )
    model = transformers.AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kwargs)
    return processor, model


def run_inference(processor: Any, model: Any, messages: list[dict]) -> str:
    """Run MedGemma inference and return the generated response."""
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to(model.device, dtype=DTYPE)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=2000)
    # Strip the input tokens to get only the generated portion
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def main() -> None:
    st.set_page_config(page_title="MedGemma Pathology", layout="wide")
    st.title("MedGemma Pathology Analyzer")
    st.caption(
        "Educational tool only. Not intended for medical diagnosis or treatment. "
        "See HAI-DEF Terms of Use."
    )

    hf_token = os.environ.get("HF_TOKEN", "")

    with st.sidebar:
        st.header("Settings")
        patch_size = st.slider("Patch size (px)", 224, 1024, 896, step=32)
        max_patches = st.slider("Max patches", 1, 200, 125)
        prompt = st.text_area("Analysis prompt", value=DEFAULT_PROMPT, height=150)

    uploaded = st.file_uploader(
        "Upload a pathology image", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )
    if uploaded is None:
        return

    image = PIL.Image.open(uploaded)
    st.subheader("Uploaded Image")
    preview = image.copy()
    preview.thumbnail((800, 800))
    st.image(preview, use_container_width=True)

    patches = extract_patches(image, patch_size=patch_size, max_patches=max_patches)
    st.subheader(f"Extracted Patches ({len(patches)})")

    if not patches:
        st.warning("No tissue-containing patches found. Try a smaller patch size.")
        return

    cols = st.columns(min(len(patches), 3))
    for i, patch in enumerate(patches[:6]):
        cols[i % 3].image(patch, caption=f"Patch {i + 1}", width=200)

    if not st.button("Analyze with MedGemma", type="primary"):
        return
    if not hf_token:
        st.error("Please set the HF_TOKEN environment variable.")
        return

    with st.spinner("Loading model (first run may take several minutes)..."):
        processor, model = load_model(hf_token)
    messages = build_messages(patches, prompt)
    with st.spinner(f"Analyzing {len(patches)} patches..."):
        response = run_inference(processor, model, messages)
    st.subheader("MedGemma Analysis")
    st.markdown(response)


if __name__ == "__main__":
    main()
