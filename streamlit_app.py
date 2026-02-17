import base64
import io
import os
import random

import numpy as np
import PIL.Image
import streamlit as st
import torch
import transformers


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


def build_messages(patches: list[np.ndarray], prompt: str) -> list[dict]:
    """Build chat-completion messages with images from patches."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for patch in patches:
        content.append({"type": "image", "image": encode_patch(patch)})
    return [{"role": "user", "content": content}]


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

    # Sidebar settings
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

    # Main area
    uploaded = st.file_uploader(
        "Upload a pathology image", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    if uploaded is not None:
        image = PIL.Image.open(uploaded)
        st.subheader("Uploaded Image")
        preview = image.copy()
        preview.thumbnail((800, 800))
        st.image(preview, use_container_width=True)

        patches = extract_patches(image, patch_size=patch_size, max_patches=max_patches)
        st.subheader(f"Extracted Patches ({len(patches)})")

        if len(patches) == 0:
            st.warning("No tissue-containing patches found. Try a smaller patch size.")
        else:
            cols = st.columns(min(len(patches), 3))
            for i, patch in enumerate(patches[: min(6, len(patches))]):
                cols[i % 3].image(patch, caption=f"Patch {i + 1}", width=200)

            if st.button("Analyze with MedGemma", type="primary"):
                if not hf_token:
                    st.error("Please enter your Hugging Face token in the sidebar.")
                else:
                    with st.spinner(
                        "Loading model (first run may take several minutes)..."
                    ):
                        processor, model = load_model(hf_token)
                    messages = build_messages(patches, prompt)
                    with st.spinner(f"Analyzing {len(patches)} patches..."):
                        response = run_inference(processor, model, messages)
                    st.subheader("MedGemma Analysis")
                    st.markdown(response)


if __name__ == "__main__":
    main()
