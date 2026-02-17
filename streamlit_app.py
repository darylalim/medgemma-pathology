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
