import base64

import numpy as np
from PIL import Image

from streamlit_app import detect_device, encode_patch, extract_patches


class TestExtractPatches:
    def test_extracts_correct_number_of_patches(self) -> None:
        img = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))
        patches = extract_patches(img, patch_size=100, max_patches=10)
        assert len(patches) == 4

    def test_filters_white_background(self) -> None:
        arr = np.ones((100, 200, 3), dtype=np.uint8) * 255
        arr[:, :100, :] = 50
        img = Image.fromarray(arr)
        patches = extract_patches(img, patch_size=100, max_patches=10)
        assert len(patches) == 1
        assert np.array(patches[0]).mean() < 100

    def test_respects_max_patches(self) -> None:
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
