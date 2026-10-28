"""Tests for input validation of images supplied as bytes or paths."""

from pathlib import Path

import pytest

from visia_spatial_intelligence.config import LimitsConfig
from visia_spatial_intelligence.validation import (
    InputValidationError,
    load_image_bytes,
    load_image_path,
)

from .conftest import make_image_bytes


def test_valid_png_and_jpeg_are_decoded_as_rgb(limits: LimitsConfig) -> None:
    png = load_image_bytes(make_image_bytes(fmt="PNG"), limits)
    jpg = load_image_bytes(make_image_bytes(fmt="JPEG"), limits)
    assert png.mode == "RGB" and png.size == (320, 240)
    assert jpg.mode == "RGB" and jpg.size == (320, 240)


def test_empty_payload_is_rejected(limits: LimitsConfig) -> None:
    with pytest.raises(InputValidationError, match="Empty input"):
        load_image_bytes(b"", limits)


def test_corrupt_bytes_are_rejected(limits: LimitsConfig) -> None:
    with pytest.raises(InputValidationError, match="Corrupt or unreadable"):
        load_image_bytes(b"not an image at all", limits)


def test_truncated_image_is_rejected(limits: LimitsConfig) -> None:
    data = make_image_bytes(fmt="PNG")
    with pytest.raises(InputValidationError):
        load_image_bytes(data[: len(data) // 3], limits)


def test_unsupported_format_is_rejected(limits: LimitsConfig) -> None:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, format="GIF")
    with pytest.raises(InputValidationError, match="Unsupported image format: GIF"):
        load_image_bytes(buffer.getvalue(), limits)


def test_oversized_payload_is_rejected() -> None:
    small_limit = LimitsConfig(max_image_bytes=100)
    with pytest.raises(InputValidationError, match="File too large"):
        load_image_bytes(make_image_bytes(), small_limit)


def test_dimensions_outside_limits_are_rejected() -> None:
    with pytest.raises(InputValidationError, match="too large"):
        load_image_bytes(make_image_bytes(320, 240), LimitsConfig(max_image_side=200))
    with pytest.raises(InputValidationError, match="too small"):
        load_image_bytes(make_image_bytes(16, 16), LimitsConfig(min_image_side=32))


def test_path_with_unsupported_extension_is_rejected(tmp_path: Path, limits: LimitsConfig) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(make_image_bytes())
    with pytest.raises(InputValidationError, match="Unsupported file extension"):
        load_image_path(target, limits)


def test_missing_path_is_rejected(tmp_path: Path, limits: LimitsConfig) -> None:
    with pytest.raises(InputValidationError, match="does not exist"):
        load_image_path(tmp_path / "missing.png", limits)


def test_valid_path_is_loaded(tmp_path: Path, limits: LimitsConfig) -> None:
    target = tmp_path / "scene.png"
    target.write_bytes(make_image_bytes())
    assert load_image_path(target, limits).size == (320, 240)
