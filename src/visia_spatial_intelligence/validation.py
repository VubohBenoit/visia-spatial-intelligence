"""Input validation for images supplied as paths or raw bytes.

Every check raises :class:`InputValidationError` with a message that is safe to show
to a user or return from an API. Nothing here touches the network or model weights.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from visia_spatial_intelligence.config import LimitsConfig

ALLOWED_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "BMP", "WEBP"})
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# Guard against decompression bombs before Pillow decodes the image.
Image.MAX_IMAGE_PIXELS = 4096 * 4096 * 2


class InputValidationError(ValueError):
    """The input is unsupported, corrupt, or exceeds configured limits."""


def _check_dimensions(width: int, height: int, limits: LimitsConfig) -> None:
    if min(width, height) < limits.min_image_side:
        raise InputValidationError(
            f"Image too small: {width}x{height}; minimum side is {limits.min_image_side}px"
        )
    if max(width, height) > limits.max_image_side:
        raise InputValidationError(
            f"Image too large: {width}x{height}; maximum side is {limits.max_image_side}px"
        )


def load_image_bytes(data: bytes, limits: LimitsConfig) -> Image.Image:
    """Decode an in-memory image into an RGB Pillow image after validating it.

    Raises:
        InputValidationError: if the payload is empty, too large, not a supported
            image format, corrupt, or outside the configured dimensions.
    """
    if not data:
        raise InputValidationError("Empty input: no image bytes received")
    if len(data) > limits.max_image_bytes:
        raise InputValidationError(
            f"File too large: {len(data)} bytes; limit is {limits.max_image_bytes} bytes"
        )
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        probe.verify()  # detects truncated or corrupted streams
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InputValidationError(f"Corrupt or unreadable image: {exc}") from exc

    if fmt not in ALLOWED_FORMATS:
        raise InputValidationError(
            f"Unsupported image format: {fmt}; allowed: {', '.join(sorted(ALLOWED_FORMATS))}"
        )

    # ``verify`` invalidates the probe, so decode again for pixel access.
    image = Image.open(io.BytesIO(data))
    _check_dimensions(image.width, image.height, limits)
    try:
        return image.convert("RGB")
    except OSError as exc:
        raise InputValidationError(f"Corrupt image pixel data: {exc}") from exc


def load_image_path(path: Path, limits: LimitsConfig) -> Image.Image:
    """Load an image from a local file after checking its extension and size.

    The caller is responsible for deciding which paths are allowed. Web-facing code
    should pass bytes through :func:`load_image_bytes` instead of paths.
    """
    if not path.is_file():
        raise InputValidationError(f"Input file does not exist: {path.name}")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise InputValidationError(
            f"Unsupported file extension '{path.suffix}'; allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )
    size = path.stat().st_size
    if size > limits.max_image_bytes:
        raise InputValidationError(
            f"File too large: {size} bytes; limit is {limits.max_image_bytes} bytes"
        )
    return load_image_bytes(path.read_bytes(), limits)
