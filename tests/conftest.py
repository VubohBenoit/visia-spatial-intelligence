"""Shared fixtures. Unit tests never download model weights or datasets."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from visia_spatial_intelligence.config import AppConfig, LimitsConfig
from visia_spatial_intelligence.schemas import DepthMap


def make_image_bytes(width: int = 320, height: int = 240, fmt: str = "PNG") -> bytes:
    """Create a deterministic synthetic RGB image as encoded bytes."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture()
def limits() -> LimitsConfig:
    return LimitsConfig()


@pytest.fixture()
def app_config() -> AppConfig:
    return AppConfig()


@pytest.fixture()
def blank_image() -> Image.Image:
    return Image.new("RGB", (640, 480), color=(20, 20, 20))


@pytest.fixture()
def flat_depth() -> DepthMap:
    """A depth map where every pixel has the same relative value."""
    values = np.full((480, 640), 0.5, dtype=np.float32)
    return DepthMap(values=values, height=480, width=640)
