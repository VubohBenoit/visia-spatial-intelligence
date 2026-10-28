"""Rendering and command-line tests that do not need model weights."""

import json
from pathlib import Path

from PIL import Image

from visia_spatial_intelligence.cli import EXIT_INPUT, main
from visia_spatial_intelligence.config import AppConfig
from visia_spatial_intelligence.schemas import BoundingBox, CorridorRegion, Detection, FrameResult
from visia_spatial_intelligence.visualization.renderer import render_frame

from .conftest import make_image_bytes


def test_render_frame_does_not_mutate_input(blank_image: Image.Image) -> None:
    original = blank_image.tobytes()
    result = FrameResult(
        frame_index=0,
        width=640,
        height=480,
        detections=[Detection("person", 0.9, BoundingBox(10, 10, 100, 200))],
    )
    annotated = render_frame(blank_image, result, CorridorRegion())
    assert blank_image.tobytes() == original
    assert annotated.size == blank_image.size
    assert annotated.tobytes() != original


def test_info_command_runs_without_loading_models(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pipeline"]["detector"]["model_id"] == "facebook/detr-resnet-50"
    assert "disclaimer" in payload


def test_analyze_rejects_invalid_file(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"garbage")
    assert main(["analyze", str(bad), "--out-dir", str(tmp_path / "out")]) == EXIT_INPUT
    assert "Invalid input" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_analyze_reports_missing_model_with_exit_code(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    from visia_spatial_intelligence.detection import detector as detector_module
    from visia_spatial_intelligence.detection.detector import ModelUnavailableError

    def _fail(self, image):  # type: ignore[no-untyped-def]
        raise ModelUnavailableError("offline")

    monkeypatch.setattr(detector_module.DetrDetector, "detect", _fail)
    good = tmp_path / "scene.png"
    good.write_bytes(make_image_bytes())
    assert main(["analyze", str(good), "--out-dir", str(tmp_path / "out"), "--no-depth"]) == 3
    assert "Model unavailable: offline" in capsys.readouterr().err
    assert AppConfig()  # defaults remain importable
