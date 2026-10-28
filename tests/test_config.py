"""Tests for YAML configuration loading and validation."""

from pathlib import Path

import pytest

from visia_spatial_intelligence.config import DEFAULT_CONFIG_PATH, ConfigError, load_config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_config_loads_and_matches_documented_defaults() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.detector.model_id == "facebook/detr-resnet-50"
    assert config.depth.model_id == "depth-anything/Depth-Anything-V2-Small-hf"
    assert "person" in config.detector.classes
    assert config.reasoning.corridor.top_y == pytest.approx(0.55)


def test_partial_yaml_falls_back_to_defaults(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, "tracker:\n  iou_threshold: 0.5\n"))
    assert config.tracker.iou_threshold == 0.5
    assert config.tracker.max_missed_frames == 5


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown keys in 'tracker'"):
        load_config(_write(tmp_path, "tracker:\n  iou_treshold: 0.5\n"))


def test_invalid_threshold_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="score_threshold"):
        load_config(_write(tmp_path, "detector:\n  score_threshold: 1.5\n"))


def test_malformed_yaml_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Malformed YAML"):
        load_config(_write(tmp_path, "detector: [unclosed\n"))


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")
