"""API tests with fake pipelines, so no model weights are downloaded."""

from __future__ import annotations

from fastapi.testclient import TestClient
from PIL import Image

from visia_spatial_intelligence.api.app import create_app
from visia_spatial_intelligence.config import AppConfig, LimitsConfig
from visia_spatial_intelligence.detection.detector import ModelUnavailableError
from visia_spatial_intelligence.pipeline import AnalysisOutput
from visia_spatial_intelligence.schemas import BoundingBox, Detection, FrameResult

from .conftest import make_image_bytes


class _FakePipeline:
    def __init__(self, fail: Exception | None = None, delay: float = 0.0) -> None:
        self._fail = fail
        self._delay = delay

    def reset(self) -> None:
        pass

    def analyze(self, image: Image.Image, frame_index: int = 0) -> AnalysisOutput:
        import time

        if self._delay:
            time.sleep(self._delay)
        if self._fail is not None:
            raise self._fail
        result = FrameResult(
            frame_index=frame_index,
            width=image.width,
            height=image.height,
            detections=[Detection("person", 0.9, BoundingBox(1, 1, 10, 10))],
        )
        return AnalysisOutput(result=result, depth=None)

    def describe(self) -> dict[str, object]:
        return {"detector": {"model_id": "fake"}}


def _client(
    pipeline: _FakePipeline | None = None,
    limits: LimitsConfig | None = None,
    timeout: float = 5.0,
) -> TestClient:
    config = AppConfig(limits=limits) if limits else AppConfig()
    app = create_app(
        config=config,
        pipeline_factory=lambda _cfg: pipeline or _FakePipeline(),  # type: ignore[arg-type,return-value]
        timeout_seconds=timeout,
    )
    return TestClient(app)


def test_health_and_info() -> None:
    client = _client()
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/info").json()
    assert "disclaimer" in info
    assert "max_image_bytes" in info["limits"]


def test_analyze_returns_result_json() -> None:
    client = _client()
    response = client.post(
        "/v1/analyze", files={"image": ("scene.png", make_image_bytes(), "image/png")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["width"] == 320 and body["height"] == 240
    assert body["detections"][0]["label"] == "person"
    assert "disclaimer" in body


def test_corrupt_image_is_rejected_with_422() -> None:
    response = _client().post(
        "/v1/analyze", files={"image": ("bad.png", b"not an image", "image/png")}
    )
    assert response.status_code == 422
    assert "Corrupt" in response.json()["detail"]


def test_unsupported_format_is_rejected_with_422() -> None:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, format="GIF")
    response = _client().post(
        "/v1/analyze", files={"image": ("x.gif", buffer.getvalue(), "image/gif")}
    )
    assert response.status_code == 422


def test_oversized_upload_is_rejected_with_413() -> None:
    small = LimitsConfig(max_image_bytes=200)
    response = _client(limits=small).post(
        "/v1/analyze", files={"image": ("big.png", make_image_bytes(), "image/png")}
    )
    assert response.status_code == 413


def test_missing_model_returns_503() -> None:
    pipeline = _FakePipeline(fail=ModelUnavailableError("offline"))
    response = _client(pipeline).post(
        "/v1/analyze", files={"image": ("scene.png", make_image_bytes(), "image/png")}
    )
    assert response.status_code == 503
    assert "offline" in response.json()["detail"]


def test_slow_analysis_times_out_with_504() -> None:
    pipeline = _FakePipeline(delay=0.5)
    response = _client(pipeline, timeout=0.05).post(
        "/v1/analyze", files={"image": ("scene.png", make_image_bytes(), "image/png")}
    )
    assert response.status_code == 504


def test_missing_image_field_is_a_client_error() -> None:
    response = _client().post("/v1/analyze", files={})
    assert response.status_code == 422
