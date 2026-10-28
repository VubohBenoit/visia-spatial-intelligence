"""FastAPI service for programmatic single-image analysis.

Endpoints:

* ``GET  /health``  liveness check.
* ``GET  /info``    pipeline configuration and runtime information (no secrets, no paths).
* ``POST /v1/analyze``  multipart upload in field ``image``; returns the frame result as JSON.

Safety properties:

* The service reads uploaded bytes only. It never accepts file paths and exposes no
  filesystem browsing or model download endpoint. Model IDs come from configuration.
* Uploads are rejected above ``limits.max_image_bytes`` (HTTP 413) before decoding.
* Invalid or corrupt images return HTTP 422. A missing model returns HTTP 503.
* Analysis is limited by ``timeout_seconds`` (HTTP 504). The worker thread cannot be
  killed, so it finishes in the background and its result is discarded.
* Requests are serialised, because the pipeline holds per-stream state.

Run locally with ``uv run --extra api uvicorn visia_spatial_intelligence.api.app:app``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from visia_spatial_intelligence.cli import DISCLAIMER, runtime_info
from visia_spatial_intelligence.config import AppConfig, load_config
from visia_spatial_intelligence.detection.detector import ModelUnavailableError
from visia_spatial_intelligence.pipeline import VisiaPipeline
from visia_spatial_intelligence.validation import InputValidationError, load_image_bytes

DEFAULT_TIMEOUT_SECONDS = 120.0
_READ_CHUNK = 64 * 1024


class _PipelineService:
    """Owns one pipeline per depth setting and serialises access to it."""

    def __init__(
        self,
        config: AppConfig,
        factory: Callable[[AppConfig], VisiaPipeline] = VisiaPipeline.from_config,
    ) -> None:
        self._config = config
        self._factory = factory
        self._pipelines: dict[bool, VisiaPipeline] = {}
        self._lock = threading.Lock()

    @property
    def config(self) -> AppConfig:
        return self._config

    def pipeline(self, depth_enabled: bool) -> VisiaPipeline:
        if depth_enabled not in self._pipelines:
            cfg = self._config.with_overrides(
                depth=replace(self._config.depth, enabled=depth_enabled)
            )
            self._pipelines[depth_enabled] = self._factory(cfg)
        return self._pipelines[depth_enabled]

    def analyse(self, data: bytes, depth_enabled: bool) -> dict[str, Any]:
        """Decode and analyse one image. Runs in a worker thread."""
        image = load_image_bytes(data, self._config.limits)
        with self._lock:
            pipeline = self.pipeline(depth_enabled)
            pipeline.reset()  # each upload is an independent single-frame request
            output = pipeline.analyze(image, frame_index=0)
        payload = output.result.to_dict()
        payload["disclaimer"] = DISCLAIMER
        return payload


def create_app(
    config: AppConfig | None = None,
    pipeline_factory: Callable[[AppConfig], VisiaPipeline] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> FastAPI:
    """Build the application. Tests pass a ``pipeline_factory`` that returns fakes."""
    service = _PipelineService(
        config or load_config(), pipeline_factory or VisiaPipeline.from_config
    )
    app = FastAPI(
        title="VisIA Spatial Intelligence API",
        version="0.1.0",
        description=DISCLAIMER,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info")
    def info() -> dict[str, Any]:
        return {
            "disclaimer": DISCLAIMER,
            "pipeline": service.pipeline(depth_enabled=service.config.depth.enabled).describe(),
            "limits": {
                "max_image_bytes": service.config.limits.max_image_bytes,
                "max_image_side": service.config.limits.max_image_side,
            },
            "timeout_seconds": timeout_seconds,
            "runtime": runtime_info(),
        }

    @app.post("/v1/analyze")
    async def analyze(
        image: Annotated[UploadFile, File(description="JPEG, PNG, BMP or WebP image")],
        depth: bool = Query(True, description="Estimate relative depth"),
    ) -> JSONResponse:
        max_bytes = service.config.limits.max_image_bytes
        data = bytearray()
        while chunk := await image.read(_READ_CHUNK):
            data.extend(chunk)
            if len(data) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds the {max_bytes}-byte limit",
                )

        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(service.analyse, bytes(data), depth),
                timeout=timeout_seconds,
            )
        except InputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail=f"Analysis exceeded {timeout_seconds:.0f} seconds",
            ) from exc
        return JSONResponse(payload)

    return app


app = create_app()
