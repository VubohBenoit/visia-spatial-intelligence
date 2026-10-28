"""Streamlit demo: upload an image, inspect detections, depth, corridor and alert reasoning.

Run with::

    uv run --extra app streamlit run app/streamlit_app.py

This is a research prototype. It is not a certified navigation or safety system.
"""

from __future__ import annotations

import io
import json
from dataclasses import replace

import streamlit as st

from visia_spatial_intelligence.cli import DISCLAIMER, runtime_info
from visia_spatial_intelligence.config import load_config
from visia_spatial_intelligence.detection.detector import ModelUnavailableError
from visia_spatial_intelligence.pipeline import VisiaPipeline
from visia_spatial_intelligence.validation import InputValidationError, load_image_bytes
from visia_spatial_intelligence.visualization.renderer import render_frame


@st.cache_resource(show_spinner=False)
def get_pipeline(depth_enabled: bool) -> VisiaPipeline:
    """Build the pipeline once per depth setting so loaded weights are reused."""
    config = load_config()
    config = config.with_overrides(depth=replace(config.depth, enabled=depth_enabled))
    return VisiaPipeline.from_config(config)


def _risk_rows(result) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
    return [
        {
            "label": a.label,
            "track": a.track_id,
            "risk": a.risk.value,
            "corridor overlap": round(a.corridor_overlap, 2),
            "bottom-centre in corridor": a.bottom_center_in_corridor,
            "relative depth": None
            if a.relative_proximity is None
            else round(a.relative_proximity, 2),
            "reasons": "; ".join(a.reasons),
        }
        for a in result.assessments
    ]


def main() -> None:
    st.set_page_config(page_title="VisIA Spatial Intelligence", layout="wide")
    st.title("VisIA Spatial Intelligence")
    st.warning(
        "Research prototype. Not a certified navigation, mobility or safety system. "
        "Do not rely on it to avoid hazards."
    )

    with st.sidebar:
        st.header("Options")
        depth_enabled = st.checkbox("Estimate relative depth", value=True)
        show_depth = st.checkbox("Overlay relative depth", value=False, disabled=not depth_enabled)
        st.caption("Images are limited to 10 MB and 4096 px on the longest side.")

    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if upload is None:
        st.info("Upload a street-scene image to start. A sample is in data/samples/.")
        return

    config = load_config()
    try:
        image = load_image_bytes(upload.getvalue(), config.limits)
    except InputValidationError as exc:
        st.error(f"Invalid input: {exc}")
        return

    pipeline = get_pipeline(depth_enabled)
    pipeline.reset()
    with st.spinner("Running detection (and depth) on CPU. This can take several seconds..."):
        try:
            output = pipeline.analyze(image, frame_index=0)
        except ModelUnavailableError as exc:
            st.error(
                f"A required model is unavailable: {exc}. "
                "Check your network connection, or the model ID in configs/default.yaml."
            )
            return

    result = output.result
    annotated = render_frame(
        image, result, config.reasoning.corridor, depth=output.depth, show_depth=show_depth
    )

    left, right = st.columns(2)
    left.subheader("Input")
    left.image(image, use_container_width=True)
    right.subheader("Annotated output")
    right.image(annotated, use_container_width=True)

    for feature, reason in result.disabled_features.items():
        st.info(f"Feature disabled: {feature}. Reason: {reason}")

    st.subheader("Alerts")
    if result.alerts:
        for alert in result.alerts:
            st.error(f"[{alert.risk.value}] {alert.message}")
            st.caption("Reasons: " + "; ".join(alert.reasons))
    else:
        st.success(
            "No alerts. A single still image cannot show that an object persists over time, "
            "so alerts are never issued for photos. The per-object reasons below show what the "
            "system observed."
        )

    st.subheader("Detections and spatial reasoning")
    st.dataframe(_risk_rows(result), use_container_width=True)

    st.subheader("Processing time and configuration")
    timings = result.timings.to_dict()
    st.json(timings)
    with st.expander("Model and pipeline configuration"):
        st.json(pipeline.describe())
        st.json(runtime_info())

    report = {
        "disclaimer": DISCLAIMER,
        "input": {"file": upload.name, "width": image.width, "height": image.height},
        "pipeline": pipeline.describe(),
        "runtime": runtime_info(),
        "result": result.to_dict(),
    }
    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download annotated image (PNG)", buffer.getvalue(), "annotated.png", "image/png"
    )
    d2.download_button(
        "Download metadata (JSON)",
        json.dumps(report, indent=2).encode("utf-8"),
        "result.json",
        "application/json",
    )


main()
