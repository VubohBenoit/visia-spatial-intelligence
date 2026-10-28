"""Draw detections, the walking corridor, optional depth overlay and alerts onto an image."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visia_spatial_intelligence.schemas import Alert, CorridorRegion, DepthMap, FrameResult

_RISK_COLOURS: dict[str, tuple[int, int, int]] = {
    "none": (150, 150, 150),
    "low": (60, 179, 113),
    "medium": (255, 165, 0),
    "high": (220, 20, 60),
}
_CORRIDOR_COLOUR = (65, 105, 225)


def _font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    return ImageFont.load_default()


def depth_overlay(image: Image.Image, depth: DepthMap, alpha: float = 0.35) -> Image.Image:
    """Blend a false-colour relative-depth map over the image (warmer = relatively closer)."""
    values = np.asarray(depth.values, dtype=np.float32)
    red = (values * 255).astype(np.uint8)
    blue = ((1.0 - values) * 255).astype(np.uint8)
    green = np.zeros_like(red)
    colour = Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")
    return Image.blend(image.convert("RGB"), colour, alpha)


def render_frame(
    image: Image.Image,
    result: FrameResult,
    corridor: CorridorRegion,
    depth: DepthMap | None = None,
    show_depth: bool = False,
) -> Image.Image:
    """Return an annotated RGB copy of ``image``. The input is never modified."""
    base = image.convert("RGB")
    if show_depth and depth is not None:
        base = depth_overlay(base, depth)

    canvas = base.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = _font()

    corners = corridor.to_polygon(result.width, result.height)
    draw.polygon(corners, fill=(*_CORRIDOR_COLOUR, 40), outline=(*_CORRIDOR_COLOUR, 255))

    # Assessments are produced in the same order as detections (see SpatialReasoner.assess).
    # Without reasoning, every box is drawn in the neutral "none" colour.
    risks = [a.risk.value for a in result.assessments] or ["none"] * len(result.detections)
    for det, risk in zip(result.detections, risks, strict=True):
        colour = _RISK_COLOURS[risk]
        box = (det.box.x1, det.box.y1, det.box.x2, det.box.y2)
        draw.rectangle(box, outline=(*colour, 255), width=3)
        prefix = f"#{det.track_id} " if det.track_id is not None else ""
        text = f"{prefix}{det.label} {det.score:.2f}"
        text_w = draw.textlength(text, font=font)
        top = max(det.box.y1 - 14, 0)
        draw.rectangle((det.box.x1, top, det.box.x1 + text_w + 6, top + 14), fill=(*colour, 230))
        draw.text((det.box.x1 + 3, top + 1), text, fill=(255, 255, 255, 255), font=font)

    _draw_alert_banner(draw, canvas.width, result.alerts, font)
    return canvas


def _draw_alert_banner(
    draw: ImageDraw.ImageDraw,
    width: int,
    alerts: list[Alert],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    if not alerts:
        return
    lines = [f"ALERT [{a.risk.value}] {a.message}" for a in alerts[:3]]
    height = 16 * len(lines) + 8
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0, 170))
    for i, line in enumerate(lines):
        draw.text((8, 4 + 16 * i), line, fill=(255, 255, 255, 255), font=font)
