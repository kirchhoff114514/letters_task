from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance


BBox = tuple[int, int, int, int]
RED = (255, 0, 0)
BLUE = (0, 102, 255)


@dataclass(frozen=True)
class StepGuidanceRender:
    annotated_image: Image.Image
    pick_bbox: BBox
    place_bbox: BBox


def render_step_guidance(
    *,
    image: Image.Image | str | Path,
    detections: dict[str, Any],
    source_box_id: int,
    placement_id: int,
    background_dim_factor: float = 0.6,
    outline_width: int = 4,
) -> StepGuidanceRender:
    """Render a guidance image for the current step.

    Non-target regions are dimmed, while the selected pick block and placement
    slot are restored to their original appearance and outlined.
    """
    if not 0.0 <= background_dim_factor <= 1.0:
        raise ValueError("background_dim_factor must be between 0.0 and 1.0")
    if outline_width < 1:
        raise ValueError("outline_width must be >= 1")

    base_image = _load_image(image).convert("RGB")
    pick_bbox = _extract_bbox(_resolve_box(detections.get("boxes", []), source_box_id))
    place_bbox = _extract_bbox(_resolve_placement(detections.get("placements", []), placement_id))

    dimmed = ImageEnhance.Brightness(base_image).enhance(background_dim_factor)
    annotated = dimmed.copy()
    annotated.paste(base_image.crop(pick_bbox), pick_bbox)
    annotated.paste(base_image.crop(place_bbox), place_bbox)

    draw = ImageDraw.Draw(annotated)
    _draw_region(draw, pick_bbox, RED, outline_width, f"PICK {source_box_id}")
    _draw_region(draw, place_bbox, BLUE, outline_width, f"PLACE {placement_id}")

    return StepGuidanceRender(
        annotated_image=annotated,
        pick_bbox=pick_bbox,
        place_bbox=place_bbox,
    )


def _load_image(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.copy()
    return Image.open(image)


def _resolve_box(boxes: list[dict[str, Any]], source_box_id: int) -> dict[str, Any]:
    if not boxes:
        raise ValueError("detections.boxes is empty")
    return _resolve_by_id(
        items=boxes,
        target_id=source_box_id,
        explicit_id_keys=("id", "box_id", "source_box_id"),
        collection_name="boxes",
    )


def _resolve_placement(placements: list[dict[str, Any]], placement_id: int) -> dict[str, Any]:
    if not placements:
        raise ValueError("detections.placements is empty")
    return _resolve_by_id(
        items=placements,
        target_id=placement_id,
        explicit_id_keys=("id", "placement_id", "slot_index"),
        collection_name="placements",
    )


def _resolve_by_id(
    *,
    items: list[dict[str, Any]],
    target_id: int,
    explicit_id_keys: tuple[str, ...],
    collection_name: str,
) -> dict[str, Any]:
    for item in items:
        if any(item.get(key) == target_id for key in explicit_id_keys):
            return item

    one_based_index = target_id - 1
    if 0 <= one_based_index < len(items):
        return items[one_based_index]

    if target_id == 0 and items:
        return items[0]

    raise ValueError(f"{collection_name} does not contain id {target_id}")


def _extract_bbox(item: dict[str, Any]) -> BBox:
    bbox = item.get("bbox")
    if bbox is None or len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox}")
    return tuple(int(value) for value in bbox)


def _draw_region(
    draw: ImageDraw.ImageDraw,
    bbox: BBox,
    color: tuple[int, int, int],
    outline_width: int,
    label: str,
) -> None:
    draw.rectangle(bbox, outline=color, width=outline_width)
    text_anchor = (bbox[0], max(0, bbox[1] - 14))
    draw.text(text_anchor, label, fill=color)
