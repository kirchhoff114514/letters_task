from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MIN_AREA = 900
DEFAULT_MAX_AREA = 30000
DEFAULT_MIN_ASPECT = 0.6
DEFAULT_MAX_ASPECT = 1.5
DEFAULT_MIN_WIDTH = 22
DEFAULT_MIN_HEIGHT = 22
DEFAULT_MAX_WIDTH = 140
DEFAULT_MAX_HEIGHT = 140
DEFAULT_IOU_THRESHOLD = 0.35
DEFAULT_MIN_SATURATION = 65
DEFAULT_MIN_CENTER_X_FRAC = 0.06
DEFAULT_MAX_CENTER_X_FRAC = 0.85
DEFAULT_MIN_CENTER_Y_FRAC = 0.20
DEFAULT_MAX_CENTER_Y_FRAC = 0.95
DEFAULT_MIN_VALUE = 40
DEFAULT_BORDER_MARGIN_FRAC = 0.10
DEFAULT_BORDER_MAX_ASPECT = 3.25
FIXED_LAYOUT_SLOT_HALF_SIZE_FRAC = 0.05
FIXED_LAYOUT_MIN_SLOT_CANDIDATES = 10
FIXED_LAYOUT_SLOTS = (
    (0.188, 0.610),
    (0.248, 0.612),
    (0.228, 0.690),
    (0.305, 0.536),
    (0.328, 0.470),
    (0.326, 0.960),
    (0.425, 0.958),
    (0.505, 0.958),
    (0.590, 0.958),
    (0.676, 0.958),
    (0.674, 0.472),
    (0.730, 0.552),
    (0.786, 0.551),
    (0.732, 0.646),
    (0.788, 0.646),
    (0.754, 0.799),
)


def order_boxes(boxes: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(boxes, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


def center_in_roi(
    bbox: list[int],
    image_shape: tuple[int, int, int],
    *,
    min_center_x_frac: float,
    max_center_x_frac: float,
    min_center_y_frac: float,
    max_center_y_frac: float,
) -> bool:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (
        min_center_x_frac * width <= cx <= max_center_x_frac * width
        and min_center_y_frac * height <= cy <= max_center_y_frac * height
    )


def passes_box_filters(
    contour: np.ndarray,
    image_shape: tuple[int, int, int],
    *,
    min_area: int,
    max_area: int,
    min_aspect: float,
    max_aspect: float,
    min_width: int,
    min_height: int,
    max_width: int,
    max_height: int,
    min_center_x_frac: float,
    max_center_x_frac: float,
    min_center_y_frac: float,
    max_center_y_frac: float,
) -> dict[str, object] | None:
    height, width = image_shape[:2]
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    if w < min_width or h < min_height or w > max_width or h > max_height:
        return None

    rect = cv2.minAreaRect(contour)
    rect_w, rect_h = rect[1]
    if rect_w <= 0 or rect_h <= 0:
        return None

    normalized_aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
    border_margin_x = width * DEFAULT_BORDER_MARGIN_FRAC
    border_margin_y = height * DEFAULT_BORDER_MARGIN_FRAC
    touches_border = (
        x <= border_margin_x
        or y <= border_margin_y
        or x + w >= width - border_margin_x
        or y + h >= height - border_margin_y
    )
    max_allowed_aspect = max_aspect
    if touches_border:
        max_allowed_aspect = max(max_allowed_aspect, DEFAULT_BORDER_MAX_ASPECT)

    if normalized_aspect > max_allowed_aspect:
        return None
    if touches_border and normalized_aspect > max_aspect and min(w, h) < 30:
        return None

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
    if len(approx) < 4 or len(approx) > 12:
        return None

    bbox = [int(x), int(y), int(x + w), int(y + h)]
    if not center_in_roi(
        bbox,
        image_shape,
        min_center_x_frac=min_center_x_frac,
        max_center_x_frac=max_center_x_frac,
        min_center_y_frac=min_center_y_frac,
        max_center_y_frac=max_center_y_frac,
    ):
        return None

    return {
        "bbox": bbox,
        "area": float(area),
        "aspect_ratio": round(normalized_aspect, 3),
    }


def extract_edge_candidates(
    gray: np.ndarray,
    image_shape: tuple[int, int, int],
    *,
    min_area: int,
    max_area: int,
    min_aspect: float,
    max_aspect: float,
    min_width: int,
    min_height: int,
    max_width: int,
    max_height: int,
    min_center_x_frac: float,
    max_center_x_frac: float,
    min_center_y_frac: float,
    max_center_y_frac: float,
) -> list[dict[str, object]]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 170)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results: list[dict[str, object]] = []
    for contour in contours:
        item = passes_box_filters(
            contour,
            image_shape,
            min_area=min_area,
            max_area=max_area,
            min_aspect=min_aspect,
            max_aspect=max_aspect,
            min_width=min_width,
            min_height=min_height,
            max_width=max_width,
            max_height=max_height,
            min_center_x_frac=min_center_x_frac,
            max_center_x_frac=max_center_x_frac,
            min_center_y_frac=min_center_y_frac,
            max_center_y_frac=max_center_y_frac,
        )
        if item is not None:
            item["source"] = "edge"
            results.append(item)
    return results


def extract_saturation_candidates(
    image: np.ndarray,
    image_shape: tuple[int, int, int],
    *,
    min_area: int,
    max_area: int,
    min_aspect: float,
    max_aspect: float,
    min_width: int,
    min_height: int,
    max_width: int,
    max_height: int,
    min_center_x_frac: float,
    max_center_x_frac: float,
    min_center_y_frac: float,
    max_center_y_frac: float,
    min_saturation: int,
) -> list[dict[str, object]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(saturation, min_saturation, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results: list[dict[str, object]] = []
    for contour in contours:
        item = passes_box_filters(
            contour,
            image_shape,
            min_area=min_area,
            max_area=max_area,
            min_aspect=min_aspect,
            max_aspect=max_aspect,
            min_width=min_width,
            min_height=min_height,
            max_width=max_width,
            max_height=max_height,
            min_center_x_frac=min_center_x_frac,
            max_center_x_frac=max_center_x_frac,
            min_center_y_frac=min_center_y_frac,
            max_center_y_frac=max_center_y_frac,
        )
        if item is not None:
            item["source"] = "saturation"
            results.append(item)
    return results


def extract_hue_candidates(
    image: np.ndarray,
    image_shape: tuple[int, int, int],
    *,
    min_area: int,
    max_area: int,
    min_aspect: float,
    max_aspect: float,
    min_width: int,
    min_height: int,
    max_width: int,
    max_height: int,
    min_center_x_frac: float,
    max_center_x_frac: float,
    min_center_y_frac: float,
    max_center_y_frac: float,
    min_saturation: int,
    min_value: int = DEFAULT_MIN_VALUE,
) -> list[dict[str, object]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    valid_sv = (saturation >= min_saturation) & (value >= min_value)

    color_ranges = (
        ("hue_red", (hue <= 12) | (hue >= 170)),
        ("hue_orange", (hue >= 8) & (hue <= 30)),
        ("hue_yellow", (hue >= 20) & (hue <= 42)),
        ("hue_green", (hue >= 35) & (hue <= 95)),
    )

    results: list[dict[str, object]] = []
    kernel = np.ones((3, 3), np.uint8)
    for source, hue_mask in color_ranges:
        mask = np.where(valid_sv & hue_mask, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            item = passes_box_filters(
                contour,
                image_shape,
                min_area=min_area,
                max_area=max_area,
                min_aspect=min_aspect,
                max_aspect=max_aspect,
                min_width=min_width,
                min_height=min_height,
                max_width=max_width,
                max_height=max_height,
                min_center_x_frac=min_center_x_frac,
                max_center_x_frac=max_center_x_frac,
                min_center_y_frac=min_center_y_frac,
                max_center_y_frac=max_center_y_frac,
            )
            if item is not None:
                item["source"] = source
                results.append(item)

    return results


def extract_fixed_layout_candidates(image: np.ndarray) -> list[dict[str, object]]:
    height, width = image.shape[:2]
    half_size = max(20, int(min(width, height) * FIXED_LAYOUT_SLOT_HALF_SIZE_FRAC))
    kernel = np.ones((3, 3), np.uint8)

    results: list[dict[str, object]] = []
    for center_x_frac, center_y_frac in FIXED_LAYOUT_SLOTS:
        center_x = int(center_x_frac * width)
        center_y = int(center_y_frac * height)
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(width, center_x + half_size)
        y2 = min(height, center_y + half_size)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        border_pixels = np.concatenate([crop[0, :, :], crop[-1, :, :], crop[:, 0, :], crop[:, -1, :]], axis=0)
        background_color = np.median(border_pixels, axis=0)
        color_diff = np.linalg.norm(crop.astype(np.float32) - background_color.astype(np.float32), axis=2)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        border_gray = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        background_gray = float(np.median(border_gray))
        gray_diff = np.abs(gray.astype(np.float32) - background_gray)

        mask = ((color_diff > 22) & ((hsv[:, :, 1] > 18) | (gray_diff > 18) | (hsv[:, :, 2] > 150))).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 40:
            continue

        x, y, box_width, box_height = cv2.boundingRect(contour)
        bbox = [x1 + int(x), y1 + int(y), x1 + int(x + box_width), y1 + int(y + box_height)]
        results.append(
            {
                "bbox": bbox,
                "area": float(area),
                "aspect_ratio": round(max(box_width, box_height) / max(1, min(box_width, box_height)), 3),
                "source": "slot",
            }
        )

    return results


def looks_like_fixed_layout(image: np.ndarray) -> bool:
    height, width = image.shape[:2]
    x1 = int(width * 0.12)
    x2 = int(width * 0.18)
    y1 = int(height * 0.45)
    y2 = int(height * 0.55)
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) >= 30.0

def contains_box(outer_box: list[int], inner_box: list[int]) -> bool:
    ox1, oy1, ox2, oy2 = outer_box
    ix1, iy1, ix2, iy2 = inner_box
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def prune_container_boxes(boxes: list[dict[str, object]]) -> list[dict[str, object]]:
    pruned: list[dict[str, object]] = []
    for item in boxes:
        container_children = [
            other
            for other in boxes
            if other is not item
            and other["area"] < item["area"]
            and contains_box(item["bbox"], other["bbox"])
            and other.get("source") != item.get("source")
        ]
        if len(container_children) >= 2:
            covered_area = sum(float(child["area"]) for child in container_children)
            if covered_area >= 0.7 * float(item["area"]):
                continue
        pruned.append(item)
    return pruned


def merge_boxes(boxes: list[dict[str, object]], iou_threshold: float) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for item in sorted(prune_container_boxes(boxes), key=lambda entry: entry["area"], reverse=True):
        if any(bbox_iou(item["bbox"], existing["bbox"]) >= iou_threshold for existing in merged):
            continue
        merged.append(item)
    return order_boxes(merged)


def detect_letter_blocks(
    image_path: str,
    *,
    min_area: int = DEFAULT_MIN_AREA,
    max_area: int = DEFAULT_MAX_AREA,
    min_aspect: float = DEFAULT_MIN_ASPECT,
    max_aspect: float = DEFAULT_MAX_ASPECT,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
    max_width: int = DEFAULT_MAX_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    min_saturation: int = DEFAULT_MIN_SATURATION,
    min_center_x_frac: float = DEFAULT_MIN_CENTER_X_FRAC,
    max_center_x_frac: float = DEFAULT_MAX_CENTER_X_FRAC,
    min_center_y_frac: float = DEFAULT_MIN_CENTER_Y_FRAC,
    max_center_y_frac: float = DEFAULT_MAX_CENTER_Y_FRAC,
) -> dict[str, object]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    fixed_layout_candidates = []
    if looks_like_fixed_layout(image):
        fixed_layout_candidates = extract_fixed_layout_candidates(image)
    if len(fixed_layout_candidates) >= FIXED_LAYOUT_MIN_SLOT_CANDIDATES:
        merged_boxes = order_boxes(fixed_layout_candidates)
        return {
            "image_path": str(Path(image_path).resolve()),
            "num_boxes": len(merged_boxes),
            "boxes": merged_boxes,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_candidates = extract_edge_candidates(
        gray,
        image.shape,
        min_area=min_area,
        max_area=max_area,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        min_width=min_width,
        min_height=min_height,
        max_width=max_width,
        max_height=max_height,
        min_center_x_frac=min_center_x_frac,
        max_center_x_frac=max_center_x_frac,
        min_center_y_frac=min_center_y_frac,
        max_center_y_frac=max_center_y_frac,
    )
    saturation_candidates = extract_saturation_candidates(
        image,
        image.shape,
        min_area=min_area,
        max_area=max_area,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        min_width=min_width,
        min_height=min_height,
        max_width=max_width,
        max_height=max_height,
        min_center_x_frac=min_center_x_frac,
        max_center_x_frac=max_center_x_frac,
        min_center_y_frac=min_center_y_frac,
        max_center_y_frac=max_center_y_frac,
        min_saturation=min_saturation,
    )
    hue_candidates = extract_hue_candidates(
        image,
        image.shape,
        min_area=min_area,
        max_area=max_area,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        min_width=min_width,
        min_height=min_height,
        max_width=max_width,
        max_height=max_height,
        min_center_x_frac=min_center_x_frac,
        max_center_x_frac=max_center_x_frac,
        min_center_y_frac=min_center_y_frac,
        max_center_y_frac=max_center_y_frac,
        min_saturation=min_saturation,
    )

    merged_boxes = merge_boxes(edge_candidates + saturation_candidates + hue_candidates, iou_threshold)
    return {
        "image_path": str(Path(image_path).resolve()),
        "num_boxes": len(merged_boxes),
        "boxes": merged_boxes,
    }


def save_overlay(image_path: str, boxes: list[dict[str, object]], output_path: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    for index, item in enumerate(boxes, start=1):
        x1, y1, x2, y2 = item["bbox"]
        source = item.get("source", "mix")
        color = (0, 255, 0) if source == "edge" else (255, 180, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            str(index),
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect candidate letter blocks using OpenCV contour and color cues.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--output-json", required=True, help="Path to write detection JSON.")
    parser.add_argument("--output-overlay", required=True, help="Path to write the overlay image.")
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA)
    parser.add_argument("--max-area", type=int, default=DEFAULT_MAX_AREA)
    parser.add_argument("--min-aspect", type=float, default=DEFAULT_MIN_ASPECT)
    parser.add_argument("--max-aspect", type=float, default=DEFAULT_MAX_ASPECT)
    parser.add_argument("--min-width", type=int, default=DEFAULT_MIN_WIDTH)
    parser.add_argument("--min-height", type=int, default=DEFAULT_MIN_HEIGHT)
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH)
    parser.add_argument("--max-height", type=int, default=DEFAULT_MAX_HEIGHT)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--min-saturation", type=int, default=DEFAULT_MIN_SATURATION)
    parser.add_argument("--min-center-x-frac", type=float, default=DEFAULT_MIN_CENTER_X_FRAC)
    parser.add_argument("--max-center-x-frac", type=float, default=DEFAULT_MAX_CENTER_X_FRAC)
    parser.add_argument("--min-center-y-frac", type=float, default=DEFAULT_MIN_CENTER_Y_FRAC)
    parser.add_argument("--max-center-y-frac", type=float, default=DEFAULT_MAX_CENTER_Y_FRAC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = detect_letter_blocks(
        args.image,
        min_area=args.min_area,
        max_area=args.max_area,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        min_width=args.min_width,
        min_height=args.min_height,
        max_width=args.max_width,
        max_height=args.max_height,
        iou_threshold=args.iou_threshold,
        min_saturation=args.min_saturation,
        min_center_x_frac=args.min_center_x_frac,
        max_center_x_frac=args.max_center_x_frac,
        min_center_y_frac=args.min_center_y_frac,
        max_center_y_frac=args.max_center_y_frac,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_overlay(args.image, result["boxes"], args.output_overlay)

    print(
        json.dumps(
            {
                "image_path": result["image_path"],
                "num_boxes": result["num_boxes"],
                "output_json": str(output_json.resolve()),
                "output_overlay": str(Path(args.output_overlay).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

