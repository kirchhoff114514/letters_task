from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TABLE_LEFT = 0.10
DEFAULT_TABLE_TOP = 0.18
DEFAULT_TABLE_RIGHT = 0.90
DEFAULT_TABLE_BOTTOM = 0.96
DEFAULT_MIN_WIDTH = 18
DEFAULT_MIN_HEIGHT = 18
DEFAULT_MAX_WIDTH = 120
DEFAULT_MAX_HEIGHT = 120
DEFAULT_MAX_ASPECT = 2.2
DEFAULT_MIN_AREA = 250
DEFAULT_MAX_AREA = 9000
DEFAULT_MIN_CONTRAST = 10.0
DEFAULT_MIN_FILL_RATIO = 0.20
DEFAULT_IOU_THRESHOLD = 0.35


def build_table_roi(
    image_shape: tuple[int, int, int],
    *,
    left_frac: float,
    top_frac: float,
    right_frac: float,
    bottom_frac: float,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1 = max(0, int(width * left_frac))
    y1 = max(0, int(height * top_frac))
    x2 = min(width, int(width * right_frac))
    y2 = min(height, int(height * bottom_frac))
    return x1, y1, x2, y2


def estimate_background(image_roi: np.ndarray) -> np.ndarray:
    # A heavy blur is enough for this mostly flat tabletop scene.
    kernel = max(31, ((min(image_roi.shape[:2]) // 6) | 1))
    return cv2.GaussianBlur(image_roi, (kernel, kernel), 0)


def build_foreground_mask(image_roi: np.ndarray, background: np.ndarray) -> np.ndarray:
    lab_image = cv2.cvtColor(image_roi, cv2.COLOR_BGR2LAB)
    lab_background = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)
    color_diff = cv2.absdiff(lab_image, lab_background)
    color_score = color_diff[:, :, 1].astype(np.uint16) + color_diff[:, :, 2].astype(np.uint16)

    gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    gray_diff = cv2.absdiff(gray, bg_gray)

    edges = cv2.Canny(gray, 40, 130)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    mask = np.where((color_score >= 18) | (gray_diff >= 15) | (edges > 0), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return mask


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


def extract_candidates(mask: np.ndarray, offset: tuple[int, int]) -> list[dict[str, object]]:
    x_offset, y_offset = offset
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    results: list[dict[str, object]] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area <= 0:
            continue
        bbox = [
            int(x + x_offset),
            int(y + y_offset),
            int(x + width + x_offset),
            int(y + height + y_offset),
        ]
        results.append(
            {
                "bbox": bbox,
                "area": float(area),
                "width": int(width),
                "height": int(height),
                "fill_ratio": round(float(area) / max(1, width * height), 3),
            }
        )
    return results


def box_ring(
    image: np.ndarray,
    bbox: list[int],
    *,
    padding: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    x1, y1, x2, y2 = bbox
    height, width = image.shape[:2]
    px1 = max(0, x1 - padding)
    py1 = max(0, y1 - padding)
    px2 = min(width, x2 + padding)
    py2 = min(height, y2 + padding)
    if px1 >= x1 or py1 >= y1 or px2 <= x2 or py2 <= y2:
        return None

    outer = image[py1:py2, px1:px2]
    inner = image[y1:y2, x1:x2]
    ring_mask = np.ones(outer.shape[:2], dtype=np.uint8)
    ring_mask[y1 - py1 : y2 - py1, x1 - px1 : x2 - px1] = 0
    return inner, outer[ring_mask == 1]


def contrast_score(image: np.ndarray, bbox: list[int]) -> float:
    samples = box_ring(image, bbox, padding=8)
    if samples is None:
        return 0.0
    inner, ring = samples
    if inner.size == 0 or ring.size == 0:
        return 0.0
    inner_lab = cv2.cvtColor(inner, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    ring_lab = cv2.cvtColor(ring.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    inner_mean = inner_lab.mean(axis=0)
    ring_mean = ring_lab.mean(axis=0)
    return float(np.linalg.norm(inner_mean - ring_mean))


def filter_candidates(
    image: np.ndarray,
    boxes: list[dict[str, object]],
    *,
    min_width: int,
    min_height: int,
    max_width: int,
    max_height: int,
    min_area: int,
    max_area: int,
    max_aspect: float,
    min_fill_ratio: float,
    min_contrast: float,
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for item in boxes:
        width = int(item["width"])
        height = int(item["height"])
        area = float(item["area"])
        fill_ratio = float(item["fill_ratio"])
        if width < min_width or height < min_height:
            continue
        if width > max_width or height > max_height:
            continue
        if area < min_area or area > max_area:
            continue
        aspect = max(width, height) / max(1, min(width, height))
        if aspect > max_aspect:
            continue
        if fill_ratio < min_fill_ratio:
            continue

        bbox = item["bbox"]
        score = contrast_score(image, bbox)
        if score < min_contrast:
            continue

        item["contrast"] = round(score, 2)
        item["aspect_ratio"] = round(aspect, 3)
        filtered.append(item)
    return filtered


def merge_candidates(boxes: list[dict[str, object]], iou_threshold: float) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for item in sorted(boxes, key=lambda entry: (entry["contrast"], entry["area"]), reverse=True):
        if any(bbox_iou(item["bbox"], existing["bbox"]) >= iou_threshold for existing in merged):
            continue
        merged.append(item)
    return sorted(merged, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def detect_letter_blocks(
    image_path: str,
    *,
    table_left: float = DEFAULT_TABLE_LEFT,
    table_top: float = DEFAULT_TABLE_TOP,
    table_right: float = DEFAULT_TABLE_RIGHT,
    table_bottom: float = DEFAULT_TABLE_BOTTOM,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
    max_width: int = DEFAULT_MAX_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
    min_area: int = DEFAULT_MIN_AREA,
    max_area: int = DEFAULT_MAX_AREA,
    max_aspect: float = DEFAULT_MAX_ASPECT,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
    min_contrast: float = DEFAULT_MIN_CONTRAST,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, object]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    x1, y1, x2, y2 = build_table_roi(
        image.shape,
        left_frac=table_left,
        top_frac=table_top,
        right_frac=table_right,
        bottom_frac=table_bottom,
    )
    image_roi = image[y1:y2, x1:x2]
    background = estimate_background(image_roi)
    mask = build_foreground_mask(image_roi, background)
    candidates = extract_candidates(mask, (x1, y1))
    filtered = filter_candidates(
        image,
        candidates,
        min_width=min_width,
        min_height=min_height,
        max_width=max_width,
        max_height=max_height,
        min_area=min_area,
        max_area=max_area,
        max_aspect=max_aspect,
        min_fill_ratio=min_fill_ratio,
        min_contrast=min_contrast,
    )
    merged = merge_candidates(filtered, iou_threshold=iou_threshold)
    return {
        "image_path": str(Path(image_path).resolve()),
        "table_roi": [x1, y1, x2, y2],
        "num_boxes": len(merged),
        "boxes": merged,
    }


def save_overlay(image_path: str, boxes: list[dict[str, object]], output_path: str, table_roi: list[int]) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    rx1, ry1, rx2, ry2 = table_roi
    cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (120, 120, 120), 1)

    for index, item in enumerate(boxes, start=1):
        x1, y1, x2, y2 = item["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(
            image,
            str(index),
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect letter blocks in a fixed tabletop scene.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--output-json", required=True, help="Path to write detection JSON.")
    parser.add_argument("--output-overlay", required=True, help="Path to write overlay image.")
    parser.add_argument("--table-left", type=float, default=DEFAULT_TABLE_LEFT)
    parser.add_argument("--table-top", type=float, default=DEFAULT_TABLE_TOP)
    parser.add_argument("--table-right", type=float, default=DEFAULT_TABLE_RIGHT)
    parser.add_argument("--table-bottom", type=float, default=DEFAULT_TABLE_BOTTOM)
    parser.add_argument("--min-width", type=int, default=DEFAULT_MIN_WIDTH)
    parser.add_argument("--min-height", type=int, default=DEFAULT_MIN_HEIGHT)
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH)
    parser.add_argument("--max-height", type=int, default=DEFAULT_MAX_HEIGHT)
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA)
    parser.add_argument("--max-area", type=int, default=DEFAULT_MAX_AREA)
    parser.add_argument("--max-aspect", type=float, default=DEFAULT_MAX_ASPECT)
    parser.add_argument("--min-fill-ratio", type=float, default=DEFAULT_MIN_FILL_RATIO)
    parser.add_argument("--min-contrast", type=float, default=DEFAULT_MIN_CONTRAST)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = detect_letter_blocks(
        args.image,
        table_left=args.table_left,
        table_top=args.table_top,
        table_right=args.table_right,
        table_bottom=args.table_bottom,
        min_width=args.min_width,
        min_height=args.min_height,
        max_width=args.max_width,
        max_height=args.max_height,
        min_area=args.min_area,
        max_area=args.max_area,
        max_aspect=args.max_aspect,
        min_fill_ratio=args.min_fill_ratio,
        min_contrast=args.min_contrast,
        iou_threshold=args.iou_threshold,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_overlay(args.image, result["boxes"], args.output_overlay, result["table_roi"])

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
