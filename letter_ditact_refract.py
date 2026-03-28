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
DEFAULT_BOARD_LEFT = 0.24
DEFAULT_BOARD_TOP = 0.42
DEFAULT_BOARD_RIGHT = 0.76
DEFAULT_BOARD_BOTTOM = 0.84
DEFAULT_BOARD_MIN_AREA_FRAC = 0.020
DEFAULT_BOARD_MAX_AREA_FRAC = 0.14
DEFAULT_BOARD_MIN_ASPECT = 1.35
DEFAULT_BOARD_MAX_ASPECT = 3.2
DEFAULT_NUM_PLACEMENTS = 5


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


def build_fractional_roi(
    image_shape: tuple[int, int, int],
    *,
    left_frac: float,
    top_frac: float,
    right_frac: float,
    bottom_frac: float,
) -> tuple[int, int, int, int]:
    return build_table_roi(
        image_shape,
        left_frac=left_frac,
        top_frac=top_frac,
        right_frac=right_frac,
        bottom_frac=bottom_frac,
    )


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


def order_rotated_box_points(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def board_payload_from_points(points: np.ndarray, area: float) -> dict[str, object]:
    ordered_points = order_rotated_box_points(points)
    center = ordered_points.mean(axis=0)
    top_edge = ordered_points[1] - ordered_points[0]
    left_edge = ordered_points[3] - ordered_points[0]
    width = float(np.linalg.norm(top_edge))
    height = float(np.linalg.norm(left_edge))
    long_side = max(width, height)
    short_side = min(width, height)
    angle = float(np.degrees(np.arctan2(top_edge[1], top_edge[0])))
    bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(np.round(ordered_points).astype(np.int32))
    return {
        "center": [round(float(center[0]), 2), round(float(center[1]), 2)],
        "size": [round(long_side, 2), round(short_side, 2)],
        "angle": round(angle, 2),
        "aspect_ratio": round(long_side / max(1.0, short_side), 3),
        "area": round(float(area), 2),
        "bbox": [int(bbox_x), int(bbox_y), int(bbox_x + bbox_w), int(bbox_y + bbox_h)],
        "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in ordered_points],
    }


def fallback_cutting_board(search_roi: tuple[int, int, int, int]) -> dict[str, object]:
    search_x1, search_y1, search_x2, search_y2 = search_roi
    roi_width = search_x2 - search_x1
    roi_height = search_y2 - search_y1
    board_width = roi_width * 0.58
    board_height = roi_height * 0.36
    center_x = (search_x1 + search_x2) / 2.0
    center_y = search_y1 + roi_height * 0.60
    half_w = board_width / 2.0
    half_h = board_height / 2.0
    points = np.array(
        [
            [center_x - half_w, center_y - half_h],
            [center_x + half_w, center_y - half_h],
            [center_x + half_w, center_y + half_h],
            [center_x - half_w, center_y + half_h],
        ],
        dtype=np.float32,
    )
    return board_payload_from_points(points, area=board_width * board_height)


def detect_cutting_board(
    image: np.ndarray,
    *,
    left_frac: float = DEFAULT_BOARD_LEFT,
    top_frac: float = DEFAULT_BOARD_TOP,
    right_frac: float = DEFAULT_BOARD_RIGHT,
    bottom_frac: float = DEFAULT_BOARD_BOTTOM,
    min_area_frac: float = DEFAULT_BOARD_MIN_AREA_FRAC,
    max_area_frac: float = DEFAULT_BOARD_MAX_AREA_FRAC,
    min_aspect: float = DEFAULT_BOARD_MIN_ASPECT,
    max_aspect: float = DEFAULT_BOARD_MAX_ASPECT,
) -> dict[str, object] | None:
    search_x1, search_y1, search_x2, search_y2 = build_fractional_roi(
        image.shape,
        left_frac=left_frac,
        top_frac=top_frac,
        right_frac=right_frac,
        bottom_frac=bottom_frac,
    )
    search = image[search_y1:search_y2, search_x1:search_x2]
    if search.size == 0:
        return None

    background = estimate_background(search)
    lab_search = cv2.cvtColor(search, cv2.COLOR_BGR2LAB)
    lab_background = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)
    diff = cv2.absdiff(lab_search, lab_background)
    color_score = diff[:, :, 1].astype(np.uint16) + diff[:, :, 2].astype(np.uint16)

    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    gray_diff = cv2.absdiff(gray, bg_gray)
    edges = cv2.Canny(gray, 45, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]
    wood_like = (
        (hue >= 6)
        & (hue <= 24)
        & (saturation >= 35)
        & (saturation <= 170)
        & (value >= 85)
    )
    bright_wood = (
        (hue >= 4)
        & (hue <= 28)
        & (saturation >= 20)
        & (saturation <= 140)
        & (value >= 120)
    )
    mask = np.where(
        wood_like | bright_wood | ((color_score >= 14) & (gray_diff >= 10) & (value >= 90)),
        255,
        0,
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback_cutting_board((search_x1, search_y1, search_x2, search_y2))

    image_area = float(image.shape[0] * image.shape[1])
    min_area = image_area * min_area_frac
    max_area = image_area * max_area_frac
    search_center = np.array([(search_x2 + search_x1) / 2.0, (search_y2 + search_y1) / 2.0], dtype=np.float32)
    target_area = image_area * 0.055
    target_aspect = 1.85

    best: dict[str, object] | None = None
    best_score = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), _ = rect
        if w <= 0 or h <= 0:
            continue
        long_side = max(w, h)
        short_side = min(w, h)
        aspect = long_side / max(1.0, short_side)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area else 0.0
        if solidity < 0.60:
            continue

        center = np.array([cx + search_x1, cy + search_y1], dtype=np.float32)
        center_distance = float(np.linalg.norm(center - search_center))
        distance_penalty = center_distance / max(1.0, np.linalg.norm([search.shape[1], search.shape[0]]))
        area_penalty = abs(area - target_area) / max(target_area, 1.0)
        aspect_penalty = abs(aspect - target_aspect)
        x, y, bw, bh = cv2.boundingRect(contour)
        touches_search_border = x <= 2 or y <= 2 or x + bw >= search.shape[1] - 2 or y + bh >= search.shape[0] - 2
        if touches_search_border:
            continue

        score = (
            area
            * max(solidity, 0.7)
            * max(0.25, 1.0 - distance_penalty)
            * max(0.20, 1.0 - 0.55 * area_penalty)
            * max(0.20, 1.0 - 0.45 * aspect_penalty)
        )
        if score <= best_score:
            continue

        points = cv2.boxPoints(rect)
        points[:, 0] += search_x1
        points[:, 1] += search_y1
        best = board_payload_from_points(points, area)
        best_score = score

    return best if best is not None else fallback_cutting_board((search_x1, search_y1, search_x2, search_y2))


def build_placement_boxes(board: dict[str, object], num_slots: int = DEFAULT_NUM_PLACEMENTS) -> list[dict[str, object]]:
    corners = np.array(board["corners"], dtype=np.float32)
    top_left, top_right, bottom_right, bottom_left = corners
    top_edge = top_right - top_left
    left_edge = bottom_left - top_left
    long_edge = top_edge if np.linalg.norm(top_edge) >= np.linalg.norm(left_edge) else left_edge
    short_edge = left_edge if np.linalg.norm(top_edge) >= np.linalg.norm(left_edge) else top_edge
    short_len = float(np.linalg.norm(short_edge))
    if short_len <= 1.0:
        return []

    board_center = corners.mean(axis=0)
    slot_size = max(18.0, short_len * 0.2)
    unit_long = long_edge / max(np.linalg.norm(long_edge), 1.0)
    spacing = np.linalg.norm(long_edge) / float(num_slots + 1)

    placements: list[dict[str, object]] = []
    for index in range(num_slots):
        center = board_center - unit_long * (spacing * ((num_slots - 1) / 2.0 - index))
        bbox = [
            int(round(center[0] - slot_size / 2.0)),
            int(round(center[1] - slot_size / 2.0)),
            int(round(center[0] + slot_size / 2.0)),
            int(round(center[1] + slot_size / 2.0)),
        ]
        placements.append(
            {
                "bbox": bbox,
                "center": [round(float(center[0]), 2), round(float(center[1]), 2)],
                "slot_index": index + 1,
                "category": "placement",
            }
        )
    return placements


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
    for item in merged:
        item["category"] = "letter"

    board = detect_cutting_board(image)
    placements = build_placement_boxes(board) if board is not None else []
    return {
        "image_path": str(Path(image_path).resolve()),
        "table_roi": [x1, y1, x2, y2],
        "num_boxes": len(merged),
        "boxes": merged,
        "cutting_board": board,
        "num_placements": len(placements),
        "placements": placements,
    }


def save_overlay(
    image_path: str,
    boxes: list[dict[str, object]],
    output_path: str,
    table_roi: list[int],
    placements: list[dict[str, object]],
    cutting_board: dict[str, object] | None,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    rx1, ry1, rx2, ry2 = table_roi
    cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (120, 120, 120), 1)

    if cutting_board is not None:
        corners = np.round(np.array(cutting_board["corners"], dtype=np.float32)).astype(np.int32)
        cv2.polylines(image, [corners], True, (0, 220, 255), 2)

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

    for placement in placements:
        x1, y1, x2, y2 = placement["bbox"]
        label = f"P{placement['slot_index']}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 120), 2)
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 120),
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
    save_overlay(
        args.image,
        result["boxes"],
        args.output_overlay,
        result["table_roi"],
        result["placements"],
        result["cutting_board"],
    )

    print(
        json.dumps(
            {
                "image_path": result["image_path"],
                "num_boxes": result["num_boxes"],
                "num_placements": result["num_placements"],
                "output_json": str(output_json.resolve()),
                "output_overlay": str(Path(args.output_overlay).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
