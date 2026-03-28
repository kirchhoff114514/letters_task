from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


DEFAULT_PADDING = 4
TARGET_CATEGORY = "letter_block"


def discover_detection_jsons(detections_dir: Path) -> list[Path]:
    return sorted(path for path in detections_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")


def crop_with_padding(image, bbox: list[int], padding: int) -> tuple[object, list[int]] | tuple[None, None]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    height, width = image.shape[:2]
    px1 = max(0, x1 - padding)
    py1 = max(0, y1 - padding)
    px2 = min(width, x2 + padding)
    py2 = min(height, y2 + padding)
    if px2 <= px1 or py2 <= py1:
        return None, None
    return image[py1:py2, px1:px2], [px1, py1, px2, py2]


def export_from_yolo_json(
    json_path: Path,
    crops_dir: Path,
    *,
    padding: int,
    start_index: int,
) -> tuple[list[dict[str, object]], int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_path = Path(data["image_path"])
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read source image referenced by {json_path}: {image_path}")

    boxes = data.get("boxes", [])
    if not isinstance(boxes, list):
        raise ValueError(f"Expected 'boxes' list in {json_path}")

    source_stem = image_path.stem
    records: list[dict[str, object]] = []
    next_index = start_index
    for item in boxes:
        if not isinstance(item, dict):
            continue
        if item.get("category") != TARGET_CATEGORY:
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        crop, padded_bbox = crop_with_padding(image, bbox, padding)
        if crop is None or padded_bbox is None or crop.size == 0:
            continue

        sample_id = f"{source_stem}_{next_index:04d}"
        image_rel_path = Path("dataset_yolo_crops") / "crops" / f"{sample_id}.png"
        output_path = crops_dir / f"{sample_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), crop)

        record = {
            "id": sample_id,
            "image": str(image_rel_path).replace("\\", "/"),
            "source_image": str(image_path),
            "source_detection_json": str(json_path),
            "bbox": [int(v) for v in bbox],
            "padded_bbox": padded_bbox,
            "category": TARGET_CATEGORY,
            "label": None,
        }
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            record["confidence"] = float(confidence)

        records.append(record)
        next_index += 1

    return records, next_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cropped letter_block images from YOLO JSON files.")
    parser.add_argument("--detections-dir", required=True, help="Directory containing YOLO JSON files.")
    parser.add_argument("--crops-dir", required=True, help="Directory to write cropped images.")
    parser.add_argument("--labels-json", required=True, help="Path to write labels manifest JSON.")
    parser.add_argument("--padding", type=int, default=DEFAULT_PADDING, help="Extra crop padding around bbox.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detections_dir = Path(args.detections_dir)
    crops_dir = Path(args.crops_dir)
    labels_json = Path(args.labels_json)

    detection_jsons = discover_detection_jsons(detections_dir)
    if not detection_jsons:
        raise FileNotFoundError(f"No detection JSON files found in: {detections_dir}")

    all_records: list[dict[str, object]] = []
    next_index = 1
    for json_path in detection_jsons:
        records, next_index = export_from_yolo_json(
            json_path,
            crops_dir,
            padding=args.padding,
            start_index=next_index,
        )
        all_records.extend(records)

    labels_json.parent.mkdir(parents=True, exist_ok=True)
    labels_json.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "num_detection_jsons": len(detection_jsons),
                "num_exported_crops": len(all_records),
                "crops_dir": str(crops_dir.resolve()),
                "labels_json": str(labels_json.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
