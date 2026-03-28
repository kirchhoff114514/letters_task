from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def serialize_result(result) -> dict[str, object]:
    boxes: list[dict[str, object]] = []
    names = result.names
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        boxes.append(
            {
                "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                "category": str(names[cls_id]),
                "confidence": round(conf, 4),
            }
        )

    return {
        "image_path": str(Path(result.path).resolve()),
        "num_boxes": len(boxes),
        "boxes": boxes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detection and export project-style JSON results.")
    parser.add_argument("--model", required=True, help="Path to YOLO weights file.")
    parser.add_argument("--source", required=True, help="Image file or directory to run detection on.")
    parser.add_argument("--output-dir", required=True, help="Directory to write JSON results.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--save-overlay", action="store_true", help="Also save YOLO-rendered overlay images.")
    parser.add_argument("--overlay-dir", help="Optional overlay output directory. Defaults to <output-dir>/overlays.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else (output_dir / "overlays")

    images = discover_images(source)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for image_path in images:
        results = model.predict(source=str(image_path), conf=args.conf, verbose=False)
        result = results[0]
        payload = serialize_result(result)

        json_path = output_dir / f"{image_path.stem}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_item = {
            "image_path": payload["image_path"],
            "num_boxes": payload["num_boxes"],
            "output_json": str(json_path.resolve()),
        }

        if args.save_overlay:
            rendered = result.plot()
            overlay_path = overlay_dir / f"{image_path.stem}_overlay.jpg"
            import cv2

            cv2.imwrite(str(overlay_path), rendered)
            summary_item["output_overlay"] = str(overlay_path.resolve())

        summary.append(summary_item)

    print(json.dumps({"num_images": len(summary), "results": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
