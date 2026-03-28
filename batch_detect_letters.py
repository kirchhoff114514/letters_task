from __future__ import annotations

import argparse
import json
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(assets_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in assets_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_output_paths(image_path: Path, output_dir: Path) -> tuple[Path, Path]:
    return output_dir / f"{image_path.stem}.json", output_dir / f"{image_path.stem}_overlay.png"


def process_image(image_path: Path, output_dir: Path) -> dict[str, object]:
    from letter_ditact_refract import detect_letter_blocks, save_overlay

    result = detect_letter_blocks(str(image_path))
    output_json, output_overlay = build_output_paths(image_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_overlay(
        str(image_path),
        result["boxes"],
        str(output_overlay),
        result["table_roi"],
        result["placements"],
        result["cutting_board"],
    )
    return {
        "image_path": str(image_path.resolve()),
        "num_boxes": result["num_boxes"],
        "num_placements": result["num_placements"],
        "output_json": str(output_json.resolve()),
        "output_overlay": str(output_overlay.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run letter block detection on all images in a directory.")
    parser.add_argument("--assets-dir", required=True, help="Directory containing source images.")
    parser.add_argument("--output-dir", required=True, help="Directory for detection JSON and overlay outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets_dir = Path(args.assets_dir)
    output_dir = Path(args.output_dir)
    images = discover_images(assets_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {assets_dir}")

    summary = [process_image(image_path, output_dir) for image_path in images]
    print(json.dumps({"num_images": len(summary), "results": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
