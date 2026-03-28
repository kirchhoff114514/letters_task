from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from torch import nn


LETTER_CATEGORIES = {"letter_block"}


class SmallLetterCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def crop_with_padding(image, bbox: list[int], padding: int = 4):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    height, width = image.shape[:2]
    px1 = max(0, x1 - padding)
    py1 = max(0, y1 - padding)
    px2 = min(width, x2 + padding)
    py2 = min(height, y2 + padding)
    if px2 <= px1 or py2 <= py1:
        return None
    return image[py1:py2, px1:px2]


def preprocess_crop(image, image_size: int) -> torch.Tensor:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)


def load_model(model_path: Path) -> tuple[nn.Module, list[str], int, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]
    image_size = int(checkpoint.get("image_size", 64))
    model = SmallLetterCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes, image_size, device


def predict_letter(model: nn.Module, classes: list[str], image_tensor: torch.Tensor, device: torch.device) -> tuple[str, float]:
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0]
        confidence, index = torch.max(probs, dim=0)
    return classes[int(index.item())], round(float(confidence.item()), 3)


def enrich_detection_results(
    image,
    data: dict[str, object],
    model: nn.Module,
    classes: list[str],
    image_size: int,
    device: torch.device,
) -> dict[str, object]:
    boxes = data.get("boxes", [])
    if not isinstance(boxes, list):
        raise ValueError("Expected 'boxes' to be a list.")

    updated_boxes: list[dict[str, object]] = []
    for item in boxes:
        if not isinstance(item, dict):
            continue
        enriched = dict(item)
        if str(enriched.get("category")) in LETTER_CATEGORIES:
            bbox = enriched.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                crop = crop_with_padding(image, bbox)
                if crop is not None and crop.size > 0:
                    image_tensor = preprocess_crop(crop, image_size)
                    letter, confidence = predict_letter(model, classes, image_tensor, device)
                    enriched["letter"] = letter
                    enriched["confidence"] = confidence
                    enriched["recognizer"] = "cnn"
        updated_boxes.append(enriched)

    result = dict(data)
    result["boxes"] = updated_boxes
    return result


def save_overlay(image, data: dict[str, object], output_path: Path) -> None:
    canvas = image.copy()
    boxes = data.get("boxes", [])
    if isinstance(boxes, list):
        for index, item in enumerate(boxes, start=1):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            category = item.get("category", "unknown")
            if str(category) in LETTER_CATEGORIES:
                color = (0, 200, 255)
                letter = str(item.get("letter", "?"))
                confidence = item.get("confidence")
                label = f"{index}:{letter} {confidence:.2f}" if isinstance(confidence, (int, float)) else f"{index}:{letter}"
            else:
                color = (0, 255, 120)
                label = str(item.get("slot_index", index))

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas,
                label,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize letters from detection JSON using a trained CNN.")
    parser.add_argument("--image", required=True, help="Path to the original image.")
    parser.add_argument("--input-json", required=True, help="Path to detection JSON.")
    parser.add_argument("--model-path", required=True, help="Path to the trained CNN model (.pt).")
    parser.add_argument("--output-json", required=True, help="Path to write enriched recognition JSON.")
    parser.add_argument("--output-overlay", help="Optional path to write overlay image with predicted letters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {args.image}")

    input_path = Path(args.input_json)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    model, classes, image_size, device = load_model(Path(args.model_path))
    result = enrich_detection_results(image, data, model, classes, image_size, device)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    overlay_value = None
    if args.output_overlay:
        overlay_path = Path(args.output_overlay)
        save_overlay(image, result, overlay_path)
        overlay_value = str(overlay_path.resolve())

    print(
        json.dumps(
            {
                "image_path": str(Path(args.image).resolve()),
                "input_json": str(input_path.resolve()),
                "output_json": str(output_path.resolve()),
                "output_overlay": overlay_value,
                "num_boxes": len(result.get("boxes", [])),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
