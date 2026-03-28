from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


DEFAULT_IMAGE_SIZE = 64
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labeled_samples(labels_json: Path) -> list[dict[str, object]]:
    data = json.loads(labels_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("labels.json must contain a list.")

    samples: list[dict[str, object]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        image_path = item.get("image")
        if not isinstance(label, str) or len(label) != 1 or not label.isalpha():
            continue
        if not isinstance(image_path, str):
            continue
        sample = dict(item)
        sample["label"] = label.upper()
        samples.append(sample)

    if not samples:
        raise ValueError("No labeled samples found in labels.json.")
    return samples


def build_class_mapping(samples: list[dict[str, object]]) -> tuple[dict[str, int], list[str]]:
    classes = sorted({str(sample["label"]) for sample in samples})
    class_to_idx = {label: index for index, label in enumerate(classes)}
    return class_to_idx, classes


def read_image_as_tensor(image_path: Path, image_size: int) -> torch.Tensor:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read crop image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return torch.from_numpy(normalized).unsqueeze(0)


class LetterCropDataset(Dataset):
    def __init__(self, samples: list[dict[str, object]], dataset_root: Path, class_to_idx: dict[str, int], image_size: int) -> None:
        self.samples = samples
        self.dataset_root = dataset_root
        self.class_to_idx = class_to_idx
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image_rel_path = Path(str(sample["image"]))
        image_path = image_rel_path if image_rel_path.is_absolute() else (self.dataset_root / image_rel_path)
        image_tensor = read_image_as_tensor(image_path, self.image_size)
        label_tensor = torch.tensor(self.class_to_idx[str(sample["label"])], dtype=torch.long)
        return image_tensor, label_tensor


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


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * targets.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += int((predictions == targets).sum().item())
            total_count += int(targets.size(0))
    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def train_model(
    labels_json: Path,
    dataset_root: Path,
    model_output: Path,
    classes_output: Path,
    *,
    image_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    val_ratio: float,
    seed: int,
) -> None:
    set_seed(seed)
    samples = load_labeled_samples(labels_json)
    class_to_idx, classes = build_class_mapping(samples)

    dataset = LetterCropDataset(samples, dataset_root, class_to_idx, image_size)
    val_size = max(1, int(len(dataset) * val_ratio)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if train_size <= 0:
        train_size = len(dataset)
        val_size = 0

    generator = torch.Generator().manual_seed(seed)
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallLetterCNN(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * targets.size(0)
            running_correct += int((logits.argmax(dim=1) == targets).sum().item())
            running_count += int(targets.size(0))

        train_loss = running_loss / max(1, running_count)
        train_acc = running_correct / max(1, running_count)

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device)
        else:
            val_loss, val_acc = train_loss, train_acc

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
            }
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 4),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 4),
                },
                ensure_ascii=False,
            )
        )

    model_output.parent.mkdir(parents=True, exist_ok=True)
    classes_output.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = model.state_dict()

    torch.save(
        {
            "model_state_dict": best_state,
            "classes": classes,
            "image_size": image_size,
        },
        model_output,
    )
    classes_output.write_text(json.dumps({"classes": classes, "history": history}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "num_samples": len(samples),
                "num_classes": len(classes),
                "model_output": str(model_output.resolve()),
                "classes_output": str(classes_output.resolve()),
                "best_val_acc": round(best_val_acc, 4),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small PyTorch CNN for letter-block classification.")
    parser.add_argument("--labels-json", required=True, help="Path to the labeled dataset JSON.")
    parser.add_argument("--dataset-root", required=True, help="Root directory for dataset image paths.")
    parser.add_argument("--model-output", required=True, help="Path to write the trained PyTorch model.")
    parser.add_argument("--classes-output", required=True, help="Path to write classes/history JSON.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        labels_json=Path(args.labels_json),
        dataset_root=Path(args.dataset_root),
        model_output=Path(args.model_output),
        classes_output=Path(args.classes_output),
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
