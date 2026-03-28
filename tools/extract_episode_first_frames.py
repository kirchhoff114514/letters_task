from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


DEFAULT_START = 0
DEFAULT_END = 999
DEFAULT_STEP = 10


def episode_name(index: int) -> str:
    return f"episode_{index:06d}"


def build_video_path(dataset_root: Path, episode_index: int) -> Path:
    chunk_index = episode_index // 1000
    chunk_name = f"chunk-{chunk_index:03d}"
    return dataset_root / "videos" / chunk_name / "observation.images.faceImg" / f"{episode_name(episode_index)}.mp4"


def build_output_path(output_dir: Path, episode_index: int) -> Path:
    return output_dir / f"{episode_name(episode_index)}.jpg"


def extract_first_frame(video_path: Path, output_path: Path) -> bool:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False

    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), frame))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the first frame of selected episode videos.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing the videos/ directory.")
    parser.add_argument("--output-dir", required=True, help="Directory to save extracted first-frame JPG files.")
    parser.add_argument("--start", type=int, default=DEFAULT_START, help="Start episode index, inclusive.")
    parser.add_argument("--end", type=int, default=DEFAULT_END, help="End episode index, inclusive.")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP, help="Episode sampling step.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    requested = list(range(args.start, args.end + 1, args.step))
    exported: list[dict[str, str]] = []
    missing: list[str] = []
    failed: list[str] = []

    for index in requested:
        video_path = build_video_path(dataset_root, index)
        output_path = build_output_path(output_dir, index)
        episode_id = episode_name(index)

        if not video_path.exists():
            missing.append(episode_id)
            continue

        if extract_first_frame(video_path, output_path):
            exported.append(
                {
                    "episode": episode_id,
                    "video_path": str(video_path),
                    "output_path": str(output_path),
                }
            )
        else:
            failed.append(episode_id)

    print(
        json.dumps(
            {
                "requested_count": len(requested),
                "exported_count": len(exported),
                "missing_count": len(missing),
                "failed_count": len(failed),
                "output_dir": str(output_dir.resolve()),
                "missing_episodes": missing,
                "failed_episodes": failed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
