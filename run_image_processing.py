import json

from image_processing import render_step_guidance


JSON_PATH = "/root/letter_tasks/output/letter_scene.json"
OUTPUT_PATH = "/root/letter_tasks/output/step_annotated.png"


def main() -> None:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        detections = json.load(f)

    result = render_step_guidance(
        image=detections["image_path"],
        detections=detections,
        source_box_id=1,
        placement_id=4,
        background_dim_factor=1,
        fill_alpha=0.5,
    )

    result.annotated_image.save(OUTPUT_PATH)
    print("pick_bbox:", result.pick_bbox)
    print("place_bbox:", result.place_bbox)
    print("saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
