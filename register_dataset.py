"""
============================================================
  XIS CV Pipeline — Step 1 (continued)
  FILE: dataset/register_dataset.py

  PURPOSE:
    - Register your COCO-format dataset with Detectron2
    - Print dataset statistics (image count, annotation count)
    - Generate DATASET_CARD.md automatically

  USAGE:
    python dataset/register_dataset.py

  EXPECTED STRUCTURE (after exporting from Roboflow as COCO JSON):
    dataset/
      train_undistorted/
        *.jpg
        _annotations.coco.json
      val_undistorted/
        *.jpg
        _annotations.coco.json
      test_undistorted/
        *.jpg
        _annotations.coco.json
============================================================
"""

import json
import os
from pathlib import Path


# ─── Configuration ───────────────────────────────────────────
DATASET_ROOT  = "dataset"
OBJECT_NAME   = "credit_card"    # ← change to your object name
SPLITS        = ["train_undistorted", "val_undistorted", "test_undistorted"]
# ─────────────────────────────────────────────────────────────


def load_coco_json(split_dir: str) -> dict | None:
    """Load the COCO annotation JSON from a split directory."""
    candidates = ["_annotations.coco.json", "annotations.json", "instances.json"]
    for name in candidates:
        path = os.path.join(split_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f), path
    return None, None


def split_stats(split_name: str, data: dict) -> dict:
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    return {
        "split": split_name,
        "images": len(images),
        "annotations": len(annotations),
        "categories": [c["name"] for c in categories],
    }


def register_with_detectron2(split_name: str, split_dir: str, json_path: str):
    """Register a COCO dataset split with Detectron2."""
    try:
        from detectron2.data.datasets import register_coco_instances
        register_coco_instances(
            name=f"xis_{split_name}",
            metadata={},
            json_file=json_path,
            image_root=split_dir,
        )
        print(f"  ✓  Registered xis_{split_name} with Detectron2")
    except ImportError:
        print("  [INFO] Detectron2 not installed yet — skipping registration")
        print("         Registration will happen inside train.py automatically")


def generate_dataset_card(stats_list: list, output_path: str):
    total_images = sum(s["images"] for s in stats_list)
    total_annots = sum(s["annotations"] for s in stats_list)

    with open(output_path, "w") as f:
        f.write("# Dataset Card\n\n")
        f.write(f"## Object\n\n**{OBJECT_NAME}**\n\n")
        f.write("## Collection Strategy\n\n")
        f.write("Images were captured using a single calibrated camera under varied:\n")
        f.write("- Lighting conditions (natural, indoor, mixed)\n")
        f.write("- Backgrounds (plain, textured, cluttered)\n")
        f.write("- Angles (top-down, slight tilt, perspective)\n")
        f.write("- Distances (30 cm – 100 cm from object)\n\n")
        f.write("All images were undistorted using intrinsic calibration parameters "
                "before labelling.\n\n")
        f.write("## Labelling Tool\n\nRoboflow (polygon segmentation masks)\n\n")
        f.write("## Statistics\n\n")
        f.write("| Split | Images | Annotations |\n|---|---|---|\n")
        for s in stats_list:
            f.write(f"| {s['split']} | {s['images']} | {s['annotations']} |\n")
        f.write(f"| **Total** | **{total_images}** | **{total_annots}** |\n\n")
        f.write("## Class Distribution\n\n")
        f.write(f"Single class: `{OBJECT_NAME}` (100%)\n\n")
        f.write("## Format\n\nCOCO JSON — instance segmentation (polygon masks)\n")

    print(f"  → {output_path}")


def main():
    print("=" * 60)
    print("  XIS CV Pipeline — Dataset Registration & Stats")
    print("=" * 60)

    stats_list = []
    for split in SPLITS:
        split_dir = os.path.join(DATASET_ROOT, split)
        if not os.path.isdir(split_dir):
            print(f"\n[SKIP] {split_dir} not found")
            continue

        data, json_path = load_coco_json(split_dir)
        if data is None:
            print(f"\n[WARN] No COCO JSON found in {split_dir}")
            continue

        stats = split_stats(split, data)
        stats_list.append(stats)
        print(f"\n  {split}:")
        print(f"    Images:      {stats['images']}")
        print(f"    Annotations: {stats['annotations']}")
        print(f"    Categories:  {stats['categories']}")

        register_with_detectron2(split, split_dir, json_path)

    if stats_list:
        card_path = "docs/DATASET_CARD.md"
        os.makedirs("docs", exist_ok=True)
        generate_dataset_card(stats_list, card_path)
        print(f"\n[DONE] Dataset card saved to {card_path}")
    else:
        print("\n[WARN] No splits found. "
              "Export your Roboflow dataset as COCO JSON first.")


if __name__ == "__main__":
    main()
