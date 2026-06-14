"""
============================================================
  XIS CV Pipeline — Step 1 (continued)
  FILE: dataset/undistort_dataset.py

  PURPOSE:
    After labelling your images and exporting them from Roboflow/CVAT,
    run this script to undistort every image in train/val/test using
    the calibration parameters from Step 1.

    Annotations (COCO JSON) are NOT modified — undistortion is applied
    purely to image pixels. Since cv2.undistort with getOptimalNewCameraMatrix
    (alpha=0) keeps image dimensions the same, existing bounding box /
    polygon coordinates remain valid.

  USAGE:
    python dataset/undistort_dataset.py

  INPUT:
    dataset/train/   dataset/val/   dataset/test/
      Each folder should contain images + _annotations.coco.json

  OUTPUT:
    dataset/train_undistorted/
    dataset/val_undistorted/
    dataset/test_undistorted/
    (annotations are copied as-is)
============================================================
"""

import cv2
import numpy as np
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm


# ─── Configuration ───────────────────────────────────────────
CALIBRATION_DIR = "calibration"
DATASET_ROOT    = "dataset"
SPLITS          = ["train", "val", "test"]
# ─────────────────────────────────────────────────────────────


def load_calibration(calib_dir: str):
    mtx  = np.load(os.path.join(calib_dir, "camera_matrix.npy"))
    dist = np.load(os.path.join(calib_dir, "dist_coeffs.npy"))
    print(f"[UNDISTORT] Loaded calibration from {calib_dir}")
    return mtx, dist


def undistort_image(img: np.ndarray, mtx, dist) -> np.ndarray:
    """Undistort a single image using stored intrinsics. Keeps original dimensions."""
    h, w = img.shape[:2]
    # alpha=0 → no black borders; image stays same size
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    return cv2.undistort(img, mtx, dist, None, newcameramtx)


def process_split(split: str, mtx, dist, dataset_root: str):
    src_dir = os.path.join(dataset_root, split)
    dst_dir = os.path.join(dataset_root, f"{split}_undistorted")

    if not os.path.isdir(src_dir):
        print(f"[SKIP] {src_dir} not found — skipping.")
        return

    os.makedirs(dst_dir, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = list(Path(src_dir).iterdir())
    images = [f for f in files if f.suffix.lower() in image_exts]
    jsons  = [f for f in files if f.suffix.lower() == ".json"]

    print(f"\n[UNDISTORT] {split}: {len(images)} images")

    for img_path in tqdm(images, desc=f"  {split}"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Could not read {img_path.name}")
            continue
        undistorted = undistort_image(img, mtx, dist)
        cv2.imwrite(str(Path(dst_dir) / img_path.name), undistorted)

    # Copy annotation JSON files unchanged
    for json_path in jsons:
        shutil.copy(str(json_path), os.path.join(dst_dir, json_path.name))
        print(f"  Copied annotation: {json_path.name}")

    print(f"  → Saved to {dst_dir}/")


def main():
    print("=" * 60)
    print("  XIS CV Pipeline — Undistort Dataset")
    print("=" * 60)

    # Load calibration
    if not os.path.exists(os.path.join(CALIBRATION_DIR, "camera_matrix.npy")):
        print("[ERROR] camera_matrix.npy not found.")
        print("  Run calibration/calibrate_camera.py first.")
        return

    mtx, dist = load_calibration(CALIBRATION_DIR)

    for split in SPLITS:
        process_split(split, mtx, dist, DATASET_ROOT)

    print("\n[DONE] All splits undistorted.")
    print("  Next step → train model with models/train.py")


if __name__ == "__main__":
    main()
