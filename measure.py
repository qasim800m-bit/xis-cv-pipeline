"""
============================================================
  XIS CV Pipeline — Step 3
  FILE: measurement/measure.py

  PURPOSE:
    - Derive pixels_per_mm from a reference object of known size
    - Measure target object width and height in mm from segmentation mask
    - Validate against physical ruler measurements
    - Compute MAE and MPE
    - Generate MEASUREMENT_REPORT.md

  USAGE:
    # Single image (end-to-end demo):
    python measurement/measure.py --image path/to/image.jpg

    # Full accuracy validation (requires CSV of ground-truth measurements):
    python measurement/measure.py --validate --gt_csv measurement/ground_truth.csv

  GROUND TRUTH CSV FORMAT (for validation):
    image_path,gt_width_mm,gt_height_mm
    dataset/test/img_001.jpg,85.6,54.0
    dataset/test/img_002.jpg,85.5,53.9
    ...

  OUTPUT:
    measurement/results/           — annotated images with mm overlays
    measurement/accuracy_report/   — error plots + tables
    docs/MEASUREMENT_REPORT.md
============================================================
"""

import cv2
import numpy as np
import argparse
import os
import json
import csv
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from datetime import datetime


# ─── Configuration ───────────────────────────────────────────
CALIBRATION_DIR       = "calibration"
MODEL_WEIGHTS         = "models/output/model_final.pth"
MODEL_CONFIG          = "models/output/config.yaml"
RESULTS_DIR           = "measurement/results"
ACCURACY_DIR          = "measurement/accuracy_report"
DOCS_DIR              = "docs"

# Reference object (placed in every measurement image)
REFERENCE_WIDTH_MM    = 85.6    # ISO/IEC 7810 ID-1 credit card
REFERENCE_HEIGHT_MM   = 54.0
CONFIDENCE_THRESH     = 0.5
CLASS_NAMES           = ["credit_card"]
NUM_CLASSES           = 1
# ─────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────

def load_calibration():
    mtx  = np.load(os.path.join(CALIBRATION_DIR, "camera_matrix.npy"))
    dist = np.load(os.path.join(CALIBRATION_DIR, "dist_coeffs.npy"))
    return mtx, dist


def undistort(img: np.ndarray, mtx, dist) -> np.ndarray:
    h, w = img.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    return cv2.undistort(img, mtx, dist, None, newcameramtx)


def load_model():
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2 import model_zoo
    except ImportError:
        raise ImportError("Detectron2 not installed. See docs/SETUP.md.")

    cfg = get_cfg()
    if os.path.exists(MODEL_CONFIG):
        cfg.merge_from_file(MODEL_CONFIG)
    else:
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.WEIGHTS   = MODEL_WEIGHTS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    return DefaultPredictor(cfg)


def get_mask_bbox(mask: np.ndarray):
    """Return (x, y, w, h) bounding box of largest contour in a binary mask."""
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return cv2.boundingRect(max(contours, key=cv2.contourArea))


# ──────────────────────────────────────────────────────────────
#  CORE MEASUREMENT PIPELINE
# ──────────────────────────────────────────────────────────────

def compute_pixels_per_mm(predictor, undistorted_img: np.ndarray) -> float | None:
    """
    Detect the reference object (first detected instance) and derive pixels_per_mm.

    Strategy:
      The reference object (e.g. credit card, 85.6 mm wide) must be present in
      the measurement image. The model detects it; we read its pixel width from
      the segmentation mask bounding box, then:

        pixels_per_mm = pixel_width_of_reference / real_width_of_reference_mm

    This ratio is valid for the entire undistorted image because undistortion
    has already removed the non-uniform spatial distortion of the lens.
    """
    outputs   = predictor(undistorted_img)
    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        return None

    # Use highest-confidence detection as the reference
    scores = instances.scores.numpy()
    best_idx = int(np.argmax(scores))
    mask = instances.pred_masks[best_idx].numpy()

    bbox = get_mask_bbox(mask)
    if bbox is None:
        return None

    _, _, ref_w_px, _ = bbox
    return ref_w_px / REFERENCE_WIDTH_MM


def measure_single_image(image_path: str, predictor, mtx, dist) -> dict:
    """
    Full pipeline for one image:
      1. Load image
      2. Undistort
      3. Infer segmentation masks
      4. Derive pixels_per_mm from reference detection
      5. Measure each detected instance
      6. Return structured results
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Cannot read: {image_path}"}

    # Step 2 — undistort (mandatory)
    undistorted = undistort(img, mtx, dist)

    # Step 3 — inference
    outputs   = predictor(undistorted)
    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        return {
            "image": image_path,
            "detections": 0,
            "measurements": [],
            "error": "No objects detected",
        }

    masks   = instances.pred_masks.numpy()
    scores  = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    # Step 4 — pixels_per_mm from the first (highest confidence) detection
    best_idx   = int(np.argmax(scores))
    ref_bbox   = get_mask_bbox(masks[best_idx])
    if ref_bbox is None:
        return {"image": image_path, "error": "Could not extract reference bbox"}

    _, _, ref_w_px, _ = ref_bbox
    pixels_per_mm = ref_w_px / REFERENCE_WIDTH_MM

    # Step 5 — measure all detected instances
    measurements = []
    for i in range(len(instances)):
        bbox = get_mask_bbox(masks[i])
        if bbox is None:
            continue
        _, _, w_px, h_px = bbox
        measurements.append({
            "instance":    i,
            "confidence":  float(scores[i]),
            "class":       CLASS_NAMES[int(classes[i])] if int(classes[i]) < len(CLASS_NAMES)
                           else f"class_{int(classes[i])}",
            "width_px":    int(w_px),
            "height_px":   int(h_px),
            "width_mm":    round(w_px  / pixels_per_mm, 2),
            "height_mm":   round(h_px  / pixels_per_mm, 2),
        })

    return {
        "image":         image_path,
        "detections":    len(instances),
        "pixels_per_mm": round(pixels_per_mm, 4),
        "measurements":  measurements,
    }


def annotate_and_save(image_path: str, result: dict, mtx, dist, out_dir: str):
    """Draw measurement overlay and save annotated image."""
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image_path)
    undistorted = undistort(img, mtx, dist)
    annotated   = undistorted.copy()

    if "measurements" not in result:
        return

    ppm = result.get("pixels_per_mm", 1)

    for m in result["measurements"]:
        colour = (52, 211, 153)
        # We re-compute the bbox from the stored px values for drawing
        # (In a real system you'd pass masks through; here we approximate)
        w_px = m["width_px"]
        h_px = m["height_px"]

        label = (f"{m['class']}  "
                 f"W:{m['width_mm']:.1f}mm  H:{m['height_mm']:.1f}mm  "
                 f"conf:{m['confidence']:.2f}")

        # Draw label at top-left corner (approximate position)
        cv2.putText(annotated, label, (20, 40 + 35 * m["instance"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

    stem     = Path(image_path).stem
    out_path = os.path.join(out_dir, f"{stem}_measured.jpg")
    cv2.imwrite(out_path, annotated)


# ──────────────────────────────────────────────────────────────
#  ACCURACY VALIDATION
# ──────────────────────────────────────────────────────────────

def load_ground_truth(csv_path: str) -> list:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "image_path":  row["image_path"],
                "gt_width_mm": float(row["gt_width_mm"]),
                "gt_height_mm": float(row["gt_height_mm"]),
            })
    return rows


def compute_accuracy(gt_rows: list, predictor, mtx, dist) -> dict:
    """Compare system measurements vs physical ground-truth for all GT rows."""
    records = []

    print(f"\n[VALIDATION] Measuring {len(gt_rows)} images ...")
    for row in gt_rows:
        path = row["image_path"]
        print(f"  → {path}")
        result = measure_single_image(path, predictor, mtx, dist)

        if "error" in result or not result.get("measurements"):
            print(f"    [SKIP] {result.get('error', 'no measurements')}")
            continue

        best = max(result["measurements"], key=lambda x: x["confidence"])
        records.append({
            "image":          path,
            "gt_width_mm":    row["gt_width_mm"],
            "gt_height_mm":   row["gt_height_mm"],
            "pred_width_mm":  best["width_mm"],
            "pred_height_mm": best["height_mm"],
            "confidence":     best["confidence"],
        })

    if not records:
        return {"error": "No valid measurements to compare"}

    gt_w   = np.array([r["gt_width_mm"]    for r in records])
    pred_w = np.array([r["pred_width_mm"]  for r in records])
    gt_h   = np.array([r["gt_height_mm"]   for r in records])
    pred_h = np.array([r["pred_height_mm"] for r in records])

    mae_w = float(np.mean(np.abs(gt_w - pred_w)))
    mae_h = float(np.mean(np.abs(gt_h - pred_h)))
    mpe_w = float(np.mean(np.abs((gt_w - pred_w) / gt_w)) * 100)
    mpe_h = float(np.mean(np.abs((gt_h - pred_h) / gt_h)) * 100)

    return {
        "records": records,
        "mae_width_mm":  round(mae_w, 3),
        "mae_height_mm": round(mae_h, 3),
        "mpe_width_pct": round(mpe_w, 3),
        "mpe_height_pct": round(mpe_h, 3),
        "n": len(records),
    }


def plot_accuracy(accuracy: dict, out_dir: str):
    """Scatter plots: ground truth vs predicted."""
    os.makedirs(out_dir, exist_ok=True)
    records = accuracy["records"]

    gt_w   = [r["gt_width_mm"]   for r in records]
    pred_w = [r["pred_width_mm"] for r in records]
    gt_h   = [r["gt_height_mm"]   for r in records]
    pred_h = [r["pred_height_mm"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Measurement Accuracy: Ground Truth vs Predicted", fontsize=13, fontweight="bold")

    for ax, gt, pred, label in [
        (axes[0], gt_w, pred_w, "Width"),
        (axes[1], gt_h, pred_h, "Height"),
    ]:
        ax.scatter(gt, pred, color="#2563EB", alpha=0.7, edgecolors="white", linewidths=0.5, s=60)
        min_v = min(min(gt), min(pred)) - 2
        max_v = max(max(gt), max(pred)) + 2
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1, label="Perfect")
        ax.set_xlabel(f"Ground Truth {label} (mm)")
        ax.set_ylabel(f"Predicted {label} (mm)")
        ax.set_title(f"{label}: MAE = {accuracy[f'mae_{label.lower()}_mm']:.2f} mm  "
                     f"| MPE = {accuracy[f'mpe_{label.lower()}_pct']:.2f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "accuracy_scatter.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  → Accuracy plot saved: {plot_path}")


def generate_measurement_report(accuracy: dict, out_dir: str):
    """Write MEASUREMENT_REPORT.md."""
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "MEASUREMENT_REPORT.md")

    with open(report_path, "w") as f:
        f.write("# Measurement Methodology & Accuracy Report\n\n")

        f.write("## Pixel-to-MM Conversion Derivation\n\n")
        f.write("### Method\n\n")
        f.write("A reference object of known real-world width is placed in every measurement image.\n")
        f.write(f"For this project, the reference is a **standard credit card (ISO/IEC 7810 ID-1)**:\n")
        f.write(f"- Width:  {REFERENCE_WIDTH_MM} mm\n")
        f.write(f"- Height: {REFERENCE_HEIGHT_MM} mm\n\n")
        f.write("The Mask R-CNN model detects and segments the reference object. "
                "The pixel width of its bounding box is extracted from the segmentation mask:\n\n")
        f.write("```\n")
        f.write("pixels_per_mm = pixel_width_of_reference / real_width_of_reference_mm\n\n")
        f.write(f"              = pixel_width / {REFERENCE_WIDTH_MM}\n")
        f.write("```\n\n")
        f.write("For any other detected object, dimensions are then:\n\n")
        f.write("```\n")
        f.write("width_mm  = pixel_width_of_object  / pixels_per_mm\n")
        f.write("height_mm = pixel_height_of_object / pixels_per_mm\n")
        f.write("```\n\n")

        f.write("## Why Undistortion Is Mandatory\n\n")
        f.write("Raw (distorted) images have non-uniform pixel-to-mm ratios across the frame.\n")
        f.write("Radial distortion compresses or stretches pixels differently at different "
                "distances from the optical centre. A pixel at the image edge represents a "
                "different real-world distance than a pixel at the centre.\n\n")
        f.write("After `cv2.undistort()`, every pixel represents the same real-world distance "
                "for a given depth, making the conversion ratio `pixels_per_mm` uniformly valid "
                "across the entire image frame.\n\n")

        if "records" in accuracy:
            f.write("## Accuracy Validation Results\n\n")
            f.write(f"Validated on **{accuracy['n']} images** measured physically "
                    "with a digital calliper.\n\n")
            f.write("| Metric | Width | Height |\n|---|---|---|\n")
            f.write(f"| MAE (mm) | {accuracy['mae_width_mm']} | {accuracy['mae_height_mm']} |\n")
            f.write(f"| MPE (%)  | {accuracy['mpe_width_pct']} | {accuracy['mpe_height_pct']} |\n\n")

            f.write("### Per-Image Error Table\n\n")
            f.write("| Image | GT W (mm) | Pred W (mm) | Err W | GT H (mm) | Pred H (mm) | Err H |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for r in accuracy["records"]:
                ew = abs(r["gt_width_mm"]  - r["pred_width_mm"])
                eh = abs(r["gt_height_mm"] - r["pred_height_mm"])
                f.write(f"| {Path(r['image']).name} "
                        f"| {r['gt_width_mm']} | {r['pred_width_mm']} | {ew:.2f} "
                        f"| {r['gt_height_mm']} | {r['pred_height_mm']} | {eh:.2f} |\n")

            f.write("\n![Accuracy Scatter](../measurement/accuracy_report/accuracy_scatter.png)\n\n")

        f.write("## Limitations\n\n")
        f.write("- The pixels_per_mm ratio assumes the reference and target objects are at the "
                "same depth (distance from camera). Parallax error increases with depth difference.\n")
        f.write("- The model must confidently detect the reference object. Low-confidence "
                "detections may yield inaccurate bounding boxes.\n")
        f.write("- Measurement accuracy is bounded by the segmentation mask quality.\n")
        f.write("- This method is monocular — depth variation is not accounted for.\n")

    print(f"  → {report_path}")


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XIS CV Pipeline — Measurement")
    parser.add_argument("--image",    type=str, help="Single image for measurement demo")
    parser.add_argument("--validate", action="store_true",
                        help="Run accuracy validation against ground-truth CSV")
    parser.add_argument("--gt_csv",   type=str,
                        default="measurement/ground_truth.csv",
                        help="Path to ground-truth CSV (required for --validate)")
    args = parser.parse_args()

    print("=" * 60)
    print("  XIS CV Pipeline — Pixel-to-MM Measurement")
    print("=" * 60)

    # Load calibration
    if not os.path.exists(os.path.join(CALIBRATION_DIR, "camera_matrix.npy")):
        print("[ERROR] Calibration not found. Run calibration/calibrate_camera.py first.")
        return

    mtx, dist = load_calibration()
    print("  ✓  Calibration loaded")

    # Load model
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"[ERROR] Model weights not found: {MODEL_WEIGHTS}")
        print("  Run models/train.py first.")
        return

    predictor = load_model()
    print("  ✓  Model loaded")

    # ── Single image demo ────────────────────────────────────
    if args.image:
        print(f"\n[DEMO] Processing: {args.image}")
        result = measure_single_image(args.image, predictor, mtx, dist)
        annotate_and_save(args.image, result, mtx, dist, RESULTS_DIR)

        print("\n  Results:")
        print(f"    Detections:     {result.get('detections', 0)}")
        print(f"    pixels_per_mm:  {result.get('pixels_per_mm', 'N/A')}")
        for m in result.get("measurements", []):
            print(f"    Instance {m['instance']}:  "
                  f"W={m['width_mm']} mm  H={m['height_mm']} mm  "
                  f"conf={m['confidence']:.2f}")

        out_json = os.path.join(RESULTS_DIR, "demo_result.json")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  → JSON result: {out_json}")

    # ── Accuracy validation ──────────────────────────────────
    if args.validate:
        if not os.path.exists(args.gt_csv):
            print(f"\n[ERROR] Ground-truth CSV not found: {args.gt_csv}")
            print("  Create it with columns: image_path, gt_width_mm, gt_height_mm")
            return

        gt_rows  = load_ground_truth(args.gt_csv)
        accuracy = compute_accuracy(gt_rows, predictor, mtx, dist)

        if "error" in accuracy:
            print(f"\n[ERROR] {accuracy['error']}")
            return

        print(f"\n[ACCURACY SUMMARY]")
        print(f"  Samples:     {accuracy['n']}")
        print(f"  MAE Width:   {accuracy['mae_width_mm']} mm")
        print(f"  MAE Height:  {accuracy['mae_height_mm']} mm")
        print(f"  MPE Width:   {accuracy['mpe_width_pct']} %")
        print(f"  MPE Height:  {accuracy['mpe_height_pct']} %")

        plot_accuracy(accuracy, ACCURACY_DIR)

        os.makedirs(DOCS_DIR, exist_ok=True)
        generate_measurement_report(accuracy, DOCS_DIR)

        acc_json = os.path.join(ACCURACY_DIR, "accuracy.json")
        os.makedirs(ACCURACY_DIR, exist_ok=True)
        with open(acc_json, "w") as f:
            json.dump(accuracy, f, indent=2, default=str)
        print(f"  → Accuracy JSON: {acc_json}")

    if not args.image and not args.validate:
        print("\nUsage examples:")
        print("  python measurement/measure.py --image path/to/image.jpg")
        print("  python measurement/measure.py --validate --gt_csv measurement/ground_truth.csv")

    print("\n[DONE] Measurement pipeline complete.")


if __name__ == "__main__":
    main()
