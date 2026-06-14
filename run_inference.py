"""
============================================================
  XIS CV Pipeline — Step 2 (continued)
  FILE: inference/run_inference.py

  PURPOSE:
    - Load trained Mask R-CNN model
    - Accept a single image path as input
    - Undistort the image using calibration parameters
    - Run segmentation inference
    - Visualise and save annotated output (masks + labels)

  USAGE:
    # Single image:
    python inference/run_inference.py --image path/to/image.jpg

    # Batch (folder):
    python inference/run_inference.py --folder path/to/folder/

    # With measurement (Step 3):
    python inference/run_inference.py --image img.jpg --measure

  OUTPUT:
    inference/outputs/  — annotated images with masks drawn
============================================================
"""

import cv2
import numpy as np
import argparse
import os
import json
from pathlib import Path


# ─── Configuration ───────────────────────────────────────────
CALIBRATION_DIR    = "calibration"
MODEL_WEIGHTS      = "models/output/model_final.pth"
MODEL_CONFIG       = "models/output/config.yaml"
OUTPUT_DIR         = "inference/outputs"
CONFIDENCE_THRESH  = 0.5
NUM_CLASSES        = 1
CLASS_NAMES        = ["credit_card"]   # ← change to your class name(s)

# Measurement config (used with --measure flag)
REFERENCE_WIDTH_MM = 85.6    # real width of your reference object (credit card)
# ─────────────────────────────────────────────────────────────


def load_calibration():
    mtx_path  = os.path.join(CALIBRATION_DIR, "camera_matrix.npy")
    dist_path = os.path.join(CALIBRATION_DIR, "dist_coeffs.npy")
    if not os.path.exists(mtx_path):
        raise FileNotFoundError(f"camera_matrix.npy not found in {CALIBRATION_DIR}")
    return np.load(mtx_path), np.load(dist_path)


def undistort(img: np.ndarray, mtx, dist) -> np.ndarray:
    h, w = img.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    return cv2.undistort(img, mtx, dist, None, newcameramtx)


def load_model(weights_path: str, config_path: str):
    """Load trained Detectron2 Mask R-CNN model."""
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2 import model_zoo
    except ImportError:
        raise ImportError("Detectron2 not installed. See docs/SETUP.md.")

    cfg = get_cfg()

    if os.path.exists(config_path):
        cfg.merge_from_file(config_path)
    else:
        # Fallback to default architecture
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.WEIGHTS   = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

    return DefaultPredictor(cfg)


def run_inference(predictor, img: np.ndarray) -> dict:
    """Run Detectron2 inference on an image."""
    outputs   = predictor(img)
    instances = outputs["instances"].to("cpu")

    result = {
        "num_detections": len(instances),
        "boxes":    instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else [],
        "masks":    instances.pred_masks.numpy()        if instances.has("pred_masks") else [],
        "scores":   instances.scores.numpy()            if instances.has("scores") else [],
        "classes":  instances.pred_classes.numpy()      if instances.has("pred_classes") else [],
    }
    return result


def mask_to_bbox(mask: np.ndarray):
    """Get bounding box (x, y, w, h) from a binary mask."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, y, w, h


def compute_measurements(mask: np.ndarray, pixels_per_mm: float | None):
    """Compute width and height from a mask."""
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return None, None, None

    x, y, w, h = bbox
    if pixels_per_mm:
        width_mm  = w / pixels_per_mm
        height_mm = h / pixels_per_mm
    else:
        width_mm = height_mm = None

    return (x, y, w, h), width_mm, height_mm


def draw_annotations(img: np.ndarray, result: dict,
                      pixels_per_mm: float | None = None) -> np.ndarray:
    """Draw masks, bounding boxes, labels and measurements onto the image."""
    annotated = img.copy()

    colours = [
        (52, 211, 153),   # emerald
        (251, 191,  36),  # amber
        (239,  68,  68),  # red
        (139,  92, 246),  # violet
        ( 34, 197, 234),  # cyan
    ]

    for i in range(result["num_detections"]):
        mask   = result["masks"][i]   # H×W bool
        score  = result["scores"][i]
        cls_id = result["classes"][i]
        colour = colours[i % len(colours)]
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

        # ── Mask overlay ───────────────────────────────────
        coloured_mask = np.zeros_like(annotated)
        coloured_mask[mask] = colour
        annotated = cv2.addWeighted(annotated, 1.0, coloured_mask, 0.45, 0)

        # ── Mask contour ───────────────────────────────────
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, colour, 2)

        # ── Bounding box + measurements ────────────────────
        bbox_result = mask_to_bbox(mask)
        if bbox_result:
            x, y, w, h = bbox_result
            cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)

            label = f"{cls_name}: {score:.2f}"
            if pixels_per_mm:
                w_mm = w / pixels_per_mm
                h_mm = h / pixels_per_mm
                label += f"  |  W:{w_mm:.1f}mm  H:{h_mm:.1f}mm"

            # Background rectangle for label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 4, y), colour, -1)
            cv2.putText(annotated, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # ── Info overlay ───────────────────────────────────────
    info = f"Detections: {result['num_detections']}  |  Undistorted + Calibrated"
    cv2.putText(annotated, info, (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


def process_image(image_path: str, predictor, mtx, dist,
                  measure: bool = False) -> dict:
    """Full pipeline: undistort → infer → annotate → save."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Undistort (mandatory)
    undistorted = undistort(img, mtx, dist)

    # Inference
    result = run_inference(predictor, undistorted)
    print(f"  Detections: {result['num_detections']}")

    # Pixel-per-mm (only if measuring and at least 1 detection)
    pixels_per_mm = None
    if measure and result["num_detections"] > 0:
        # Use the first detected object as reference (assumes it IS the reference object)
        first_mask = result["masks"][0]
        bbox = mask_to_bbox(first_mask)
        if bbox:
            _, _, ref_w_px, _ = bbox
            pixels_per_mm = ref_w_px / REFERENCE_WIDTH_MM
            print(f"  pixels_per_mm: {pixels_per_mm:.4f}")

    # Annotate
    annotated = draw_annotations(undistorted, result, pixels_per_mm)

    # Save
    stem    = Path(image_path).stem
    out_path = os.path.join(OUTPUT_DIR, f"{stem}_annotated.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"  → Saved: {out_path}")

    # Return structured output
    output = {
        "image": image_path,
        "detections": result["num_detections"],
        "output_image": out_path,
    }

    if measure and result["num_detections"] > 0 and pixels_per_mm:
        measurements = []
        for i in range(result["num_detections"]):
            bbox_result = mask_to_bbox(result["masks"][i])
            if bbox_result:
                _, _, w_px, h_px = bbox_result
                measurements.append({
                    "instance": i,
                    "confidence": float(result["scores"][i]),
                    "width_mm":  round(w_px / pixels_per_mm, 2),
                    "height_mm": round(h_px / pixels_per_mm, 2),
                    "width_px":  w_px,
                    "height_px": h_px,
                })
        output["measurements"] = measurements
        output["pixels_per_mm"] = round(pixels_per_mm, 4)

    return output


def main():
    parser = argparse.ArgumentParser(description="XIS CV Pipeline — Inference")
    parser.add_argument("--image",  type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--measure", action="store_true",
                        help="Also compute mm measurements (Step 3)")
    args = parser.parse_args()

    if not args.image and not args.folder:
        print("Usage:")
        print("  python inference/run_inference.py --image path/to/image.jpg")
        print("  python inference/run_inference.py --folder path/to/folder/")
        return

    print("=" * 60)
    print("  XIS CV Pipeline — Inference")
    print("=" * 60)

    # Load calibration
    print("\n[1] Loading calibration ...")
    mtx, dist = load_calibration()

    # Load model
    print("[2] Loading model ...")
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"[ERROR] Weights not found: {MODEL_WEIGHTS}")
        print("  Run models/train.py first.")
        return
    predictor = load_model(MODEL_WEIGHTS, MODEL_CONFIG)
    print("  ✓  Model loaded")

    # Determine images to process
    if args.image:
        image_paths = [args.image]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = [str(p) for p in Path(args.folder).iterdir()
                       if p.suffix.lower() in exts]

    print(f"\n[3] Processing {len(image_paths)} image(s) ...")
    all_outputs = []
    for path in image_paths:
        print(f"\n  → {path}")
        output = process_image(path, predictor, mtx, dist, measure=args.measure)
        all_outputs.append(output)

        if "measurements" in output:
            for m in output["measurements"]:
                print(f"     Instance {m['instance']}: "
                      f"W={m['width_mm']} mm  H={m['height_mm']} mm  "
                      f"(conf={m['confidence']:.2f})")

    # Save JSON summary
    summary_path = os.path.join(OUTPUT_DIR, "inference_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_outputs, f, indent=2)
    print(f"\n[DONE] Results saved → {summary_path}")


if __name__ == "__main__":
    main()
