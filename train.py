"""
============================================================
  XIS CV Pipeline — Step 2
  FILE: models/train.py

  PURPOSE:
    Train a Mask R-CNN (ResNet-50-FPN) segmentation model on your
    custom COCO dataset using Detectron2.

    Model selection rationale:
      Mask R-CNN is a two-stage instance segmentation model that
      outputs both bounding boxes and per-instance pixel masks.
      It is well established in industrial measurement pipelines,
      supports transfer learning from COCO pretrained weights,
      and is NOT a YOLO variant or Roboflow model — satisfying
      the assessment requirement.

  USAGE:
    python models/train.py

  PRE-REQUISITES:
    1. pip install 'git+https://github.com/facebookresearch/detectron2.git'
    2. Run dataset/undistort_dataset.py first
    3. Each split folder must contain images + _annotations.coco.json

  OUTPUT:
    models/output/         — checkpoints + metrics
    models/output/metrics.json
    docs/TRAINING_REPORT.md
============================================================
"""

import os
import json
import logging
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless — no display needed

# ─── Configuration ───────────────────────────────────────────
DATASET_ROOT    = "dataset"
OUTPUT_DIR      = "models/output"
DOCS_DIR        = "docs"

# Dataset split dirs (undistorted versions)
TRAIN_DIR       = os.path.join(DATASET_ROOT, "train_undistorted")
VAL_DIR         = os.path.join(DATASET_ROOT, "val_undistorted")
TEST_DIR        = os.path.join(DATASET_ROOT, "test_undistorted")

# Training hyperparameters — adjust based on your dataset size
NUM_CLASSES     = 1          # number of object classes (credit card = 1)
MAX_ITER        = 3000       # training iterations (increase for larger datasets)
BATCH_SIZE      = 2          # images per batch (lower if GPU OOM)
LEARNING_RATE   = 0.001
WARMUP_ITERS    = 200
EVAL_PERIOD     = 500        # evaluate every N iterations
CHECKPOINT_PERIOD = 500
NUM_WORKERS     = 2

# Model architecture
BASE_MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# ─────────────────────────────────────────────────────────────


def find_coco_json(split_dir: str) -> str:
    """Find the COCO JSON file in a split directory."""
    candidates = ["_annotations.coco.json", "annotations.json", "instances.json"]
    for name in candidates:
        path = os.path.join(split_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No COCO JSON found in {split_dir}")


def register_datasets():
    """Register train/val/test splits with Detectron2."""
    from detectron2.data.datasets import register_coco_instances

    splits = {
        "xis_train": TRAIN_DIR,
        "xis_val":   VAL_DIR,
        "xis_test":  TEST_DIR,
    }

    for name, split_dir in splits.items():
        if not os.path.isdir(split_dir):
            print(f"[WARN] {split_dir} not found — skipping {name}")
            continue
        try:
            json_path = find_coco_json(split_dir)
            register_coco_instances(name, {}, json_path, split_dir)
            print(f"  ✓  Registered {name} from {split_dir}")
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")


def build_config():
    """Build Detectron2 training configuration."""
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_MODEL))

    # Datasets
    cfg.DATASETS.TRAIN = ("xis_train",)
    cfg.DATASETS.TEST  = ("xis_val",)

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS

    # Pretrained weights from COCO
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(BASE_MODEL)

    # Solver (learning rate schedule)
    cfg.SOLVER.IMS_PER_BATCH   = BATCH_SIZE
    cfg.SOLVER.BASE_LR         = LEARNING_RATE
    cfg.SOLVER.MAX_ITER        = MAX_ITER
    cfg.SOLVER.WARMUP_ITERS    = WARMUP_ITERS
    cfg.SOLVER.STEPS           = (int(MAX_ITER * 0.66), int(MAX_ITER * 0.88))
    cfg.SOLVER.GAMMA           = 0.1   # LR decay factor at each step
    cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD

    # Model head — match your number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # Evaluation
    cfg.TEST.EVAL_PERIOD = EVAL_PERIOD

    # Output
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    return cfg


def plot_metrics(metrics_file: str):
    """Parse Detectron2 metrics.json and plot loss + mAP curves."""
    if not os.path.exists(metrics_file):
        return

    iterations, total_loss, val_map = [], [], []

    with open(metrics_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if "total_loss" in entry:
                    iterations.append(entry["iteration"])
                    total_loss.append(entry["total_loss"])
                if "segm/AP" in entry:
                    val_map.append((entry["iteration"], entry["segm/AP"]))
            except json.JSONDecodeError:
                continue

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Metrics — Mask R-CNN", fontsize=14, fontweight="bold")

    # Loss curve
    axes[0].plot(iterations, total_loss, color="#2563EB", linewidth=1.5)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Total Loss")
    axes[0].grid(True, alpha=0.3)

    # mAP curve
    if val_map:
        iters_map, maps = zip(*val_map)
        axes[1].plot(iters_map, maps, color="#16A34A", linewidth=1.5, marker="o", markersize=4)
        axes[1].set_title("Validation mAP@0.5:0.95 (Segmentation)")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("mAP")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No mAP data yet", ha="center", va="center",
                     transform=axes[1].transAxes)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    print(f"  → Saved training curves: {plot_path}")
    plt.close()


def evaluate_on_test(cfg):
    """Run evaluation on the held-out test set."""
    from detectron2.engine import DefaultPredictor
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    # Load best checkpoint
    cfg_test = cfg.clone()
    cfg_test.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
    cfg_test.DATASETS.TEST = ("xis_test",)

    if not os.path.exists(cfg_test.MODEL.WEIGHTS):
        print("[WARN] model_final.pth not found — skipping test evaluation")
        return {}

    predictor = DefaultPredictor(cfg_test)

    evaluator   = COCOEvaluator("xis_test", output_dir=OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg_test, "xis_test")
    results     = inference_on_dataset(predictor.model, test_loader, evaluator)

    print("\n[TEST RESULTS]")
    print(json.dumps(results, indent=2))
    return results


def generate_training_report(cfg, results: dict):
    """Write TRAINING_REPORT.md."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    report_path = os.path.join(DOCS_DIR, "TRAINING_REPORT.md")

    with open(report_path, "w") as f:
        f.write("# Model Training Report\n\n")
        f.write("## Architecture\n\n")
        f.write("| Parameter | Value |\n|---|---|\n")
        f.write(f"| Model | Mask R-CNN |\n")
        f.write(f"| Backbone | ResNet-50 + FPN |\n")
        f.write(f"| Pretrained On | COCO (80 classes) |\n")
        f.write(f"| Fine-tuned For | {NUM_CLASSES} class(es) |\n")
        f.write(f"| Framework | Detectron2 |\n\n")

        f.write("### Model Selection Rationale\n\n")
        f.write("Mask R-CNN was selected for the following reasons:\n")
        f.write("- It is a dedicated instance segmentation model, producing per-pixel masks "
                "required for accurate boundary extraction.\n")
        f.write("- It is NOT a YOLO variant or Roboflow model — satisfying the assessment constraint.\n")
        f.write("- Strong COCO pretrained weights allow high accuracy with small custom datasets "
                "via transfer learning.\n")
        f.write("- The two-stage (RPN + RoI head) design gives precise masks even on "
                "difficult object boundaries.\n\n")

        f.write("## Hyperparameters\n\n")
        f.write("| Hyperparameter | Value |\n|---|---|\n")
        f.write(f"| Max Iterations | {MAX_ITER} |\n")
        f.write(f"| Batch Size | {BATCH_SIZE} |\n")
        f.write(f"| Base Learning Rate | {LEARNING_RATE} |\n")
        f.write(f"| LR Warmup Iterations | {WARMUP_ITERS} |\n")
        f.write(f"| LR Decay Steps | {int(MAX_ITER*0.66)}, {int(MAX_ITER*0.88)} |\n")
        f.write(f"| LR Decay Factor | 0.1 |\n")
        f.write(f"| ROI Batch Size | 128 |\n\n")

        f.write("## Augmentation Strategy\n\n")
        f.write("Detectron2 default augmentations were applied:\n")
        f.write("- Random horizontal flip (p=0.5)\n")
        f.write("- Multi-scale resize (shorter edge: 640–800 px)\n\n")

        f.write("## Test Set Metrics\n\n")
        if results:
            bbox = results.get("bbox", {})
            segm = results.get("segm", {})
            f.write("| Metric | BBox | Segmentation |\n|---|---|---|\n")
            f.write(f"| mAP@0.5:0.95 | {bbox.get('AP', 'N/A'):.1f} | "
                    f"{segm.get('AP', 'N/A'):.1f} |\n")
            f.write(f"| mAP@0.5 | {bbox.get('AP50', 'N/A'):.1f} | "
                    f"{segm.get('AP50', 'N/A'):.1f} |\n")
            f.write(f"| mAP@0.75 | {bbox.get('AP75', 'N/A'):.1f} | "
                    f"{segm.get('AP75', 'N/A'):.1f} |\n")
        else:
            f.write("_(Run test evaluation to populate this table)_\n\n")

        f.write("\n## Training Curves\n\n")
        f.write("![Training Curves](../models/output/training_curves.png)\n\n")
        f.write("## Limitations\n\n")
        f.write("- Performance depends on dataset diversity; more images will improve robustness.\n")
        f.write("- Model assumes the camera used at inference is the same calibrated camera.\n")

    print(f"  → {report_path}")


def main():
    print("=" * 60)
    print("  XIS CV Pipeline — Model Training (Mask R-CNN)")
    print("=" * 60)

    # Detectron2 import check
    try:
        import detectron2
        print(f"  Detectron2 version: {detectron2.__version__}")
    except ImportError:
        print("\n[ERROR] Detectron2 not installed.")
        print("  Run: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
        print("  See docs/SETUP.md for full instructions.")
        return

    from detectron2.engine import DefaultTrainer
    from detectron2.utils.logger import setup_logger
    setup_logger()

    # Register datasets
    print("\n[STEP 1] Registering datasets ...")
    register_datasets()

    # Build config
    print("\n[STEP 2] Building config ...")
    cfg = build_config()

    # Save config
    config_path = os.path.join(OUTPUT_DIR, "config.yaml")
    with open(config_path, "w") as f:
        f.write(cfg.dump())
    print(f"  Config saved → {config_path}")

    # Train
    print(f"\n[STEP 3] Training for {MAX_ITER} iterations ...")
    print(f"  Batch size: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"  Checkpoints every {CHECKPOINT_PERIOD} iterations")
    print(f"  Eval every {EVAL_PERIOD} iterations\n")

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Plot curves
    print("\n[STEP 4] Plotting training curves ...")
    plot_metrics(os.path.join(OUTPUT_DIR, "metrics.json"))

    # Test evaluation
    print("\n[STEP 5] Evaluating on test set ...")
    results = evaluate_on_test(cfg)

    # Generate report
    print("\n[STEP 6] Generating training report ...")
    generate_training_report(cfg, results)

    print("\n[DONE] Training complete.")
    print(f"  Model weights → {OUTPUT_DIR}/model_final.pth")
    print("  Next step → run inference/run_inference.py")


if __name__ == "__main__":
    main()
