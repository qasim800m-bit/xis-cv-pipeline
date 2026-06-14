# Setup & Installation Guide

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 20.04 / Windows 10 / macOS 12 | Ubuntu 22.04 |
| Python | 3.9 | 3.10 |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU-only slow) | NVIDIA GPU, 6+ GB VRAM |
| CUDA | — | 11.7 or 12.1 |

---

## Step-by-Step Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/xis-cv-pipeline.git
cd xis-cv-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install PyTorch (match your CUDA version)

```bash
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install Detectron2

```bash
# Linux / macOS (GPU):
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Windows (CPU fallback):
pip install 'git+https://github.com/facebookresearch/detectron2.git'
# If that fails, install prebuilt wheel from:
# https://github.com/facebookresearch/detectron2/issues/9
```

Verify:
```bash
python -c "import detectron2; print(detectron2.__version__)"
```

---

## Running the Full Pipeline

### Step 1 — Camera Calibration

1. Print a checkerboard or display one on a screen.
   Default: 8×6 inner corners, 25 mm squares.
   Edit `CHECKERBOARD` and `SQUARE_SIZE_MM` in `calibration/calibrate_camera.py` if different.

2. Take 20–30 photos from varied angles and distances. Save to `calibration/images/`.

3. Run:
```bash
python calibration/calibrate_camera.py
```

Outputs: `calibration/camera_matrix.npy`, `calibration/dist_coeffs.npy`

---

### Step 1b — Collect & Label Dataset

1. Collect 70+ images of your chosen object using the same camera.
2. Go to [roboflow.com](https://roboflow.com), create a project (Instance Segmentation).
3. Upload images, draw polygon masks around your object.
4. Export as **COCO JSON** format.
5. Download and place splits into:
   ```
   dataset/train/    ← images + _annotations.coco.json
   dataset/val/      ← images + _annotations.coco.json
   dataset/test/     ← images + _annotations.coco.json
   ```
6. Undistort all images:
```bash
python dataset/undistort_dataset.py
```

---

### Step 2 — Train Model

```bash
python models/train.py
```

Training takes 30–120 minutes depending on GPU. Progress is logged to `models/output/`.

---

### Step 3 — Run Inference

```bash
# Single image:
python inference/run_inference.py --image path/to/image.jpg

# With measurement:
python inference/run_inference.py --image path/to/image.jpg --measure
```

---

### Step 3b — Accuracy Validation

1. Physically measure 10+ object instances with a calliper.
2. Fill in `measurement/ground_truth.csv`:
   ```
   image_path,gt_width_mm,gt_height_mm
   dataset/test_undistorted/img_001.jpg,85.6,54.0
   ...
   ```
3. Run:
```bash
python measurement/measure.py --validate --gt_csv measurement/ground_truth.csv
```

---

### End-to-End Demo (Step 3 — required deliverable)

```bash
python demo.py --image path/to/any_image.jpg
```

Outputs annotated image + JSON to `demo_outputs/`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Detectron2 not found` | Re-run the pip install command from step 5 |
| `camera_matrix.npy not found` | Run calibration script first |
| `model_final.pth not found` | Run training script first |
| Low mAP | Collect more diverse images; increase `MAX_ITER` in train.py |
| High reprojection error | Recapture calibration images with better lighting + angles |
| CUDA out of memory | Reduce `BATCH_SIZE` in models/train.py to 1 |
