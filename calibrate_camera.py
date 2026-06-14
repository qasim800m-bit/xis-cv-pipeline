"""
============================================================
  XIS CV Pipeline — Step 1
  FILE: calibration/calibrate_camera.py

  PURPOSE:
    - Detect checkerboard corners in calibration images
    - Compute intrinsic camera matrix and distortion coefficients
    - Save calibration parameters for use in all later steps
    - Report reprojection error

  USAGE:
    1. Print a checkerboard pattern (or display on screen).
       Default expects 8x6 inner corners. Edit CHECKERBOARD below if different.
    2. Place 20+ photos of the checkerboard (varied angles/distances)
       inside:  calibration/images/
    3. Run:
         python calibration/calibrate_camera.py

  OUTPUT:
    calibration/camera_matrix.npy      — intrinsic matrix K (3x3)
    calibration/dist_coeffs.npy        — distortion coefficients
    calibration/calibration_report.txt — reprojection error + parameters
    calibration/undistorted_samples/   — sample undistorted images to verify
============================================================
"""

import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION  ← edit these to match your checkerboard
# ─────────────────────────────────────────────────────────────
CHECKERBOARD      = (8, 6)       # (columns-1, rows-1) inner corners
SQUARE_SIZE_MM    = 25.0         # real-world size of one square in mm
CALIB_IMAGES_DIR  = "calibration/images"
OUTPUT_DIR        = "calibration"
SAMPLES_DIR       = "calibration/undistorted_samples"
# ─────────────────────────────────────────────────────────────


def find_calibration_images(directory: str) -> list:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(images)


def detect_corners(images: list, board: tuple):
    """Detect checkerboard corners in all images."""

    # Prepare 3-D object points for one board view: (0,0,0), (1,0,0) … scaled by square size
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []   # 3-D points in real-world space
    imgpoints = []   # 2-D points in image plane
    good_images = []
    failed_images = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"\n[CALIBRATION] Scanning {len(images)} images for checkerboard {board[0]}x{board[1]} ...")

    for path in images:
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, board, None)

        if ret:
            # Sub-pixel refinement
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            good_images.append(path)
            print(f"  ✓  {os.path.basename(path)}")
        else:
            failed_images.append(path)
            print(f"  ✗  {os.path.basename(path)}  [corners not found]")

    print(f"\n  Detected: {len(good_images)} / {len(images)} images")
    return objpoints, imgpoints, good_images, failed_images, gray.shape[::-1]


def run_calibration(objpoints, imgpoints, image_size):
    """Run OpenCV camera calibration."""
    print("\n[CALIBRATION] Running cv2.calibrateCamera ...")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print(f"  Reprojection Error (RMS): {ret:.4f} px")
    if ret < 0.3:
        print("  ✓  Excellent calibration (< 0.3 px)")
    elif ret < 0.5:
        print("  ✓  Acceptable calibration (< 0.5 px)")
    else:
        print("  ⚠  High reprojection error — capture more varied images")

    return ret, mtx, dist, rvecs, tvecs


def compute_per_image_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """Compute per-image reprojection error for the report."""
    errors = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        errors.append(err)
    return errors


def save_calibration(mtx, dist, ret, output_dir):
    """Save camera matrix and distortion coefficients."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "camera_matrix.npy"), mtx)
    np.save(os.path.join(output_dir, "dist_coeffs.npy"), dist)

    # Also save as JSON for readability
    calib_data = {
        "reprojection_error_rms": float(ret),
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "checkerboard": list(CHECKERBOARD),
        "square_size_mm": SQUARE_SIZE_MM,
    }
    with open(os.path.join(output_dir, "calibration_params.json"), "w") as f:
        json.dump(calib_data, f, indent=2)

    print(f"\n[CALIBRATION] Saved:")
    print(f"  → {output_dir}/camera_matrix.npy")
    print(f"  → {output_dir}/dist_coeffs.npy")
    print(f"  → {output_dir}/calibration_params.json")


def save_report(ret, mtx, dist, per_image_errors, good_images, failed_images, output_dir):
    """Write a human-readable calibration report."""
    report_path = os.path.join(output_dir, "CALIBRATION_REPORT.md")

    with open(report_path, "w") as f:
        f.write("# Camera Calibration Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"| Parameter | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Checkerboard Size | {CHECKERBOARD[0]}×{CHECKERBOARD[1]} inner corners |\n")
        f.write(f"| Square Size | {SQUARE_SIZE_MM} mm |\n")
        f.write(f"| Images Used | {len(good_images)} |\n")
        f.write(f"| Images Failed | {len(failed_images)} |\n")
        f.write(f"| **RMS Reprojection Error** | **{ret:.4f} px** |\n\n")

        rating = "Excellent ✓" if ret < 0.3 else ("Acceptable ✓" if ret < 0.5 else "Poor ⚠")
        f.write(f"**Quality:** {rating}\n\n")

        f.write("## Intrinsic Camera Matrix (K)\n\n")
        f.write("```\n")
        f.write(f"fx={mtx[0,0]:.2f}   0       cx={mtx[0,2]:.2f}\n")
        f.write(f"0       fy={mtx[1,1]:.2f}   cy={mtx[1,2]:.2f}\n")
        f.write(f"0       0       1\n")
        f.write("```\n\n")

        f.write("## Distortion Coefficients\n\n")
        f.write(f"k1={dist[0,0]:.6f}, k2={dist[0,1]:.6f}, "
                f"p1={dist[0,2]:.6f}, p2={dist[0,3]:.6f}, k3={dist[0,4]:.6f}\n\n")

        f.write("## Per-Image Reprojection Errors\n\n")
        f.write("| Image | Error (px) |\n|---|---|\n")
        for img_path, err in zip(good_images, per_image_errors):
            f.write(f"| {os.path.basename(img_path)} | {err:.4f} |\n")

        if failed_images:
            f.write("\n## Failed Images (Corners Not Detected)\n\n")
            for p in failed_images:
                f.write(f"- {os.path.basename(p)}\n")

        f.write("\n## Why Undistortion Is Mandatory for Measurement\n\n")
        f.write("Lens distortion — especially radial distortion — causes straight lines to appear\n")
        f.write("curved and stretches pixels non-uniformly across the image. A pixel near the\n")
        f.write("image border represents a different real-world distance than a pixel at the centre.\n")
        f.write("If pixel-to-mm conversion is applied to a distorted image, measurements will be\n")
        f.write("systematically wrong at all but the image centre. Undistortion maps every pixel\n")
        f.write("back to its ideal position, making the conversion ratio uniform and valid.\n")

    print(f"  → {report_path}")


def save_undistorted_samples(good_images, mtx, dist, samples_dir, n=5):
    """Save a few undistorted samples so you can visually verify."""
    os.makedirs(samples_dir, exist_ok=True)
    sample_paths = good_images[:min(n, len(good_images))]

    for path in sample_paths:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # Side-by-side comparison
        comparison = np.hstack([img, undistorted])
        cv2.putText(comparison, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "UNDISTORTED", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out_path = os.path.join(samples_dir, f"check_{os.path.basename(path)}")
        cv2.imwrite(out_path, comparison)

    print(f"  → {samples_dir}/ ({len(sample_paths)} comparison images saved)")


def main():
    print("=" * 60)
    print("  XIS CV Pipeline — Camera Calibration")
    print("=" * 60)

    # Validate input directory
    if not os.path.isdir(CALIB_IMAGES_DIR):
        print(f"\n[ERROR] Directory not found: {CALIB_IMAGES_DIR}")
        print(f"  Create it and add 20+ checkerboard images.")
        return

    images = find_calibration_images(CALIB_IMAGES_DIR)
    if len(images) < 10:
        print(f"\n[ERROR] Only {len(images)} images found. Need at least 10 (recommended 20+).")
        return

    # Step 1 — detect corners
    objpoints, imgpoints, good_images, failed_images, image_size = \
        detect_corners(images, CHECKERBOARD)

    if len(good_images) < 10:
        print(f"\n[ERROR] Only {len(good_images)} usable images. "
              "Ensure good lighting and the full checkerboard is visible.")
        return

    # Step 2 — calibrate
    ret, mtx, dist, rvecs, tvecs = run_calibration(objpoints, imgpoints, image_size)

    # Step 3 — per-image errors
    per_image_errors = compute_per_image_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)

    # Step 4 — save outputs
    save_calibration(mtx, dist, ret, OUTPUT_DIR)
    save_report(ret, mtx, dist, per_image_errors, good_images, failed_images, OUTPUT_DIR)
    save_undistorted_samples(good_images, mtx, dist, SAMPLES_DIR)

    print("\n[DONE] Calibration complete.")
    print(f"  RMS Reprojection Error: {ret:.4f} px")
    print("  Next step → collect object images and run dataset/undistort_dataset.py")


if __name__ == "__main__":
    main()
