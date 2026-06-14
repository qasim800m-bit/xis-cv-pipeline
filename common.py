"""
============================================================
  XIS CV Pipeline — Shared Utilities
  FILE: utils/common.py

  Reusable functions imported by other modules.
============================================================
"""

import cv2
import numpy as np
import os


def load_calibration(calibration_dir: str = "calibration"):
    """Load camera matrix and distortion coefficients from disk."""
    mtx_path  = os.path.join(calibration_dir, "camera_matrix.npy")
    dist_path = os.path.join(calibration_dir, "dist_coeffs.npy")

    if not os.path.exists(mtx_path) or not os.path.exists(dist_path):
        raise FileNotFoundError(
            f"Calibration files not found in {calibration_dir}. "
            "Run calibration/calibrate_camera.py first."
        )

    mtx  = np.load(mtx_path)
    dist = np.load(dist_path)
    return mtx, dist


def undistort_image(img: np.ndarray, mtx: np.ndarray, dist: np.ndarray,
                    keep_original_size: bool = True) -> np.ndarray:
    """
    Undistort an image using stored intrinsic calibration parameters.

    Args:
        img:                Input BGR image (H × W × 3)
        mtx:                3×3 camera matrix
        dist:               1×5 distortion coefficients
        keep_original_size: If True (default), output is same size as input.
                            alpha=0 → no black borders.

    Returns:
        Undistorted BGR image, same shape as input.
    """
    h, w = img.shape[:2]
    alpha = 0 if keep_original_size else 1
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return undistorted


def get_mask_bbox(mask: np.ndarray):
    """
    Compute bounding box of the largest contour in a binary mask.

    Args:
        mask: H×W bool or uint8 mask

    Returns:
        (x, y, w, h) or None if no contour found
    """
    mask_u8 = (mask * 255).astype(np.uint8) if mask.dtype == bool else mask
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return cv2.boundingRect(max(contours, key=cv2.contourArea))


def px_to_mm(pixels: float, pixels_per_mm: float) -> float:
    """Convert a pixel measurement to millimetres."""
    return round(pixels / pixels_per_mm, 2)


def compute_pixels_per_mm(pixel_width: float, real_width_mm: float) -> float:
    """
    Derive the pixels-per-mm ratio from a reference object.

    Args:
        pixel_width:   Width of the reference object in pixels
        real_width_mm: Known real-world width of the reference object in mm

    Returns:
        pixels_per_mm ratio
    """
    return pixel_width / real_width_mm


def find_images_in_dir(directory: str) -> list:
    """Return sorted list of image paths in a directory."""
    import glob
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)
