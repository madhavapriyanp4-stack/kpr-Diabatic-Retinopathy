"""
Fundus image preprocessing: crop to the retina's bounding circle and
apply CLAHE (contrast-limited adaptive histogram equalization) so lesions
stand out against the background. Mirrors the preprocessing steps used
in the reference literature (green-channel CLAHE + circular crop).
"""
import cv2
import numpy as np


def crop_to_retina(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Remove the black border around the retina and center it."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def apply_clahe(img: np.ndarray, clip_limit: float = 2.5, tile: int = 8) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB color space (preserves color)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_fundus(img: np.ndarray, size: int = 380) -> np.ndarray:
    """Full pipeline: crop -> CLAHE -> resize to a square input."""
    img = crop_to_retina(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img
