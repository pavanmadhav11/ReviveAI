import time
import logging
import cv2
import numpy as np
from scratch_detection import detect_scratches
from ai_restoration import enhance_resolution


def denoise(image, method="bilateral", strength=5):
    strength = int(np.clip(strength, 1, 15))
    if method in ("nlmeans", "auto"):
        # Non-local means averages similar patches across the photo, so it
        # removes film grain without smearing edges like a large Gaussian blur.
        h = float(np.clip(2.0 + strength * 0.42, 2.5, 8))
        if image.ndim == 2:
            return cv2.fastNlMeansDenoising(image, None, h, 7, 21)
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    if method == "gaussian":
        return cv2.GaussianBlur(image, (0, 0), max(0.35, strength / 7))
    if method == "median":
        radius = max(1, int(round(strength / 2)))
        return cv2.medianBlur(image, 2 * radius + 1)
    # Bilateral preserves some boundaries while smoothing similarly colored regions.
    return cv2.bilateralFilter(image, 5 + 2 * strength, 10 + 3 * strength, 8 + 2 * strength)


def enhance_contrast(image, clip_limit=1.55, tile_grid=8):
    grid = int(np.clip(tile_grid, 2, 16))
    clahe = cv2.createCLAHE(clipLimit=float(np.clip(clip_limit, 1, 5)), tileGridSize=(grid, grid))
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen(image, amount=0.45):
    alpha = float(np.clip(amount, 0, 0.55))
    blur = cv2.GaussianBlur(image, (0, 0), 1.3)
    return np.clip(image.astype(np.float32) + alpha * (image.astype(np.float32) - blur.astype(np.float32)), 0, 255).astype(np.uint8)


def restore(image, options=None):
    options = options or {}
    mode = options.get("mode", "full")
    operations = {
        "quick": {"denoise": True, "contrast": True, "scratch": False, "sharpen": False},
        "scratch": {"denoise": False, "contrast": False, "scratch": True, "sharpen": False},
        "full": {"denoise": True, "contrast": True, "scratch": True, "sharpen": True},
        "custom": {k: bool(options.get(k, False)) for k in ("denoise", "contrast", "scratch", "sharpen")},
    }.get(mode, None)
    if operations is None:
        raise ValueError("mode must be quick, scratch, full, or custom")
    start = time.perf_counter()
    stages = {"original": image.copy()}
    current = image.copy()
    # Build the damage mask before denoising/CLAHE changes the scratch contrast.
    damage_mask = damage_overlay = None
    if operations["scratch"]:
        damage_mask, damage_overlay = detect_scratches(image, options.get("sensitivity", 70))
    if operations["denoise"]:
        current = denoise(current, options.get("noiseMethod", "bilateral"), options.get("strength", 2))
        stages["denoised"] = current.copy()
    if operations["contrast"] and bool(options.get("contrast", True)):
        current = enhance_contrast(current, options.get("clipLimit", 1.55), options.get("tileGrid", 8))
        stages["contrast"] = current.copy()
    if operations["scratch"]:
        stages["scratchMask"] = damage_mask
        stages["scratchOverlay"] = damage_overlay
        method = options.get("inpaintMethod", "telea")
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        radius = float(np.clip(options.get("radius", 3), 1, 8))
        current = cv2.inpaint(current, damage_mask, radius, flag)
        stages["inpainted"] = current.copy()
    if operations["sharpen"]:
        current = sharpen(current, options.get("sharpness", 0.22))
        stages["sharpened"] = current.copy()
    try:
        current = enhance_resolution(current)
        stages["clarity"] = current.copy()
    except Exception as exc:
        # The classical restoration remains usable if the local model or GPU
        # runtime is unavailable on another computer.
        logging.warning("Local clarity enhancement unavailable: %s", exc)
    stages["final"] = current.copy()
    return stages, round((time.perf_counter() - start) * 1000, 2), operations
