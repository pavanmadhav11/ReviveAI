"""Directional, contrast-based detection for thin scratches and crease lines."""
import cv2
import numpy as np
import logging
from ai_scratch_detection import predict_damage_mask


def _line_kernel(length: int, angle: float) -> np.ndarray:
    """Return a centered one-pixel line structuring element at the given angle."""
    size = length if length % 2 else length + 1
    kernel = np.zeros((size, size), np.uint8)
    center = size // 2
    reach = center - 1
    radians = np.deg2rad(angle)
    dx, dy = int(round(reach * np.cos(radians))), int(round(reach * np.sin(radians)))
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 3)
    return kernel


def _directional_responses(gray: np.ndarray):
    """Return separate bright/dark line responses for each principal orientation."""
    h, w = gray.shape
    length = int(np.clip(round(min(h, w) * 0.055), 11, 31))
    if length % 2 == 0:
        length += 1
    responses = []
    # 30-degree spacing catches steep creases that sit between the main diagonals.
    for angle in (0, 30, 45, 60, 90, 120, 135, 150):
        kernel = _line_kernel(length, angle)
        bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        responses.append((angle, cv2.max(bright, dark), kernel))
    return responses


def _elongation(contour) -> float:
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 3:
        x, y, w, h = cv2.boundingRect(contour)
        return max(w, h) / max(1, min(w, h))
    covariance = np.cov(points, rowvar=False)
    values = np.linalg.eigvalsh(covariance)
    return float(np.sqrt(max(values[-1], 1e-6) / max(values[0], 1e-6)))


def detect_scratches(image: np.ndarray, sensitivity: int = 70):
    """Detect long thin bright/dark defects at multiple orientations.

    The sensitivity slider lowers the contrast threshold. The detector remains
    heuristic: image texture and fine facial details can still be selected, so
    its mask is returned separately for review.
    """
    sensitivity = int(np.clip(sensitivity, 1, 100))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0.4) if min(gray.shape) >= 3 else gray
    try:
        learned_mask = predict_damage_mask(image, sensitivity)
        mask_fraction = float(np.mean(learned_mask > 0))
        # The learned model was trained on real aged-photo damage and can
        # overlook synthetic high-contrast test lines on nearly flat images.
        # Keep the deterministic directional detector for that low-texture case.
        low_texture = float(np.std(gray)) < 30.0
        if 0.0001 < mask_fraction < 0.20 and not low_texture:
            overlay = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            overlay[learned_mask > 0] = (30, 70, 255)
            overlay = cv2.addWeighted(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.62, overlay, 0.38, 0)
            return learned_mask, overlay
        logging.warning("Learned scratch mask coverage %.3f or low-texture input; using classical detector", mask_fraction)
    except Exception as exc:
        logging.warning("Learned scratch detector unavailable; using classical detector: %s", exc)

    responses = _directional_responses(gray)

    # Local response statistics accommodate faded and unevenly lit photographs.
    response = np.maximum.reduce([item[1] for item in responses])
    mean, std = cv2.meanStdDev(response)
    global_floor = max(14, int(46 - sensitivity * 0.27))
    adaptive_floor = int(mean[0, 0] + max(3.0, std[0, 0] * (0.58 - sensitivity * 0.0035)))
    threshold = max(global_floor, adaptive_floor)
    h, w = gray.shape
    min_length = max(12, int(min(h, w) * 0.07))
    max_area = max(30, int(h * w * 0.04))
    mask = np.zeros((h, w), np.uint8)
    for _, direction_response, _kernel in responses:
        raw = cv2.threshold(direction_response, threshold, 255, cv2.THRESH_BINARY)[1]
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, _kernel)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        for label in range(1, count):
            _, _, cw, ch, area = stats[label]
            if area < 4 or area > max_area:
                continue
            component = np.uint8(labels == label) * 255
            contours = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(contour)
            length, width = rect[1]
            major, minor = max(length, width), max(1.0, min(length, width))
            elongated = _elongation(contour)
            local_contrast = float(np.mean(direction_response[labels == label]))
            if major >= min_length and (major / minor >= 3.0 or elongated >= 2.5) and local_contrast >= threshold * 1.25:
                cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)

    # Segment long continuous creases from the aggregate directional response.
    binary = cv2.threshold(response, max(threshold, 24), 255, cv2.THRESH_BINARY)[1]
    segments = cv2.HoughLinesP(binary, 1, np.pi / 180,
                               threshold=max(28, int(min(h, w) * 0.62)),
                               minLineLength=max(min_length, int(min(h, w) * 0.18)),
                               maxLineGap=max(4, int(min(h, w) * 0.055)))
    if segments is not None:
        # Keep the repair core close to the actual crack. A thick mask causes
        # Telea to borrow too much nearby content and can reshape faces.
        width = max(2, min(3, int(round(min(h, w) * 0.005))))
        for x1, y1, x2, y2 in segments[:, 0, :]:
            length = int(np.hypot(x2 - x1, y2 - y1))
            if length < min_length:
                continue
            n = max(2, length // 2)
            xs = np.clip(np.linspace(x1, x2, n).astype(int), 0, w - 1)
            ys = np.clip(np.linspace(y1, y2, n).astype(int), 0, h - 1)
            values = response[ys, xs]
            # Require sustained, strong line evidence to reject accidental grain alignments.
            if np.mean(values >= max(threshold, 24) * 1.4) >= 0.52:
                cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, width)

    # Bridge one-pixel gaps and widen just enough to cover the visible defect.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    overlay = image.copy()
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = (30, 70, 255)
    overlay = cv2.addWeighted(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.62, overlay, 0.38, 0)
    return mask, overlay
