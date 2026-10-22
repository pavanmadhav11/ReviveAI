"""Local Real-ESRGAN enhancement, kept behind the DIP service API."""
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


_RUNTIME = Path(__file__).resolve().parent / "ai-runtime" / "runtime"
_EXECUTABLE = _RUNTIME / "realesrgan-ncnn-vulkan.exe"
_MODELS = _RUNTIME / "models"


def available() -> bool:
    return _EXECUTABLE.is_file() and (_MODELS / "realesrgan-x4plus.param").is_file()


def enhance_resolution(image: np.ndarray) -> np.ndarray:
    """Upscale conservatively, limiting model detail that can look invented.

    Real-ESRGAN is useful for recovering edges, but its strong texture prior can
    make faces and hair look artificial. Keep a Lanczos upscale as the base and
    add only a small, bounded part of the model's residual.
    """
    if not available():
        raise RuntimeError("Local restoration model is not installed")

    # Bound output memory for very large scans; the 2x model then produces a
    # result close to the source pixel count instead of a 100 MP image.
    h, w = image.shape[:2]
    scale_down = min(1.0, 3500.0 / max(h, w), np.sqrt(6_000_000 / (h * w)))
    if scale_down < 1.0:
        image = cv2.resize(image, (int(w * scale_down), int(h * scale_down)), interpolation=cv2.INTER_AREA)

    with tempfile.TemporaryDirectory(prefix="reviveai-") as temp_dir:
        temp = Path(temp_dir)
        source, target = temp / "source.png", temp / "restored.png"
        if not cv2.imwrite(str(source), image):
            raise RuntimeError("Could not prepare image for local enhancement")
        command = [
            str(_EXECUTABLE), "-i", str(source), "-o", str(target),
            "-n", "realesrgan-x4plus", "-s", "2", "-t", "128",
            "-m", str(_MODELS), "-f", "png",
        ]
        run = subprocess.run(command, cwd=_RUNTIME, capture_output=True, text=True, timeout=120)
        if run.returncode != 0 or not target.is_file():
            detail = (run.stderr or run.stdout or "Unknown model error").strip()
            raise RuntimeError(f"Local enhancement failed: {detail[-500:]}")
        enhanced = cv2.imread(str(target), cv2.IMREAD_COLOR)
        if enhanced is None:
            raise RuntimeError("Local enhancement produced an unreadable image")

    baseline = cv2.resize(image, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    residual = enhanced.astype(np.float32) - baseline.astype(np.float32)
    # Bound hallucinated contrast and avoid the high-frequency, waxy look from
    # the default model mix. The baseline retains the photo's actual structure.
    residual = np.clip(residual, -22.0, 22.0)
    conservative = baseline.astype(np.float32) + residual * 0.18
    return np.clip(conservative, 0, 255).astype(np.uint8)
