"""Microsoft's pretrained old-photo scratch detector, run locally on CPU."""
from pathlib import Path
import os
import sys

import cv2
import numpy as np


_ROOT = Path(__file__).resolve().parent / "ai-runtime" / "old-photo-scratch"
_WEIGHTS = _ROOT / "FT_Epoch_latest.pt"
_UPSTREAM = _ROOT / "upstream"
_MODEL = None
_TORCH = None


def available() -> bool:
    return _WEIGHTS.is_file()


def _load_model():
    global _MODEL, _TORCH
    if _MODEL is not None:
        return _MODEL
    if not available():
        raise RuntimeError("Old-photo scratch model weights are missing")

    sys.path.insert(0, str(_UPSTREAM))
    import torch
    from detection_models.networks import UNet

    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
    model = UNet(in_channels=1, out_channels=1, depth=4, conv_num=2, wf=6,
                 padding=True, batch_norm=True, up_mode="upsample",
                 with_tanh=False, sync_bn=False, antialiasing=True)
    checkpoint = torch.load(_WEIGHTS, map_location="cpu", weights_only=True)
    state = checkpoint["model_state"]
    if state and next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    _MODEL, _TORCH = model, torch
    return model


def predict_damage_mask(image: np.ndarray, sensitivity: int = 72) -> np.ndarray:
    """Return a source-aligned 0/255 scratch mask from the trained U-Net."""
    model = _load_model()
    torch = _TORCH
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Match the upstream detector's scale_256 inference while retaining the
    # source aspect ratio, then restore the mask to the exact source geometry.
    scale = min(1.0, 256.0 / max(h, w))
    target_h = max(16, int(round((h * scale) / 16) * 16))
    target_w = max(16, int(round((w * scale) / 16) * 16))
    small = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    tensor = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None]
    tensor = (tensor - 0.5) / 0.5
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    threshold = float(np.clip(0.65 - int(sensitivity) * 0.0035, 0.25, 0.62))
    predicted = (probability >= threshold).astype(np.uint8) * 255
    mask = cv2.resize(predicted, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask
