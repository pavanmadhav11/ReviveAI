import cv2
import numpy as np

def evaluate(reference, restored):
    if reference is None or restored is None:
        raise ValueError("reference and restored images are required")
    if reference.shape[:2] != restored.shape[:2]:
        restored = cv2.resize(restored, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    if reference.ndim != restored.ndim:
        raise ValueError("reference and restored color representations must match")
    a, b = reference.astype(np.float64), restored.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    # JSON has no standard Infinity value; the UI renders null PSNR as ∞ for an exact match.
    psnr = None if mse == 0 else float(10 * np.log10(255 ** 2 / mse))
    # Standard luminance/contrast/structure SSIM components, averaged over 11x11 windows.
    x, y = a, b
    if x.ndim == 3:
        x, y = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.cvtColor(y.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    x, y = x.astype(np.float64), y.astype(np.float64)
    mu_x, mu_y = cv2.GaussianBlur(x, (11, 11), 1.5), cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x*x, (11, 11), 1.5) - mu_x*mu_x
    sigma_y = cv2.GaussianBlur(y*y, (11, 11), 1.5) - mu_y*mu_y
    sigma_xy = cv2.GaussianBlur(x*y, (11, 11), 1.5) - mu_x*mu_y
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    ssim = float(np.mean(((2*mu_x*mu_y+c1)*(2*sigma_xy+c2))/((mu_x*mu_x+mu_y*mu_y+c1)*(sigma_x+sigma_y+c2))))
    return {"mse": mse, "psnr": psnr, "ssim": ssim}
