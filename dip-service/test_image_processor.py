import unittest
import cv2
import numpy as np
from evaluation import evaluate
from image_processor import denoise, enhance_contrast, restore
from scratch_detection import detect_scratches


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((96, 128, 3), 125, dtype=np.uint8)
        cv2.line(self.image, (10, 15), (100, 80), (225, 225, 225), 2)

    def test_denoise_filters_preserve_shape(self):
        for method in ("gaussian", "median", "bilateral"):
            with self.subTest(method=method):
                self.assertEqual(denoise(self.image, method).shape, self.image.shape)

    def test_clahe_preserves_color_shape(self):
        self.assertEqual(enhance_contrast(self.image).shape, self.image.shape)

    def test_scratch_mask_is_binary_and_aligned(self):
        mask, overlay = detect_scratches(self.image, 70)
        self.assertEqual(mask.shape, self.image.shape[:2])
        self.assertEqual(overlay.shape, self.image.shape)
        self.assertTrue(set(np.unique(mask)).issubset({0, 255}))

    def test_detects_crossing_scratches_in_both_polarities(self):
        rng = np.random.default_rng(4)
        gray = np.clip(125 + rng.normal(0, 8, (180, 240)), 0, 255).astype(np.uint8)
        photo = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.line(photo, (4, 45), (235, 45), (220, 220, 220), 2)
        cv2.line(photo, (80, 4), (80, 175), (35, 35, 35), 2)
        cv2.line(photo, (15, 165), (225, 15), (220, 220, 220), 2)
        mask, _ = detect_scratches(photo, 90)
        for x, y in ((120, 45), (80, 90), (120, 90)):
            with self.subTest(point=(x, y)):
                patch = mask[max(0, y - 4):y + 5, max(0, x - 4):x + 5]
                self.assertGreater(float(np.mean(patch > 0)), 0.5)
        self.assertLess(float(np.mean(mask > 0)), 0.15)

    def test_full_pipeline_emits_actual_stages(self):
        stages, duration, _ = restore(self.image, {"mode": "full"})
        self.assertIn("denoised", stages)
        self.assertIn("contrast", stages)
        self.assertIn("scratchMask", stages)
        self.assertIn("inpainted", stages)
        self.assertIn("final", stages)
        final = stages["final"]
        self.assertAlmostEqual(final.shape[1] / final.shape[0], self.image.shape[1] / self.image.shape[0], places=2)
        if "clarity" in stages:
            self.assertGreaterEqual(final.shape[1], self.image.shape[1])
        self.assertGreaterEqual(duration, 0)

    def test_metrics_exact_and_noisy(self):
        self.assertEqual(evaluate(self.image, self.image)["mse"], 0)
        noisy = np.clip(self.image.astype(np.int16) + 5, 0, 255).astype(np.uint8)
        self.assertGreater(evaluate(self.image, noisy)["mse"], 0)


if __name__ == "__main__":
    unittest.main()
