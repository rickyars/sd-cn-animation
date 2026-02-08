"""
Image processing utilities for the animation pipeline.
Handles color correction and saturation control.
"""
import numpy as np
import cv2


class ImageProcessor:
    """Image processing operations for animation pipeline"""

    def __init__(self):
        """Initialize the image processor"""
        pass

    def limit_saturation(self, image, max_saturation=160):
        """
        Prevents oversaturation by capping saturation values in HSV color space.

        Args:
            image: RGB numpy array image
            max_saturation: Maximum saturation value (0-255)

        Returns:
            RGB image with limited saturation
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, max_saturation)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


    def resize_to_even_dimensions(self, image, divisor=8):
        """
        Resize image to dimensions divisible by the given divisor

        Args:
            image: Input image as numpy array
            divisor: Ensure dimensions are divisible by this number (default: 8)

        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_h = (h // divisor) * divisor
        target_w = (w // divisor) * divisor

        if target_h != h or target_w != w:
            return cv2.resize(image, (target_w, target_h))
        return image

    def match_image_sizes(self, images):
        """
        Resize all images to the smallest common dimensions

        Args:
            images: List of images as numpy arrays

        Returns:
            List of resized images
        """
        if not images:
            return []

        # Find minimum dimensions
        min_h = min(img.shape[0] for img in images)
        min_w = min(img.shape[1] for img in images)

        # Resize all images to minimum dimensions
        return [cv2.resize(img, (min_w, min_h)) for img in images]

    def match_histograms(self, source, reference):
        """
        Match the histogram of source to reference image (per-channel).
        Equivalent to skimage.exposure.match_histograms used in reference implementation.
        Prevents color drift over time by anchoring to the initial frame.

        Args:
            source: Source image (numpy array, uint8)
            reference: Reference image to match to (numpy array, uint8)

        Returns:
            Color-matched image (numpy array, uint8)
        """
        result = np.zeros_like(source)
        for c in range(3):
            src_vals = source[:, :, c].ravel()
            ref_vals = reference[:, :, c].ravel()

            # Compute CDFs
            src_counts, _ = np.histogram(src_vals, bins=256, range=(0, 256))
            ref_counts, _ = np.histogram(ref_vals, bins=256, range=(0, 256))
            src_cdf = np.cumsum(src_counts).astype(np.float64)
            ref_cdf = np.cumsum(ref_counts).astype(np.float64)
            src_cdf /= src_cdf[-1]
            ref_cdf /= ref_cdf[-1]

            # Build mapping: for each source intensity, find closest reference intensity
            mapping = np.interp(src_cdf, ref_cdf, np.arange(256))
            result[:, :, c] = mapping[source[:, :, c]].astype(np.uint8)

        return result

    def apply_color_correction_pipeline(self, image, saturation_limit=200):
        """
        Apply a minimal color correction pipeline

        Args:
            image: Input image to correct
            saturation_limit: Maximum saturation value

        Returns:
            Color-corrected image
        """
        # Limit extreme saturation values to prevent color blowout
        return self.limit_saturation(image, max_saturation=saturation_limit)