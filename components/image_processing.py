"""
Image processing utilities for the animation pipeline.
Handles color correction, saturation control, and histogram matching.
"""
import numpy as np
import cv2
import skimage.exposure


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

    def apply_histogram_matching(self, image, reference, strength=1.0):
        """
        Matches the color distribution of an image to a reference image.

        Args:
            image: Source RGB numpy array image
            reference: Reference RGB numpy array image
            strength: Blending factor (0.0 = no effect, 1.0 = full matching)

        Returns:
            Color-matched RGB image
        """
        # Apply full histogram matching
        matched = skimage.exposure.match_histograms(
            image, reference, channel_axis=-1
        )

        # Blend between original and matched based on strength
        if strength < 1.0:
            result = (image * (1 - strength) + matched * strength).astype(np.float32)
            result = np.clip(result, 0, 255).astype(np.uint8)
            return result
        else:
            return matched

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

    def apply_color_correction_pipeline(self, image, reference,
                                        saturation_limit=160,
                                        histogram_strength=0.7):
        """
        Apply a complete color correction pipeline

        Args:
            image: Input image to correct
            reference: Reference image for histogram matching
            saturation_limit: Maximum saturation value
            histogram_strength: Strength of histogram matching

        Returns:
            Color-corrected image
        """
        # First limit extreme saturation values to prevent color blowout
        image = self.limit_saturation(image, max_saturation=saturation_limit)

        # Then apply partial histogram matching to maintain consistency
        image = self.apply_histogram_matching(
            image,
            reference,
            strength=histogram_strength
        )

        return image