"""
Visualization utilities for the animation pipeline.
Creates visual representations of flow, occlusion masks, and other debug outputs.
"""
import numpy as np
import cv2
import torch

class VisualizationHandler:
    """Visualization utilities for animation pipeline"""

    def __init__(self):
        """Initialize the visualization handler"""
        pass

    def create_flow_visualization(self, flow):
        """
        Create a color-coded visualization of optical flow

        Args:
            flow: Optical flow array of shape [H, W, 2]

        Returns:
            RGB visualization of the flow field
        """
        # Calculate flow magnitude and angle
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        mag, ang = cv2.cartToPolar(fx, fy)

        # Normalize magnitude for better visualization
        mag = np.clip(mag / 30, 0, 1)

        # Create HSV image
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = angle
        hsv[..., 1] = 255  # Saturation = max
        hsv[..., 2] = np.minimum(mag * 255, 255).astype(np.uint8)  # Value = magnitude

        # Convert to RGB for display
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return flow_vis

    def create_tensor_visualization(self, image_np, target_h=None, target_w=None):
        """
        Convert numpy image to visualization tensor in the format expected by ComfyUI

        Args:
            image_np: Numpy image to convert
            target_h: Target height for resizing (optional)
            target_w: Target width for resizing (optional)

        Returns:
            Tensor in format [C, H, W] with values 0-1
        """
        # Ensure image is in uint8 range
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        # Resize if dimensions are provided
        if target_h is not None and target_w is not None:
            image_np = cv2.resize(image_np, (target_w, target_h))

        # Convert to tensor in the format expected by ComfyUI
        tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return tensor

    def prepare_visualization_batch(self, visualization_images, target_h, target_w):
        """
        Prepare a batch of visualization tensors for ComfyUI

        Args:
            visualization_images: List of numpy images
            target_h: Target height for all images
            target_w: Target width for all images

        Returns:
            Tensor batch in format [B, H, W, C] with values 0-1
        """
        tensors = []

        # Process each image
        for img in visualization_images:
            # Skip if None
            if img is None:
                # Create a blank placeholder
                placeholder = np.zeros((target_h, target_w, 3), dtype=np.float32)
                tensor = self.create_tensor_visualization(placeholder, target_h, target_w)
            else:
                tensor = self.create_tensor_visualization(img, target_h, target_w)

            tensors.append(tensor)

        # Stack tensors into batch
        batch = torch.stack(tensors)

        # Convert to ComfyUI format [B, H, W, C]
        batch = batch.permute(0, 2, 3, 1)

        return batch

    def create_occlusion_visualization(self, mask, colormap=cv2.COLORMAP_INFERNO):
        """
        Create a colored visualization of an occlusion mask

        Args:
            mask: Grayscale mask as numpy array
            colormap: OpenCV colormap to apply

        Returns:
            RGB visualization of the mask
        """
        # Ensure mask is grayscale
        if len(mask.shape) > 2 and mask.shape[2] > 1:
            mask = np.mean(mask, axis=2)

        # Normalize to 0-255 range
        mask_norm = np.clip(mask, 0, 255).astype(np.uint8)

        # Apply colormap
        colored_mask = cv2.applyColorMap(mask_norm, colormap)

        # Convert from BGR to RGB
        colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)

        return colored_mask

    def create_side_by_side(self, image1, image2, axis=1):
        """
        Create a side by side or top/bottom visualization of two images

        Args:
            image1: First image as numpy array
            image2: Second image as numpy array
            axis: 1 for side by side, 0 for top/bottom

        Returns:
            Combined image
        """
        # Ensure both images have the same shape
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        if axis == 1:  # Side by side
            # Use same height, adjust widths
            if h1 != h2:
                # Resize second image to match height of first
                image2 = cv2.resize(image2, (int(w2 * h1 / h2), h1))
            return np.concatenate((image1, image2), axis=axis)
        else:  # Top/bottom
            # Use same width, adjust heights
            if w1 != w2:
                # Resize second image to match width of first
                image2 = cv2.resize(image2, (w1, int(h2 * w1 / w2)))
            return np.concatenate((image1, image2), axis=axis)

    def draw_flow_vectors(self, img, flow, step=16, scale=1.0, color=(0, 255, 0)):
        """
        Draw optical flow vectors on an image

        Args:
            img: Background image
            flow: Optical flow field
            step: Grid step size for drawing vectors
            scale: Scale factor for vector magnitude
            color: RGB color for flow lines

        Returns:
            Image with flow vectors drawn
        """
        h, w = img.shape[:2]
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T

        # Create lines
        lines = np.vstack([x, y, x+fx*scale, y+fy*scale]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)

        # Make a copy of the image
        vis = img.copy()

        # Draw flow lines
        cv2.polylines(vis, lines, 0, color)

        # Draw start points
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, color, -1)

        return vis