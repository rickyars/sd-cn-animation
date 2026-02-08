"""
FloweR model handler for optical flow prediction.
This module handles loading, using, and clearing the FloweR model.
"""
import os
import torch
import gc
import numpy as np
import cv2
import requests

from ..FloweR.model import FloweR
from ..utils import flow_utils


class FloweRHandler:
    """Handler for the FloweR optical flow prediction model"""

    def __init__(self):
        """Initialize the FloweR handler"""
        self.model = None
        self.device = None

    def clear_memory(self):
        """Clear FloweR model from memory"""
        if self.model is not None:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            self.model = None

    def load_model(self, width, height):
        """
        Load the FloweR model for optical flow prediction

        Args:
            width: Target width for flow prediction
            height: Target height for flow prediction
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define model path - directly in models folder
        model_path = os.path.join('models', 'FloweR_0.1.2.pth')
        remote_model_path = 'https://drive.google.com/uc?id=1-UYsTXkdUkHLgtPK1Y5_7kKzCgzL_Z6o'

        # Download model if needed
        if not os.path.isfile(model_path):
            try:
                from basicsr.utils.download_util import load_file_from_url
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                load_file_from_url(remote_model_path, file_name=model_path)
            except ImportError:
                print(f"Downloading FloweR model to {model_path}...")
                r = requests.get(remote_model_path, allow_redirects=True)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, 'wb') as f:
                    f.write(r.content)

        # Load the model
        self.model = FloweR(input_size=(height, width))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"FloweR model loaded on {self.device}")
        return self.model

    def predict_flow(self, clip_frames):
        """
        Predict optical flow, occlusion mask, and next frame using FloweR

        Args:
            clip_frames: Input frames tensor of shape [window_size, height, width, channels]

        Returns:
            Tuple of (optical flow, occlusion mask, predicted next frame)
        """
        if self.model is None:
            raise ValueError("FloweR model not loaded. Call load_model first.")

        # Prepare input for FloweR
        clip_frames_torch = torch.from_numpy(clip_frames).to(self.device, dtype=torch.float32)
        clip_frames_torch = flow_utils.frames_norm(clip_frames_torch)

        # Predict with FloweR model
        with torch.no_grad():
            pred_data = self.model(clip_frames_torch.unsqueeze(0))[0]

        # Process FloweR outputs
        pred_flow = flow_utils.flow_renorm(pred_data[..., :2]).cpu().numpy()
        pred_occl = flow_utils.occl_renorm(pred_data[..., 2:3]).cpu().numpy().repeat(3, axis=-1)
        pred_next = flow_utils.frames_renorm(pred_data[..., 3:6]).cpu().numpy()

        return pred_flow, pred_occl, pred_next

    def process_flow(self, pred_flow, pred_occl, pred_next, org_size,
                     occlusion_mask_multiplier=10.0, occlusion_mask_blur=0.0):
        """
        Process flow predictions - simplified to match reference implementation

        Note: pred_next already contains flow-guided warping from FloweR model,
        so we don't need to warp again. We just process the occlusion mask.

        Args:
            pred_flow: Raw predicted optical flow (for visualization only)
            pred_occl: Raw predicted occlusion mask
            pred_next: Flow-guided predicted next frame (already warped internally by FloweR)
            org_size: Original size (width, height) to resize outputs
            occlusion_mask_multiplier: Multiplier for occlusion mask strength (default: 10.0)
            occlusion_mask_blur: Gaussian blur sigma for mask smoothing (default: 0.0)

        Returns:
            Dictionary with processed outputs
        """
        # Clip FIRST, then resize - matches reference implementation order
        pred_occl = np.clip(pred_occl * occlusion_mask_multiplier, 0, 255).astype(np.uint8)
        pred_next = np.clip(pred_next, 0, 255).astype(np.uint8)

        # Resize to original dimensions
        pred_flow = cv2.resize(pred_flow, org_size)
        pred_occl = cv2.resize(pred_occl, org_size)
        pred_next = cv2.resize(pred_next, org_size)

        # Optional blur if specified
        if occlusion_mask_blur > 0:
            # Calculate blur kernel size based on image dimensions (match reference)
            h, w = pred_occl.shape[:2]
            blur_filter_size = min(w, h) // 15 | 1  # Ensure odd number
            pred_occl = cv2.GaussianBlur(pred_occl, (blur_filter_size, blur_filter_size),
                                         occlusion_mask_blur, cv2.BORDER_REFLECT)

        # Get grayscale occlusion for inpainting
        pred_occl_gray = np.mean(pred_occl, axis=2).astype(np.uint8)

        return {
            'flow': pred_flow,
            'occlusion': pred_occl,
            'occlusion_gray': pred_occl_gray,
            'predicted_next': pred_next
        }