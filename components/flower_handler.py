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

    def process_flow(self, pred_flow, pred_occl, pred_next, prev_frame, org_size,
                     occlusion_mask_multiplier=5.0, occlusion_flow_multiplier=1.0,
                     occlusion_difo_multiplier=1.0, occlusion_mask_blur=5.0,
                     occlusion_difs_multiplier=2.0):
        """
        Process flow predictions and create warped frame

        Args:
            pred_flow: Raw predicted optical flow
            pred_occl: Raw predicted occlusion mask
            pred_next: Raw predicted next frame
            prev_frame: Previous frame as reference
            org_size: Original size (width, height) to resize outputs
            occlusion_*: Various occlusion parameters

        Returns:
            Dictionary with processed flow outputs
        """
        # Apply multipliers to flow and occlusion
        pred_flow = pred_flow * occlusion_flow_multiplier

        # General multiplier for the occlusion mask
        pred_occl = np.clip(pred_occl * occlusion_mask_multiplier, 0, 255).astype(np.uint8)

        # Additional processing for flow
        flow_magnitude = np.linalg.norm(pred_flow, axis=-1, keepdims=True)
        difo_factor = 1.0 / (1.0 + flow_magnitude * 0.05 * occlusion_difo_multiplier)
        pred_flow = pred_flow * difo_factor

        # Resize to original dimensions
        pred_flow = cv2.resize(pred_flow, org_size)
        pred_occl = cv2.resize(pred_occl, org_size)
        pred_next = cv2.resize(pred_next, org_size)

        # Clean up and ensure proper ranges
        pred_next = np.clip(pred_next, 0, 255).astype(np.uint8)

        # Process occlusion mask with difs multiplier
        pred_occl = cv2.GaussianBlur(pred_occl, (21, 21), 2, cv2.BORDER_REFLECT_101)
        pred_occl = (np.abs(pred_occl / 255.0) ** 1.5) * 255.0 * occlusion_difs_multiplier
        pred_occl = np.clip(pred_occl, 0, 255).astype(np.uint8)

        # Apply optical flow to warp the previous frame
        h, w = pred_flow.shape[:2]
        flow_map = pred_flow.copy()
        flow_map[:, :, 0] += np.arange(w)
        flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]

        warped_frame = cv2.remap(prev_frame, flow_map, None, cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_REFLECT_101)

        # Blend warped frame with predicted next frame using occlusion mask
        alpha_mask = pred_occl / 255.0
        blended_frame = pred_next.astype(float) * alpha_mask + warped_frame.astype(float) * (1 - alpha_mask)
        blended_frame = np.clip(blended_frame, 0, 255).astype(np.uint8)

        # Get grayscale occlusion for inpainting
        pred_occl_gray = np.mean(pred_occl, axis=2).astype(np.uint8)

        return {
            'flow': pred_flow,
            'occlusion': pred_occl,
            'occlusion_gray': pred_occl_gray,
            'predicted_next': pred_next,
            'warped_frame': warped_frame,
            'blended_frame': blended_frame
        }