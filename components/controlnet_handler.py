"""
ControlNet handler for managing ControlNet processing in the animation pipeline.
Uses standard ComfyUI copy() + set_cond_hint() flow for compatibility with all
ControlNet types including ControlNet++ from comfyui-advanced-controlnet.
"""
import torch
import numpy as np
import cv2


class ControlNetHandler:
    """Handler for ControlNet operations in the animation pipeline"""

    def __init__(self):
        """Initialize the ControlNet handler"""
        self.tile_preprocessor = "ColorFix"  # Default preprocessor
        self.tile_blur_strength = 5.0  # Default blur strength

    def set_preprocessor(self, preprocessor_type, blur_strength=5.0):
        self.tile_preprocessor = preprocessor_type
        self.tile_blur_strength = blur_strength

    def _preprocess_image(self, img_np):
        """Apply tile preprocessing to an image and return [B, C, H, W] tensor 0-1"""
        img_processed = img_np.copy()

        if self.tile_preprocessor == "ColorFix":
            img_processed = self.tile_colorfix_preprocessor(img_processed, blur_strength=self.tile_blur_strength)
        elif self.tile_preprocessor == "Basic":
            img_processed = self.tile_basic_preprocessor(img_processed, blur_strength=self.tile_blur_strength)

        # Convert uint8 numpy [H, W, C] to float [B, C, H, W] tensor (ComfyUI convention)
        # ComfyUI IMAGE type is [B, H, W, C] 0-1, but set_cond_hint expects [B, C, H, W]
        # (the standard apply nodes do image.movedim(-1, 1) before passing to set_cond_hint)
        img_tensor = torch.from_numpy(img_processed).float() / 255.0  # [H, W, C] 0-1
        img_tensor = img_tensor.unsqueeze(0)  # [1, H, W, C]
        img_tensor = img_tensor.movedim(-1, 1)  # [1, C, H, W] — matches ComfyUI convention
        return img_processed, img_tensor

    def apply_controlnet(self, img_np, positive, negative, control_net, strength,
                         start_percent=0.0, end_percent=1.0, preprocessing_step="unknown"):
        """
        Apply ControlNet to conditioning using standard ComfyUI copy()+set_cond_hint() flow.
        Compatible with all ControlNet types including ControlNet++.

        Returns: (new_positive, new_negative, c_net_copy)
            c_net_copy is the ControlNet copy to be used for updating hints in subsequent frames.
        """
        if control_net is None or img_np is None or strength == 0:
            return positive, negative, None

        processed_img, control_hint = self._preprocess_image(img_np)

        # Use standard ComfyUI flow: copy() + set_cond_hint()
        # copy() creates a new ControlNet sharing the model weights (no VRAM duplication)
        # set_cond_hint() properly handles all ControlNet types:
        #   - Standard ControlNet: stores tensor directly
        #   - ControlNet++ (advanced): set_cond_hint_inject wraps into PlusPlusInputGroup
        c_net = control_net.copy()

        # Fix: ControlNetPlusPlusAdvanced.copy() creates a NEW control_model_wrapped
        # ModelPatcher in its __init__ instead of sharing the original's (unlike standard
        # ControlNet.copy() which passes None to skip creation, then assigns the original).
        # This orphan ModelPatcher triggers ComfyUI's "memory leak" detection when GC'd.
        # Patch the copy to share the original's ModelPatcher, matching standard behavior.
        if hasattr(control_net, 'control_model_wrapped') and hasattr(c_net, 'control_model_wrapped'):
            if c_net.control_model_wrapped is not control_net.control_model_wrapped:
                c_net.control_model_wrapped = control_net.control_model_wrapped

        c_net.set_cond_hint(control_hint, strength, (start_percent, end_percent))

        # Build new conditioning with ControlNet attached
        # Uses same pattern as ComfyUI's ControlNetApplyAdvanced node
        #
        # Conditioning can be in two formats:
        #   1. Standard ComfyUI: list of [tensor, dict] pairs (from CONDITIONING type)
        #   2. Internal/guider format: list of dicts with 'cross_attn' key (from guider.original_conds)
        # We handle both and always output format #1 for set_conds() compatibility.
        cnets = {}
        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                if isinstance(t, (list, tuple)) and len(t) >= 2:
                    # Standard format: [tensor, dict]
                    cond_tensor = t[0]
                    d = t[1].copy()
                elif isinstance(t, dict):
                    # Internal guider format: dict with 'cross_attn' key
                    d = t.copy()
                    cond_tensor = d.pop('cross_attn', None)
                else:
                    print(f"Unexpected conditioning format: {type(t)}")
                    continue

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net_for_cond = cnets[prev_cnet]
                else:
                    c_net_for_cond = c_net
                    c_net_for_cond.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net_for_cond

                d['control'] = c_net_for_cond
                d['control_apply_to_uncond'] = False
                c.append([cond_tensor, d])
            out.append(c)

        # Return the c_net copy so it can be updated with new hints per-frame
        return out[0], out[1], c_net

    def update_controlnet_hint(self, c_net, img_np, strength, start_percent=0.0, end_percent=1.0):
        """
        Update the ControlNet hint image for a new frame (reuses existing copy).
        Uses set_cond_hint() which updates cond_hint_original.
        The cached cond_hint is regenerated during the next forward pass.
        """
        if c_net is None or img_np is None:
            return

        _, control_hint = self._preprocess_image(img_np)
        c_net.set_cond_hint(control_hint, strength, (start_percent, end_percent))

    def tile_basic_preprocessor(self, image, blur_strength=5.0):
        """Basic tile preprocessor - applies Gaussian blur"""
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        result = image.copy()
        if blur_strength > 0:
            kernel_size = max(3, int(blur_strength * 2)) | 1
            result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
        return result

    def tile_colorfix_preprocessor(self, image, blur_strength=5.0):
        """Enhanced tile preprocessor - blurs luminance only, preserving colors"""
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        img_copy = image.copy()
        lab = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        if blur_strength > 0:
            kernel_size = max(3, int(blur_strength * 2)) | 1
            l_channel = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), 0)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
