"""
ControlNet handler for managing ControlNet processing in the animation pipeline.
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
        """
        Set the tile preprocessor type and blur strength

        Args:
            preprocessor_type: Type of preprocessor ("None", "Basic", "ColorFix")
            blur_strength: Strength of blur (0.0-25.0)
        """
        self.tile_preprocessor = preprocessor_type
        self.tile_blur_strength = blur_strength

    def apply_controlnet(self, img_np, positive, negative, control_net, strength,
                         start_percent=0.0, end_percent=1.0, preprocessing_step="unknown"):
        """
        Apply ControlNet to conditioning based on an image
        """
        if control_net is None or img_np is None or strength == 0:
            print(f"[{preprocessing_step}] Skipping ControlNet: None or strength=0")
            return positive, negative, None

        # Make a copy of the image to avoid modifying the original
        img_processed = img_np.copy()

        # Apply tile preprocessing if enabled
        if self.tile_preprocessor != "None":
            print(
                f"[{preprocessing_step}] Applying {self.tile_preprocessor} preprocessing with blur={self.tile_blur_strength}")

            # Apply the selected tile preprocessor
            if self.tile_preprocessor == "ColorFix":
                img_processed = self.tile_colorfix_preprocessor(
                    img_processed,
                    blur_strength=self.tile_blur_strength
                )
            elif self.tile_preprocessor == "Basic":
                img_processed = self.tile_basic_preprocessor(
                    img_processed,
                    blur_strength=self.tile_blur_strength
                )
        else:
            print(f"[{preprocessing_step}] No preprocessing applied (set to None)")

        # Keep processed for return
        processed_for_return = img_processed.copy()

        # Convert numpy array to tensor in the format expected by ControlNet
        img_tensor = torch.from_numpy(img_processed).float() / 255.0

        # CRITICAL: Ensure proper tensor format for ControlNet
        # If tensor is [H, W, C], convert to [B, C, H, W]
        if len(img_tensor.shape) == 3:
            if img_tensor.shape[2] == 3 or img_tensor.shape[2] == 1:  # [H, W, C]
                img_tensor = img_tensor.permute(2, 0, 1)  # Convert to [C, H, W]

        # Ensure batch dimension
        if len(img_tensor.shape) == 3:  # [C, H, W]
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension [B, C, H, W]

        # Move to device
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_tensor = img_tensor.to(target_device)

        # IMPORTANT: Use the ComfyUI-style control hint
        control_hint = img_tensor

        # Pre-create wrapper classes outside the loop to avoid memory leaks
        class PlusPlusInputWrapper:
            def __init__(self, image_tensor, control_type="tile", input_strength=1.0):
                self.image = image_tensor
                self.control_type = control_type
                self.strength = input_strength
            
            def clone(self):
                return PlusPlusInputWrapper(self.image, self.control_type, self.strength)
        
        class ControlHintWrapper:
            def __init__(self, hint_tensor, input_strength=1.0):
                # For Plus Plus models, create a controls dict with PlusPlusInput objects
                plus_input = PlusPlusInputWrapper(hint_tensor, "tile", input_strength)
                self.controls = {'tile': plus_input}

        # Process each positive conditioning item
        new_positive = []
        for i, t in enumerate(positive):
            try:
                # Handle different conditioning formats
                if isinstance(t, dict):
                    # CFGGuider format: dict with 'cross_attn' and other keys
                    if 'cross_attn' in t:
                        # Convert to ComfyUI conditioning format [tensor, dict]
                        cond_tensor = t['cross_attn']
                        cond_dict = {
                            'pooled_output': t.get('pooled_output'),
                            'model_conds': t.get('model_conds', {})
                        }
                        n = [cond_tensor, cond_dict.copy()]
                    else:
                        print(f"Unknown dict conditioning format for item {i}: {list(t.keys())}")
                        continue
                elif isinstance(t, (list, tuple)) and len(t) >= 2:
                    # Standard ComfyUI format: [tensor, dict]
                    n = [t[0], t[1].copy()]
                else:
                    print(f"Unexpected conditioning format for item {i}: {type(t)}")
                    continue
            except (IndexError, KeyError, TypeError) as e:
                print(f"Error processing positive conditioning item {i}: {e}")
                print(f"Item type: {type(t)}")
                continue
            
            # OPTIMIZATION: Only copy the control net once, not per conditioning item
            # This prevents repeated model loading and memory leaks
            c_net = control_net.copy() if i == 0 else control_net

            # Set the conditioning hint - handle different ControlNet types
            if hasattr(c_net, 'get_control_advanced') or 'PlusPlus' in str(type(c_net)):
                # ControlNet Plus Plus format - needs controls dict with PlusPlusInput objects
                c_net.cond_hint_original = ControlHintWrapper(control_hint, strength)
            else:
                # Standard ControlNet format - direct tensor
                c_net.cond_hint_original = control_hint
            
            c_net.strength = strength
            c_net.timestep_percent_range = (start_percent, end_percent)

            # Link to previous controlnet if one exists
            if 'control' in n[1]:
                c_net.previous_controlnet = n[1]['control']

            # Set the control net in the conditioning
            n[1]['control'] = c_net
            n[1]['control_apply_to_uncond'] = False
            new_positive.append(n)

        # Process each negative conditioning item
        new_negative = []
        for i, t in enumerate(negative):
            try:
                # Handle different conditioning formats
                if isinstance(t, dict):
                    # CFGGuider format: dict with 'cross_attn' and other keys
                    if 'cross_attn' in t:
                        # Convert to ComfyUI conditioning format [tensor, dict]
                        cond_tensor = t['cross_attn']
                        cond_dict = {
                            'pooled_output': t.get('pooled_output'),
                            'model_conds': t.get('model_conds', {})
                        }
                        n = [cond_tensor, cond_dict.copy()]
                    else:
                        print(f"Unknown dict conditioning format for item {i}: {list(t.keys())}")
                        continue
                elif isinstance(t, (list, tuple)) and len(t) >= 2:
                    # Standard ComfyUI format: [tensor, dict]
                    n = [t[0], t[1].copy()]
                else:
                    print(f"Unexpected conditioning format for item {i}: {type(t)}")
                    continue
            except (IndexError, KeyError, TypeError) as e:
                print(f"Error processing negative conditioning item {i}: {e}")
                print(f"Item type: {type(t)}")
                continue
            
            # OPTIMIZATION: Only copy the control net once, not per conditioning item
            # This prevents repeated model loading and memory leaks
            c_net = control_net.copy() if i == 0 else control_net

            # Set the conditioning hint - handle different ControlNet types
            if hasattr(c_net, 'get_control_advanced') or 'PlusPlus' in str(type(c_net)):
                # ControlNet Plus Plus format - reuse wrapper classes defined earlier
                c_net.cond_hint_original = ControlHintWrapper(control_hint, strength)
            else:
                # Standard ControlNet format - direct tensor
                c_net.cond_hint_original = control_hint
            
            c_net.strength = strength
            c_net.timestep_percent_range = (start_percent, end_percent)

            # Link to previous controlnet if one exists
            if 'control' in n[1]:
                c_net.previous_controlnet = n[1]['control']

            # Set the control net in the conditioning
            n[1]['control'] = c_net
            new_negative.append(n)

        # Explicit cleanup to prevent memory leaks
        del PlusPlusInputWrapper, ControlHintWrapper
        
        return new_positive, new_negative, processed_for_return

    def tile_basic_preprocessor(self, image, blur_strength=5.0):
        """
        Basic tile preprocessor (tile_resample equivalent)
        Applies Gaussian blur to the image

        Args:
            image: Input image as numpy array
            blur_strength: Strength of blur (0.0-25.0)

        Returns:
            Processed image
        """
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Make a copy to avoid modifying the original
        result = image.copy()

        # Apply Gaussian blur - key for tile preprocessor
        if blur_strength > 0:
            # Calculate kernel size based on blur strength (must be odd)
            kernel_size = max(3, int(blur_strength * 2)) | 1  # Ensure odd number
            result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)

        return result

    def tile_colorfix_preprocessor(self, image, blur_strength=5.0):
        """
        Enhanced tile preprocessor with color preservation (tile_colorfix equivalent)

        Args:
            image: Input image as numpy array
            blur_strength: Strength of blur (0.0-25.0)

        Returns:
            Processed image
        """
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Make a copy to avoid modifying the original
        img_copy = image.copy()

        # Convert to LAB color space to preserve colors while blurring
        # LAB separates luminance from color information
        lab = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LAB)

        # Split channels
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Only blur the L (luminance) channel
        if blur_strength > 0:
            kernel_size = max(3, int(blur_strength * 2)) | 1  # Ensure odd number
            l_channel = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), 0)

        # Merge channels back
        lab = cv2.merge([l_channel, a_channel, b_channel])

        # Convert back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return result