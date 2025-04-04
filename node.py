import os
import torch
import numpy as np
import cv2
import gc
import comfy.sample
from PIL import Image

# Import flow_utils from utils folder
from .utils import flow_utils
from .FloweR.model import FloweR


class Txt2VidNode:
    def __init__(self):
        self.FloweR_model = None
        self.DEVICE = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),  # Diffusion model
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),  # Initial latent
                "vae": ("VAE",),
                "control_net": ("CONTROL_NET",),  # Keep ControlNet input
                "num_frames": ("INT", {
                    "default": 16,
                    "min": 2,
                    "max": 1000,
                    "step": 1
                }),
                "cn_frame_send": (["None", "Current Frame", "Previous Frame"], {
                    "default": "Previous Frame",
                }),
                "tile_preprocessor": (["None", "Basic", "ColorFix"], {
                    "default": "ColorFix",
                    "label": "Tile Preprocessor Type"
                }),
                "tile_blur_strength": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 25.0,
                    "step": 0.1,
                    "label": "Tile Preprocessor Blur"
                }),
                "controlnet_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "controlnet_start_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "controlnet_end_percent": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "sampling_steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 150,
                    "step": 1
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
                                  "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "ddim"],),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"],),
                "processing_strength": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "fix_frame_strength": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "occlusion_mask_blur": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1
                }),
                "occlusion_mask_multiplier": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "occlusion_flow_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "occlusion_difo_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "occlusion_difs_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("first_pass_frames", "second_pass_frames", "flow_visualization", "occlusion_mask",
                    "warped_frame", "blended_frame", "preprocessed_controlnet_input")
    FUNCTION = "generate_frames"
    CATEGORY = "animation"

    def FloweR_clear_memory(self):
        """Clear FloweR model from memory"""
        if self.FloweR_model is not None:
            del self.FloweR_model
            gc.collect()
            torch.cuda.empty_cache()
            self.FloweR_model = None

    def FloweR_load_model(self, w, h):
        """Load the FloweR model for optical flow prediction"""
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                import requests
                print(f"Downloading FloweR model to {model_path}...")
                r = requests.get(remote_model_path, allow_redirects=True)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, 'wb') as f:
                    f.write(r.content)

        # Load the model
        self.FloweR_model = FloweR(input_size=(h, w))
        self.FloweR_model.load_state_dict(torch.load(model_path, map_location=self.DEVICE))
        self.FloweR_model = self.FloweR_model.to(self.DEVICE)
        self.FloweR_model.eval()

    def tile_basic_preprocessor(self, image, blur_strength=5.0):
        """
        Basic tile preprocessor (tile_resample equivalent)
        Applies Gaussian blur to the image
        """
        import cv2
        import numpy as np

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
        """
        import cv2
        import numpy as np

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

    def apply_controlnet(self, img_np, positive, negative, control_net, strength,
                         start_percent=0.0, end_percent=1.0, preprocessing_step="unknown"):
        """Apply ControlNet to conditioning based on an image"""
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
        if len(img_tensor.shape) == 3 and img_tensor.shape[2] == 3:  # [H, W, C]
            img_tensor = img_tensor.permute(2, 0, 1)  # Convert to [C, H, W]
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]

        # Get the device
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'device'):
            target_device = self.model.model.device

        img_tensor = img_tensor.to(target_device)

        # Prepare control hint - move channels dimension to position 1 as expected
        control_hint = img_tensor.movedim(-1, 1) if img_tensor.dim() == 4 and img_tensor.shape[-1] in [1, 3,
                                                                                                       4] else img_tensor

        # Create new conditioning list
        new_positive = []

        # Process each conditioning item
        for t in positive:
            # Create a new conditioning entry with a copy of the cond dict
            n = [t[0], t[1].copy()]

            # Create a copy of the control_net and set the conditioning hint and strength
            c_net = control_net.copy()

            # Set the conditioning hint (image), strength, and timestep range
            c_net.cond_hint_original = control_hint
            c_net.strength = strength
            c_net.timestep_percent_range = (start_percent, end_percent)

            # Link to previous controlnet if one exists
            if 'control' in t[1]:
                c_net.previous_controlnet = t[1]['control']

            # Set the control net in the conditioning
            n[1]['control'] = c_net

            # Apply to uncond (negative prompt) as well
            n[1]['control_apply_to_uncond'] = True

            # Add to the new conditioning list
            new_positive.append(n)

        return new_positive, negative, processed_for_return

    def encode_with_vae(self, pil_image, vae, device):
        """Simplified VAE encoding function for PIL images"""
        # Convert PIL to numpy
        img = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0

        # Convert to tensor with batch dimension
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        # Encode with VAE
        with torch.no_grad():
            latent = vae.encode(img_tensor)

        return latent

    def inpaint_with_mask(self, image_pil, mask_pil, strength, steps, seed, device):
        """Inpainting using ComfyUI's approach with latent noise mask"""
        # Get the latent tensor
        image_latent = self.encode_with_vae(image_pil, self.vae, device)

        # Process mask
        if mask_pil.mode != 'L':
            mask_pil = mask_pil.convert('L')
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0

        # Convert to tensor
        mask_tensor = torch.from_numpy(mask_np).to(device)

        # Add the noise_mask to the latent dict EXACTLY as in SetLatentNoiseMask
        noise_mask = mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1]))

        # Generate noise
        noise = comfy.sample.prepare_noise(image_latent, seed, None)

        # Sample using standard method
        samples = comfy.sample.sample(
            self.model,
            noise,
            steps,
            self.cfg,
            self.sampler_name,
            self.scheduler,
            self.positive,
            self.negative,
            image_latent,
            denoise=strength,
            noise_mask=noise_mask
        )

        return {"samples": samples}

    def sample_diffusion(self, latent, strength, steps, seed):
        """Sample the diffusion model using ComfyUI's samplers"""

        # Get batch indices if available
        batch_inds = latent.get("batch_index", None)

        # Prepare noise using ComfyUI's method
        noise = comfy.sample.prepare_noise(latent["samples"], seed, batch_inds)

        # Run the sampler
        samples = comfy.sample.sample(
            self.model,
            noise,
            steps,
            self.cfg,
            self.sampler_name,
            self.scheduler,
            self.positive,
            self.negative,
            latent["samples"],
            denoise=strength
        )

        return {"samples": samples}

    def limit_saturation(self, image, max_saturation=160):
        """
        Prevents oversaturation by capping saturation values in HSV color space.

        Args:
            image: RGB numpy array image
            max_saturation: Maximum saturation value (0-255)

        Returns:
            RGB image with limited saturation
        """
        import cv2
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
        import skimage.exposure
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

    def create_flow_visualization(self, flow):
        """Create a color-coded visualization of optical flow"""
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

    def generate_frames(self, model, positive, negative, latent, vae, control_net, num_frames, cn_frame_send,
                        controlnet_strength, controlnet_start_percent, controlnet_end_percent,
                        sampling_steps, cfg, sampler_name, scheduler, processing_strength, fix_frame_strength,
                        occlusion_mask_blur, occlusion_mask_multiplier, occlusion_flow_multiplier,
                        occlusion_difo_multiplier, occlusion_difs_multiplier, seed, tile_preprocessor,
                        tile_blur_strength):
        """Generate a sequence of latent frames with iterative diffusion"""
        # Store model and conditioning for internal use
        self.model = model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.cfg = cfg
        self.occlusion_mask_blur = occlusion_mask_blur
        self.occlusion_mask_multiplier = occlusion_mask_multiplier
        self.occlusion_flow_multiplier = occlusion_flow_multiplier
        self.occlusion_difo_multiplier = occlusion_difo_multiplier
        self.occlusion_difs_multiplier = occlusion_difs_multiplier

        # Store the selected tile preprocessor type
        self.tile_preprocessor = tile_preprocessor
        self.tile_blur_strength = tile_blur_strength

        # Initialize collections for debug visualizations
        flow_visualizations = []
        occlusion_masks = []
        warped_frames = []
        blended_frames = []
        preprocessed_controlnet_inputs = []

        # Convert cn_frame_send string to numeric value
        cn_frame_options = ["None", "Current Frame", "Previous Frame"]
        self.cn_frame_value = cn_frame_options.index(cn_frame_send)

        # Use CUDA device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Step 1: Get initial frame by decoding the input latent
        print("Decoding initial latent")
        # Ensure latent is on the correct device
        latent_on_device = {"samples": latent["samples"].to(device)}
        init_frame_tensor = vae.decode(latent_on_device["samples"])

        # Convert tensor to numpy for processing
        init_frame = init_frame_tensor[0].cpu()
        if init_frame.shape[0] == 3:  # [C, H, W]
            init_frame_np = init_frame.permute(1, 2, 0).numpy() * 255
        else:  # Assume [H, W, C]
            init_frame_np = init_frame.numpy() * 255

        init_frame_np = np.clip(init_frame_np, 0, 255).astype(np.uint8)
        print(f"Initial frame shape: {init_frame_np.shape}")

        # Get dimensions
        height, width = init_frame_np.shape[:2]

        # Setup for FloweR model - ensure size is divisible by 128
        size = (width // 128) * 128, (height // 128) * 128
        if size[0] == 0 or size[1] == 0:
            size = 128, 128
        org_size = (width, height)

        print(f"Loading FloweR model with size: {size}")
        self.FloweR_load_model(size[0], size[1])

        # Initialize frames buffer for FloweR (needs 4 frames for context)
        clip_frames = np.zeros((4, size[1], size[0], 3), dtype=np.uint8)
        prev_frame = init_frame_np.copy()

        # Initialize latent frames array with initial latent
        first_pass_latents = [latent["samples"].to(device)]  # First pass latents
        second_pass_latents = [latent["samples"].to(device)]  # Final refined latents

        # For the first frame visualization (placeholder):
        h, w = init_frame_np.shape[:2]
        target_h = (h // 8) * 8
        target_w = (w // 8) * 8

        # ComfyUI expects IMAGE type in format [B, H, W, C] with values 0-1
        placeholder = np.zeros((target_h, target_w, 3), dtype=np.float32)
        flow_visualizations.append(torch.from_numpy(placeholder).permute(2, 0, 1).float())  # Convert to [C, H, W]
        occlusion_masks.append(torch.from_numpy(placeholder).permute(2, 0, 1).float())
        warped_frames.append(torch.from_numpy(placeholder).permute(2, 0, 1).float())
        blended_frames.append(torch.from_numpy(placeholder).permute(2, 0, 1).float())

        # Generate the sequence
        print(f"Generating {num_frames} frames")
        for i in range(num_frames - 1):
            print(f"Processing frame {i + 2}/{num_frames}")

            # Update clip frames with previous frame
            clip_frames = np.roll(clip_frames, -1, axis=0)
            prev_frame_resized = cv2.resize(prev_frame, size)
            clip_frames[-1] = prev_frame_resized

            # Prepare input for FloweR
            clip_frames_torch = torch.from_numpy(clip_frames).to(self.DEVICE, dtype=torch.float32)
            clip_frames_torch = flow_utils.frames_norm(clip_frames_torch)

            # Predict next frame with FloweR
            with torch.no_grad():
                pred_data = self.FloweR_model(clip_frames_torch.unsqueeze(0))[0]

            # Process FloweR outputs
            pred_flow = flow_utils.flow_renorm(pred_data[..., :2]).cpu().numpy()
            pred_occl = flow_utils.occl_renorm(pred_data[..., 2:3]).cpu().numpy().repeat(3, axis=-1)
            pred_next = flow_utils.frames_renorm(pred_data[..., 3:6]).cpu().numpy()

            # Apply multipliers to flow and occlusion
            pred_flow = pred_flow * self.occlusion_flow_multiplier

            # Additional processing for flow
            flow_magnitude = np.linalg.norm(pred_flow, axis=-1, keepdims=True)
            difo_factor = 1.0 / (1.0 + flow_magnitude * 0.05 * self.occlusion_difo_multiplier)
            pred_flow = pred_flow * difo_factor

            # Resize to original dimensions
            pred_flow = cv2.resize(pred_flow, org_size)
            pred_occl = cv2.resize(pred_occl, org_size)
            pred_next = cv2.resize(pred_next, org_size)

            # Clean up and ensure proper ranges
            pred_next = np.clip(pred_next, 0, 255).astype(np.uint8)

            # Process occlusion mask with difs multiplier
            pred_occl = cv2.GaussianBlur(pred_occl, (21, 21), 2, cv2.BORDER_REFLECT_101)
            pred_occl = (np.abs(pred_occl / 255.0) ** 1.5) * 255.0 * self.occlusion_difs_multiplier
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

            # Create flow visualization
            flow_vis = self.create_flow_visualization(pred_flow)
            flow_vis = cv2.resize(flow_vis, (target_w, target_h))  # Ensure consistent size
            flow_vis = np.clip(flow_vis, 0, 255).astype(np.uint8)
            flow_tensor = torch.from_numpy(flow_vis).permute(2, 0, 1).float() / 255.0
            flow_visualizations.append(flow_tensor)

            # Same for other visualizations
            occlusion_vis = np.clip(pred_occl, 0, 255).astype(np.uint8)
            occlusion_vis = cv2.resize(occlusion_vis, (target_w, target_h))
            occlusion_tensor = torch.from_numpy(occlusion_vis).permute(2, 0, 1).float() / 255.0
            occlusion_masks.append(occlusion_tensor)

            # Create warped frame visualization
            warped_vis = np.clip(warped_frame, 0, 255).astype(np.uint8)
            warped_vis = cv2.resize(warped_vis, (target_w, target_h))  # Add this line
            warped_tensor = torch.from_numpy(warped_vis).permute(2, 0, 1).float() / 255.0
            warped_frames.append(warped_tensor)

            # Create blended frame visualization
            blended_vis = np.clip(blended_frame, 0, 255).astype(np.uint8)
            blended_vis = cv2.resize(blended_vis, (target_w, target_h))  # Add this line
            blended_tensor = torch.from_numpy(blended_vis).permute(2, 0, 1).float() / 255.0
            blended_frames.append(blended_tensor)

            # Use the blended frame as the base for further processing
            pred_next = blended_frame

            # Get grayscale occlusion for inpainting
            pred_occl_gray = np.mean(pred_occl, axis=2).astype(np.uint8)

            # IMPORTANT: Create new conditioning for each pass instead of modifying the same one
            # This prevents ControlNet effects from compounding

            # --- First Pass: Inpainting Phase ---
            # Create fresh conditioning for first pass
            first_pass_positive = positive.copy()
            first_pass_negative = negative.copy()

            # Apply ControlNet for first pass
            if self.cn_frame_value == 1:  # Current Frame
                first_pass_positive, first_pass_negative, proc_img = self.apply_controlnet(
                    pred_next, first_pass_positive, first_pass_negative, control_net,
                    controlnet_strength, controlnet_start_percent, controlnet_end_percent,
                    preprocessing_step="first_pass_current"
                )
                if proc_img is not None:
                    # Resize to match other output images if needed
                    proc_img = cv2.resize(proc_img, (target_w, target_h))
                    # Convert to tensor in the format expected by ComfyUI
                    proc_tensor = torch.from_numpy(proc_img).permute(2, 0, 1).float() / 255.0
                    # Add to the collection
                    preprocessed_controlnet_inputs.append(proc_tensor)
            elif self.cn_frame_value == 2:  # Previous Frame
                first_pass_positive, first_pass_negative, proc_img = self.apply_controlnet(
                    prev_frame, first_pass_positive, first_pass_negative, control_net,
                    controlnet_strength, controlnet_start_percent, controlnet_end_percent,
                    preprocessing_step="first_pass_previous"
                )
                if proc_img is not None:
                    proc_img = cv2.resize(proc_img, (target_w, target_h))
                    proc_tensor = torch.from_numpy(proc_img).permute(2, 0, 1).float() / 255.0
                    preprocessed_controlnet_inputs.append(proc_tensor)

            # Store the current positive/negative for inpainting
            self.positive = first_pass_positive
            self.negative = first_pass_negative

            # Convert to PIL for inpainting
            pred_next_pil = Image.fromarray(pred_next)
            pred_occl_pil = Image.fromarray(pred_occl_gray)

            # First diffusion pass - focusing on occluded areas (mode 4)
            frame_seed = seed + i if seed != 0 else i
            inpaint_result = self.inpaint_with_mask(
                pred_next_pil,
                pred_occl_pil,
                processing_strength,
                sampling_steps,
                frame_seed,
                device
            )

            first_pass_latent = inpaint_result["samples"]
            first_pass_latents.append(first_pass_latent.to(device))

            # Second diffusion pass - using the latent directly from first pass
            fixed_frame_result = self.sample_diffusion(
                {"samples": first_pass_latent},  # Use direct latent from first pass
                fix_frame_strength,
                sampling_steps,
                frame_seed + 10000
            )

            # Store this frame's latent - ensure it's on device
            second_pass_latent = fixed_frame_result["samples"]
            second_pass_latents.append(second_pass_latent.to(device))

            # Decode for next iteration
            final_frame_tensor = vae.decode(second_pass_latent)
            final_frame = final_frame_tensor[0].cpu()
            if final_frame.shape[0] == 3:  # [C, H, W]
                final_frame_np = final_frame.permute(1, 2, 0).numpy() * 255
            else:  # Assume [H, W, C]
                final_frame_np = final_frame.numpy() * 255

            # Apply color correction techniques
            final_frame_np = np.clip(final_frame_np, 0, 255).astype(np.uint8)

            # First limit extreme saturation values to prevent color blowout
            final_frame_np = self.limit_saturation(final_frame_np, max_saturation=160)

            # Then apply partial histogram matching to maintain consistency
            # Use a moderate strength (0.7) to balance between color accuracy and consistency
            final_frame_np = self.apply_histogram_matching(
                final_frame_np,
                init_frame_np,
                strength=0.7  # Adjust this value as needed (0.0-1.0)
            )

            # Update previous frame for next iteration
            prev_frame = final_frame_np.copy()

        # Stack all latents for output
        stacked_first_pass = torch.cat(first_pass_latents, dim=0)  # Stack first pass latents
        stacked_second_pass = torch.cat(second_pass_latents, dim=0)  # Stack second pass latents

        # Stack tensors into batches
        flow_batch = torch.stack(flow_visualizations)
        occlusion_batch = torch.stack(occlusion_masks)
        warped_batch = torch.stack(warped_frames)
        blended_batch = torch.stack(blended_frames)

        # If needed, convert format for ComfyUI compatibility
        flow_batch = flow_batch.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        occlusion_batch = occlusion_batch.permute(0, 2, 3, 1)
        warped_batch = warped_batch.permute(0, 2, 3, 1)
        blended_batch = blended_batch.permute(0, 2, 3, 1)

        # Add placeholder if no ControlNet was applied
        if len(preprocessed_controlnet_inputs) == 0:
            placeholder = torch.zeros_like(flow_visualizations[0])
            preprocessed_controlnet_inputs.append(placeholder)

        # At the end of generate_frames, prepare the tensor for output:
        proc_batch = torch.stack(preprocessed_controlnet_inputs)
        proc_batch = proc_batch.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        print(f"Flow batch shape: {flow_batch.shape}")

        # Clean up
        self.FloweR_clear_memory()
        print("Animation generation complete")

        # Return all visualizations
        return (
            {"samples": stacked_first_pass},
            {"samples": stacked_second_pass},
            flow_batch,
            occlusion_batch,
            warped_batch,
            blended_batch,
            proc_batch
        )

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "SDCNAnimation": Txt2VidNode
}

# Display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDCNAnimation": "SD-CN Animation"
}