"""
Main node implementation for SD-CN Animation.
Now uses modular components for better organization.
"""
import os
import torch
import numpy as np
import cv2
from PIL import Image

from .components import (
    FloweRHandler,
    ControlNetHandler,
    DiffusionHandler,
    ImageProcessor,
    VisualizationHandler
)
from .utils import latent_utils


class Txt2VidNode:
    """
    Text to Video node for ComfyUI
    Generates video frames from an initial image using text prompts and ControlNet
    """

    def __init__(self):
        """Initialize the node and its components"""
        # Initialize all component handlers
        self.flower = FloweRHandler()
        self.controlnet = ControlNetHandler()
        self.diffusion = DiffusionHandler()
        self.image_processor = ImageProcessor()
        self.visualization = VisualizationHandler()

        # Frame tracking
        self.cn_frame_value = 2  # Default: Previous Frame

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
                    "max": 100.0,
                    "step": 0.1
                }),
                "occlusion_mask_multiplier": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
                "occlusion_flow_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
                "occlusion_difo_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
                "occlusion_difs_multiplier": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 100.0,
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

    def generate_frames(self, model, positive, negative, latent, vae, control_net, num_frames, cn_frame_send,
                        controlnet_strength, controlnet_start_percent, controlnet_end_percent,
                        sampling_steps, cfg, sampler_name, scheduler, processing_strength, fix_frame_strength,
                        occlusion_mask_blur, occlusion_mask_multiplier, occlusion_flow_multiplier,
                        occlusion_difo_multiplier, occlusion_difs_multiplier, seed, tile_preprocessor,
                        tile_blur_strength):
        """
        Generate a sequence of latent frames with iterative diffusion

        Args:
            All parameters from the node interface

        Returns:
            Tuple of (first_pass_latents, second_pass_latents, flow_visualization, occlusion_mask,
                     warped_frame, blended_frame, preprocessed_controlnet_input)
        """
        # Initialize component parameters
        self.diffusion.set_model_params(
            model, vae, positive, negative, cfg, sampler_name, scheduler
        )

        self.controlnet.set_preprocessor(
            tile_preprocessor, tile_blur_strength
        )

        # Convert cn_frame_send string to numeric value
        cn_frame_options = ["None", "Current Frame", "Previous Frame"]
        self.cn_frame_value = cn_frame_options.index(cn_frame_send)

        # Use CUDA device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Initialize collections for debug visualizations
        flow_visualizations = []
        occlusion_masks = []
        warped_frames = []
        blended_frames = []
        preprocessed_controlnet_inputs = []

        # === STEP 1: Get initial frame by decoding the input latent ===
        print("Decoding initial latent")
        # Ensure latent is on the correct device
        latent_on_device = latent_utils.ensure_latent_on_device(latent, device)

        # Decode latent to image
        init_frame_np = self.diffusion.decode_latent(latent_on_device["samples"])
        print(f"Initial frame shape: {init_frame_np.shape}")

        # Get dimensions
        height, width = init_frame_np.shape[:2]

        # Setup for FloweR model - ensure size is divisible by 128
        size = (width // 128) * 128, (height // 128) * 128
        if size[0] == 0 or size[1] == 0:
            size = 128, 128
        org_size = (width, height)

        # === STEP 2: Load FloweR model for flow prediction ===
        print(f"Loading FloweR model with size: {size}")
        self.flower.load_model(size[0], size[1])

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

        # === STEP 3: Generate the sequence ===
        print(f"Generating {num_frames} frames")
        for i in range(num_frames - 1):
            print(f"Processing frame {i + 2}/{num_frames}")

            # Update clip frames with previous frame
            clip_frames = np.roll(clip_frames, -1, axis=0)
            prev_frame_resized = cv2.resize(prev_frame, size)
            clip_frames[-1] = prev_frame_resized

            # === STEP 4: Predict flow using FloweR ===
            pred_flow, pred_occl, pred_next = self.flower.predict_flow(clip_frames)

            # Process flow predictions to get warped frame
            flow_result = self.flower.process_flow(
                pred_flow, pred_occl, pred_next, prev_frame, org_size,
                occlusion_mask_multiplier, occlusion_flow_multiplier,
                occlusion_difo_multiplier, occlusion_mask_blur,
                occlusion_difs_multiplier
            )

            # Extract processed results
            pred_flow = flow_result['flow']
            pred_occl = flow_result['occlusion']
            pred_occl_gray = flow_result['occlusion_gray']
            pred_next = flow_result['predicted_next']
            warped_frame = flow_result['warped_frame']
            blended_frame = flow_result['blended_frame']

            # Create flow visualization for debug output
            flow_vis = self.visualization.create_flow_visualization(pred_flow)
            flow_vis = cv2.resize(flow_vis, (target_w, target_h))  # Ensure consistent size
            flow_vis = np.clip(flow_vis, 0, 255).astype(np.uint8)
            flow_tensor = torch.from_numpy(flow_vis).permute(2, 0, 1).float() / 255.0
            flow_visualizations.append(flow_tensor)

            # Prepare other visualizations
            occlusion_vis = np.clip(pred_occl, 0, 255).astype(np.uint8)
            occlusion_vis = cv2.resize(occlusion_vis, (target_w, target_h))
            occlusion_tensor = torch.from_numpy(occlusion_vis).permute(2, 0, 1).float() / 255.0
            occlusion_masks.append(occlusion_tensor)

            warped_vis = np.clip(warped_frame, 0, 255).astype(np.uint8)
            warped_vis = cv2.resize(warped_vis, (target_w, target_h))
            warped_tensor = torch.from_numpy(warped_vis).permute(2, 0, 1).float() / 255.0
            warped_frames.append(warped_tensor)

            blended_vis = np.clip(blended_frame, 0, 255).astype(np.uint8)
            blended_vis = cv2.resize(blended_vis, (target_w, target_h))
            blended_tensor = torch.from_numpy(blended_vis).permute(2, 0, 1).float() / 255.0
            blended_frames.append(blended_tensor)

            # Use the blended frame as the base for further processing
            pred_next = blended_frame

            # === STEP 5: Apply ControlNet based on settings ===
            # Create fresh conditioning for first pass
            first_pass_positive = positive.copy()
            first_pass_negative = negative.copy()

            # Apply ControlNet for first pass depending on settings
            if self.cn_frame_value == 1:  # Current Frame
                first_pass_positive, first_pass_negative, proc_img = self.controlnet.apply_controlnet(
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
                first_pass_positive, first_pass_negative, proc_img = self.controlnet.apply_controlnet(
                    prev_frame, first_pass_positive, first_pass_negative, control_net,
                    controlnet_strength, controlnet_start_percent, controlnet_end_percent,
                    preprocessing_step="first_pass_previous"
                )
                if proc_img is not None:
                    proc_img = cv2.resize(proc_img, (target_w, target_h))
                    proc_tensor = torch.from_numpy(proc_img).permute(2, 0, 1).float() / 255.0
                    preprocessed_controlnet_inputs.append(proc_tensor)

            # Update diffusion conditioning
            self.diffusion.positive = first_pass_positive
            self.diffusion.negative = first_pass_negative

            # === STEP 6: First diffusion pass - focusing on occluded areas ===
            # Convert to PIL for inpainting
            pred_next_pil = Image.fromarray(pred_next)
            pred_occl_pil = Image.fromarray(pred_occl_gray)

            # Calculate seed for this frame
            frame_seed = seed + i if seed != 0 else i

            # First diffusion pass - inpainting mode
            inpaint_result = self.diffusion.inpaint_with_mask(
                pred_next_pil,
                pred_occl_pil,
                processing_strength,
                sampling_steps,
                frame_seed,
                device
            )

            first_pass_latent = inpaint_result["samples"]
            first_pass_latents.append(first_pass_latent.to(device))

            # === STEP 7: Second diffusion pass - refining the entire frame ===
            fixed_frame_result = self.diffusion.sample_diffusion(
                {"samples": first_pass_latent},  # Use direct latent from first pass
                fix_frame_strength,
                sampling_steps,
                frame_seed + 10000
            )

            # Store this frame's latent - ensure it's on device
            second_pass_latent = fixed_frame_result["samples"]
            second_pass_latents.append(second_pass_latent.to(device))

            # Decode for next iteration
            final_frame_np = self.diffusion.decode_latent(second_pass_latent)

            # === STEP 8: Apply color correction ===
            final_frame_np = self.image_processor.apply_color_correction_pipeline(
                final_frame_np,
                init_frame_np,
                saturation_limit=160,
                histogram_strength=0.7
            )

            # Update previous frame for next iteration
            prev_frame = final_frame_np.copy()

        # === STEP 9: Prepare output tensors ===
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

        # Prepare ControlNet preprocessed inputs
        proc_batch = torch.stack(preprocessed_controlnet_inputs)
        proc_batch = proc_batch.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        # Clean up
        self.flower.clear_memory()
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