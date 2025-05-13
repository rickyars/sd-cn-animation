"""
Advanced node implementation for SD-CN Animation.
Takes direct inputs for samplers, sigmas, etc. for more control.
"""
import torch
import numpy as np
import cv2
from PIL import Image
import comfy.utils
from comfy.utils import ProgressBar

from .components import (
    FloweRHandler,
    ControlNetHandler,
    DiffusionHandler,
    ImageProcessor,
    VisualizationHandler
)
from .utils import latent_utils


class Txt2VidNodeAdvanced:
    """
    Advanced Text to Video node for ComfyUI
    Generates video frames from an initial image using text prompts and ControlNet
    With direct control over samplers and sigmas
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
                # Direct inputs for advanced sampling control
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "first_pass_sigmas": ("SIGMAS",),
                "second_pass_sigmas": ("SIGMAS",),
                "latent": ("LATENT",),
                "vae": ("VAE",),

                # ControlNet inputs - identical to original node
                "control_net": ("CONTROL_NET",),
                "cn_frame_send": (["None", "Current Frame", "Previous Frame"], {
                    "default": "Previous Frame",
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

                # Animation parameters - identical to original node
                "num_frames": ("INT", {
                    "default": 16,
                    "min": 2,
                    "max": 1000,
                    "step": 1
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

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("first_pass_frames", "second_pass_frames", "flow_visualization", "occlusion_mask", "warped_frame", "blended_frame")
    FUNCTION = "generate_frames_advanced"
    CATEGORY = "animation"

    def generate_frames_advanced(self, noise, guider, sampler, first_pass_sigmas, second_pass_sigmas,
                               latent, vae, control_net, cn_frame_send, controlnet_strength,
                               controlnet_start_percent, controlnet_end_percent,
                               num_frames, tile_preprocessor, tile_blur_strength,
                               occlusion_mask_blur, occlusion_mask_multiplier, occlusion_flow_multiplier,
                               occlusion_difo_multiplier, occlusion_difs_multiplier, seed):
        """
        Generate a sequence of latent frames with iterative diffusion
        Using direct control over samplers and sigma schedules

        Args:
            All parameters from the node interface

        Returns:
            Tuple of (first_pass_latents, second_pass_latents, flow_visualization, occlusion_mask,
                    warped_frame, blended_frame, preprocessed_controlnet_input)
        """
        # Extract model from the guider
        model = guider.model_patcher

        # Initialize diffusion handler with the VAE
        self.diffusion.model = model
        self.diffusion.vae = vae

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

        # Initialize progress tracking
        pbar = ProgressBar(num_frames)

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
            sampling_guider = guider  # Default to original guider

            # Only apply ControlNet if it exists and strength > 0
            if control_net is not None and controlnet_strength > 0:
                # Get current conditioning from guider
                positive_cond = guider.positive_cond
                negative_cond = guider.negative_cond

                # Apply ControlNet based on settings
                if self.cn_frame_value == 1:  # Current Frame
                    modified_positive, _, proc_img = self.controlnet.apply_controlnet(
                        pred_next, positive_cond, negative_cond, control_net,
                        controlnet_strength, controlnet_start_percent, controlnet_end_percent,
                        preprocessing_step="first_pass_current"
                    )

                    # Create a new guider with the modified conditioning
                    # We need to copy the original guider's settings
                    new_guider = comfy.samplers.CFGGuider(model)
                    new_guider.set_conds(modified_positive, negative_cond)
                    new_guider.set_cfg(guider.cfg_scale)
                    sampling_guider = new_guider

                elif self.cn_frame_value == 2:  # Previous Frame
                    modified_positive, _, proc_img = self.controlnet.apply_controlnet(
                        prev_frame, positive_cond, negative_cond, control_net,
                        controlnet_strength, controlnet_start_percent, controlnet_end_percent,
                        preprocessing_step="first_pass_previous"
                    )

                    # Create a new guider with the modified conditioning
                    new_guider = comfy.samplers.CFGGuider(model)
                    new_guider.set_conds(modified_positive, negative_cond)
                    new_guider.set_cfg(guider.cfg_scale)
                    sampling_guider = new_guider
            else:
                print(f"Skipping ControlNet: None or strength={controlnet_strength}")


            # === STEP 6: First diffusion pass - focusing on occluded areas ===
            # Convert to PIL for inpainting
            pred_next_pil = Image.fromarray(pred_next)
            pred_occl_pil = Image.fromarray(pred_occl_gray)

            # Encode the blended frame using diffusion handler for consistency
            blended_latent = self.diffusion.encode_with_vae(pred_next_pil, device)

            # Prepare mask for inpainting
            mask_np = np.array(pred_occl_pil).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np).to(device)

            # This next line is important - it reshapes the mask to have a channel dimension
            noise_mask = mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1]))

            # Create a latent for the first diffusion pass that includes the mask
            first_pass_input = {
                "samples": blended_latent,
                "noise_mask": mask_tensor
            }

            # Calculate seed for this frame
            frame_seed = noise.seed + i if noise.seed != 0 else i

            # Create first pass noise tensor directly
            first_noise_tensor = comfy.sample.prepare_noise(
                first_pass_input["samples"],
                frame_seed
            )

            # Apply the first pass diffusion using guider.sample (exactly as in SamplerCustomAdvanced)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            first_pass_samples = sampling_guider.sample(
                first_noise_tensor,
                first_pass_input["samples"],
                sampler,
                first_pass_sigmas,
                denoise_mask=noise_mask,
                disable_pbar=disable_pbar,
                seed=frame_seed
            )

            # Store first pass result - make sure it's on the same device as the first element
            first_pass_samples = first_pass_samples.to(device)
            first_pass_latents.append(first_pass_samples)

            # === STEP 7: Second diffusion pass - refining the entire frame ===
            # Create latent for the second pass
            second_pass_input = {
                "samples": first_pass_samples
            }

            # For the second pass - create a new noise object with different seed
            second_seed = frame_seed + 10000
            second_noise_tensor = comfy.sample.prepare_noise(
                second_pass_input["samples"],
                second_seed
            )
            # Apply the second pass diffusion
            second_pass_samples = sampling_guider.sample(
                second_noise_tensor,
                second_pass_input["samples"],
                sampler,
                second_pass_sigmas,
                denoise_mask=None,
                disable_pbar=disable_pbar,
                seed=second_seed
            )

            # Store second pass result - make sure it's on the same device as the first element
            second_pass_samples = second_pass_samples.to(device)
            second_pass_latents.append(second_pass_samples)

            # Decode for next iteration
            final_frame_np = self.diffusion.decode_latent(second_pass_samples)

            # === STEP 8: Apply color correction ===
            final_frame_np = self.image_processor.apply_color_correction_pipeline(
                final_frame_np,
                init_frame_np,
                saturation_limit=160,
                histogram_strength=0.7
            )

            # Update previous frame for next iteration
            prev_frame = final_frame_np.copy()

            # Update progress
            pbar.update(1)

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
            blended_batch
        )


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "SDCNAnimationAdvanced": Txt2VidNodeAdvanced
}

# Display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDCNAnimationAdvanced": "SD-CN Animation Advanced"
}