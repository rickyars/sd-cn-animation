"""
Advanced node implementation for SD-CN Animation.
Uses standard ComfyUI guider.sample() per frame with a single reused ControlNet copy.
"""
import torch
import numpy as np
import cv2
from PIL import Image
import comfy.utils
import comfy.samplers
import comfy.sample
import comfy.model_management
import latent_preview
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
    """Advanced Text to Video node using FloweR flow + two-pass diffusion."""

    def __init__(self):
        self.flower = FloweRHandler()
        self.controlnet = ControlNetHandler()
        self.diffusion = DiffusionHandler()
        self.image_processor = ImageProcessor()
        self.visualization = VisualizationHandler()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "processing_strength": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "fix_frame_strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "num_frames": ("INT", {
                    "default": 16, "min": 2, "max": 1000, "step": 1
                }),
                "occlusion_mask_multiplier": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 50.0, "step": 0.5
                }),
                "occlusion_mask_blur": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1
                }),
                "color_correction": ("BOOLEAN", {
                    "default": True
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2147483647
                }),
            },
            "optional": {
                "cn_config": ("SDCN_CN_CONFIG",),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("latent_frames", "debug_frames", "debug_first_pass", "debug_flower_pred", "debug_occlusion_mask")
    FUNCTION = "generate_frames_advanced"
    CATEGORY = "animation"

    def _setup_controlnet_conds(self, base_guider, control_net, img_np,
                                controlnet_strength, controlnet_start_percent,
                                controlnet_end_percent):
        """Create a single ControlNet copy and build conditioning once.
        Returns (positive_cond, negative_cond, c_net_copy) to be reused across frames.
        """
        positive_cond = base_guider.original_conds.get("positive", [])
        negative_cond = base_guider.original_conds.get("negative", [])
        if not isinstance(positive_cond, list):
            positive_cond = [positive_cond] if positive_cond is not None else []
        if not isinstance(negative_cond, list):
            negative_cond = [negative_cond] if negative_cond is not None else []

        modified_positive, modified_negative, c_net_copy = self.controlnet.apply_controlnet(
            img_np, positive_cond, negative_cond, control_net,
            controlnet_strength, controlnet_start_percent, controlnet_end_percent,
            preprocessing_step="frame"
        )

        return modified_positive, modified_negative, c_net_copy

    def _build_guider_for_frame(self, base_guider, cn_positive, cn_negative):
        """Build a guider using pre-built ControlNet conditioning."""
        frame_guider = comfy.samplers.CFGGuider(base_guider.model_patcher)
        frame_guider.set_conds(cn_positive, cn_negative)
        frame_guider.set_cfg(base_guider.cfg)
        return frame_guider

    def _sample_frame(self, guider_to_use, sampler, sigmas, latent_image,
                      noise_seed, denoise_mask=None):
        """Sample one frame using guider.sample(), optionally with inpainting mask."""
        noise_tensor = comfy.sample.prepare_noise(latent_image, noise_seed)

        latent_fixed = comfy.sample.fix_empty_latent_channels(
            guider_to_use.model_patcher, latent_image
        )

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        callback = latent_preview.prepare_callback(
            guider_to_use.model_patcher, sigmas.shape[-1] - 1
        )

        samples = guider_to_use.sample(
            noise_tensor, latent_fixed, sampler, sigmas,
            denoise_mask=denoise_mask, callback=callback,
            disable_pbar=disable_pbar, seed=noise_seed
        )
        samples = samples.to(comfy.model_management.intermediate_device())
        return samples

    @staticmethod
    def _truncate_sigmas(sigmas, strength):
        """Truncate sigma schedule like A1111: take the last (strength * total_steps) sigmas.

        Given a full schedule of N steps (N+1 values including terminal 0),
        A1111 computes: steps_used = int(strength * N), then uses the last steps_used sigmas.
        """
        total_steps = sigmas.shape[0] - 1  # exclude terminal 0
        steps_used = max(1, int(strength * total_steps))
        # Take last steps_used sigmas + terminal 0
        truncated = sigmas[-(steps_used + 1):]
        return truncated

    def generate_frames_advanced(self, noise, guider, sampler, sigmas,
                                latent, vae,
                                processing_strength, fix_frame_strength,
                                num_frames,
                                occlusion_mask_multiplier, occlusion_mask_blur,
                                color_correction, seed,
                                cn_config=None):
        # Unpack ControlNet config (if provided)
        control_net = None
        controlnet_strength = 0
        controlnet_start_percent = 0.0
        controlnet_end_percent = 1.0
        if cn_config is not None:
            control_net = cn_config["control_net"]
            controlnet_strength = cn_config["strength"]
            controlnet_start_percent = cn_config["start_percent"]
            controlnet_end_percent = cn_config["end_percent"]
            self.controlnet.set_preprocessor(
                cn_config["tile_preprocessor"], cn_config["tile_blur_strength"]
            )

        # Split the single sigma schedule into two passes (A1111-style truncation)
        first_pass_sigmas = self._truncate_sigmas(sigmas, processing_strength)
        second_pass_sigmas = self._truncate_sigmas(sigmas, fix_frame_strength)

        model = guider.model_patcher
        self.diffusion.model = model
        self.diffusion.vae = vae

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Decode initial frame
        latent_on_device = latent_utils.ensure_latent_on_device(latent, device)
        init_frame_np = self.diffusion.decode_latent(latent_on_device["samples"])
        height, width = init_frame_np.shape[:2]
        print(f"Initial frame: {width}x{height}")

        # Load FloweR
        size = (width // 128) * 128, (height // 128) * 128
        if size[0] == 0 or size[1] == 0:
            size = 128, 128
        org_size = (width, height)

        self.flower.load_model(size[0], size[1])

        init_resized = cv2.resize(init_frame_np, size)
        clip_frames = np.stack([init_resized] * 4, axis=0)
        prev_frame = init_frame_np.copy()

        second_pass_latents = [latent["samples"].to(device)]
        init_frame_tensor = torch.from_numpy(init_frame_np.astype(np.float32) / 255.0)
        debug_frames = [init_frame_tensor]
        debug_first_pass_frames = [init_frame_tensor]
        debug_flower_frames = [init_frame_tensor]
        debug_mask_frames = [torch.zeros_like(init_frame_tensor)]  # black mask for frame 1 (no occlusion)

        pbar = ProgressBar(num_frames)
        use_controlnet = control_net is not None and controlnet_strength > 0

        # Create a single ControlNet copy and conditioning ONCE before the loop.
        # Each frame we only update the hint image on this same copy.
        cn_positive = cn_negative = c_net_copy = None
        if use_controlnet:
            # Pre-load ControlNet into VRAM
            if hasattr(control_net, 'get_models'):
                cn_models = control_net.get_models()
                comfy.model_management.load_models_gpu(cn_models)
                print(f"Pre-loaded ControlNet into VRAM")

            cn_positive, cn_negative, c_net_copy = self._setup_controlnet_conds(
                guider, control_net, prev_frame,
                controlnet_strength, controlnet_start_percent, controlnet_end_percent
            )

        total_steps = sigmas.shape[0] - 1
        print(f"Full sigmas ({total_steps} steps): {sigmas.tolist()}")
        print(f"First pass (strength={processing_strength}): {first_pass_sigmas.shape[0]-1} steps, sigmas={first_pass_sigmas.tolist()}")
        print(f"Second pass (strength={fix_frame_strength}): {second_pass_sigmas.shape[0]-1} steps, sigmas={second_pass_sigmas.tolist()}")
        print(f"Generating {num_frames} frames")
        for i in range(num_frames - 1):
            print(f"Processing frame {i + 2}/{num_frames}")

            clip_frames = np.roll(clip_frames, -1, axis=0)
            clip_frames[-1] = cv2.resize(prev_frame, size)

            pred_flow, pred_occl, pred_next = self.flower.predict_flow(clip_frames)
            flow_result = self.flower.process_flow(
                pred_flow, pred_occl, pred_next, org_size,
                occlusion_mask_multiplier, occlusion_mask_blur
            )
            pred_occl_gray = flow_result['occlusion_gray']
            pred_next = flow_result['predicted_next']

            # Collect debug: raw FloweR prediction and occlusion mask
            debug_flower_frames.append(torch.from_numpy(pred_next.astype(np.float32) / 255.0))
            # Convert grayscale mask to 3-channel for IMAGE output
            mask_rgb = np.stack([pred_occl_gray] * 3, axis=-1)
            debug_mask_frames.append(torch.from_numpy(mask_rgb.astype(np.float32) / 255.0))

            if i == 0:
                mask_min, mask_max, mask_mean = pred_occl_gray.min(), pred_occl_gray.max(), pred_occl_gray.mean()
                print(f"  Occlusion mask stats: min={mask_min}, max={mask_max}, mean={mask_mean:.1f}")
                print(f"  noise_mask shape: will be [1, 1, {pred_occl_gray.shape[0]}, {pred_occl_gray.shape[1]}]")
                print(f"  blended_latent shape: will be {self.diffusion.encode_with_vae(Image.fromarray(pred_next), device).shape}")

            # Update hint image on existing ControlNet copy (no new copies created)
            if use_controlnet:
                self.controlnet.update_controlnet_hint(
                    c_net_copy, prev_frame,
                    controlnet_strength, controlnet_start_percent, controlnet_end_percent
                )
                frame_guider = self._build_guider_for_frame(
                    guider, cn_positive, cn_negative
                )
            else:
                frame_guider = guider

            # First pass: inpaint occluded areas (composite-at-end)
            pred_next_pil = Image.fromarray(pred_next)
            blended_latent = self.diffusion.encode_with_vae(pred_next_pil, device)

            frame_seed = noise.seed + i if noise.seed != 0 else i

            # Denoise the full image without any mask
            first_pass_raw = self._sample_frame(
                frame_guider, sampler, first_pass_sigmas,
                blended_latent, frame_seed
            )

            # Composite unmasked (non-occluded) regions back from original
            mask_np = np.array(pred_occl_gray).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
            latent_mask = torch.nn.functional.interpolate(
                mask_tensor, size=blended_latent.shape[2:], mode='bilinear', align_corners=False
            )
            first_pass_raw = first_pass_raw.to(device)
            blended_latent = blended_latent.to(device)
            latent_mask = latent_mask.to(device)
            first_pass_samples = first_pass_raw * latent_mask + blended_latent * (1.0 - latent_mask)

            # Decode, optionally histogram match, re-encode
            first_pass_frame_np = self.diffusion.decode_latent(first_pass_samples.to(device))
            if color_correction:
                first_pass_frame_np = self.image_processor.match_histograms(first_pass_frame_np, init_frame_np)
            first_pass_frame_np = np.clip(first_pass_frame_np, 0, 255).astype(np.uint8)
            debug_first_pass_frames.append(torch.from_numpy(first_pass_frame_np.astype(np.float32) / 255.0))
            second_pass_input = self.diffusion.encode_with_vae(
                Image.fromarray(first_pass_frame_np), device
            )

            # Second pass: refine entire frame (no mask)
            second_seed = frame_seed + 10000
            second_pass_samples = self._sample_frame(
                frame_guider, sampler, second_pass_sigmas,
                second_pass_input, second_seed
            )

            second_pass_samples = second_pass_samples.to(device)
            second_pass_latents.append(second_pass_samples)

            final_frame_np = self.diffusion.decode_latent(second_pass_samples)
            if color_correction:
                final_frame_np = self.image_processor.match_histograms(final_frame_np, init_frame_np)
            final_frame_np = np.clip(final_frame_np, 0, 255).astype(np.uint8)

            debug_frames.append(torch.from_numpy(final_frame_np.astype(np.float32) / 255.0))
            prev_frame = final_frame_np.copy()
            pbar.update(1)

        stacked_latents = torch.cat(second_pass_latents, dim=0)
        stacked_images = torch.stack(debug_frames, dim=0)
        stacked_first_pass = torch.stack(debug_first_pass_frames, dim=0)
        stacked_flower = torch.stack(debug_flower_frames, dim=0)
        stacked_masks = torch.stack(debug_mask_frames, dim=0)

        # Clean up ControlNet state (frees cached hint tensors)
        if c_net_copy is not None:
            c_net_copy.cleanup()

        self.flower.clear_memory()
        print("Animation generation complete")

        return ({"samples": stacked_latents}, stacked_images, stacked_first_pass, stacked_flower, stacked_masks)


NODE_CLASS_MAPPINGS = {
    "SDCNAnimationAdvanced": Txt2VidNodeAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDCNAnimationAdvanced": "SD-CN Animation Advanced"
}
