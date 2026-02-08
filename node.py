"""
Main node implementation for SD-CN Animation.
Now uses modular components for better organization.
"""
import torch
import numpy as np
import cv2
from PIL import Image
import comfy.utils
import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.model_management
import comfy.model_patcher
import comfy.hooks
from comfy.utils import ProgressBar
from comfy.samplers import calculate_sigmas, sampler_object, preprocess_conds_hooks, get_total_hook_groups_in_conds, filter_registered_hooks_on_conds, cast_to_load_options

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

        # Frame tracking - always use Previous Frame (most stable)
        self.cn_frame_value = 2

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
                "occlusion_mask_multiplier": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "Occlusion Mask Strength"
                }),
                "occlusion_mask_blur": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "Occlusion Mask Blur"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_frames",)
    FUNCTION = "generate_frames"
    CATEGORY = "animation"

    def generate_frames(self, model, positive, negative, latent, vae, control_net, num_frames,
                        controlnet_strength, controlnet_start_percent, controlnet_end_percent,
                        sampling_steps, cfg, sampler_name, scheduler, processing_strength, fix_frame_strength,
                        occlusion_mask_multiplier, occlusion_mask_blur, seed, tile_preprocessor, tile_blur_strength):
        """
        Generate a sequence of latent frames with iterative diffusion
        """
        # Initialize diffusion handler with VAE (still used for encode/decode)
        self.diffusion.set_model_params(
            model, vae, positive, negative, cfg, sampler_name, scheduler
        )

        self.controlnet.set_preprocessor(
            tile_preprocessor, tile_blur_strength
        )

        # Always use Previous Frame for ControlNet (most stable)
        self.cn_frame_value = 2

        # Use CUDA device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # === STEP 1: Get initial frame by decoding the input latent ===
        print("Decoding initial latent")
        latent_on_device = latent_utils.ensure_latent_on_device(latent, device)
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
        # Pre-fill with initial frame so FloweR sees a still scene initially
        # (zeros would make FloweR think everything changed → 97% occlusion on frame 1)
        init_resized = cv2.resize(init_frame_np, size)
        clip_frames = np.stack([init_resized] * 4, axis=0)
        prev_frame = init_frame_np.copy()

        # Initialize latent frames array with initial latent
        first_pass_latents = [latent["samples"].to(device)]
        second_pass_latents = [latent["samples"].to(device)]

        # Initialize progress tracking
        pbar = ProgressBar(num_frames)

        # === STEP 3: Build sampling context ===
        # Compute sigmas for both passes
        model_sampling = model.get_model_object("model_sampling")

        # First pass sigmas (high strength - inpainting)
        first_pass_steps = sampling_steps
        if processing_strength <= 0.0:
            first_pass_sigmas = torch.FloatTensor([])
        elif processing_strength > 0.9999:
            first_pass_sigmas = calculate_sigmas(model_sampling, scheduler, first_pass_steps)
        else:
            new_steps = int(first_pass_steps / processing_strength)
            sigmas = calculate_sigmas(model_sampling, scheduler, new_steps)
            first_pass_sigmas = sigmas[-(first_pass_steps + 1):]

        # Second pass sigmas (low strength - refinement)
        second_pass_steps = sampling_steps
        if fix_frame_strength <= 0.0:
            second_pass_sigmas = torch.FloatTensor([])
        elif fix_frame_strength > 0.9999:
            second_pass_sigmas = calculate_sigmas(model_sampling, scheduler, second_pass_steps)
        else:
            new_steps = int(second_pass_steps / fix_frame_strength)
            sigmas = calculate_sigmas(model_sampling, scheduler, new_steps)
            second_pass_sigmas = sigmas[-(second_pass_steps + 1):]

        # Create sampler object
        sampler_obj = sampler_object(sampler_name)

        # === STEP 4: Setup ControlNet and build guider (once) ===
        working_positive = list(map(lambda a: a.copy(), comfy.sampler_helpers.convert_cond(positive)))
        working_negative = list(map(lambda a: a.copy(), comfy.sampler_helpers.convert_cond(negative)))
        c_net_copy = None  # Will hold the ControlNet copy for per-frame updates

        if control_net is not None and controlnet_strength > 0:
            print(f"Setting up ControlNet guider (strength={controlnet_strength})")
            cn_frame = prev_frame
            working_positive, working_negative, c_net_copy = self.controlnet.apply_controlnet(
                cn_frame, working_positive, working_negative, control_net,
                controlnet_strength, controlnet_start_percent, controlnet_end_percent,
                preprocessing_step="initial_setup"
            )
        else:
            print(f"Skipping ControlNet: None or strength={controlnet_strength}")

        # Create CFGGuider
        sampling_guider = comfy.samplers.CFGGuider(model)
        sampling_guider.set_conds(working_positive, working_negative)
        sampling_guider.set_cfg(cfg)

        # === STEP 5: Prepare sampling context ONCE ===
        print("Preparing sampling context (loading models once)...")

        # Build conditioning
        sampling_guider.conds = {}
        for k in sampling_guider.original_conds:
            sampling_guider.conds[k] = list(map(lambda a: a.copy(), sampling_guider.original_conds[k]))
        preprocess_conds_hooks(sampling_guider.conds)

        # Set up model options
        orig_model_options = sampling_guider.model_options
        sampling_guider.model_options = comfy.model_patcher.create_model_options_clone(sampling_guider.model_options)
        orig_hook_mode = sampling_guider.model_patcher.hook_mode
        if get_total_hook_groups_in_conds(sampling_guider.conds) <= 1:
            sampling_guider.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
        comfy.sampler_helpers.prepare_model_patcher(sampling_guider.model_patcher, sampling_guider.conds, sampling_guider.model_options)
        filter_registered_hooks_on_conds(sampling_guider.conds, sampling_guider.model_options)

        # Use dummy noise shape for memory estimation
        dummy_noise = comfy.sample.prepare_noise(latent["samples"].to(device), seed)

        # prepare_sampling: loads models to GPU and returns inner_model
        sampling_guider.inner_model, sampling_guider.conds, sampling_guider.loaded_models = \
            comfy.sampler_helpers.prepare_sampling(
                sampling_guider.model_patcher, dummy_noise.shape,
                sampling_guider.conds, sampling_guider.model_options
            )

        load_device = sampling_guider.model_patcher.load_device
        model_dtype = sampling_guider.model_patcher.model_dtype()
        cast_to_load_options(sampling_guider.model_options, device=load_device, dtype=model_dtype)

        # pre_run: initialize model state
        sampling_guider.model_patcher.pre_run()
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        print("Models loaded and ready - entering frame generation loop")

        # === STEP 6: Generate frames with models staying loaded ===
        try:
            print(f"Generating {num_frames} frames")
            for i in range(num_frames - 1):
                print(f"Processing frame {i + 2}/{num_frames}")

                # Update clip frames with previous frame
                clip_frames = np.roll(clip_frames, -1, axis=0)
                prev_frame_resized = cv2.resize(prev_frame, size)
                clip_frames[-1] = prev_frame_resized

                # === Predict flow using FloweR ===
                pred_flow, pred_occl, pred_next = self.flower.predict_flow(clip_frames)

                flow_result = self.flower.process_flow(
                    pred_flow, pred_occl, pred_next, org_size,
                    occlusion_mask_multiplier, occlusion_mask_blur
                )

                pred_occl_gray = flow_result['occlusion_gray']
                pred_next = flow_result['predicted_next']

                # === Update ControlNet hint with previous frame ===
                if c_net_copy is not None:
                    self.controlnet.update_controlnet_hint(
                        c_net_copy, prev_frame, controlnet_strength,
                        controlnet_start_percent, controlnet_end_percent
                    )

                # === First diffusion pass - inpainting occluded areas ===
                pred_next_pil = Image.fromarray(pred_next)
                pred_occl_pil = Image.fromarray(pred_occl_gray)

                blended_latent = self.diffusion.encode_with_vae(pred_next_pil, device)

                mask_np = np.array(pred_occl_pil).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_np).to(device)
                noise_mask = mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1]))

                frame_seed = seed + i if seed != 0 else i
                first_noise_tensor = comfy.sample.prepare_noise(blended_latent, frame_seed)

                # Prepare denoise mask
                latent_shape = blended_latent.shape
                prepared_mask = comfy.sampler_helpers.prepare_mask(noise_mask, latent_shape, load_device)

                # Call inner_sample directly - no model loading/unloading!
                first_noise_on_device = first_noise_tensor.to(load_device)
                blended_on_device = blended_latent.to(load_device)
                first_sigmas_on_device = first_pass_sigmas.to(load_device)

                # Rebuild conds from original before each inner_sample call
                # (process_conds inside inner_sample mutates self.conds in place;
                #  without rebuilding, conds accumulate stale state across calls)
                sampling_guider.conds = {}
                for k in sampling_guider.original_conds:
                    sampling_guider.conds[k] = list(map(lambda a: a.copy(), sampling_guider.original_conds[k]))
                preprocess_conds_hooks(sampling_guider.conds)
                filter_registered_hooks_on_conds(sampling_guider.conds, sampling_guider.model_options)

                first_pass_samples = sampling_guider.inner_sample(
                    first_noise_on_device, blended_on_device, load_device,
                    sampler_obj, first_sigmas_on_device,
                    prepared_mask, None, disable_pbar, frame_seed
                )

                first_pass_samples = first_pass_samples.to(device)
                first_pass_latents.append(first_pass_samples)

                # === Intermediate: decode to pixels, histogram match, re-encode ===
                # Reference decodes between passes and applies histogram matching
                first_pass_frame_np = self.diffusion.decode_latent(first_pass_samples)
                first_pass_frame_np = self.image_processor.match_histograms(first_pass_frame_np, init_frame_np)
                first_pass_frame_np = np.clip(first_pass_frame_np, 0, 255).astype(np.uint8)
                first_pass_pil = Image.fromarray(first_pass_frame_np)
                second_pass_input = self.diffusion.encode_with_vae(first_pass_pil, device)

                # === Second diffusion pass - refining the entire frame ===
                second_seed = frame_seed + 10000
                second_noise_tensor = comfy.sample.prepare_noise(second_pass_input, second_seed)

                second_noise_on_device = second_noise_tensor.to(load_device)
                second_pass_input_on_device = second_pass_input.to(load_device)
                second_sigmas_on_device = second_pass_sigmas.to(load_device)

                # Rebuild conds from original before each inner_sample call
                sampling_guider.conds = {}
                for k in sampling_guider.original_conds:
                    sampling_guider.conds[k] = list(map(lambda a: a.copy(), sampling_guider.original_conds[k]))
                preprocess_conds_hooks(sampling_guider.conds)
                filter_registered_hooks_on_conds(sampling_guider.conds, sampling_guider.model_options)

                second_pass_samples = sampling_guider.inner_sample(
                    second_noise_on_device, second_pass_input_on_device, load_device,
                    sampler_obj, second_sigmas_on_device,
                    None, None, disable_pbar, second_seed
                )

                second_pass_samples = second_pass_samples.to(device)
                second_pass_latents.append(second_pass_samples)

                # Decode for next iteration
                final_frame_np = self.diffusion.decode_latent(second_pass_samples)

                # Match histograms to initial frame to prevent color drift
                # (reference does this after BOTH passes)
                final_frame_np = self.image_processor.match_histograms(final_frame_np, init_frame_np)
                final_frame_np = np.clip(final_frame_np, 0, 255).astype(np.uint8)

                prev_frame = final_frame_np.copy()
                pbar.update(1)

        finally:
            # === STEP 7: Cleanup ONCE after all frames ===
            # Match the exact cleanup sequence from outer_sample() + guider.sample()
            # outer_sample() cleanup:
            sampling_guider.model_patcher.cleanup()
            comfy.sampler_helpers.cleanup_models(sampling_guider.conds, sampling_guider.loaded_models)
            del sampling_guider.inner_model
            del sampling_guider.loaded_models

            # guider.sample() cleanup - restore state to pre-sampling condition:
            cast_to_load_options(sampling_guider.model_options, device=sampling_guider.model_patcher.offload_device)
            sampling_guider.model_options = orig_model_options  # Restore original model_options
            sampling_guider.model_patcher.hook_mode = orig_hook_mode
            sampling_guider.model_patcher.restore_hook_patches()
            del sampling_guider.conds
            # Do NOT clear original_conds - let ComfyUI manage the guider lifecycle

        # === STEP 8: Prepare output ===
        stacked_latents = torch.cat(second_pass_latents, dim=0)

        self.flower.clear_memory()
        print("Animation generation complete")

        return ({"samples": stacked_latents},)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "SDCNAnimation": Txt2VidNode
}

# Display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDCNAnimation": "SD-CN Animation"
}
