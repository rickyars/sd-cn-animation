"""
Standard node implementation for SD-CN Animation.
Uses KSampler-style inputs (MODEL, CONDITIONING, sampler_name, scheduler, etc.)
for compatibility with workflows like z-image that don't use the advanced sampling API.
"""
import comfy.utils
import comfy.samplers
import comfy.sample
import comfy.model_management
import latent_preview

from .components import ControlNetHandler
from .animation_core import run_animation_loop, truncate_sigmas


class Txt2VidNode:
    """Text to Video node using FloweR flow + two-pass diffusion.
    Uses KSampler-style inputs (MODEL, CONDITIONING, sampler_name, scheduler).
    """

    def __init__(self):
        self.controlnet = ControlNetHandler()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True
                }),
                "steps": ("INT", {
                    "default": 25, "min": 1, "max": 10000
                }),
                "cfg": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
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
            },
            "optional": {
                "cn_config": ("SDCN_CN_CONFIG",),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent_frames", "frames")
    FUNCTION = "generate_frames"
    CATEGORY = "animation"

    def generate_frames(self, model, positive, negative, latent, vae,
                       seed, steps, cfg, sampler_name, scheduler,
                       processing_strength, fix_frame_strength,
                       num_frames,
                       occlusion_mask_multiplier, occlusion_mask_blur,
                       color_correction,
                       cn_config=None):
        # Compute full sigma schedule from model, then truncate for each pass
        model_sampling = model.get_model_object("model_sampling")
        full_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
        full_sigmas = full_sigmas.to(model.load_device)

        first_pass_sigmas = truncate_sigmas(full_sigmas, processing_strength)
        second_pass_sigmas = truncate_sigmas(full_sigmas, fix_frame_strength)

        # Set up ControlNet state (populated lazily via on_init callback)
        use_controlnet = False
        control_net = None
        controlnet_strength = 0
        controlnet_start_percent = 0.0
        controlnet_end_percent = 1.0
        # Mutable conditioning — may be replaced by CN setup
        cond_state = {"positive": positive, "negative": negative}
        cn_state = {"c_net_copy": None}

        if cn_config is not None:
            control_net = cn_config["control_net"]
            controlnet_strength = cn_config["strength"]
            controlnet_start_percent = cn_config["start_percent"]
            controlnet_end_percent = cn_config["end_percent"]
            self.controlnet.set_preprocessor(
                cn_config["tile_preprocessor"], cn_config["tile_blur_strength"]
            )
            use_controlnet = controlnet_strength > 0

            if use_controlnet and hasattr(control_net, 'get_models'):
                comfy.model_management.load_models_gpu(control_net.get_models())

        # Called once after initial frame is decoded
        def on_init(init_frame_np):
            if not use_controlnet:
                return
            cn_pos, cn_neg, c_net_copy = self.controlnet.apply_controlnet(
                init_frame_np, positive, negative, control_net,
                controlnet_strength, controlnet_start_percent, controlnet_end_percent,
                preprocessing_step="frame"
            )
            cond_state["positive"] = cn_pos
            cond_state["negative"] = cn_neg
            cn_state["c_net_copy"] = c_net_copy

        # Build per-frame sampling function using KSampler internals
        sampler_obj = comfy.samplers.sampler_object(sampler_name)

        def sample_frame(latent_image, frame_seed, frame_sigmas):
            noise_tensor = comfy.sample.prepare_noise(latent_image, frame_seed)
            latent_fixed = comfy.sample.fix_empty_latent_channels(model, latent_image)

            cfg_guider = comfy.samplers.CFGGuider(model)
            cfg_guider.set_conds(cond_state["positive"], cond_state["negative"])
            cfg_guider.set_cfg(cfg)

            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            callback = latent_preview.prepare_callback(
                model, frame_sigmas.shape[-1] - 1
            )
            samples = cfg_guider.sample(
                noise_tensor, latent_fixed, sampler_obj, frame_sigmas,
                callback=callback, disable_pbar=disable_pbar, seed=frame_seed
            )
            return samples.to(comfy.model_management.intermediate_device())

        # Called before each frame — updates ControlNet hint image
        def on_before_sample(prev_frame_np):
            self.controlnet.update_controlnet_hint(
                cn_state["c_net_copy"], prev_frame_np,
                controlnet_strength, controlnet_start_percent, controlnet_end_percent
            )

        try:
            result = run_animation_loop(
                sample_frame_fn=sample_frame,
                vae=vae,
                model_patcher=model,
                latent=latent,
                first_pass_sigmas=first_pass_sigmas,
                second_pass_sigmas=second_pass_sigmas,
                num_frames=num_frames,
                base_seed=seed,
                occlusion_mask_multiplier=occlusion_mask_multiplier,
                occlusion_mask_blur=occlusion_mask_blur,
                color_correction=color_correction,
                on_init=on_init if use_controlnet else None,
                on_before_sample=on_before_sample if use_controlnet else None,
            )
        finally:
            if cn_state["c_net_copy"] is not None:
                cn_state["c_net_copy"].cleanup()

        return result


NODE_CLASS_MAPPINGS = {
    "SDCNAnimation": Txt2VidNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDCNAnimation": "SD-CN Animation"
}
