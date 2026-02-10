"""
Shared animation loop for SD-CN Animation.
Both node_advanced.py (advanced API) and node.py (KSampler API) delegate to this.
"""
import torch
import numpy as np
import cv2
from PIL import Image
from comfy.utils import ProgressBar

from .components import FloweRHandler, DiffusionHandler, ImageProcessor
from .utils import latent_utils


def truncate_sigmas(sigmas, strength):
    """Truncate sigma schedule like A1111: take the last (strength * total_steps) sigmas.

    Given a full schedule of N steps (N+1 values including terminal 0),
    A1111 computes: steps_used = int(strength * N), then uses the last steps_used sigmas.
    """
    total_steps = sigmas.shape[0] - 1  # exclude terminal 0
    steps_used = max(1, int(strength * total_steps))
    return sigmas[-(steps_used + 1):]


def run_animation_loop(
    sample_frame_fn,
    vae,
    model_patcher,
    latent,
    first_pass_sigmas,
    second_pass_sigmas,
    num_frames,
    base_seed,
    occlusion_mask_multiplier,
    occlusion_mask_blur,
    color_correction,
    on_init=None,
    on_before_sample=None,
):
    """Core animation loop shared by both node types.

    Args:
        sample_frame_fn: callable(latent_image, seed, sigmas) -> sampled latent tensor
        vae: ComfyUI VAE model
        model_patcher: ModelPatcher (for DiffusionHandler)
        latent: Initial LATENT dict with "samples" key
        first_pass_sigmas: Pre-truncated sigmas for first pass (inpainting)
        second_pass_sigmas: Pre-truncated sigmas for second pass (refinement)
        num_frames: Total number of frames to generate
        base_seed: Base seed for noise generation
        occlusion_mask_multiplier: Multiplier for FloweR occlusion mask
        occlusion_mask_blur: Gaussian blur sigma for mask smoothing
        color_correction: Whether to apply histogram matching
        on_init: Optional callable(init_frame_np) called once after decoding the initial
            frame, before the loop starts. Used to set up ControlNet with the first frame.
        on_before_sample: Optional callable(prev_frame_np) called before each frame's
            sampling pass. Used to update ControlNet hints per-frame.

    Returns:
        Tuple of (latent_dict, image_tensor)
    """
    diffusion = DiffusionHandler()
    diffusion.model = model_patcher
    diffusion.vae = vae
    image_processor = ImageProcessor()
    flower = FloweRHandler()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Decode initial frame
    latent_on_device = latent_utils.ensure_latent_on_device(latent, device)
    init_frame_np = diffusion.decode_latent(latent_on_device["samples"])
    height, width = init_frame_np.shape[:2]

    # Load FloweR
    size = (width // 128) * 128, (height // 128) * 128
    if size[0] == 0 or size[1] == 0:
        size = 128, 128
    org_size = (width, height)

    flower.load_model(size[0], size[1])

    clip_frames = np.zeros((4, size[1], size[0], 3), dtype=np.uint8)
    prev_frame = init_frame_np.copy()

    # Let nodes set up ControlNet etc. with the initial frame
    if on_init is not None:
        on_init(init_frame_np)

    second_pass_latents = [latent["samples"].to(device)]
    init_frame_tensor = torch.from_numpy(init_frame_np.astype(np.float32) / 255.0)
    output_frames = [init_frame_tensor]

    pbar = ProgressBar(num_frames)

    print(f"SD-CN Animation: {num_frames} frames @ {width}x{height}, "
          f"first pass {first_pass_sigmas.shape[0]-1} steps, "
          f"second pass {second_pass_sigmas.shape[0]-1} steps")

    for i in range(num_frames - 1):
        print(f"  Frame {i + 2}/{num_frames}")

        clip_frames = np.roll(clip_frames, -1, axis=0)
        clip_frames[-1] = cv2.resize(prev_frame, size)

        pred_flow, pred_occl, pred_next = flower.predict_flow(clip_frames)
        flow_result = flower.process_flow(
            pred_flow, pred_occl, pred_next, org_size,
            occlusion_mask_multiplier, occlusion_mask_blur
        )
        pred_occl_gray = flow_result['occlusion_gray']
        pred_next = flow_result['predicted_next']

        # Allow nodes to update ControlNet hints etc. before sampling
        if on_before_sample is not None:
            on_before_sample(prev_frame)

        # First pass: denoise full image, then composite-at-end
        pred_next_pil = Image.fromarray(pred_next)
        blended_latent = diffusion.encode_with_vae(pred_next_pil, device)

        frame_seed = base_seed + i if base_seed != 0 else i

        first_pass_raw = sample_frame_fn(blended_latent, frame_seed, first_pass_sigmas)

        # Composite: keep non-occluded from original, use diffused for occluded
        mask_np = pred_occl_gray.astype(np.float32) / 255.0
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
        mask_t = torch.nn.functional.interpolate(
            mask_t, size=blended_latent.shape[2:], mode='bilinear', align_corners=False
        )
        first_pass_raw = first_pass_raw.to(device)
        blended_latent = blended_latent.to(device)
        first_pass_samples = first_pass_raw * mask_t + blended_latent * (1.0 - mask_t)

        # Second pass: refine entire frame (no mask), fed directly from first pass latent
        second_seed = frame_seed + 10000
        second_pass_samples = sample_frame_fn(first_pass_samples, second_seed, second_pass_sigmas)

        second_pass_samples = second_pass_samples.to(device)
        second_pass_latents.append(second_pass_samples)

        # Single decode + optional color correction on final output
        final_frame_np = diffusion.decode_latent(second_pass_samples)
        if color_correction:
            final_frame_np = image_processor.match_histograms(final_frame_np, init_frame_np)
        final_frame_np = np.clip(final_frame_np, 0, 255).astype(np.uint8)

        output_frames.append(torch.from_numpy(final_frame_np.astype(np.float32) / 255.0))
        prev_frame = final_frame_np.copy()
        pbar.update(1)

    stacked_latents = torch.cat(second_pass_latents, dim=0)
    stacked_images = torch.stack(output_frames, dim=0)

    flower.clear_memory()

    return ({"samples": stacked_latents}, stacked_images)
