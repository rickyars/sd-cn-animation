import os
import torch
import numpy as np
import cv2
import gc
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
                    "default": 0.35,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "fix_frame_strength": ("FLOAT", {
                    "default": 0.35,
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

    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("frames", "flow_visualization", "occlusion_mask", "warped_frame", "blended_frame")
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

    def apply_controlnet(self, img_np, positive, negative, control_net, strength, start_percent=0.0, end_percent=1.0):
        """Apply ControlNet to conditioning based on an image"""
        if control_net is None or img_np is None:
            return positive, negative

        # Convert numpy array to tensor in the format expected by ControlNet
        img_tensor = torch.from_numpy(img_np).float() / 255.0
        if len(img_tensor.shape) == 3 and img_tensor.shape[2] == 3:  # [H, W, C]
            img_tensor = img_tensor.permute(2, 0, 1)  # Convert to [C, H, W]
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]

        # Get the device - in ComfyUI, we need to check model components
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Alternative ways to get device in ComfyUI:
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'device'):
            target_device = self.model.model.device
        elif hasattr(self.model, 'first_stage_model') and hasattr(self.model.first_stage_model, 'device'):
            target_device = self.model.first_stage_model.device

        img_tensor = img_tensor.to(target_device)

        # Create a copy of the controlnet to avoid modifying the original
        cn_copy = control_net.copy()

        # Set the conditioning hint
        cn_copy.cond_hint_original = img_tensor
        cn_copy.strength = strength
        cn_copy.start_percent = start_percent
        cn_copy.end_percent = end_percent

        # Apply to positive conditioning only
        new_positive = []
        for t in positive:
            d = t[1].copy()
            d['control'] = cn_copy  # Attach the ControlNet object directly
            d['control_apply_to_uncond'] = False
            new_positive.append([t[0], d])

        # Return the modified positive conditioning and original negative conditioning
        return new_positive, negative

    def inpaint_with_mask(self, image_pil, mask_pil, strength, steps, seed, device):
        """Perform inpainting on the provided image with the given mask"""
        import comfy.samplers

        # Make sure mask is in grayscale mode
        if mask_pil.mode != 'L':
            mask_pil = mask_pil.convert('L')

        # Apply Gaussian blur to mask if needed
        if self.occlusion_mask_blur > 0:
            from PIL import ImageFilter
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(self.occlusion_mask_blur))

        # Adjust mask intensity
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(mask_pil)
        mask_pil = enhancer.enhance(self.occlusion_mask_multiplier)

        # Ensure the mask is properly thresholded
        mask_np = np.array(mask_pil)
        mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np)

        # Convert to latent space
        image_latent = self.encode_with_vae(image_pil, self.vae, device)

        # Create mask tensor in latent space
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

        # Resize mask to match latent dimensions
        from torch.nn.functional import interpolate
        latent_mask = interpolate(
            mask_tensor,
            size=image_latent.shape[2:],
            mode='bilinear',
            align_corners=False
        ).to(device)  # Ensure mask is on the same device as latent

        # Create noise
        torch.manual_seed(seed)
        noise = torch.randn_like(image_latent)

        # Ensure all tensors are on the same device
        image_latent = image_latent.to(device)
        latent_mask = latent_mask.to(device)
        noise = noise.to(device)

        # Create noised image latent for inpainting
        # This applies noise only to the masked areas
        # Low strength (e.g., 0.15) will keep more of the original image
        # High strength (e.g., 0.85) will generate more new content
        noised_latent = image_latent * (1 - latent_mask) + (
                    image_latent * (1 - strength) + noise * strength) * latent_mask

        # Sample the diffusion model
        # For inpainting, we want to use the provided strength directly
        # Lower values (0.15) preserve more of the original
        # Higher values (0.85) create more new content
        samples = comfy.sample.sample(
            self.model,
            noise,
            steps,
            self.cfg,
            self.sampler_name,
            self.scheduler,
            self.positive,
            self.negative,
            noised_latent,
            denoise=strength
        )

        return {"samples": samples}

    def encode_with_vae(self, pil_image, vae, device):
        """Encode PIL image to latent using ComfyUI's approach"""
        # Convert PIL to numpy
        img = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0

        # Create tensor with correct shape
        img_tensor = torch.from_numpy(img)

        # Move to device
        img_tensor = img_tensor.to(device)

        # Ensure dimensions are multiples of 8
        h, w = img_tensor.shape[0], img_tensor.shape[1]
        x = (w // 8) * 8
        y = (h // 8) * 8

        # Crop if necessary
        if w != x or h != y:
            x_offset = (w % 8) // 2
            y_offset = (h % 8) // 2
            img_tensor = img_tensor[y_offset:y + y_offset, x_offset:x + x_offset, :]

        # Add batch dimension
        pixels = img_tensor.unsqueeze(0)

        # Encode with VAE (only using RGB channels)
        with torch.no_grad():
            latent = vae.encode(pixels[:, :, :, :3])

        return latent

    def sample_diffusion(self, latent, strength, steps, seed):
        """Sample the diffusion model using ComfyUI's samplers"""
        import comfy.samplers
        import comfy.sample

        # Set random seed
        torch.manual_seed(seed)

        # Get the device from the latent
        device = latent["samples"].device

        # Create noise on the same device
        noise = torch.randn_like(latent["samples"]).to(device)

        # Create noisy latent
        if strength > 0:
            noised_latent = latent["samples"] * (1 - strength) + noise * strength
        else:
            noised_latent = latent["samples"]

        # Use ComfyUI's sampling method
        samples = comfy.sample.sample(
            self.model,
            noise,
            steps,
            self.cfg,
            self.sampler_name,
            self.scheduler,
            self.positive,
            self.negative,
            noised_latent,
            denoise=strength
        )

        return {"samples": samples}

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
                        occlusion_difo_multiplier, occlusion_difs_multiplier, seed):
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

        # Initialize collections for debug visualizations
        flow_visualizations = []
        occlusion_masks = []
        warped_frames = []
        blended_frames = []

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
        all_latents = [latent["samples"].to(device)]

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

            warped_frame = cv2.remap(prev_frame, flow_map, None, cv2.INTER_LINEAR,
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

            # Before first inpainting pass
            if self.cn_frame_value == 1:  # Current Frame
                self.positive, self.negative = self.apply_controlnet(
                    pred_next, positive, negative, control_net,
                    controlnet_strength, controlnet_start_percent, controlnet_end_percent
                )
            elif self.cn_frame_value == 2:  # Previous Frame
                self.positive, self.negative = self.apply_controlnet(
                    prev_frame, positive, negative, control_net,
                    controlnet_strength, controlnet_start_percent, controlnet_end_percent
                )

            # Convert to PIL for inpainting
            pred_next_pil = Image.fromarray(pred_next)
            pred_occl_pil = Image.fromarray(pred_occl_gray)

            # First diffusion pass - focusing on occluded areas (mode 4)
            # This properly recreates the inpainting approach from the original code
            # processing_strength controls how much to regenerate in masked areas
            frame_seed = seed + i if seed != 0 else i
            inpaint_result = self.inpaint_with_mask(
                pred_next_pil,
                pred_occl_pil,
                processing_strength,  # Use the user-provided value directly
                sampling_steps,
                frame_seed,
                device
            )

            # Decode the inpainted result for visualization and next step
            inpaint_frame_tensor = vae.decode(inpaint_result["samples"])
            inpaint_frame = inpaint_frame_tensor[0].cpu()
            if inpaint_frame.shape[0] == 3:  # [C, H, W]
                inpaint_frame_np = inpaint_frame.permute(1, 2, 0).numpy() * 255
            else:  # Assume [H, W, C]
                inpaint_frame_np = inpaint_frame.numpy() * 255

            inpaint_frame_np = np.clip(inpaint_frame_np, 0, 255).astype(np.uint8)

            # Convert back to PIL for second pass
            inpaint_frame_pil = Image.fromarray(inpaint_frame_np)

            # Get the frame for the second pass ControlNet application
            if self.cn_frame_value == 1:  # Current Frame
                self.positive, self.negative = self.apply_controlnet(
                    inpaint_frame_np, positive, negative, control_net,
                    controlnet_strength, controlnet_start_percent, controlnet_end_percent
                )
            elif self.cn_frame_value == 2:  # Previous Frame (still the same as before)
                self.positive, self.negative = self.apply_controlnet(
                    prev_frame, positive, negative, control_net,
                    controlnet_strength, controlnet_start_percent, controlnet_end_percent
                )

            # Second diffusion pass - overall refinement (mode 0)
            # Apply the fix_frame_strength to the entire frame without a mask
            # fix_frame_strength should be low (e.g., 0.15) to maintain consistency
            fixed_frame_latent = self.encode_with_vae(inpaint_frame_pil, vae, device)
            fixed_frame_result = self.sample_diffusion(
                {"samples": fixed_frame_latent},
                fix_frame_strength,  # Use the user-provided value directly
                sampling_steps,
                frame_seed + 10000
            )

            # Store this frame's latent - ensure it's on device
            all_latents.append(fixed_frame_result["samples"].to(device))

            # Decode for next iteration
            final_frame_tensor = vae.decode(fixed_frame_result["samples"])
            final_frame = final_frame_tensor[0].cpu()
            if final_frame.shape[0] == 3:  # [C, H, W]
                final_frame_np = final_frame.permute(1, 2, 0).numpy() * 255
            else:  # Assume [H, W, C]
                final_frame_np = final_frame.numpy() * 255

            final_frame_np = np.clip(final_frame_np, 0, 255).astype(np.uint8)

            # Match histogram to first frame for consistency
            import skimage.exposure
            final_frame_np = skimage.exposure.match_histograms(
                final_frame_np,
                init_frame_np,
                channel_axis=-1
            )
            final_frame_np = np.clip(final_frame_np, 0, 255).astype(np.uint8)

            # Update previous frame for next iteration
            prev_frame = final_frame_np.copy()

        # Stack all latents for output
        stacked_latents = torch.cat(all_latents, dim=0)

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

        print(f"Flow batch shape: {flow_batch.shape}")

        # Clean up
        self.FloweR_clear_memory()
        print("Animation generation complete")

        # Return all visualizations
        return (
            {"samples": stacked_latents},
            flow_batch,
            occlusion_batch,
            warped_batch,
            blended_batch
        )

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "SDCNAnimation": Txt2VidNode
}

# Display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDCNAnimation": "SD-CN Animation"
}