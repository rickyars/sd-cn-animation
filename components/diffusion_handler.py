"""
Diffusion process handler for the animation pipeline.
This module handles VAE encoding/decoding and diffusion sampling.
"""
import torch
import numpy as np
from PIL import Image
import comfy.sample


class DiffusionHandler:
    """Handler for diffusion operations in the animation pipeline"""

    def __init__(self, model=None, vae=None, cfg=7.0,
                 sampler_name="euler", scheduler="normal"):
        """
        Initialize the diffusion handler

        Args:
            model: Stable Diffusion model
            vae: VAE model
            cfg: Classifier-free guidance scale
            sampler_name: Name of sampler to use
            scheduler: Name of scheduler to use
        """
        self.model = model
        self.vae = vae
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.positive = None
        self.negative = None

    def set_model_params(self, model, vae, positive, negative,
                         cfg=None, sampler_name=None, scheduler=None):
        """
        Set or update model parameters

        Args:
            model: Stable Diffusion model
            vae: VAE model
            positive: Positive conditioning
            negative: Negative conditioning
            cfg: Classifier-free guidance scale (optional)
            sampler_name: Name of sampler to use (optional)
            scheduler: Name of scheduler to use (optional)
        """
        self.model = model
        self.vae = vae
        self.positive = positive
        self.negative = negative

        if cfg is not None:
            self.cfg = cfg
        if sampler_name is not None:
            self.sampler_name = sampler_name
        if scheduler is not None:
            self.scheduler = scheduler

    def encode_with_vae(self, pil_image, device=None):
        """
        Encode a PIL image to latent space using VAE

        Args:
            pil_image: PIL image to encode
            device: Device to use for encoding (optional)

        Returns:
            Encoded latent tensor
        """
        if self.vae is None:
            raise ValueError("VAE not set. Call set_model_params first.")

        # Set device if provided, otherwise use CUDA if available
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert PIL to numpy
        img = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0

        # Convert to tensor with batch dimension
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        # Encode with VAE
        with torch.no_grad():
            latent = self.vae.encode(img_tensor)

        return latent

    def decode_latent(self, latent):
        """
        Decode a latent tensor to a numpy image array

        Args:
            latent: Latent tensor to decode

        Returns:
            Decoded image as numpy array
        """
        if self.vae is None:
            raise ValueError("VAE not set. Call set_model_params first.")

        # Decode with VAE
        with torch.no_grad():
            decoded = self.vae.decode(latent)

        # Convert tensor to numpy
        if decoded.dim() == 4:  # [B, C, H, W]
            decoded = decoded[0]  # Take first batch item

        # Handle different tensor layouts
        if decoded.shape[0] == 3:  # [C, H, W]
            decoded_np = decoded.permute(1, 2, 0).cpu().numpy() * 255
        else:  # Assume [H, W, C]
            decoded_np = decoded.cpu().numpy() * 255

        return np.clip(decoded_np, 0, 255).astype(np.uint8)

    def inpaint_with_mask(self, image_pil, mask_pil, strength, steps, seed, device=None):
        """
        Inpainting using ComfyUI's approach with latent noise mask

        Args:
            image_pil: PIL image to inpaint
            mask_pil: PIL mask image (L mode)
            strength: Denoising strength (0.0-1.0)
            steps: Number of diffusion steps
            seed: Random seed
            device: Device to use (optional)

        Returns:
            Dictionary with resulting samples
        """
        if self.model is None or self.vae is None:
            raise ValueError("Model or VAE not set. Call set_model_params first.")

        # Set device if provided, otherwise use CUDA if available
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get the latent tensor
        image_latent = self.encode_with_vae(image_pil, device)

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
        """
        Sample the diffusion model using ComfyUI's samplers

        Args:
            latent: Input latent dict with "samples" key
            strength: Denoising strength (0.0-1.0)
            steps: Number of diffusion steps
            seed: Random seed

        Returns:
            Dictionary with resulting samples
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model_params first.")

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