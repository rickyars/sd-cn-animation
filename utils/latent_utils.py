"""
Utilities for handling latent operations in the animation pipeline.
Handles common operations like stacking, batching, and conversion.
"""
import torch
import numpy as np


def stack_latents(latent_list):
    """
    Stack a list of latent tensors along batch dimension

    Args:
        latent_list: List of latent tensors to stack

    Returns:
        Stacked latent tensor
    """
    if not latent_list:
        raise ValueError("Empty latent list provided")

    # Ensure all latents are on the same device
    device = latent_list[0].device
    aligned_latents = [lat.to(device) for lat in latent_list]

    # Stack along batch dimension
    return torch.cat(aligned_latents, dim=0)


def ensure_latent_on_device(latent, device=None):
    """
    Ensure latent is on the specified device

    Args:
        latent: Latent dict or tensor
        device: Target device (if None, uses CUDA if available)

    Returns:
        Latent dict or tensor on target device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(latent, dict) and "samples" in latent:
        # Handle dict format
        latent_on_device = {
            "samples": latent["samples"].to(device)
        }

        # Copy other keys
        for k, v in latent.items():
            if k != "samples" and isinstance(v, torch.Tensor):
                latent_on_device[k] = v.to(device)
            elif k != "samples":
                latent_on_device[k] = v

        return latent_on_device
    elif isinstance(latent, torch.Tensor):
        # Handle direct tensor
        return latent.to(device)
    else:
        raise ValueError(f"Unsupported latent type: {type(latent)}")


def prepare_latent_dict(tensor):
    """
    Convert a tensor to a latent dict format expected by ComfyUI

    Args:
        tensor: Latent tensor

    Returns:
        Dict with "samples" key containing the tensor
    """
    return {"samples": tensor}


def create_empty_latent(batch_size, height, width, channels=4, device=None):
    """
    Create an empty latent tensor with the given dimensions

    Args:
        batch_size: Number of items in batch
        height: Latent height (original height / 8)
        width: Latent width (original width / 8)
        channels: Number of latent channels (default: 4)
        device: Target device

    Returns:
        Empty latent tensor initialized with zeros
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent = torch.zeros((batch_size, channels, height, width), device=device)
    return latent


def split_latent_batch(latent_batch):
    """
    Split a batched latent tensor into a list of individual latents

    Args:
        latent_batch: Batched latent tensor [B, C, H, W]

    Returns:
        List of individual latent tensors
    """
    return [latent_batch[i:i + 1] for i in range(latent_batch.size(0))]