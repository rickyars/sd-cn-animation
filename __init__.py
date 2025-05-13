"""
SD-CN Animation - Text-to-video generation using Stable Diffusion and ControlNet
"""
from .node import NODE_CLASS_MAPPINGS as SDCN_NCM, NODE_DISPLAY_NAME_MAPPINGS as SDCN_NDCM
from .node_advanced import NODE_CLASS_MAPPINGS as ADV_NCM, NODE_DISPLAY_NAME_MAPPINGS as ADV_NDCM

NODE_CLASS_MAPPINGS = {**SDCN_NCM, **ADV_NCM}
NODE_DISPLAY_NAME_MAPPINGS = {**SDCN_NDCM, **ADV_NDCM}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Version info
__version__ = "0.3.0"