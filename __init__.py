"""
SD-CN Animation - Text-to-video generation using Stable Diffusion and ControlNet
"""
from .node_advanced import NODE_CLASS_MAPPINGS as _ADVANCED_NODES
from .node_advanced import NODE_DISPLAY_NAME_MAPPINGS as _ADVANCED_NAMES
from .node_controlnet import NODE_CLASS_MAPPINGS as _CN_NODES
from .node_controlnet import NODE_DISPLAY_NAME_MAPPINGS as _CN_NAMES

NODE_CLASS_MAPPINGS = {**_ADVANCED_NODES, **_CN_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_ADVANCED_NAMES, **_CN_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "0.5.0"
