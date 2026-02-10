"""
ControlNet configuration node for SD-CN Animation.
Bundles all ControlNet-related parameters into a single config object
that can be optionally connected to the main animation node.
"""


class SDCNControlNetConfig:
    """Configures ControlNet settings for use with SD-CN Animation."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "tile_preprocessor": (["None", "Basic", "ColorFix"], {
                    "default": "ColorFix"
                }),
                "tile_blur_strength": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 25.0, "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("SDCN_CN_CONFIG",)
    RETURN_NAMES = ("cn_config",)
    FUNCTION = "build_config"
    CATEGORY = "animation"

    def build_config(self, control_net, strength, start_percent, end_percent,
                     tile_preprocessor, tile_blur_strength):
        return ({
            "control_net": control_net,
            "strength": strength,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "tile_preprocessor": tile_preprocessor,
            "tile_blur_strength": tile_blur_strength,
        },)


NODE_CLASS_MAPPINGS = {
    "SDCNControlNetConfig": SDCNControlNetConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDCNControlNetConfig": "SD-CN ControlNet Config",
}
