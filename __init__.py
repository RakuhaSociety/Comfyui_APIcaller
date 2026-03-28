"""
Comfyui_APIcaller - 多API供应商支持的ComfyUI节点

支持的功能:
- Nano Banana 图像编辑 (WaveSpeed, Lingke)
- Nano Banana 文生图 (WaveSpeed, Lingke)

可扩展架构，方便添加新的API供应商
"""

from .config import APICallerSettings
from .nodes import NanoBananaEdit, NanoBananaText2Img, GrokVideoNode, Sora2VideoNode, Veo31VideoNode, HailuoVideoNode, APIKeyPoolNode, OpenAILLM


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "APIcaller_Settings": APICallerSettings,
    "APIcaller_NanoBananaEdit": NanoBananaEdit,
    "APIcaller_NanoBananaText2Img": NanoBananaText2Img,
    "APIcaller_GrokVideo": GrokVideoNode,
    "APIcaller_Sora2Video": Sora2VideoNode,
    "APIcaller_Veo31Video": Veo31VideoNode,
    "APIcaller_HailuoVideo": HailuoVideoNode,
    "APIcaller_KeyPool": APIKeyPoolNode,
    "APIcaller_OpenAILLM": OpenAILLM,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "APIcaller_Settings": "🔧 Custom Provider",
    "APIcaller_NanoBananaEdit": "🍌 Nano Banana Edit",
    "APIcaller_NanoBananaText2Img": "🍌 Nano Banana Text2Img",
    "APIcaller_GrokVideo": "🎬 Grok Video Generator",
    "APIcaller_Sora2Video": "🎬 Sora 2 Video Generator",
    "APIcaller_Veo31Video": "🎬 Veo 3.1 Video Generator",
    "APIcaller_HailuoVideo": "🎬 Hailuo Video Generator",
    "APIcaller_KeyPool": "🔑 API Key Pool",
    "APIcaller_OpenAILLM": "🤖 OpenAI LLM",
}

__all__ = {
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
}

# ComfyUI版本信息
__version__ = "1.0.0"

# 打印加载信息
print(f"[APIcaller] Comfyui_APIcaller v{__version__} loaded successfully!")
print(f"[APIcaller] Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
