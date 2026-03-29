"""
配置管理模块
"""


PROVIDER_TYPE_LIST = ["lingke", "kie", "wavespeed", "openai"]

PROVIDER_TYPE_DISPLAY = {
    "lingke": "Lingke (灵客)",
    "kie": "Kie",
    "wavespeed": "WaveSpeed",
    "openai": "OpenAI (标准格式)",
}


def create_provider_instance(custom_provider: dict):
    """根据 custom_provider dict 创建对应的 Provider 实例"""
    from .providers import get_provider
    from .providers.provider_lingke import LingkeProvider

    provider_type = custom_provider.get("provider_type", "lingke")
    api_key = custom_provider.get("api_key", "")
    base_url = custom_provider.get("base_url", "")

    try:
        provider_instance = get_provider(provider_type)
    except Exception:
        provider_instance = LingkeProvider()

    provider_instance.api_key = api_key
    if base_url:
        provider_instance.base_url = base_url
    return provider_instance


class APICallerSettings:
    """
    自定义供应商节点
    允许用户输入自定义的 API Key 和 Base URL，接入其他节点后覆盖供应商选择
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "自定义API Key"}),
                "base_url": ("STRING", {"default": "", "multiline": False, "placeholder": "自定义Base URL，如 https://api.example.com"}),
                "provider_type": (PROVIDER_TYPE_LIST, {"default": "lingke"}),
            },
        }

    RETURN_TYPES = ("CUSTOM_PROVIDER",)
    RETURN_NAMES = ("custom_provider",)
    FUNCTION = "create_provider"
    CATEGORY = "APIcaller"

    def create_provider(self, api_key: str, base_url: str, provider_type: str = "lingke"):
        if not api_key.strip():
            print("[APIcaller] 警告: 自定义供应商未设置API Key")
        if not base_url.strip():
            print("[APIcaller] 警告: 自定义供应商未设置Base URL")
        
        result = {
            "api_key": api_key.strip(),
            "base_url": base_url.strip().rstrip("/"),
            "provider_type": provider_type,
        }
        display = PROVIDER_TYPE_DISPLAY.get(provider_type, provider_type)
        print(f"[APIcaller] 自定义供应商配置: {display}, base_url={result['base_url']}")
        return (result,)
