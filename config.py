"""
配置管理模块
"""


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
            },
        }

    RETURN_TYPES = ("CUSTOM_PROVIDER",)
    RETURN_NAMES = ("custom_provider",)
    FUNCTION = "create_provider"
    CATEGORY = "APIcaller"

    def create_provider(self, api_key: str, base_url: str):
        if not api_key.strip():
            print("[APIcaller] 警告: 自定义供应商未设置API Key")
        if not base_url.strip():
            print("[APIcaller] 警告: 自定义供应商未设置Base URL")
        
        result = {
            "api_key": api_key.strip(),
            "base_url": base_url.strip().rstrip("/"),
        }
        print(f"[APIcaller] 自定义供应商配置: base_url={result['base_url']}")
        return (result,)
