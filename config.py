"""
配置管理模块
管理API密钥和供应商配置
"""
import os
import json
from typing import Dict, Any, Optional


CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'apicaller_config.json')


def get_config() -> Dict[str, Any]:
    """获取完整配置"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"providers": {}}
    except json.JSONDecodeError:
        return {"providers": {}}


def save_config(config: Dict[str, Any]) -> None:
    """保存配置"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def get_provider_config(provider_name: str) -> Dict[str, Any]:
    """获取特定供应商的配置"""
    config = get_config()
    return config.get("providers", {}).get(provider_name, {})


def set_provider_config(provider_name: str, provider_config: Dict[str, Any]) -> None:
    """设置特定供应商的配置"""
    config = get_config()
    if "providers" not in config:
        config["providers"] = {}
    config["providers"][provider_name] = provider_config
    save_config(config)


def get_api_key(provider_name: str) -> Optional[str]:
    """获取供应商的API密钥"""
    provider_config = get_provider_config(provider_name)
    return provider_config.get("api_key", "")


def set_api_key(provider_name: str, api_key: str) -> None:
    """设置供应商的API密钥"""
    provider_config = get_provider_config(provider_name)
    provider_config["api_key"] = api_key
    set_provider_config(provider_name, provider_config)


# 供应商列表定义
PROVIDER_LIST = ["lingke", "kie", "wavespeed"]

# 供应商显示名称映射
PROVIDER_DISPLAY_NAMES = {
    "lingke": "Lingke (灵客)",
    "kie": "Kie",
    "wavespeed": "WaveSpeed",
}

# 供应商默认基础URL映射 - 选择provider后自动使用对应的URL
PROVIDER_BASE_URLS = {
    "lingke": "https://lingkeapi.com",
    "kie": "https://api.kie.ai",
    "wavespeed": "https://api.wavespeed.ai",
}


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
