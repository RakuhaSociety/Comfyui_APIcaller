"""
供应商模块
"""
from .base_provider import BaseProvider
from .provider_wavespeed import WaveSpeedProvider
from .provider_lingke import LingkeProvider
from .provider_kie import KieProvider


# 供应商注册表
PROVIDERS = {
    "wavespeed": WaveSpeedProvider,
    "lingke": LingkeProvider,
    "kie": KieProvider,
}


def get_provider(provider_name: str) -> BaseProvider:
    """
    获取供应商实例
    
    Args:
        provider_name: 供应商名称
        
    Returns:
        供应商实例
        
    Raises:
        ValueError: 如果供应商不存在
    """
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(PROVIDERS.keys())}")
    
    return PROVIDERS[provider_name]()


def list_providers() -> list:
    """
    列出所有可用的供应商
    
    Returns:
        供应商名称列表
    """
    return list(PROVIDERS.keys())


__all__ = [
    'BaseProvider',
    'WaveSpeedProvider', 
    'LingkeProvider',
    'KieProvider',
    'PROVIDERS',
    'get_provider',
    'list_providers',
]
