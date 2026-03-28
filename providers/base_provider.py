"""
API供应商基类
定义所有供应商必须实现的接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import torch


class BaseProvider(ABC):
    """
    API供应商基类
    所有供应商都需要继承此类并实现相应的方法
    """
    
    # 供应商名称
    name: str = "base"
    
    # 供应商显示名称
    display_name: str = "Base Provider"
    
    # 默认API基础URL
    default_base_url: str = ""
    
    # 默认超时时间（秒）
    default_timeout: int = 300
    
    def __init__(self):
        self.api_key: str = ""
        self.base_url: str = self.default_base_url
        self.timeout: int = self.default_timeout
    
    def configure(self, api_key: str, base_url: Optional[str] = None, timeout: Optional[int] = None):
        """
        配置供应商
        
        Args:
            api_key: API密钥
            base_url: 可选的自定义基础URL
            timeout: 可选的超时时间
        """
        self.api_key = api_key
        if base_url:
            self.base_url = base_url
        if timeout:
            self.timeout = timeout
    
    def get_headers(self) -> Dict[str, str]:
        """
        获取请求头
        
        Returns:
            请求头字典
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    @abstractmethod
    def nano_banana_edit(
        self,
        prompt: str,
        images: List[torch.Tensor],
        aspect_ratio: str = "1:1",
        resolution: str = "1k",
        output_format: str = "png",
        **kwargs
    ) -> Tuple[Optional[torch.Tensor], str, str]:
        """
        Nano Banana 图像编辑
        
        Args:
            prompt: 提示词
            images: 输入图像列表
            aspect_ratio: 宽高比
            resolution: 分辨率
            output_format: 输出格式
            **kwargs: 其他供应商特定参数
            
        Returns:
            (生成的图像tensor, 响应信息, 图像URL)
        """
        pass
    
    @abstractmethod
    def nano_banana_text2img(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        resolution: str = "1k",
        output_format: str = "png",
        **kwargs
    ) -> Tuple[Optional[torch.Tensor], str, str]:
        """
        Nano Banana 文生图
        
        Args:
            prompt: 提示词
            aspect_ratio: 宽高比
            resolution: 分辨率
            output_format: 输出格式
            **kwargs: 其他供应商特定参数
            
        Returns:
            (生成的图像tensor, 响应信息, 图像URL)
        """
        pass
    
    def generate_video(
        self,
        prompt: str,
        image: Optional[torch.Tensor] = None,
        model: str = "grok-video-3",
        aspect_ratio: str = "3:2",
        duration: int = 5,
        **kwargs
    ) -> Tuple[str, str]:
        """
        生成视频接口
        默认返回未实现提示
        Returns: (video_url, response_json)
        """
        return "", "This provider does not support video generation."

    def get_supported_features(self) -> Dict[str, bool]:
        """
        获取供应商支持的功能
        
        Returns:
            功能支持字典
        """
        return {
            "nano_banana_edit": True,
            "nano_banana_text2img": True,
            "generate_video": False,
        }
    
    def get_available_models(self) -> List[str]:
        """
        获取可用的模型列表
        
        Returns:
            模型名称列表
        """
        return []
    
    def get_available_aspect_ratios(self) -> List[str]:
        """
        获取可用的宽高比列表
        
        Returns:
            宽高比列表
        """
        return ["1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
    
    def get_available_resolutions(self) -> List[str]:
        """
        获取可用的分辨率列表
        
        Returns:
            分辨率列表
        """
        return ["1k", "2k", "4k"]
