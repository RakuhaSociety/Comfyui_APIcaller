"""
WaveSpeed API供应商
https://wavespeed.ai
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import json
import torch
import ssl
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO
from PIL import Image

from .base_provider import BaseProvider
from ..utils import pil2tensor, tensor2pil, image_to_base64, download_image, create_blank_image


class WaveSpeedProvider(BaseProvider):
    """
    WaveSpeed API供应商实现
    API文档: https://wavespeed.ai/docs
    """
    
    name = "wavespeed"
    display_name = "WaveSpeed"
    default_base_url = "https://api.wavespeed.ai"
    default_timeout = 300
    
    def __init__(self):
        super().__init__()
        self.max_poll_attempts = 150
        self.poll_interval = 2
        self._session = None
    
    def _get_session(self) -> requests.Session:
        """获取带有重试机制的session"""
        if self._session is None:
            self._session = requests.Session()
            
            # 配置重试策略
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
        
        return self._session
    
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        发起HTTP请求，带有SSL错误处理
        """
        session = self._get_session()
        kwargs.setdefault('timeout', self.timeout)
        kwargs.setdefault('headers', self.get_headers())
        
        try:
            if method.upper() == 'GET':
                return session.get(url, **kwargs)
            elif method.upper() == 'POST':
                return session.post(url, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
        except requests.exceptions.SSLError as e:
            # SSL错误时，尝试禁用SSL验证（不推荐用于生产环境）
            print(f"[WaveSpeed] SSL错误，尝试禁用SSL验证: {str(e)}")
            kwargs['verify'] = False
            if method.upper() == 'GET':
                return session.get(url, **kwargs)
            else:
                return session.post(url, **kwargs)
    
    def get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _upload_image_url(self, image_tensor: torch.Tensor) -> Optional[str]:
        """
        将图像转换为base64 data URL格式
        WaveSpeed接受data URL格式的图像
        """
        base64_str = image_to_base64(image_tensor)
        if base64_str:
            return f"data:image/png;base64,{base64_str}"
        return None
    
    def _poll_for_result(self, request_id: str, pbar=None) -> Tuple[Optional[Dict], Optional[str]]:
        """
        轮询获取任务结果
        
        Args:
            request_id: 任务ID
            pbar: 进度条对象
            
        Returns:
            (结果数据, 错误信息)
        """
        attempts = 0
        
        while attempts < self.max_poll_attempts:
            time.sleep(self.poll_interval)
            attempts += 1
            
            try:
                response = self._make_request(
                    'GET',
                    f"{self.base_url}/api/v3/predictions/{request_id}/result"
                )
                
                if response.status_code != 200:
                    continue
                
                result = response.json()
                
                if result.get("code") != 200:
                    continue
                
                data = result.get("data", {})
                status = data.get("status", "")
                
                # 更新进度
                if pbar:
                    progress = min(90, 30 + (attempts * 60 // self.max_poll_attempts))
                    pbar.update_absolute(progress)
                
                if status == "completed":
                    return data, None
                elif status == "failed":
                    error = data.get("error", "Unknown error")
                    return None, f"任务失败: {error}"
                    
            except Exception as e:
                print(f"[WaveSpeed] 轮询错误: {str(e)}")
        
        return None, "任务超时"
    
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
        
        API端点: POST /api/v3/google/nano-banana-pro/edit
        """
        pbar = kwargs.get('pbar')
        
        if not self.api_key:
            return create_blank_image(), "API密钥未设置", ""
        
        try:
            if pbar:
                pbar.update_absolute(10)
            
            # 准备图像URL列表
            image_urls = []
            for img in images:
                if img is not None:
                    img_url = self._upload_image_url(img)
                    if img_url:
                        image_urls.append(img_url)
            
            if not image_urls:
                return create_blank_image(), "未提供输入图像", ""
            
            # 构建请求payload
            payload = {
                "prompt": prompt,
                "images": image_urls,
                "resolution": resolution,
                "output_format": output_format,
                "enable_sync_mode": False,
                "enable_base64_output": False
            }
            
            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio
            
            if pbar:
                pbar.update_absolute(20)
            
            # 提交任务
            response = self._make_request(
                'POST',
                f"{self.base_url}/api/v3/google/nano-banana-pro/edit",
                json=payload
            )
            
            if response.status_code != 200:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                print(f"[WaveSpeed] {error_msg}")
                return create_blank_image(), error_msg, ""
            
            result = response.json()
            
            if result.get("code") != 200:
                error_msg = f"API返回错误: {result.get('message', '未知错误')}"
                return create_blank_image(), error_msg, ""
            
            request_id = result.get("data", {}).get("id")
            if not request_id:
                return create_blank_image(), "未获取到任务ID", ""
            
            if pbar:
                pbar.update_absolute(30)
            
            # 轮询获取结果
            task_result, error = self._poll_for_result(request_id, pbar)
            
            if error:
                return create_blank_image(), error, ""
            
            if not task_result:
                return create_blank_image(), "未获取到结果", ""
            
            # 获取输出图像
            outputs = task_result.get("outputs", [])
            if not outputs:
                return create_blank_image(), "未生成图像", ""
            
            # 下载第一张图像
            image_url = outputs[0]
            image_data, download_error = download_image(image_url, self.timeout)
            
            if download_error:
                return create_blank_image(), f"下载图像失败: {download_error}", image_url
            
            # 转换为tensor
            pil_image = Image.open(BytesIO(image_data))
            image_tensor = pil2tensor(pil_image)
            
            if pbar:
                pbar.update_absolute(100)
            
            response_info = {
                "provider": self.display_name,
                "model": "nano-banana-pro/edit",
                "task_id": request_id,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "image_url": image_url
            }
            
            return image_tensor, json.dumps(response_info, ensure_ascii=False, indent=2), image_url
            
        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[WaveSpeed] {error_msg}")
            return create_blank_image(), error_msg, ""
    
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
        
        API端点: POST /api/v3/google/nano-banana-pro/text-to-image
        """
        pbar = kwargs.get('pbar')
        
        if not self.api_key:
            return create_blank_image(), "API密钥未设置", ""
        
        try:
            if pbar:
                pbar.update_absolute(10)
            
            # 构建请求payload
            payload = {
                "prompt": prompt,
                "resolution": resolution,
                "output_format": output_format,
                "enable_sync_mode": False,
                "enable_base64_output": False
            }
            
            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio
            
            if pbar:
                pbar.update_absolute(20)
            
            # 提交任务
            response = self._make_request(
                'POST',
                f"{self.base_url}/api/v3/google/nano-banana-pro/text-to-image",
                json=payload
            )
            
            if response.status_code != 200:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                print(f"[WaveSpeed] {error_msg}")
                return create_blank_image(), error_msg, ""
            
            result = response.json()
            
            if result.get("code") != 200:
                error_msg = f"API返回错误: {result.get('message', '未知错误')}"
                return create_blank_image(), error_msg, ""
            
            request_id = result.get("data", {}).get("id")
            if not request_id:
                return create_blank_image(), "未获取到任务ID", ""
            
            if pbar:
                pbar.update_absolute(30)
            
            # 轮询获取结果
            task_result, error = self._poll_for_result(request_id, pbar)
            
            if error:
                return create_blank_image(), error, ""
            
            if not task_result:
                return create_blank_image(), "未获取到结果", ""
            
            # 获取输出图像
            outputs = task_result.get("outputs", [])
            if not outputs:
                return create_blank_image(), "未生成图像", ""
            
            # 下载第一张图像
            image_url = outputs[0]
            image_data, download_error = download_image(image_url, self.timeout)
            
            if download_error:
                return create_blank_image(), f"下载图像失败: {download_error}", image_url
            
            # 转换为tensor
            pil_image = Image.open(BytesIO(image_data))
            image_tensor = pil2tensor(pil_image)
            
            if pbar:
                pbar.update_absolute(100)
            
            response_info = {
                "provider": self.display_name,
                "model": "nano-banana-pro/text-to-image",
                "task_id": request_id,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "image_url": image_url
            }
            
            return image_tensor, json.dumps(response_info, ensure_ascii=False, indent=2), image_url
            
        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[WaveSpeed] {error_msg}")
            return create_blank_image(), error_msg, ""
    
    def get_available_models(self) -> List[str]:
        return [
            "nano-banana-pro/edit",
            "nano-banana-pro/text-to-image",
            "nano-banana-pro/edit-ultra",
            "nano-banana-pro/text-to-image-multi",
        ]
    
    def get_available_resolutions(self) -> List[str]:
        return ["1k", "2k", "4k"]
