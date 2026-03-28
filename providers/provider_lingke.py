"""
Lingke (灵客) API供应商
https://lingkeapi.com
使用Gemini原生格式的API
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
import re
import torch
import base64
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO
from PIL import Image

from .base_provider import BaseProvider
from ..utils import pil2tensor, tensor2pil, image_to_base64, download_image, create_blank_image, base64_to_pil


class LingkeProvider(BaseProvider):
    """
    Lingke (灵客) API供应商实现
    使用Gemini原生格式的API
    API文档: https://lingke112.apifox.cn
    """
    
    name = "lingke"
    display_name = "Lingke (灵客)"
    default_base_url = "https://lingkeapi.com"
    default_timeout = 300
    
    # Veo 3.1 可用模型列表 - 只需在这里修改
    VEO31_MODELS = [
        "veo3.1",
        "veo3.1-4k",
        "veo3.1-fast",
        "veo3.1-pro",
        "veo3.1-pro-4k",
        "veo_3_1",
        "veo_3_1-4K",
        "veo_3_1-fast",
        "veo_3_1-fast-4K",
    ]
    
    # 可用的模型映射
    MODEL_ENDPOINTS = {
        "gemini-2.0-flash-exp-image-generation": "/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent",
        "gemini-3-pro-image-preview": "/v1beta/models/gemini-3-pro-image-preview:generateContent",
        "gemini-2.5-flash-image": "/v1beta/models/gemini-2.5-flash-image:generateContent",
        "gemini-2.5-flash-image-preview": "/v1beta/models/gemini-2.5-flash-image-preview:generateContent",
    }
    
    def __init__(self):
        super().__init__()
        self._session = None
        # 轮询间隔2秒，450次≈15分钟
        self.max_poll_attempts = 450
        self.poll_interval = 2
    
    def _get_session(self) -> requests.Session:
        """获取带有重试机制的session"""
        if self._session is None:
            self._session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
        return self._session
    
    def get_headers(self, include_content_type: bool = True) -> Dict[str, str]:
        """构建请求头，GET场景可关闭Content-Type以贴近官方示例"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        if include_content_type:
            headers["Content-Type"] = "application/json"
        return headers
    
    def _get_model_endpoint(self, model: str) -> str:
        """获取模型对应的API端点"""
        # 如果模型名在映射中，使用映射的端点
        if model in self.MODEL_ENDPOINTS:
            return self.MODEL_ENDPOINTS[model]
        # 否则构建默认端点
        return f"/v1beta/models/{model}:generateContent"
    
    def _extract_image_from_response(self, response_data: Dict) -> Tuple[Optional[torch.Tensor], str, str]:
        """
        从Gemini API响应中提取图像
        
        Returns:
            (图像tensor, 文本响应, 图像URL或base64片段)
        """
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                return None, "API返回空结果", ""
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            text_response = ""
            image_tensor = None
            image_info = ""
            
            for part in parts:
                # 提取文本
                if "text" in part:
                    text_response += part["text"]
                
                # 提取内联图像数据
                if "inlineData" in part or "inline_data" in part:
                    inline_data = part.get("inlineData") or part.get("inline_data", {})
                    mime_type = inline_data.get("mimeType") or inline_data.get("mime_type", "image/png")
                    data = inline_data.get("data", "")
                    
                    if data:
                        try:
                            image_bytes = base64.b64decode(data)
                            pil_image = Image.open(BytesIO(image_bytes))
                            image_tensor = pil2tensor(pil_image)
                            image_info = f"base64_image ({len(data)} chars)"
                        except Exception as e:
                            print(f"[Lingke] 解析图像数据失败: {str(e)}")
            
            return image_tensor, text_response, image_info
            
        except Exception as e:
            print(f"[Lingke] 解析响应失败: {str(e)}")
            return None, str(e), ""
    
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
        Nano Banana 图像编辑 (使用Gemini原生格式)
        
        API端点: /v1beta/models/{model}:generateContent
        """
        pbar = kwargs.get('pbar')
        model = kwargs.get('model', 'gemini-3-pro-image-preview')
        
        if not self.api_key:
            return create_blank_image(), "API密钥未设置", ""
        
        try:
            if pbar:
                pbar.update_absolute(10)
            
            # 构建请求parts
            parts = []
            
            # 添加文本提示
            parts.append({
                "text": prompt
            })
            
            # 添加图像
            images_added = 0
            for img in images:
                if img is not None:
                    batch_size = img.shape[0]
                    for i in range(batch_size):
                        single_image = img[i:i+1]
                        img_base64 = image_to_base64(single_image)
                        if img_base64:
                            parts.append({
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": img_base64
                                }
                            })
                            images_added += 1
            
            print(f"[Lingke] 添加了 {images_added} 张图像")
            
            # 构建Gemini原生格式的payload
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": parts
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio,
                        "imageSize": resolution.upper()
                    }
                }
            }
            
            if pbar:
                pbar.update_absolute(30)
            
            # 获取API端点
            endpoint = self._get_model_endpoint(model)
            url = f"{self.base_url}{endpoint}"
            
            print(f"[Lingke] 请求URL: {url}")
            print(f"[Lingke] 使用模型: {model}")
            print(f"[Lingke] 图片比例: {aspect_ratio}, 分辨率: {resolution}")
            
            # 发送请求
            session = self._get_session()
            response = session.post(
                url,
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if pbar:
                pbar.update_absolute(70)
            
            if response.status_code != 200:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                print(f"[Lingke] {error_msg}")
                return create_blank_image(), error_msg, ""
            
            result = response.json()
            
            # 从响应中提取图像
            image_tensor, text_response, image_info = self._extract_image_from_response(result)
            
            if pbar:
                pbar.update_absolute(100)
            
            if image_tensor is not None:
                response_info = {
                    "provider": self.display_name,
                    "model": model,
                    "endpoint": endpoint,
                    "images_processed": images_added,
                    "text_response": text_response[:200] if text_response else "",
                }
                return image_tensor, json.dumps(response_info, ensure_ascii=False, indent=2), image_info
            else:
                # 没有找到图像
                return create_blank_image(), f"未能从响应中提取图像。响应: {text_response}", ""
            
        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[Lingke] {error_msg}")
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
        Nano Banana 文生图 (使用Gemini原生格式)
        """
        pbar = kwargs.get('pbar')
        model = kwargs.get('model', 'gemini-3-pro-image-preview')
        
        if not self.api_key:
            return create_blank_image(), "API密钥未设置", ""
        
        try:
            if pbar:
                pbar.update_absolute(10)
            
            # 构建Gemini原生格式的payload
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio,
                        "imageSize": resolution.upper()
                    }
                }
            }
            
            if pbar:
                pbar.update_absolute(30)
            
            # 获取API端点
            endpoint = self._get_model_endpoint(model)
            url = f"{self.base_url}{endpoint}"
            
            print(f"[Lingke] 请求URL: {url}")
            print(f"[Lingke] 使用模型: {model}")
            print(f"[Lingke] 图片比例: {aspect_ratio}, 分辨率: {resolution}")
            
            # 发送请求
            session = self._get_session()
            response = session.post(
                url,
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )
            
            if pbar:
                pbar.update_absolute(70)
            
            if response.status_code != 200:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                print(f"[Lingke] {error_msg}")
                return create_blank_image(), error_msg, ""
            
            result = response.json()
            
            # 从响应中提取图像
            image_tensor, text_response, image_info = self._extract_image_from_response(result)
            
            if pbar:
                pbar.update_absolute(100)
            
            if image_tensor is not None:
                response_info = {
                    "provider": self.display_name,
                    "model": model,
                    "endpoint": endpoint,
                    "text_response": text_response[:200] if text_response else "",
                }
                return image_tensor, json.dumps(response_info, ensure_ascii=False, indent=2), image_info
            else:
                return create_blank_image(), f"未能从响应中提取图像。响应: {text_response}", ""
            
        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[Lingke] {error_msg}")
            return create_blank_image(), error_msg, ""

    def _poll_video_result(self, video_id: str, pbar=None) -> Tuple[Optional[Dict], Optional[str]]:
        attempts = 0
        session = self._get_session()

        while attempts < self.max_poll_attempts:
            time.sleep(self.poll_interval)
            attempts += 1

            try:
                poll_url = f"{self.base_url}/v1/video/query?id={video_id}"
                response = session.get(
                    poll_url,
                    headers=self.get_headers(include_content_type=False),
                    timeout=self.timeout
                )

                if response.status_code != 200:
                    print(f"[Lingke] 轮询状态码: {response.status_code}")
                    continue

                result = response.json()
                data = result.get("data", {}) if isinstance(result.get("data"), dict) else result

                if pbar:
                    progress = min(90, 30 + (attempts * 60 // self.max_poll_attempts))
                    pbar.update_absolute(progress)

                status = (
                    data.get("status") or
                    data.get("state") or
                    data.get("task_status") or
                    ""
                )
                status = str(status).lower()

                print(f"[Lingke] 轮询 {attempts}: 状态 = {status}")

                if status in ["success", "succeeded", "completed", "done"]:
                    return data, None
                if status in ["fail", "failed", "error"]:
                    fail_msg = data.get("message") or data.get("error") or "Unknown error"
                    return None, f"任务失败: {fail_msg}"

            except Exception as e:
                print(f"[Lingke] 轮询错误: {str(e)}")

        return None, "任务超时"

    def generate_video(
        self,
        prompt: str,
        image: Optional[torch.Tensor] = None,
        model: str = "grok-video-3",
        aspect_ratio: str = "3:2",
        duration: int = 5,
        **kwargs
    ) -> Tuple[str, str]:
        pbar = kwargs.get("pbar")

        if not self.api_key:
            return "", "API密钥未设置"

        try:
            if pbar:
                pbar.update_absolute(10)

            size_map = {
                "16:9": "1280x720",
                "9:16": "720x1280",
                "3:2": "1200x800",
                "2:3": "800x1200",
                "1:1": "1024x1024",
            }
            size = kwargs.get("size") or size_map.get(aspect_ratio, "1200x800")

            images_payload: List[str] = []
            image_urls: List[str] = kwargs.get("image_urls") or []

            if image_urls:
                images_payload.extend(image_urls)
                image_debug = [
                    {"index": idx, "type": "url", "length": len(url)}
                    for idx, url in enumerate(images_payload)
                ]
            else:
                if image is not None:
                    batch_size = image.shape[0]
                    for i in range(batch_size):
                        single_image = image[i:i+1]
                        img_base64 = image_to_base64(single_image)
                        if img_base64:
                            data_uri = f"data:image/png;base64,{img_base64}"
                            images_payload.append(data_uri)
                image_debug = [
                    # {"index": idx, "type": "base64-data-uri", "length": len(img_str)}
                    {"type": "base64", "size": len(img_str), "preview": img_str[:30] + "..."}
                    for idx, img_str in enumerate(images_payload)
                ]

            payload = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "size": size,
                "images": images_payload,
                "duration": duration,
            }

            if pbar:
                pbar.update_absolute(30)

            url = f"{self.base_url}/v1/video/create"
            print(f"[Lingke] 请求URL: {url}")
            print(f"[Lingke] 使用模型: {model}")
            print(f"[Lingke] payload: {json.dumps({key: val for key, val in payload.items() if key != 'images'}, ensure_ascii=False)}")
            print(f"[Lingke] images: {image_debug}")

            session = self._get_session()
            response = session.post(
                url,
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )

            if pbar:
                pbar.update_absolute(50)

            if response.status_code != 200:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                print(f"[Lingke] {error_msg}")
                return "", error_msg

            result = response.json()
            data = result.get("data") if isinstance(result.get("data"), dict) else result
            video_id = (
                result.get("id") or
                result.get("task_id") or
                (data.get("id") if isinstance(data, dict) else None) or
                (data.get("task_id") if isinstance(data, dict) else None)
            )

            if not video_id:
                return "", f"未获取到任务ID。响应: {json.dumps(result, ensure_ascii=False)}"

            if pbar:
                pbar.update_absolute(60)

            task_result, error = self._poll_video_result(video_id, pbar)

            if error:
                return "", error

            if not task_result:
                return "", "未获取到结果"

            video_url = (
                task_result.get("video_url") or
                task_result.get("videoUrl") or
                task_result.get("url") or
                task_result.get("result")
            )

            if isinstance(video_url, list):
                video_url = video_url[0] if video_url else ""

            if not video_url:
                return "", f"未生成视频。响应: {json.dumps(task_result, ensure_ascii=False)[:500]}"

            if pbar:
                pbar.update_absolute(100)

            response_info = {
                "provider": self.display_name,
                "model": model,
                "task_id": video_id,
                "aspect_ratio": aspect_ratio,
                "size": size,
                "duration": duration,
                "video_url": video_url
            }

            return video_url, json.dumps(response_info, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[Lingke] {error_msg}")
            return "", error_msg

    def generate_video_sora2(
        self,
        prompt: str,
        image: Optional[torch.Tensor] = None,
        orientation: str = "portrait",
        size: str = "large",
        duration: int = 10,
        watermark: bool = False,
        model: str = "sora-2-all",
        **kwargs
    ) -> Tuple[str, str]:
        """
        Sora 2 视频生成
        
        API端点: POST /v1/video/create
        模型: sora-2-all
        轮询端点: GET /v1/video/query?id={task_id}
        """
        pbar = kwargs.get("pbar")

        if not self.api_key:
            return "", "API密钥未设置"

        try:
            if pbar:
                pbar.update_absolute(10)

            # 处理图像输入
            images_payload: List[str] = []
            image_urls: List[str] = kwargs.get("image_urls") or []

            if image_urls:
                images_payload.extend(image_urls)
                image_debug = [
                    {"index": idx, "type": "url", "length": len(url)}
                    for idx, url in enumerate(images_payload)
                ]
            else:
                if image is not None:
                    batch_size = image.shape[0]
                    for i in range(batch_size):
                        single_image = image[i:i+1]
                        img_base64 = image_to_base64(single_image)
                        if img_base64:
                            data_uri = f"data:image/png;base64,{img_base64}"
                            images_payload.append(data_uri)

                image_debug = [
                    # {"index": idx, "type": "base64-data-uri", "length": len(img_str)}
                    {"type": "base64", "size": len(img_str), "preview": img_str[:30] + "..."}
                    for idx, img_str in enumerate(images_payload)
                ]

            # 构建请求payload
            payload = {
                "model": model,
                "prompt": prompt,
                "images": images_payload,
                "orientation": orientation,
                "size": size,
                "duration": duration,
                "watermark": watermark,
                "private": True
            }

            if pbar:
                pbar.update_absolute(30)

            url = f"{self.base_url}/v1/video/create"
            print(f"[Lingke] 请求URL: {url}")
            print(f"[Lingke] 使用模型: {model}")
            print(f"[Lingke] payload: {json.dumps({key: val for key, val in payload.items() if key != 'images'}, ensure_ascii=False)}")
            print(f"[Lingke] images: {image_debug}")

            session = self._get_session()
            response = session.post(
                url,
                headers=self.get_headers(),
                json=payload,
                timeout=self.timeout
            )

            if pbar:
                pbar.update_absolute(50)

            if response.status_code != 200:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                print(f"[Lingke] {error_msg}")
                return "", error_msg

            result = response.json()
            print(f"[Lingke] 创建任务响应: {json.dumps(result, ensure_ascii=False)[:500]}")
            
            # Sora 2 返回格式: {"id": "sora-2-all:task_xxx", ...}
            data = result.get("data") if isinstance(result.get("data"), dict) else result
            task_id = (
                result.get("id") or
                result.get("task_id") or
                (data.get("id") if isinstance(data, dict) else None) or
                (data.get("task_id") if isinstance(data, dict) else None)
            )

            if not task_id:
                return "", f"未获取到任务ID。响应: {json.dumps(result, ensure_ascii=False)}"

            print(f"[Lingke] 任务ID: {task_id}")

            if pbar:
                pbar.update_absolute(60)

            # 轮询获取结果
            task_result, error = self._poll_sora2_result(task_id, pbar)

            if error:
                return "", error

            if not task_result:
                return "", "未获取到结果"

            # 提取视频URL
            video_url = (
                task_result.get("video_url") or
                task_result.get("videoUrl") or
                task_result.get("url") or
                task_result.get("result")
            )

            if isinstance(video_url, list):
                video_url = video_url[0] if video_url else ""

            if not video_url:
                return "", f"未生成视频。响应: {json.dumps(task_result, ensure_ascii=False)[:500]}"

            if pbar:
                pbar.update_absolute(100)

            response_info = {
                "provider": self.display_name,
                "model": "sora-2-all",
                "task_id": task_id,
                "orientation": orientation,
                "size": size,
                "duration": duration,
                "video_url": video_url
            }

            return video_url, json.dumps(response_info, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[Lingke] {error_msg}")
            return "", error_msg

    def _poll_sora2_result(self, task_id: str, pbar=None) -> Tuple[Optional[Dict], Optional[str]]:
        """
        轮询获取 Sora 2 任务结果
        
        使用端点: GET /v1/video/query?id={task_id}
        """
        attempts = 0
        
        while attempts < self.max_poll_attempts:
            time.sleep(self.poll_interval)
            attempts += 1
            
            try:
                poll_url = f"{self.base_url}/v1/video/query?id={task_id}"
                session = self._get_session()
                response = session.get(
                    poll_url,
                    headers=self.get_headers(include_content_type=False),
                    timeout=self.timeout
                )

                if response.status_code != 200:
                    print(f"[Lingke] 轮询状态码: {response.status_code}")
                    continue

                result = response.json()
                data = result.get("data", {}) if isinstance(result.get("data"), dict) else result

                if pbar:
                    progress = min(90, 60 + (attempts * 30 // self.max_poll_attempts))
                    pbar.update_absolute(progress)

                status = (
                    data.get("status") or
                    data.get("state") or
                    data.get("task_status") or
                    result.get("status") or
                    ""
                )
                status = str(status).lower()

                print(f"[Lingke] 轮询 {attempts}: 状态 = {status}")

                if status in ["success", "succeeded", "completed", "done"]:
                    return data, None
                if status in ["fail", "failed", "error"]:
                    fail_msg = data.get("message") or data.get("error") or "Unknown error"
                    return None, f"任务失败: {fail_msg}"

            except Exception as e:
                print(f"[Lingke] 轮询错误: {str(e)}")

        return None, "任务超时"
    
    def generate_video_veo31(
        self,
        prompt: str,
        model: str = "veo3.1-fast",
        aspect_ratio: str = "16:9",
        image_start: Optional[torch.Tensor] = None,
        image_end: Optional[torch.Tensor] = None,
        enable_upsample: bool = True,
        enhance_prompt: bool = True,
        watermark: bool = False,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Veo 3.1 视频生成 (fast/pro)

        API端点: POST /v1/video/create
        轮询端点: GET /v1/video/query?id={task_id}
        """
        pbar = kwargs.get("pbar")

        if not self.api_key:
            return "", "API密钥未设置"

        try:
            if pbar:
                pbar.update_absolute(10)

            images_payload: List[str] = []
            image_urls: List[str] = kwargs.get("image_urls") or []

            if image_urls:
                # 优先使用URL，贴近官方示例
                images_payload.extend(image_urls[:2])
                image_debug = [
                    {"index": idx, "type": "url", "length": len(url)}
                    # {"type": "url", "size": len(url), "preview": url[:30] + "..."}
                    for idx, url in enumerate(images_payload)
                ]
            else:
                # 按文档要求使用 base64 数据，添加 data URI 前缀
                for tensor in [image_start, image_end]:
                    if tensor is not None:
                        # 仅取第一张，避免批量额外上传
                        single = tensor[0:1]
                        img_base64 = image_to_base64(single)
                        if img_base64:
                            data_uri = f"data:image/png;base64,{img_base64}"
                            images_payload.append(data_uri)

                # 记录图片数量和大小，避免输出完整base64
                image_debug = [
                    # {"index": idx, "type": "base64-data-uri", "length": len(img_str)}
                    {"type": "base64", "size": len(img_str), "preview": img_str[:30] + "..."}
                    for idx, img_str in enumerate(images_payload)
                ]

            payload = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "images": images_payload,
                "enable_upsample": enable_upsample,
                "enhance_prompt": enhance_prompt,
                "watermark": watermark,
            }

            if pbar:
                pbar.update_absolute(30)

            url = f"{self.base_url}/v1/video/create"
            # url = f"{self.base_url}/v1/videos"
            safe_headers = self.get_headers().copy()
            # 避免打印完整密钥，只保留前缀便于确认生效
            if "Authorization" in safe_headers:
                auth_val = safe_headers["Authorization"]
                safe_headers["Authorization"] = auth_val[:12] + "..." if len(auth_val) > 12 else "***"

            print(f"[Lingke] 请求URL: {url}")
            print(f"[Lingke] 使用模型: {model}")
            print(f"[Lingke] 请求头: {json.dumps(safe_headers, ensure_ascii=False)}")
            print(f"[Lingke] payload: {json.dumps({key: val for key, val in payload.items() if key != 'images'}, ensure_ascii=False)}")
            print(f"[Lingke] images: {image_debug}")

            session = self._get_session()
            max_retries = 3
            response = None
            for attempt in range(1, max_retries + 1):
                response = session.post(
                    url,
                    headers=self.get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
                if pbar:
                    pbar.update_absolute(30 + attempt * 5)

                if response.status_code == 500:
                    print(f"[Lingke] 第{attempt}次请求返回500，服务器繁忙，等待后重试...")
                    try:
                        print(f"[Lingke] 响应头: {response.headers}")
                        print(f"[Lingke] 响应体: {response.text[:1000]}")
                    except Exception:
                        pass
                    if attempt < max_retries:
                        time.sleep(self.poll_interval)
                        continue
                break

            if pbar:
                pbar.update_absolute(50)

            if not response or response.status_code != 200:
                error_msg = f"API错误: {response.status_code if response else '无响应'}"
                print(f"[Lingke] {error_msg}")
                try:
                    if response:
                        print(f"[Lingke] 响应头: {response.headers}")
                        print(f"[Lingke] 响应体: {response.text[:2000]}")
                except Exception:
                    pass
                return "", error_msg

            result = response.json()
            print(f"[Lingke] 创建任务响应: {json.dumps(result, ensure_ascii=False)[:500]}")

            data = result.get("data") if isinstance(result.get("data"), dict) else result
            task_id = (
                result.get("id") or
                result.get("task_id") or
                (data.get("id") if isinstance(data, dict) else None) or
                (data.get("task_id") if isinstance(data, dict) else None)
            )

            if not task_id:
                return "", f"未获取到任务ID。响应: {json.dumps(result, ensure_ascii=False)}"

            print(f"[Lingke] 任务ID: {task_id}")

            if pbar:
                pbar.update_absolute(60)

            task_result, error = self._poll_sora2_result(task_id, pbar)

            if error:
                return "", error

            if not task_result:
                return "", "未获取到结果"

            video_url = (
                task_result.get("video_url") or
                task_result.get("videoUrl") or
                task_result.get("url") or
                task_result.get("result")
            )

            if isinstance(video_url, list):
                video_url = video_url[0] if video_url else ""

            if not video_url:
                return "", f"未生成视频。响应: {json.dumps(task_result, ensure_ascii=False)[:500]}"

            if pbar:
                pbar.update_absolute(100)

            response_info = {
                "provider": self.display_name,
                "model": model,
                "task_id": task_id,
                "aspect_ratio": aspect_ratio,
                "enable_upsample": enable_upsample,
                "enhance_prompt": enhance_prompt,
                "video_url": video_url
            }

            return video_url, json.dumps(response_info, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[Lingke] {error_msg}")
            return "", error_msg

    def generate_video_hailuo(
        self,
        prompt: str,
        model: str = "MiniMax-Hailuo-2.3-Fast",
        duration: int = 10,
        resolution: str = "768P",
        prompt_optimizer: bool = True,
        image_start: Optional[torch.Tensor] = None,
        image_end: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Hailuo 视频生成 (Lingke)

        API端点: POST /minimax/v1/video_generation
        轮询: 复用 _poll_sora2_result (GET /v1/video/query)
        模型: MiniMax-Hailuo-2.3-Fast / MiniMax-Hailuo-2.3
        """
        pbar = kwargs.get("pbar")

        if not self.api_key:
            return "", "API密钥未设置"

        try:
            if pbar:
                pbar.update_absolute(10)

            # 处理首尾帧图像
            image_urls: List[str] = kwargs.get("image_urls") or []
            first_frame_image = None
            last_frame_image = None

            if image_urls:
                if len(image_urls) >= 1:
                    first_frame_image = image_urls[0]
                if len(image_urls) >= 2:
                    last_frame_image = image_urls[1]
            else:
                for idx, tensor in enumerate([image_start, image_end]):
                    if tensor is not None:
                        single = tensor[0:1]
                        img_base64 = image_to_base64(single)
                        if img_base64:
                            data_uri = f"data:image/png;base64,{img_base64}"
                            if idx == 0:
                                first_frame_image = data_uri
                            else:
                                last_frame_image = data_uri

            if pbar:
                pbar.update_absolute(20)

            payload: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
                "prompt_optimizer": prompt_optimizer,
            }

            if first_frame_image:
                payload["first_frame_image"] = first_frame_image
            if last_frame_image:
                payload["last_frame_image"] = last_frame_image

            if pbar:
                pbar.update_absolute(30)

            url = f"{self.base_url}/minimax/v1/video_generation"

            safe_headers = self.get_headers().copy()
            if "Authorization" in safe_headers:
                auth_val = safe_headers["Authorization"]
                safe_headers["Authorization"] = auth_val[:12] + "..." if len(auth_val) > 12 else "***"

            print(f"[Lingke] Hailuo 请求URL: {url}")
            print(f"[Lingke] Hailuo 使用模型: {model}")
            print(f"[Lingke] Hailuo 请求头: {json.dumps(safe_headers, ensure_ascii=False)}")
            # 不打印图片完整内容
            safe_payload = {k: (v[:50] + '...' if isinstance(v, str) and len(v) > 200 else v) for k, v in payload.items()}
            print(f"[Lingke] Hailuo payload: {json.dumps(safe_payload, ensure_ascii=False)}")

            session = self._get_session()
            max_retries = 3
            response = None
            for attempt in range(1, max_retries + 1):
                response = session.post(
                    url,
                    headers=self.get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
                if pbar:
                    pbar.update_absolute(30 + attempt * 5)

                if response.status_code == 500:
                    print(f"[Lingke] Hailuo 第{attempt}次请求返回500，等待后重试...")
                    try:
                        print(f"[Lingke] 响应体: {response.text[:1000]}")
                    except Exception:
                        pass
                    if attempt < max_retries:
                        time.sleep(self.poll_interval)
                        continue
                break

            if pbar:
                pbar.update_absolute(50)

            if not response or response.status_code != 200:
                error_msg = f"API错误: {response.status_code if response else '无响应'}"
                print(f"[Lingke] Hailuo {error_msg}")
                try:
                    if response:
                        print(f"[Lingke] 响应体: {response.text[:2000]}")
                except Exception:
                    pass
                return "", error_msg

            result = response.json()
            print(f"[Lingke] Hailuo 创建任务响应: {json.dumps(result, ensure_ascii=False)[:500]}")

            data = result.get("data") if isinstance(result.get("data"), dict) else result
            task_id = (
                result.get("id") or
                result.get("task_id") or
                (data.get("id") if isinstance(data, dict) else None) or
                (data.get("task_id") if isinstance(data, dict) else None)
            )

            if not task_id:
                return "", f"未获取到任务ID。响应: {json.dumps(result, ensure_ascii=False)}"

            print(f"[Lingke] Hailuo 任务ID: {task_id}")

            if pbar:
                pbar.update_absolute(60)

            task_result, error = self._poll_sora2_result(task_id, pbar)

            if error:
                return "", error

            if not task_result:
                return "", "未获取到结果"

            video_url = (
                task_result.get("video_url") or
                task_result.get("videoUrl") or
                task_result.get("url") or
                task_result.get("result")
            )

            if isinstance(video_url, list):
                video_url = video_url[0] if video_url else ""

            if not video_url:
                return "", f"未生成视频。响应: {json.dumps(task_result, ensure_ascii=False)[:500]}"

            if pbar:
                pbar.update_absolute(100)

            response_info = {
                "provider": self.display_name,
                "model": model,
                "task_id": task_id,
                "duration": duration,
                "resolution": resolution,
                "prompt_optimizer": prompt_optimizer,
                "video_url": video_url
            }

            return video_url, json.dumps(response_info, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[Lingke] Hailuo {error_msg}")
            return "", error_msg

    def get_available_models(self) -> List[str]:
        return [
            "gemini-3-pro-image-preview",
            "gemini-2.5-flash-image",
            "gemini-2.5-flash-image-preview",
            "gemini-2.0-flash-exp-image-generation",
            "grok-video-3",
            "sora-2-all",
            *self.VEO31_MODELS,
            "MiniMax-Hailuo-2.3-Fast",
            "MiniMax-Hailuo-2.3",
        ]
    
    def get_available_resolutions(self) -> List[str]:
        # Lingke使用Gemini格式，分辨率由模型决定
        return ["auto"]
