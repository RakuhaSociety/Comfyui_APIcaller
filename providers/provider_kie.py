"""
Kie API供应商
https://api.kie.ai
异步任务型API
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import json
import torch
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO
from PIL import Image

from .base_provider import BaseProvider
from ..utils import pil2tensor, tensor2pil, image_to_base64, download_image, create_blank_image


class KieProvider(BaseProvider):
    """
    Kie API供应商实现
    API文档: https://api.kie.ai
    
    使用异步任务模式：
    1. POST /api/v1/jobs/createTask 创建任务
    2. GET /api/v1/jobs/getTask/{taskId} 轮询任务状态
    """
    
    name = "kie"
    display_name = "Kie"
    default_base_url = "https://api.kie.ai"
    default_timeout = 300
    
    # 文件上传使用不同的域名
    UPLOAD_BASE_URL = "https://kieai.redpandaai.co"
    
    def __init__(self):
        super().__init__()
        # 轮询间隔2秒，150次≈5分钟；加长到450次≈15分钟
        self.max_poll_attempts = 450
        self.poll_interval = 2
        self._session = None
    
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

    def generate_video(
        self,
        prompt: str,
        image: Optional[torch.Tensor] = None,
        model: str = "grok-imagine/image-to-video",
        aspect_ratio: str = "3:2",
        duration: int = 6,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Grok Video (图转视频)
        模型: grok-imagine/image-to-video
        流程: 上传图片 -> createTask -> 轮询 -> 返回视频URL
        """
        pbar = kwargs.get('pbar')

        if not self.api_key:
            return "", "API密钥未设置"

        # 支持节点层预上传的 image_urls
        pre_uploaded_urls = kwargs.get('image_urls')

        if image is None and not pre_uploaded_urls:
            return "", "需要输入图像"

        try:
            if pbar:
                pbar.update_absolute(5)

            # 如果节点层已经上传过，直接使用
            if pre_uploaded_urls:
                image_urls = pre_uploaded_urls
                print(f"[Kie] 使用节点层预上传的 {len(image_urls)} 张图像URL")
            else:
                # 上传图像
                image_urls = []
                batch_size = image.shape[0]
                for i in range(batch_size):
                    single_image = image[i:i+1]
                    print(f"[Kie] 正在上传图像 {i + 1}...")
                    img_url, upload_error = self._upload_image(single_image, i)
                    if upload_error:
                        print(f"[Kie] 上传图像失败: {upload_error}")
                        return "", f"上传图像失败: {upload_error}"
                    if img_url:
                        image_urls.append(img_url)

            if not image_urls:
                return "", "未获取到上传后的图像URL"

            if pbar:
                pbar.update_absolute(20)

            payload = {
                "model": model,
                "input": {
                    "image_urls": image_urls,
                    "prompt": prompt,
                    "mode": "normal",
                    "duration": str(duration),
                    "aspect_ratio": aspect_ratio
                }
            }

            url = f"{self.base_url}/api/v1/jobs/createTask"
            print(f"[Kie] 请求URL: {url}")
            print(f"[Kie] 使用模型: {model}")

            response = self._make_request('POST', url, json=payload)

            if response.status_code != 200:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                print(f"[Kie] {error_msg}")
                return "", error_msg

            result = response.json()
            print(f"[Kie] 创建任务响应: {json.dumps(result, ensure_ascii=False)[:500]}" )

            data = result.get("data") if result else None
            task_id = (
                result.get("taskId") or 
                result.get("task_id") or 
                result.get("id") or 
                (data.get("taskId") if isinstance(data, dict) else None) or
                (data.get("task_id") if isinstance(data, dict) else None) or
                (data.get("id") if isinstance(data, dict) else None)
            )

            if not task_id:
                return "", f"未获取到任务ID。响应: {json.dumps(result, ensure_ascii=False)}"

            print(f"[Kie] 任务ID: {task_id}")

            if pbar:
                pbar.update_absolute(40)

            task_result, error = self._poll_for_result(task_id, pbar)

            if error:
                return "", error

            if not task_result:
                return "", "未获取到结果"

            # 显式提取尝试，避免长表达式求值问题
            outputs = None
            
            # 1. 检查根节点常见字段
            # Note: The log shows resultUrls is used by this provider
            target_keys = ["resultUrls", "result_urls", "video_url", "videoUrl", "url", "result", "outputs", "videos", "video"]
            
            for key in target_keys:
                if key in task_result and task_result[key]:
                    outputs = task_result[key]
                    break
            
            # 2. 检查 output 嵌套字段
            if not outputs:
                output_node = task_result.get("output")
                if isinstance(output_node, dict):
                    for key in ["video_urls", "video_url", "videoUrl", "url"]:
                        if output_node.get(key):
                            outputs = output_node[key]
                            break

            if isinstance(outputs, str):
                outputs = [outputs]

            if isinstance(outputs, dict):
                # 某些返回可能是 {"url": "..."}
                outputs = list(outputs.values())

            if not outputs:
                print(f"[Kie] 视频URL提取失败。可用字段: {list(task_result.keys())}")
                return "", f"未生成视频。响应: {json.dumps(task_result, ensure_ascii=False)[:500]}"

            video_url = outputs[0]

            if pbar:
                pbar.update_absolute(100)

            response_info = {
                "provider": self.display_name,
                "model": model,
                "task_id": task_id,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "video_url": video_url
            }

            return video_url, json.dumps(response_info, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[Kie] {error_msg}")
            return "", error_msg

    def generate_video_veo31(
        self,
        prompt: str,
        model: str = "veo3.1-fast",
        aspect_ratio: str = "16:9",
        image_start: Optional[torch.Tensor] = None,
        image_end: Optional[torch.Tensor] = None,
        watermark: str = "",
        seeds: int = 0,
        enable_fallback: bool = False,
        enable_translation: bool = True,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Veo 3.1 视频生成 (Kie)

        API端点: POST /api/v1/veo/generate
        模型: veo3_fast / veo3_pro
        """
        pbar = kwargs.get('pbar')

        if not self.api_key:
            return "", "API密钥未设置"

        try:
            if pbar:
                pbar.update_absolute(5)

            # 支持节点层预上传的 image_urls
            pre_uploaded_urls = kwargs.get('image_urls')
            if pre_uploaded_urls:
                image_urls = pre_uploaded_urls
                print(f"[Kie] Veo3.1 使用节点层预上传的 {len(image_urls)} 张图像URL")
            else:
                image_urls = []
                image_index = 0
                for img in [image_start, image_end]:
                    if img is not None:
                        batch_size = img.shape[0]
                        for i in range(batch_size):
                            single_image = img[i:i+1]
                            print(f"[Kie] 正在上传图像 {image_index + 1}...")
                            img_url, upload_error = self._upload_image(single_image, image_index)
                            if upload_error:
                                print(f"[Kie] 上传图像失败: {upload_error}")
                                return "", f"上传图像失败: {upload_error}"
                            if img_url:
                                image_urls.append(img_url)
                                image_index += 1

            if pbar:
                pbar.update_absolute(20)

            model_map = {
                "veo3.1-fast": "veo3_fast",
                "veo3.1-pro": "veo3_pro",
            }
            mapped_model = model_map.get(model, "veo3_fast")

            payload = {
                "prompt": prompt,
                "imageUrls": image_urls,
                "model": mapped_model,
                "watermark": watermark,
                "aspect_ratio": aspect_ratio,
                "seeds": seeds,
                "enableFallback": enable_fallback,
                "enableTranslation": enable_translation,
                "generationType": "REFERENCE_2_VIDEO",
            }

            url = f"{self.base_url}/api/v1/veo/generate"
            print(f"[Kie] 请求URL: {url}")
            print(f"[Kie] 使用模型: {mapped_model}")

            response = self._make_request('POST', url, json=payload)

            if response.status_code != 200:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                print(f"[Kie] {error_msg}")
                return "", error_msg

            result = response.json()
            print(f"[Kie] 创建任务响应: {json.dumps(result, ensure_ascii=False)[:500]}")

            outputs = (
                result.get("resultUrls") or
                result.get("result_urls") or
                result.get("videoUrls") or
                result.get("video_urls") or
                result.get("url") or
                result.get("result")
            )

            if isinstance(outputs, str):
                outputs = [outputs]

            if isinstance(outputs, dict):
                outputs = list(outputs.values())

            if not outputs:
                data = result.get("data") if isinstance(result.get("data"), dict) else result
                task_id = (
                    result.get("taskId") or
                    result.get("task_id") or
                    result.get("id") or
                    (data.get("taskId") if isinstance(data, dict) else None) or
                    (data.get("task_id") if isinstance(data, dict) else None) or
                    (data.get("id") if isinstance(data, dict) else None)
                )

                if not task_id:
                    return "", f"未获取到任务ID或视频URL。响应: {json.dumps(result, ensure_ascii=False)}"

                if pbar:
                    pbar.update_absolute(40)

                task_result, error = self._poll_for_result(task_id, pbar)

                if error:
                    return "", error

                if not task_result:
                    return "", "未获取到结果"

                outputs = (
                    task_result.get("resultUrls") or
                    task_result.get("result_urls") or
                    task_result.get("outputs") or
                    task_result.get("videos") or
                    task_result.get("video") or
                    task_result.get("url") or
                    task_result.get("result")
                )

                if isinstance(outputs, str):
                    outputs = [outputs]

                if isinstance(outputs, dict):
                    outputs = list(outputs.values())

                if not outputs:
                    print(f"[Kie] 视频URL提取失败。可用字段: {list(task_result.keys())}")
                    return "", f"未生成视频。响应: {json.dumps(task_result, ensure_ascii=False)[:500]}"

                video_url = outputs[0]

                if pbar:
                    pbar.update_absolute(100)

                response_info = {
                    "provider": self.display_name,
                    "model": mapped_model,
                    "task_id": task_id,
                    "aspect_ratio": aspect_ratio,
                    "watermark": watermark,
                    "video_url": video_url
                }

                return video_url, json.dumps(response_info, ensure_ascii=False, indent=2)

            video_url = outputs[0]

            if pbar:
                pbar.update_absolute(100)

            response_info = {
                "provider": self.display_name,
                "model": mapped_model,
                "aspect_ratio": aspect_ratio,
                "watermark": watermark,
                "video_url": video_url
            }

            return video_url, json.dumps(response_info, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[Kie] {error_msg}")
            return "", error_msg
    
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
            print(f"[Kie] SSL错误，尝试禁用SSL验证: {str(e)}")
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
    
    def _image_to_data_url(self, image_tensor: torch.Tensor) -> Optional[str]:
        """
        将图像tensor转换为data URL格式
        """
        base64_str = image_to_base64(image_tensor)
        if base64_str:
            return f"data:image/png;base64,{base64_str}"
        return None
    
    def _upload_image(self, image_tensor: torch.Tensor, index: int = 0) -> Tuple[Optional[str], Optional[str]]:
        """
        上传图像到Kie服务器获取URL
        
        Args:
            image_tensor: 图像tensor
            index: 图像索引，用于生成唯一文件名
            
        Returns:
            (图像URL, 错误信息)
        """
        try:
            # 获取base64数据（带data URL前缀）
            data_url = self._image_to_data_url(image_tensor)
            if not data_url:
                return None, "无法转换图像为base64"
            
            # 生成唯一文件名
            import uuid
            filename = f"comfyui_{uuid.uuid4().hex[:8]}_{index}.png"
            
            # 构建上传请求
            payload = {
                "base64Data": data_url,
                "uploadPath": "images",
                "fileName": filename
            }
            
            # 发送上传请求 - 使用专门的上传域名
            upload_url = f"{self.UPLOAD_BASE_URL}/api/file-base64-upload"
            print(f"[Kie] 上传URL: {upload_url}")
            
            response = self._make_request(
                'POST',
                upload_url,
                json=payload
            )
            
            if response.status_code != 200:
                return None, f"上传失败: {response.status_code} - {response.text[:500]}"
            
            result = response.json()
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            
            # 尝试多种可能的返回字段
            image_url = (
                data.get("downloadUrl") or  # Kie返回的实际字段
                data.get("url") or
                data.get("fileUrl") or
                result.get("url") or
                result.get("downloadUrl") or
                result.get("fileUrl")
            )
            
            if not image_url:
                return None, f"上传成功但未获取到URL。响应: {json.dumps(result, ensure_ascii=False)}"
            
            print(f"[Kie] 图像上传成功: {image_url}")
            return image_url, None
            
        except Exception as e:
            return None, f"上传异常: {str(e)}"
    
    def _poll_for_result(self, task_id: str, pbar=None) -> Tuple[Optional[Dict], Optional[str]]:
        """
        轮询获取任务结果
        
        Args:
            task_id: 任务ID
            pbar: 进度条对象
            
        Returns:
            (结果数据, 错误信息)
            
        使用端点: GET /api/v1/jobs/recordInfo?taskId={taskId}
        返回格式: {data: {state, resultJson, failMsg}}
        """
        attempts = 0
        
        while attempts < self.max_poll_attempts:
            time.sleep(self.poll_interval)
            attempts += 1
            
            try:
                # 正确的轮询端点
                poll_url = f"{self.base_url}/api/v1/jobs/recordInfo?taskId={task_id}"
                response = self._make_request('GET', poll_url)
                
                if response.status_code != 200:
                    print(f"[Kie] 轮询状态码: {response.status_code}")
                    continue
                
                result = response.json()
                data = result.get("data", {}) if isinstance(result.get("data"), dict) else {}
                
                # 更新进度
                if pbar:
                    progress = min(90, 30 + (attempts * 60 // self.max_poll_attempts))
                    pbar.update_absolute(progress)
                
                # 检查任务状态 - Kie使用state字段
                state = data.get("state", "")
                
                print(f"[Kie] 轮询 {attempts}: 状态 = {state}")
                
                if state == "success":
                    # 解析resultJson获取结果
                    result_json_str = data.get("resultJson", "{}")
                    try:
                        result_data = json.loads(result_json_str) if isinstance(result_json_str, str) else result_json_str
                        return result_data, None
                    except json.JSONDecodeError:
                        return {"raw": result_json_str}, None
                        
                elif state == "fail":
                    fail_msg = data.get("failMsg", "Unknown error")
                    return None, f"任务失败: {fail_msg}"
                    
                elif state in ["pending", "processing", "running", "queued", ""]:
                    # 任务仍在处理中
                    continue
                    
            except Exception as e:
                print(f"[Kie] 轮询错误: {str(e)}")
        
        return None, "任务超时"
    
    def nano_banana_edit(
        self,
        prompt: str,
        images: List[torch.Tensor],
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
        output_format: str = "png",
        **kwargs
    ) -> Tuple[Optional[torch.Tensor], str, str]:
        """
        Nano Banana 图像编辑
        
        API端点: POST /api/v1/jobs/createTask
        模型: nano-banana-pro
        
        流程: 先上传图像获取URL，再创建任务
        """
        pbar = kwargs.get('pbar')
        model = kwargs.get('model', 'nano-banana-pro')
        
        if not self.api_key:
            return create_blank_image(), "API密钥未设置", ""
        
        try:
            if pbar:
                pbar.update_absolute(5)
            
            # 先上传图像获取URL
            image_urls = []
            image_index = 0
            for img in images:
                if img is not None:
                    batch_size = img.shape[0]
                    for i in range(batch_size):
                        single_image = img[i:i+1]
                        print(f"[Kie] 正在上传图像 {image_index + 1}...")
                        img_url, upload_error = self._upload_image(single_image, image_index)
                        if upload_error:
                            print(f"[Kie] 上传图像失败: {upload_error}")
                            return create_blank_image(), f"上传图像失败: {upload_error}", ""
                        if img_url:
                            image_urls.append(img_url)
                            image_index += 1
            
            if not image_urls:
                return create_blank_image(), "未提供输入图像", ""
            
            print(f"[Kie] 成功上传 {len(image_urls)} 张图像")
            
            if pbar:
                pbar.update_absolute(20)
            
            # 确保resolution格式正确 (e.g. 1K, 2K)
            res_str = resolution.upper() if resolution else "1K"
            
            # 构建请求payload
            payload = {
                "model": model,
                "input": {
                    "prompt": prompt,
                    "image_input": image_urls,
                    "output_format": output_format,
                    "aspect_ratio": aspect_ratio,
                    "resolution": res_str
                }
            }
            
            # 提交任务
            url = f"{self.base_url}/api/v1/jobs/createTask"
            print(f"[Kie] 请求URL: {url}")
            print(f"[Kie] 使用模型: {model}")
            print(f"[Kie] 图片比例: {aspect_ratio}, 分辨率: {res_str}")
            
            response = self._make_request('POST', url, json=payload)
            
            if response.status_code != 200:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                print(f"[Kie] {error_msg}")
                return create_blank_image(), error_msg, ""
            
            result = response.json()
            print(f"[Kie] 创建任务响应: {json.dumps(result, ensure_ascii=False)[:500]}")
            
            # 获取任务ID - 安全处理None值
            data = result.get("data") if result else None
            task_id = (
                result.get("taskId") or 
                result.get("task_id") or 
                result.get("id") or 
                (data.get("taskId") if isinstance(data, dict) else None) or
                (data.get("task_id") if isinstance(data, dict) else None) or
                (data.get("id") if isinstance(data, dict) else None)
            )
            if not task_id:
                return create_blank_image(), f"未获取到任务ID。响应: {json.dumps(result, ensure_ascii=False)}", ""
            
            print(f"[Kie] 任务ID: {task_id}")
            
            if pbar:
                pbar.update_absolute(30)
            
            # 轮询获取结果
            task_result, error = self._poll_for_result(task_id, pbar)
            
            if error:
                return create_blank_image(), error, ""
            
            if not task_result:
                return create_blank_image(), "未获取到结果", ""
            
            # 显式提取尝试 (Nano Banana Edit)
            outputs = None
            
            # 1. 检查根节点常见字段
            target_keys = ["resultUrls", "result_urls", "output", "outputs", "images", "imgs", "url", "video_url"]
            
            for key in target_keys:
                if key in task_result and task_result[key]:
                    val = task_result[key]
                    if isinstance(val, list) and val:
                        outputs = val
                        break
                    elif isinstance(val, str) and val.startswith("http"):
                        outputs = [val]
                        break

            # 2. 检查 output 嵌套字段
            if not outputs:
                output_node = task_result.get("output")
                if isinstance(output_node, dict):
                    for key in ["image_urls", "images", "url", "video_urls"]:
                        if output_node.get(key):
                             val = output_node[key]
                             if isinstance(val, list) and val:
                                outputs = val
                                break
                             elif isinstance(val, str) and val.startswith("http"):
                                outputs = [val]
                                break

            # 3. 之前的逻辑作为最后的备选
            if not outputs:
                outputs = (
                    task_result.get("resultUrls") or  # Kie的实际返回字段
                    task_result.get("result_urls") or
                    task_result.get("output", {}).get("image_urls") if isinstance(task_result.get("output"), dict) else None or
                    task_result.get("outputs") or
                    task_result.get("images") or
                    []
                )
            
            # 如果outputs是字符串，转换为列表
            if isinstance(outputs, str):
                outputs = [outputs]
            
            # 如果没找到，尝试其他字段
            if not outputs:
                single_output = (
                    task_result.get("output") if isinstance(task_result.get("output"), str) else
                    task_result.get("result") if isinstance(task_result.get("result"), str) else
                    None
                )
                if single_output and isinstance(single_output, str) and single_output.startswith("http"):
                    outputs = [single_output]
            
            if not outputs:
                return create_blank_image(), f"未生成图像。响应: {json.dumps(task_result, ensure_ascii=False)[:500]}", ""
            
            print(f"[Kie] 获取到 {len(outputs)} 个结果URL")
            
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
                "model": model,
                "task_id": task_id,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "image_url": image_url
            }
            
            return image_tensor, json.dumps(response_info, ensure_ascii=False, indent=2), image_url
            
        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[Kie] {error_msg}")
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
        
        API端点: POST /api/v1/jobs/createTask
        模型: google/nano-banana (文生图模型)
        """
        pbar = kwargs.get('pbar')
        model = kwargs.get('model', 'google/nano-banana')
        
        if not self.api_key:
            return create_blank_image(), "API密钥未设置", ""
        
        try:
            if pbar:
                pbar.update_absolute(10)
            
            # 构建请求payload
            res_str = resolution.upper() if resolution else "1K"
            payload = {
                "model": model,
                "input": {
                    "prompt": prompt,
                    "output_format": output_format,
                    "image_size": aspect_ratio,
                    "resolution": res_str
                }
            }
            
            if pbar:
                pbar.update_absolute(20)
            
            # 提交任务
            url = f"{self.base_url}/api/v1/jobs/createTask"
            print(f"[Kie] 请求URL: {url}")
            print(f"[Kie] 使用模型: {model}")
            print(f"[Kie] 图片比例: {aspect_ratio}, 分辨率: {res_str}")
            
            response = self._make_request('POST', url, json=payload)
            
            if response.status_code != 200:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                print(f"[Kie] {error_msg}")
                return create_blank_image(), error_msg, ""
            
            result = response.json()
            print(f"[Kie] 创建任务响应: {json.dumps(result, ensure_ascii=False)[:500]}")
            
            # 获取任务ID - 安全处理None值
            data = result.get("data") if result else None
            task_id = (
                result.get("taskId") or 
                result.get("task_id") or 
                result.get("id") or 
                (data.get("taskId") if isinstance(data, dict) else None) or
                (data.get("task_id") if isinstance(data, dict) else None) or
                (data.get("id") if isinstance(data, dict) else None)
            )
            if not task_id:
                return create_blank_image(), f"未获取到任务ID。响应: {json.dumps(result, ensure_ascii=False)}", ""
            
            print(f"[Kie] 任务ID: {task_id}")
            
            if pbar:
                pbar.update_absolute(30)
            
            # 轮询获取结果
            task_result, error = self._poll_for_result(task_id, pbar)
            
            if error:
                return create_blank_image(), error, ""
            
            if not task_result:
                return create_blank_image(), "未获取到结果", ""
            
            # 显式提取尝试 (Nano Banana Text2Img)
            outputs = None
            
            # 1. 检查根节点常见字段
            target_keys = ["resultUrls", "result_urls", "output", "outputs", "images", "imgs", "url", "video_url"]
            
            for key in target_keys:
                if key in task_result and task_result[key]:
                    val = task_result[key]
                    if isinstance(val, list) and val:
                        outputs = val
                        break
                    elif isinstance(val, str) and val.startswith("http"):
                        outputs = [val]
                        break
            
            # 2. 检查 output 嵌套字段
            if not outputs:
                output_node = task_result.get("output")
                if isinstance(output_node, dict):
                    for key in ["image_urls", "images", "url", "video_urls"]:
                        if output_node.get(key):
                             val = output_node[key]
                             if isinstance(val, list) and val:
                                outputs = val
                                break
                             elif isinstance(val, str) and val.startswith("http"):
                                outputs = [val]
                                break

            # 3. 之前的逻辑作为最后的备选 (处理 weird structure)
            if not outputs:
                outputs = (
                    task_result.get("resultUrls") or
                    task_result.get("result_urls") or
                    task_result.get("output", {}).get("image_urls") if isinstance(task_result.get("output"), dict) else None or
                    task_result.get("outputs") or
                    task_result.get("images") or
                    []
                )
            
            if isinstance(outputs, str):
                outputs = [outputs]
            
            if not outputs:
                single_output = (
                    task_result.get("output") if isinstance(task_result.get("output"), str) else
                    task_result.get("result") if isinstance(task_result.get("result"), str) else
                    None
                )
                if single_output and isinstance(single_output, str) and single_output.startswith("http"):
                    outputs = [single_output]
            
            if not outputs:
                return create_blank_image(), f"未生成图像。响应: {json.dumps(task_result, ensure_ascii=False)[:500]}", ""
            
            print(f"[Kie] 获取到 {len(outputs)} 个结果URL")
            
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
                "model": model,
                "task_id": task_id,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "image_url": image_url
            }
            
            return image_tensor, json.dumps(response_info, ensure_ascii=False, indent=2), image_url
            
        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[Kie] {error_msg}")
            return create_blank_image(), error_msg, ""

    def generate_video_sora2(
        self,
        prompt: str,
        image: Optional[torch.Tensor] = None,
        orientation: str = "portrait",
        size: str = "large",
        duration: int = 10,
        watermark: bool = False,
        model: str = "sora-2-image-to-video-stable",
        **kwargs
    ) -> Tuple[str, str]:
        """
        Sora 2 视频生成 (图生视频)
        
        API端点: POST /api/v1/jobs/createTask
        模型: sora-2-image-to-video
        """
        pbar = kwargs.get('pbar')

        if not self.api_key:
            return "", "API密钥未设置"

        try:
            if pbar:
                pbar.update_absolute(5)

            # 支持节点层预上传的 image_urls
            pre_uploaded_urls = kwargs.get('image_urls')
            if pre_uploaded_urls:
                image_urls = pre_uploaded_urls
                print(f"[Kie] Sora2 使用节点层预上传的 {len(image_urls)} 张图像URL")
            else:
                # 上传图像获取URL
                image_urls = []
                if image is not None:
                    batch_size = image.shape[0]
                    for i in range(batch_size):
                        single_image = image[i:i+1]
                        print(f"[Kie] 正在上传图像 {i + 1}...")
                        img_url, upload_error = self._upload_image(single_image, i)
                        if upload_error:
                            print(f"[Kie] 上传图像失败: {upload_error}")
                            return "", f"上传图像失败: {upload_error}"
                        if img_url:
                            image_urls.append(img_url)

            if pbar:
                pbar.update_absolute(20)

            # 将 orientation 映射到 Kie 的 aspect_ratio
            aspect_ratio_map = {
                "portrait": "portrait",
                "landscape": "landscape",
            }
            aspect_ratio = aspect_ratio_map.get(orientation, "landscape")

            # 将 duration 映射到 n_frames (Kie使用帧数)
            # 假设大约 5 秒 = 10 帧, 10 秒 = 10 帧, 15 秒 = 15 帧, 20 秒 = 20 帧
            n_frames_map = {
                5: "5",
                10: "10",
                15: "15",
                20: "20",
            }
            n_frames = n_frames_map.get(duration, "10")

            # 构建请求payload
            payload = {
                "model": model,
                "input": {
                    "prompt": prompt,
                    "image_urls": image_urls,
                    "aspect_ratio": aspect_ratio,
                    "n_frames": n_frames,
                    "remove_watermark": not watermark
                }
            }

            url = f"{self.base_url}/api/v1/jobs/createTask"
            print(f"[Kie] 请求URL: {url}")
            print(f"[Kie] 使用模型: {model}")

            response = self._make_request('POST', url, json=payload)

            if response.status_code != 200:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                print(f"[Kie] {error_msg}")
                return "", error_msg

            result = response.json()
            print(f"[Kie] 创建任务响应: {json.dumps(result, ensure_ascii=False)[:500]}")

            data = result.get("data") if result else None
            task_id = (
                result.get("taskId") or 
                result.get("task_id") or 
                result.get("id") or 
                (data.get("taskId") if isinstance(data, dict) else None) or
                (data.get("task_id") if isinstance(data, dict) else None) or
                (data.get("id") if isinstance(data, dict) else None)
            )

            if not task_id:
                return "", f"未获取到任务ID。响应: {json.dumps(result, ensure_ascii=False)}"

            print(f"[Kie] 任务ID: {task_id}")

            if pbar:
                pbar.update_absolute(40)

            # 轮询获取结果
            task_result, error = self._poll_for_result(task_id, pbar)

            if error:
                return "", error

            if not task_result:
                return "", "未获取到结果"

            # 提取视频URL
            outputs = None
            target_keys = ["resultUrls", "result_urls", "video_url", "videoUrl", "url", "result", "outputs", "videos", "video"]
            
            for key in target_keys:
                if key in task_result and task_result[key]:
                    outputs = task_result[key]
                    break
            
            if not outputs:
                output_node = task_result.get("output")
                if isinstance(output_node, dict):
                    for key in ["video_urls", "video_url", "videoUrl", "url"]:
                        if output_node.get(key):
                            outputs = output_node[key]
                            break

            if isinstance(outputs, str):
                outputs = [outputs]

            if isinstance(outputs, dict):
                outputs = list(outputs.values())

            if not outputs:
                print(f"[Kie] 视频URL提取失败。可用字段: {list(task_result.keys())}")
                return "", f"未生成视频。响应: {json.dumps(task_result, ensure_ascii=False)[:500]}"

            video_url = outputs[0]

            if pbar:
                pbar.update_absolute(100)

            response_info = {
                "provider": self.display_name,
                # "model": "sora-2-image-to-video",
                "model": "sora-2-image-to-video-stable",
                "task_id": task_id,
                "orientation": orientation,
                "n_frames": n_frames,
                "video_url": video_url
            }

            return video_url, json.dumps(response_info, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[Kie] {error_msg}")
            return "", error_msg
    
    def generate_video_hailuo(
        self,
        prompt: str,
        model: str = "2-3-image-to-video-standard",
        duration: int = 6,
        resolution: str = "768P",
        image_start: Optional[torch.Tensor] = None,
        image_end: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Hailuo 视频生成 (Kie)

        API端点: POST /api/v1/jobs/createTask
        轮询: GET /api/v1/jobs/recordInfo?taskId={taskId}
        模型: hailuo/2-3-image-to-video-standard / hailuo/2-3-image-to-video-pro
        """
        pbar = kwargs.get('pbar')

        if not self.api_key:
            return "", "API密钥未设置"

        try:
            if pbar:
                pbar.update_absolute(5)

            # 支持节点层预上传的 image_urls
            pre_uploaded_urls = kwargs.get('image_urls')
            image_url = None
            end_image_url = None
            if pre_uploaded_urls:
                print(f"[Kie] Hailuo 使用节点层预上传的 {len(pre_uploaded_urls)} 张图像URL")
                if len(pre_uploaded_urls) >= 1:
                    image_url = pre_uploaded_urls[0]
                if len(pre_uploaded_urls) >= 2:
                    end_image_url = pre_uploaded_urls[1]
            else:
                # 上传图像获取URL
                if image_start is not None:
                    single_image = image_start[0:1]
                    print(f"[Kie] Hailuo 正在上传首帧图像...")
                    img_url, upload_error = self._upload_image(single_image, 0)
                    if upload_error:
                        print(f"[Kie] Hailuo 上传首帧图像失败: {upload_error}")
                        return "", f"上传首帧图像失败: {upload_error}"
                    if img_url:
                        image_url = img_url

                if image_end is not None:
                    single_image = image_end[0:1]
                    print(f"[Kie] Hailuo 正在上传尾帧图像...")
                    img_url, upload_error = self._upload_image(single_image, 1)
                    if upload_error:
                        print(f"[Kie] Hailuo 上传尾帧图像失败: {upload_error}")
                        return "", f"上传尾帧图像失败: {upload_error}"
                    if img_url:
                        end_image_url = img_url

            if pbar:
                pbar.update_absolute(20)

            # Kie的hailuo模型需要加上 hailuo/ 前缀
            full_model = f"hailuo/{model}" if not model.startswith("hailuo/") else model

            input_data: Dict[str, Any] = {
                "prompt": prompt,
                "duration": str(duration),
                "resolution": resolution,
                "prompt_optimizer": kwargs.get("prompt_optimizer", True),
            }

            if image_url:
                input_data["image_url"] = image_url
            if end_image_url:
                input_data["end_image_url"] = end_image_url

            payload = {
                "model": full_model,
                "input": input_data,
            }

            url = f"{self.base_url}/api/v1/jobs/createTask"
            print(f"[Kie] Hailuo 请求URL: {url}")
            print(f"[Kie] Hailuo 使用模型: {full_model}")
            print(f"[Kie] Hailuo payload: {json.dumps({k: v for k, v in payload.items()}, ensure_ascii=False)}")

            response = self._make_request('POST', url, json=payload)

            if response.status_code != 200:
                error_msg = f"API错误: {response.status_code} - {response.text}"
                print(f"[Kie] Hailuo {error_msg}")
                return "", error_msg

            result = response.json()
            print(f"[Kie] Hailuo 创建任务响应: {json.dumps(result, ensure_ascii=False)[:500]}")

            # 尝试直接获取结果
            outputs = (
                result.get("resultUrls") or
                result.get("result_urls") or
                result.get("videoUrls") or
                result.get("video_urls") or
                result.get("url") or
                result.get("result")
            )

            if isinstance(outputs, str):
                outputs = [outputs]

            if isinstance(outputs, dict):
                outputs = list(outputs.values())

            if not outputs:
                # 需要轮询
                data = result.get("data") if isinstance(result.get("data"), dict) else result
                task_id = (
                    result.get("taskId") or
                    result.get("task_id") or
                    result.get("id") or
                    (data.get("taskId") if isinstance(data, dict) else None) or
                    (data.get("task_id") if isinstance(data, dict) else None) or
                    (data.get("id") if isinstance(data, dict) else None)
                )

                if not task_id:
                    return "", f"未获取到任务ID或视频URL。响应: {json.dumps(result, ensure_ascii=False)}"

                print(f"[Kie] Hailuo 任务ID: {task_id}")

                if pbar:
                    pbar.update_absolute(40)

                task_result, error = self._poll_for_result(task_id, pbar)

                if error:
                    return "", error

                if not task_result:
                    return "", "未获取到结果"

                outputs = (
                    task_result.get("resultUrls") or
                    task_result.get("result_urls") or
                    task_result.get("outputs") or
                    task_result.get("videos") or
                    task_result.get("video") or
                    task_result.get("url") or
                    task_result.get("result")
                )

                if isinstance(outputs, str):
                    outputs = [outputs]

                if isinstance(outputs, dict):
                    outputs = list(outputs.values())

                if not outputs:
                    print(f"[Kie] Hailuo 视频URL提取失败。可用字段: {list(task_result.keys())}")
                    return "", f"未生成视频。响应: {json.dumps(task_result, ensure_ascii=False)[:500]}"

                video_url = outputs[0]

                if pbar:
                    pbar.update_absolute(100)

                response_info = {
                    "provider": self.display_name,
                    "model": full_model,
                    "task_id": task_id,
                    "duration": duration,
                    "resolution": resolution,
                    "video_url": video_url
                }

                return video_url, json.dumps(response_info, ensure_ascii=False, indent=2)

            video_url = outputs[0]

            if pbar:
                pbar.update_absolute(100)

            response_info = {
                "provider": self.display_name,
                "model": full_model,
                "duration": duration,
                "resolution": resolution,
                "video_url": video_url
            }

            return video_url, json.dumps(response_info, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[Kie] Hailuo {error_msg}")
            return "", error_msg

    def get_available_models(self) -> List[str]:
        return [
            "nano-banana-pro",
            "google/nano-banana",
            "grok-imagine/image-to-video",
            # "sora-2-image-to-video",
            "sora-2-image-to-video-stable",
            "veo3.1-fast",
            "veo3.1-pro",
            "hailuo/2-3-image-to-video-standard",
            "hailuo/2-3-image-to-video-pro",
            "hailuo/02-image-to-video-standard",
            "hailuo/02-image-to-video-pro",
        ]
    
    def get_available_resolutions(self) -> List[str]:
        return ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
