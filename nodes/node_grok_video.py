"""
Grok Video 视频生成节点
支持 Lingke 和 Kie 供应商
"""
import torch
from typing import Tuple, Optional, List
import json

from ..providers import get_provider
from ..utils import create_blank_image, save_video_to_temp, VideoAdapter, EmptyVideoAdapter
import os
import json


class GrokVideoNode:
    """
    Grok Video 视频生成节点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cat eating fish"}),
                "provider": (["lingke", "kie"], {"default": "lingke"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "model": ("STRING", {"default": "grok-video-3-10s", "multiline": False}),
                "aspect_ratio": (["16:9", "9:16", "3:2", "2:3", "1:1"], {"default": "3:2"}),
                "size": (["720p", "1080p"], {"default": "720p"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15}),
                "use_kie_upload": ("BOOLEAN", {"default": False}),
                "kie_api_key": ("STRING", {"default": "", "placeholder": "Kie API Key，用于上传取URL"}),
                "api_key": ("STRING", {"default": "", "placeholder": "留空使用已保存的密钥"}),
                "custom_provider": ("CUSTOM_PROVIDER",),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response_info", "video_url")
    FUNCTION = "generate_video"
    CATEGORY = "APIcaller"

    def generate_video(
        self,
        prompt: str,
        provider: str,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
        image5: Optional[torch.Tensor] = None,
        model: str = "grok-video-3-10s",
        aspect_ratio: str = "3:2",
        size: str = "720p",
        duration: int = 5,
        use_kie_upload: bool = False,
        kie_api_key: str = "",
        api_key: str = "",
        custom_provider: dict = None,
    ):
        
        try:
            # 收集所有非空图像
            all_images = [img for img in [image1, image2, image3, image4, image5] if img is not None]

            # 获取供应商实例（自定义供应商优先）
            if custom_provider and custom_provider.get("api_key") and custom_provider.get("base_url"):
                from ..config import create_provider_instance
                provider_instance = create_provider_instance(custom_provider)
                print(f"[APIcaller] 使用自定义供应商: {custom_provider['base_url']}")
            else:
                provider_instance = get_provider(provider)
            
            # 如果提供了API密钥，临时设置
            if api_key.strip():
                provider_instance.api_key = api_key.strip()
            
            # 兼容Kie模型名称
            if provider == "kie" and model == "grok-video-3-10s":
                model = "grok-imagine/image-to-video"
                
            # 图像URL转换（通过Kie上传）
            image_urls: Optional[List[str]] = None
            if provider == "lingke" and use_kie_upload and all_images:
                kie_provider = get_provider("kie")
                if kie_api_key.strip():
                    kie_provider.api_key = kie_api_key.strip()
                urls: List[str] = []
                for i, img in enumerate(all_images):
                    single_image = img[0:1]
                    img_url, upload_error = kie_provider._upload_image(single_image, i)  # type: ignore[attr-defined]
                    if upload_error:
                        return (
                            EmptyVideoAdapter(),
                            "",
                            json.dumps({"error": f"Kie上传失败: {upload_error}"}, ensure_ascii=False),
                            "",
                        )
                    if img_url:
                        urls.append(img_url)
                image_urls = urls
                all_images = []

            # 将多张图像合并为一个batch tensor传给provider
            merged_image = None
            if all_images:
                tensors = [img[0:1] for img in all_images]
                merged_image = torch.cat(tensors, dim=0)

            # 调用视频生成API
            video_url = ""
            response_json = ""
            
            task_id = "unknown"
            
            if hasattr(provider_instance, "generate_video"):
                video_url, response_json = provider_instance.generate_video(
                    prompt=prompt,
                    image=merged_image,
                    model=model,
                    aspect_ratio=aspect_ratio,
                    duration=duration,
                    image_urls=image_urls,
                    size=size,
                )
                # Try to extract task_id from response for parity
                try:
                    res_data = json.loads(response_json)
                    task_id = res_data.get("task_id", res_data.get("id", "unknown"))
                except:
                    pass
            else:
                return (EmptyVideoAdapter(), "", json.dumps({"error": f"Provider {provider} does not support video generation"}), "")

            if not video_url:
                print(f"[APIcaller] Failed to get video URL. Response: {response_json}")
                return (EmptyVideoAdapter(), task_id, response_json, "")
                
            # 使用 VideoAdapter 封装 URL 或 本地路径
            # 推荐先下载保存，确保稳定性
            video_path = ""
            temp_filename, save_error = save_video_to_temp(video_url)
            
            if not save_error and temp_filename:
                import folder_paths
                video_path = os.path.join(folder_paths.get_temp_directory(), temp_filename)
                video_adapter = VideoAdapter(video_path)
            else:
                # 如果下载失败，尝试使用URL模式 (虽然 create_blank_image 可能更好)
                video_adapter = VideoAdapter(video_url)
                
            return (video_adapter, task_id, response_json, video_url)

        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[APIcaller] {error_msg}")
            # Ensure we return valid tuple size even on error
            return (EmptyVideoAdapter(), "", json.dumps({"error": error_msg}), "")
