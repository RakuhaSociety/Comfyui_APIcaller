"""
Sora 2 视频生成节点
支持 Lingke 和 Kie 供应商
"""
import torch
from typing import Tuple, Optional, List
import json

from ..config import create_provider_instance
from ..providers import get_provider
from ..utils import create_blank_image, save_video_to_temp, VideoAdapter, EmptyVideoAdapter
import os


class Sora2VideoNode:
    """
    Sora 2 视频生成节点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "make animate"}),
                "custom_provider": ("CUSTOM_PROVIDER",),
                "orientation": (["portrait", "landscape"], {"default": "portrait"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "model": ("STRING", {"default": "sora-2-all"}),
                "size": (["small", "medium", "large"], {"default": "large"}),
                "duration": ("INT", {"default": 10, "min": 5, "max": 20}),
                "watermark": ("BOOLEAN", {"default": False}),
                "use_kie_upload": ("BOOLEAN", {"default": False}),
                "kie_api_key": ("STRING", {"default": "", "placeholder": "Kie API Key，用于上传取URL"}),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response_info", "video_url")
    FUNCTION = "generate_video"
    CATEGORY = "APIcaller"

    def generate_video(
        self,
        prompt: str,
        custom_provider: dict,
        image: Optional[torch.Tensor] = None,
        model: str = "sora-2-all",
        orientation: str = "portrait",
        size: str = "large",
        duration: int = 10,
        watermark: bool = False,
        use_kie_upload: bool = False,
        kie_api_key: str = "",
    ):
        
        try:
            # 使用自定义供应商
            if not custom_provider.get("api_key") or not custom_provider.get("base_url"):
                return (EmptyVideoAdapter(), "", json.dumps({"error": "请在 Custom Provider 节点中设置 API Key 和 Base URL"}), "")
            
            provider_instance = create_provider_instance(custom_provider)
            provider_type = custom_provider.get("provider_type", "lingke")
            print(f"[APIcaller] 使用供应商: {custom_provider['base_url']}")
            
            # 图像URL转换（通过Kie上传）
            image_urls: Optional[List[str]] = None
            if provider_type == "lingke" and use_kie_upload and image is not None:
                kie_provider = get_provider("kie")
                if kie_api_key.strip():
                    kie_provider.api_key = kie_api_key.strip()
                urls: List[str] = []
                batch_size = image.shape[0]
                for i in range(batch_size):
                    single_image = image[i:i+1]
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
                # 让下游用URL，不再传tensor
                image = None

            # 调用视频生成API
            video_url = ""
            response_json = ""
            task_id = "unknown"
            
            if hasattr(provider_instance, "generate_video_sora2"):
                video_url, response_json = provider_instance.generate_video_sora2(
                    prompt=prompt,
                    image=image,
                    model=model,
                    orientation=orientation,
                    size=size,
                    duration=duration,
                    watermark=watermark,
                    image_urls=image_urls,
                )
                # Try to extract task_id from response
                try:
                    res_data = json.loads(response_json)
                    task_id = res_data.get("task_id", res_data.get("id", "unknown"))
                except:
                    pass
            else:
                return (EmptyVideoAdapter(), "", json.dumps({"error": "Provider does not support Sora 2 video generation"}), "")

            if not video_url:
                print(f"[APIcaller] Failed to get video URL. Response: {response_json}")
                return (EmptyVideoAdapter(), task_id, response_json, "")
                
            # 使用 VideoAdapter 封装 URL 或 本地路径
            video_path = ""
            temp_filename, save_error = save_video_to_temp(video_url)
            
            if not save_error and temp_filename:
                import folder_paths
                video_path = os.path.join(folder_paths.get_temp_directory(), temp_filename)
                video_adapter = VideoAdapter(video_path)
            else:
                video_adapter = VideoAdapter(video_url)
                
            return (video_adapter, task_id, response_json, video_url)

        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[APIcaller] {error_msg}")
            return (EmptyVideoAdapter(), "", json.dumps({"error": error_msg}), "")
