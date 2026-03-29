"""
Hailuo 视频生成节点
支持 Lingke 和 Kie 供应商，可选首帧与尾帧图像
"""
import torch
from typing import Tuple, Optional, List
import json
import os

from ..providers import get_provider
from ..utils import save_video_to_temp, VideoAdapter, EmptyVideoAdapter


class HailuoVideoNode:
    """
    Hailuo 视频生成节点
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "make animate"}),
                "provider": (["lingke", "kie"], {"default": "lingke"}),
                "model": (
                    [
                        "MiniMax-Hailuo-2.3-Fast",
                        "MiniMax-Hailuo-2.3",
                        "2-3-image-to-video-standard",
                        "2-3-image-to-video-pro",
                        "02-image-to-video-standard",
                        "02-image-to-video-pro",
                    ],
                    {"default": "MiniMax-Hailuo-2.3-Fast"},
                ),
                "duration": ("INT", {"default": 10, "min": 1, "max": 30, "step": 1}),
                "resolution": (["768P", "1080P"], {"default": "768P"}),
            },
            "optional": {
                "image_start": ("IMAGE",),   # 首帧
                "image_end": ("IMAGE",),     # 尾帧
                "prompt_optimizer": ("BOOLEAN", {"default": True}),
                "use_kie_upload": ("BOOLEAN", {"default": False}),
                "kie_api_key": ("STRING", {"default": "", "placeholder": "Kie API Key，用于上传取URL"}),
                "api_key": ("STRING", {"default": "", "placeholder": "留空使用已保存的密钥"}),
                "custom_provider": ("CUSTOM_PROVIDER",),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response_info", "video_url")
    FUNCTION = "generate_video"
    CATEGORY = "APIcaller"

    def _extract_single_image(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor[0:1]

    def generate_video(
        self,
        prompt: str,
        provider: str,
        model: str = "MiniMax-Hailuo-2.3-Fast",
        duration: int = 10,
        resolution: str = "768P",
        image_start: Optional[torch.Tensor] = None,
        image_end: Optional[torch.Tensor] = None,
        prompt_optimizer: bool = True,
        use_kie_upload: bool = False,
        kie_api_key: str = "",
        api_key: str = "",
        custom_provider: dict = None,
    ):
        try:
            # 获取供应商实例（自定义供应商优先）
            if custom_provider and custom_provider.get("api_key") and custom_provider.get("base_url"):
                from ..config import create_provider_instance
                provider_instance = create_provider_instance(custom_provider)
                print(f"[APIcaller] 使用自定义供应商: {custom_provider['base_url']}")
            else:
                provider_instance = get_provider(provider)

            if api_key.strip():
                provider_instance.api_key = api_key.strip()

            start_img = self._extract_single_image(image_start)
            end_img = self._extract_single_image(image_end)

            image_urls = None
            if provider == "lingke" and use_kie_upload:
                # 使用 Kie 上传接口获取 URL 再给 Lingke 调用
                kie_provider = get_provider("kie")
                if kie_api_key.strip():
                    kie_provider.api_key = kie_api_key.strip()
                urls: List[str] = []
                image_index = 0
                for tensor in [start_img, end_img]:
                    if tensor is None:
                        continue
                    img_url, upload_error = kie_provider._upload_image(tensor, image_index)
                    if upload_error:
                        return (
                            EmptyVideoAdapter(),
                            "",
                            json.dumps({"error": f"Kie上传失败: {upload_error}"}, ensure_ascii=False),
                            "",
                        )
                    if img_url:
                        urls.append(img_url)
                        image_index += 1
                image_urls = urls
                start_img = None
                end_img = None

            video_url = ""
            response_json = ""
            task_id = "unknown"

            if hasattr(provider_instance, "generate_video_hailuo"):
                video_url, response_json = provider_instance.generate_video_hailuo(
                    prompt=prompt,
                    model=model,
                    duration=duration,
                    resolution=resolution,
                    prompt_optimizer=prompt_optimizer,
                    image_start=start_img,
                    image_end=end_img,
                    image_urls=image_urls,
                )
                try:
                    res_data = json.loads(response_json)
                    task_id = res_data.get("task_id", res_data.get("id", "unknown"))
                except Exception:
                    pass
            else:
                return (
                    EmptyVideoAdapter(),
                    "",
                    json.dumps({"error": f"Provider {provider} does not support Hailuo video generation"}),
                    "",
                )

            if not video_url:
                print(f"[APIcaller] Hailuo Failed to get video URL. Response: {response_json}")
                return (EmptyVideoAdapter(), task_id, response_json, "")

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
            print(f"[APIcaller] Hailuo {error_msg}")
            return (EmptyVideoAdapter(), "", json.dumps({"error": error_msg}), "")
