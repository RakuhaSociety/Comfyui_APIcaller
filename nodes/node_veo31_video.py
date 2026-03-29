"""
Veo 3.1 视频生成节点
支持 Lingke 和 Kie 供应商（fast / pro），可选首帧与尾帧图像
"""
import torch
from typing import Tuple, Optional, List
import json
import os

from ..config import create_provider_instance
from ..providers import get_provider
from ..providers.provider_lingke import LingkeProvider
from ..utils import save_video_to_temp, VideoAdapter, EmptyVideoAdapter


class Veo31VideoNode:
    """
    Veo 3.1 视频生成节点
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "make animate"}),
                "custom_provider": ("CUSTOM_PROVIDER",),
                "model": (LingkeProvider.VEO31_MODELS, {"default": LingkeProvider.VEO31_MODELS[0]}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "3:2", "2:3"], {"default": "16:9"}),
            },
            "optional": {
                "image_start": ("IMAGE",),
                "image_end": ("IMAGE",),
                "enable_upsample": ("BOOLEAN", {"default": True}),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "use_kie_upload": ("BOOLEAN", {"default": False}),
                "kie_api_key": ("STRING", {"default": "", "placeholder": "Kie API Key，用于上传取URL"}),
                "enable_translation": ("BOOLEAN", {"default": True}),
                "seeds": ("INT", {"default": 0, "min": 0, "max": 2**31-1}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response_info", "video_url")
    FUNCTION = "generate_video"
    CATEGORY = "APIcaller"

    def _extract_single_image(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        # 只取 batch 中的第一张
        return tensor[0:1]

    def generate_video(
        self,
        prompt: str,
        custom_provider: dict,
        model: str = "veo_3_1-fast",
        aspect_ratio: str = "16:9",
        image_start: Optional[torch.Tensor] = None,
        image_end: Optional[torch.Tensor] = None,
        enable_upsample: bool = True,
        enhance_prompt: bool = True,
        watermark: bool = False,
        use_kie_upload: bool = False,
        kie_api_key: str = "",
        enable_translation: bool = True,
        seeds: int = 0,
    ):
        try:
            # 使用自定义供应商
            if not custom_provider.get("api_key") or not custom_provider.get("base_url"):
                return (EmptyVideoAdapter(), "", json.dumps({"error": "请在 Custom Provider 节点中设置 API Key 和 Base URL"}), "")
            
            provider_instance = create_provider_instance(custom_provider)
            provider_type = custom_provider.get("provider_type", "lingke")
            print(f"[APIcaller] 使用供应商: {custom_provider['base_url']}")

            # 只取首尾帧各一张，避免冗余
            start_img = self._extract_single_image(image_start)
            end_img = self._extract_single_image(image_end)

            image_urls = None
            if use_kie_upload or provider_type == "kie":
                # 使用 Kie 的上传接口获取 URL
                if provider_type == "kie":
                    kie_provider = provider_instance
                else:
                    kie_provider = get_provider("kie")
                    if kie_api_key.strip():
                        kie_provider.api_key = kie_api_key.strip()
                urls: List[str] = []
                image_index = 0
                for tensor in [start_img, end_img]:
                    if tensor is None:
                        continue
                    img_url, upload_error = kie_provider._upload_image(tensor, image_index)  # type: ignore[attr-defined]
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
                # 避免再走base64上传，交由 URL 使用
                start_img = None
                end_img = None

            video_url = ""
            response_json = ""
            task_id = "unknown"

            if hasattr(provider_instance, "generate_video_veo31"):
                video_url, response_json = provider_instance.generate_video_veo31(
                    prompt=prompt,
                    model=model,
                    aspect_ratio=aspect_ratio,
                    image_start=start_img,
                    image_end=end_img,
                    image_urls=image_urls,
                    enable_upsample=enable_upsample,
                    enhance_prompt=enhance_prompt,
                    watermark=watermark,
                    enable_translation=enable_translation,
                    seeds=seeds,
                )
                try:
                    res_data = json.loads(response_json)
                    task_id = res_data.get("task_id", res_data.get("id", "unknown"))
                except Exception:
                    pass
            else:
                return (EmptyVideoAdapter(), "", json.dumps({"error": "Provider does not support Veo 3.1 video generation"}), "")

            if not video_url:
                print(f"[APIcaller] Failed to get video URL. Response: {response_json}")
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
            print(f"[APIcaller] {error_msg}")
            return (EmptyVideoAdapter(), "", json.dumps({"error": error_msg}), "")