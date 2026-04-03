"""
GPT Image 节点
文生图 (Text-to-Image) 和 图生图 (Image-to-Image)
支持 Lingke 和 Kie 供应商，Lingke 额外支持蒙版 (mask) 模式
"""
import torch
import comfy.utils
import time
from typing import Tuple, Optional, List
import json

from ..config import create_provider_instance
from ..providers import get_provider
from ..utils import create_blank_image


class GPTImageText2Img:
    """
    GPT Image 文生图节点
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful sunset over the ocean"}),
                "custom_provider": ("CUSTOM_PROVIDER",),
                "model": ("STRING", {"default": "gpt-image-1.5", "multiline": False, "placeholder": "如 gpt-image-1.5"}),
            },
            "optional": {
                "size": (["1024x1024", "1024x1536", "1536x1024", "auto"], {"default": "1024x1024"}),
                "aspect_ratio": ("STRING", {"default": "", "placeholder": "Kie用，如 1:1, 3:2（留空使用size）"}),
                "quality": (["medium", "high", "low", "auto"], {"default": "medium"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
                "use_kie_upload": ("BOOLEAN", {"default": False}),
                "kie_api_key": ("STRING", {"default": "", "placeholder": "Kie API Key，用于上传取URL"}),
                "error_retry": ("BOOLEAN", {"default": False}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate"
    CATEGORY = "APIcaller/GPTImage"

    def generate(
        self,
        prompt: str,
        custom_provider: dict,
        model: str = "gpt-image-1.5",
        size: str = "1024x1024",
        aspect_ratio: str = "",
        quality: str = "medium",
        n: int = 1,
        use_kie_upload: bool = False,
        kie_api_key: str = "",
        error_retry: bool = False,
        max_retries: int = 3,
    ) -> Tuple[torch.Tensor, str, str]:
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)

        try:
            if not custom_provider.get("api_key") or not custom_provider.get("base_url"):
                return (create_blank_image(), "请在 Custom Provider 节点中设置 API Key 和 Base URL", "")

            provider_instance = create_provider_instance(custom_provider)
            provider_type = custom_provider.get("provider_type", "lingke")
            print(f"[APIcaller] GPT Image T2I 使用供应商: {custom_provider['base_url']}")

            attempts = max_retries if error_retry else 1
            last_error = ""
            for attempt in range(1, attempts + 1):
                if attempt > 1:
                    print(f"[APIcaller] 第 {attempt}/{attempts} 次重试...")
                    time.sleep(2)

                pbar = comfy.utils.ProgressBar(100)
                pbar.update_absolute(5)

                result_image, response, image_url = provider_instance.gpt_image_t2i(
                    prompt=prompt,
                    model=model,
                    size=size,
                    aspect_ratio=aspect_ratio.strip() if aspect_ratio.strip() else None,
                    quality=quality,
                    n=n,
                    pbar=pbar,
                )

                if image_url:
                    return (result_image, response, image_url)
                last_error = response

            return (create_blank_image(), last_error, "")

        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[APIcaller] {error_msg}")
            return (create_blank_image(), error_msg, "")


class GPTImageImg2Img:
    """
    GPT Image 图生图节点
    支持多图输入 + 蒙版 (Lingke mask 模式)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Edit the image..."}),
                "custom_provider": ("CUSTOM_PROVIDER",),
                "model": ("STRING", {"default": "gpt-image-1.5", "multiline": False, "placeholder": "如 gpt-image-1.5"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "mask": ("IMAGE",),
                "size": (["1024x1024", "1024x1536", "1536x1024", "auto"], {"default": "1024x1024"}),
                "aspect_ratio": ("STRING", {"default": "", "placeholder": "Kie用，如 1:1, 3:2（留空使用size）"}),
                "quality": (["medium", "high", "low", "auto"], {"default": "medium"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
                "use_kie_upload": ("BOOLEAN", {"default": False}),
                "kie_api_key": ("STRING", {"default": "", "placeholder": "Kie API Key，用于上传取URL"}),
                "error_retry": ("BOOLEAN", {"default": False}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate"
    CATEGORY = "APIcaller/GPTImage"

    def generate(
        self,
        prompt: str,
        custom_provider: dict,
        model: str = "gpt-image-1.5",
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        mask=None,
        size: str = "1024x1024",
        aspect_ratio: str = "",
        quality: str = "medium",
        n: int = 1,
        use_kie_upload: bool = False,
        kie_api_key: str = "",
        error_retry: bool = False,
        max_retries: int = 3,
    ) -> Tuple[torch.Tensor, str, str]:
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)

        try:
            if not custom_provider.get("api_key") or not custom_provider.get("base_url"):
                return (create_blank_image(), "请在 Custom Provider 节点中设置 API Key 和 Base URL", "")

            provider_instance = create_provider_instance(custom_provider)
            provider_type = custom_provider.get("provider_type", "lingke")
            print(f"[APIcaller] GPT Image I2I 使用供应商: {custom_provider['base_url']}")

            # 收集图像
            raw_images = [image1, image2, image3, image4]
            images = [img for img in raw_images if img is not None]

            if not images:
                return (create_blank_image(), "请至少提供一张输入图像", "")

            # Kie 模式：先上传图像获取 URL
            image_urls: Optional[List[str]] = None
            if use_kie_upload or provider_type == "kie":
                if provider_type == "kie":
                    kie_provider = provider_instance
                else:
                    kie_provider = get_provider("kie")
                    if kie_api_key.strip():
                        kie_provider.api_key = kie_api_key.strip()

                urls: List[str] = []
                img_index = 0
                for img in images:
                    single = img[0:1]
                    img_url, upload_error = kie_provider._upload_image(single, img_index)
                    if upload_error:
                        return (create_blank_image(), f"Kie上传失败: {upload_error}", "")
                    if img_url:
                        urls.append(img_url)
                    img_index += 1
                image_urls = urls
                images = []  # 已上传，不再传 tensor

            attempts = max_retries if error_retry else 1
            last_error = ""
            for attempt in range(1, attempts + 1):
                if attempt > 1:
                    print(f"[APIcaller] 第 {attempt}/{attempts} 次重试...")
                    time.sleep(2)

                pbar = comfy.utils.ProgressBar(100)
                pbar.update_absolute(5)

                result_image, response, image_url = provider_instance.gpt_image_i2i(
                    prompt=prompt,
                    images=images,
                    mask=mask,
                    model=model,
                    size=size,
                    aspect_ratio=aspect_ratio.strip() if aspect_ratio.strip() else None,
                    quality=quality,
                    n=n,
                    image_urls=image_urls,
                    pbar=pbar,
                )

                if image_url:
                    return (result_image, response, image_url)
                last_error = response

            return (create_blank_image(), last_error, "")

        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[APIcaller] {error_msg}")
            return (create_blank_image(), error_msg, "")
