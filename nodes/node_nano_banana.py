"""
Nano Banana 节点
支持多供应商的图像生成和编辑
"""
import comfy.utils
import time
from typing import Tuple, Optional
import torch

from ..config import get_provider_config, PROVIDER_LIST, PROVIDER_DISPLAY_NAMES, PROVIDER_BASE_URLS
from ..providers import get_provider
from ..utils import create_blank_image


# 供应商对应的模型列表
PROVIDER_MODELS = {
    "wavespeed": [
        "nano-banana-pro/edit",
        "nano-banana-pro/edit-ultra",
    ],
    "lingke": [
        "nano-banana-2",
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash-image",
        "nano-banana",
        "nano-banana-hd",
        "gemini-2.5-flash-image-preview",
    ],
}

# 合并所有模型为一个列表（用于UI显示）
ALL_MODELS = []
for provider, models in PROVIDER_MODELS.items():
    for model in models:
        ALL_MODELS.append(f"[{provider}] {model}")


class NanoBananaEdit:
    """
    Nano Banana 图像编辑节点
    支持多API供应商
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Edit this image..."}),
                "provider": (PROVIDER_LIST, {"default": "lingke"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "model": ("STRING", {"default": "auto", "placeholder": "留空使用默认模型"}),
                "aspect_ratio": (["auto", "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {"default": "auto"}),
                "resolution": (["1k", "2k", "4k"], {"default": "1k"}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "api_key": ("STRING", {"default": "", "placeholder": "留空使用已保存的密钥"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "max_tokens": ("INT", {"default": 32768, "min": 1, "max": 65536}),
                "error_retry": ("BOOLEAN", {"default": False}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
                "batch_mode": ("BOOLEAN", {"default": False}),
                "custom_provider": ("CUSTOM_PROVIDER",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "process"
    CATEGORY = "APIcaller/NanoBanana"
    
    def process(
        self,
        prompt: str,
        provider: str,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        model: str = "auto",
        aspect_ratio: str = "auto",
        resolution: str = "1k",
        output_format: str = "png",
        api_key: str = "",
        temperature: float = 1.0,
        top_p: float = 0.95,
        seed: int = 0,
        max_tokens: int = 32768,
        error_retry: bool = False,
        max_retries: int = 3,
        batch_mode: bool = False,
        custom_provider: dict = None,
    ) -> Tuple[torch.Tensor, str, str]:
        """处理图像编辑请求"""
        
        # 创建进度条
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)
        
        try:
            # 自定义供应商优先
            if custom_provider and custom_provider.get("api_key") and custom_provider.get("base_url"):
                from ..providers.provider_lingke import LingkeProvider
                provider_instance = LingkeProvider()
                provider_instance.api_key = custom_provider["api_key"]
                provider_instance.base_url = custom_provider["base_url"]
                print(f"[APIcaller] 使用自定义供应商: {custom_provider['base_url']}")
            else:
                # 获取供应商配置
                provider_config = get_provider_config(provider)
                
                # 确定API密钥
                final_api_key = api_key.strip() if api_key.strip() else provider_config.get("api_key", "")
                
                if not final_api_key:
                    display_name = PROVIDER_DISPLAY_NAMES.get(provider, provider)
                    error_msg = f"请先设置 {display_name} 的API密钥"
                    return (create_blank_image(), error_msg, "")
                
                # 使用供应商的默认base_url，选择provider后自动使用对应的URL
                base_url = PROVIDER_BASE_URLS.get(provider, "")
                print(f"[APIcaller] 使用供应商: {provider}, Base URL: {base_url}")
                
                # 获取供应商实例
                provider_instance = get_provider(provider)
                provider_instance.configure(final_api_key, base_url)
            
            # 收集图像
            raw_inputs = [image1, image2, image3, image4]
            non_null = [(i, img) for i, img in enumerate(raw_inputs, 1) if img is not None]
            
            if not non_null:
                return (create_blank_image(), "请至少提供一张输入图像", "")
            
            # 确定模型
            if model == "auto" or not model.strip():
                available_models = provider_instance.get_available_models()
                model = available_models[0] if available_models else ""
            
            # 处理宽高比
            final_aspect_ratio = None if aspect_ratio == "auto" else aspect_ratio
            
            # === 批次模式 ===
            if batch_mode:
                return self._process_batch(
                    provider_instance, prompt, non_null, final_aspect_ratio,
                    resolution, output_format, model, temperature, top_p,
                    seed, max_tokens, error_retry, max_retries,
                )
            
            # === 普通模式 ===
            images = [img for _, img in non_null]
            
            # 调用供应商API（含重试逻辑）
            attempts = max_retries if error_retry else 1
            last_error = ""
            for attempt in range(1, attempts + 1):
                if attempt > 1:
                    print(f"[APIcaller] 第 {attempt}/{attempts} 次重试...")
                    time.sleep(2)
                
                pbar = comfy.utils.ProgressBar(100)
                pbar.update_absolute(5)
                result_image, response, image_url = provider_instance.nano_banana_edit(
                    prompt=prompt,
                    images=images,
                    aspect_ratio=final_aspect_ratio,
                    resolution=resolution,
                    output_format=output_format,
                    pbar=pbar,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    max_tokens=max_tokens,
                )
                
                # 检查是否成功（image_url非空表示成功）
                if image_url:
                    return (result_image, response, image_url)
                
                last_error = response
                if attempt < attempts:
                    print(f"[APIcaller] 调用失败: {response}，准备重试")
            
            return (result_image, last_error, image_url)
            
        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[APIcaller] {error_msg}")
            return (create_blank_image(), error_msg, "")

    def _process_batch(self, provider_instance, prompt, non_null, aspect_ratio,
                       resolution, output_format, model, temperature, top_p,
                       seed, max_tokens, error_retry, max_retries):
        """批次模式处理"""
        # 解析提示词行
        prompt_lines = [l for l in prompt.split('\n') if l.strip()]
        prompt_count = len(prompt_lines)
        
        # 获取每个image输入的批次数
        batch_sizes = {}
        for idx, img in non_null:
            if img.dim() == 4:
                batch_sizes[f"image{idx}"] = img.shape[0]
            else:
                batch_sizes[f"image{idx}"] = 1
        
        # 找出所有“多批次”的尺寸 (>1)
        multi_sizes = [s for s in batch_sizes.values() if s > 1]
        if prompt_count > 1:
            multi_sizes.append(prompt_count)
        
        if not multi_sizes:
            batch_count = 1
        else:
            batch_count = multi_sizes[0]
            if not all(s == batch_count for s in multi_sizes):
                detail = ", ".join(f"{k}={v}" for k, v in batch_sizes.items() if v > 1)
                if prompt_count > 1:
                    detail += f", prompt行数={prompt_count}"
                return (create_blank_image(), f"批次数不匹配: {detail}，所有多批次输入必须数量一致", "")
        
        # 广播提示词
        if prompt_count <= 1:
            prompts = [prompt_lines[0] if prompt_lines else prompt] * batch_count
        else:
            prompts = prompt_lines
        
        print(f"[APIcaller] 批次模式: 共 {batch_count} 次调用")
        
        result_images = []
        all_responses = []
        all_urls = []
        
        for i in range(batch_count):
            print(f"[APIcaller] === 批次 {i+1}/{batch_count} ===")
            
            # 为本次调用收集图像
            batch_images = []
            for idx, img in non_null:
                if img.dim() == 4 and img.shape[0] > 1:
                    batch_images.append(img[i:i+1])
                else:
                    batch_images.append(img)
            
            # 含重试的单次调用
            attempts = max_retries if error_retry else 1
            last_error = ""
            success = False
            for attempt in range(1, attempts + 1):
                if attempt > 1:
                    print(f"[APIcaller] 批次 {i+1} 第 {attempt}/{attempts} 次重试...")
                    time.sleep(2)
                
                pbar = comfy.utils.ProgressBar(100)
                pbar.update_absolute(5)
                result_image, response, image_url = provider_instance.nano_banana_edit(
                    prompt=prompts[i],
                    images=batch_images,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    output_format=output_format,
                    pbar=pbar,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    max_tokens=max_tokens,
                )
                
                if image_url:
                    success = True
                    break
                last_error = response
            
            if success:
                result_images.append(result_image)
                all_responses.append(response)
                all_urls.append(image_url)
            else:
                result_images.append(create_blank_image())
                all_responses.append(f"[批次{i+1}失败] {last_error}")
                all_urls.append("")
        
        combined_image = torch.cat(result_images, dim=0)
        combined_response = "\n---\n".join(all_responses)
        combined_urls = "\n".join(all_urls)
        return (combined_image, combined_response, combined_urls)


class NanoBananaText2Img:
    """
    Nano Banana 文生图节点
    支持多API供应商
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful sunset over the ocean"}),
                "provider": (PROVIDER_LIST, {"default": "lingke"}),
            },
            "optional": {
                "model": ("STRING", {"default": "auto", "placeholder": "留空使用默认模型"}),
                "aspect_ratio": (["auto", "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {"default": "1:1"}),
                "resolution": (["1k", "2k", "4k"], {"default": "1k"}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "api_key": ("STRING", {"default": "", "placeholder": "留空使用已保存的密钥"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "max_tokens": ("INT", {"default": 32768, "min": 1, "max": 65536}),
                "error_retry": ("BOOLEAN", {"default": False}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
                "batch_mode": ("BOOLEAN", {"default": False}),
                "custom_provider": ("CUSTOM_PROVIDER",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "process"
    CATEGORY = "APIcaller/NanoBanana"
    
    def process(
        self,
        prompt: str,
        provider: str,
        model: str = "auto",
        aspect_ratio: str = "1:1",
        resolution: str = "1k",
        output_format: str = "png",
        api_key: str = "",
        temperature: float = 1.0,
        top_p: float = 0.95,
        seed: int = 0,
        max_tokens: int = 32768,
        error_retry: bool = False,
        max_retries: int = 3,
        batch_mode: bool = False,
        custom_provider: dict = None,
    ) -> Tuple[torch.Tensor, str, str]:
        """处理文生图请求"""
        
        # 创建进度条
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)
        
        try:
            # 自定义供应商优先
            if custom_provider and custom_provider.get("api_key") and custom_provider.get("base_url"):
                from ..providers.provider_lingke import LingkeProvider
                provider_instance = LingkeProvider()
                provider_instance.api_key = custom_provider["api_key"]
                provider_instance.base_url = custom_provider["base_url"]
                print(f"[APIcaller] 使用自定义供应商: {custom_provider['base_url']}")
            else:
                # 获取供应商配置
                provider_config = get_provider_config(provider)
                
                # 确定API密钥
                final_api_key = api_key.strip() if api_key.strip() else provider_config.get("api_key", "")
                
                if not final_api_key:
                    display_name = PROVIDER_DISPLAY_NAMES.get(provider, provider)
                    error_msg = f"请先设置 {display_name} 的API密钥"
                    return (create_blank_image(), error_msg, "")
                
                # 使用供应商的默认base_url，选择provider后自动使用对应的URL
                base_url = PROVIDER_BASE_URLS.get(provider, "")
                print(f"[APIcaller] 使用供应商: {provider}, Base URL: {base_url}")
                
                # 获取供应商实例
                provider_instance = get_provider(provider)
                provider_instance.configure(final_api_key, base_url)
            
            # 确定模型
            if model == "auto" or not model.strip():
                available_models = provider_instance.get_available_models()
                # 对于文生图，优先选择text-to-image模型
                text2img_models = [m for m in available_models if "text-to-image" in m or "text2img" in m]
                model = text2img_models[0] if text2img_models else (available_models[0] if available_models else "")
            
            # 处理宽高比
            final_aspect_ratio = None if aspect_ratio == "auto" else aspect_ratio
            
            # === 批次模式 ===
            if batch_mode:
                prompt_lines = [l for l in prompt.split('\n') if l.strip()]
                if len(prompt_lines) <= 1:
                    return self._text2img_single_call(
                        provider_instance, prompt, final_aspect_ratio,
                        resolution, output_format, model, temperature,
                        top_p, seed, max_tokens, error_retry, max_retries,
                    )
                
                batch_count = len(prompt_lines)
                print(f"[APIcaller] 文生图批次模式: 共 {batch_count} 次调用")
                
                result_images = []
                all_responses = []
                all_urls = []
                
                for i in range(batch_count):
                    print(f"[APIcaller] === 批次 {i+1}/{batch_count} ===")
                    img, resp, url = self._text2img_single_call(
                        provider_instance, prompt_lines[i], final_aspect_ratio,
                        resolution, output_format, model, temperature,
                        top_p, seed, max_tokens, error_retry, max_retries,
                    )
                    result_images.append(img)
                    all_responses.append(resp)
                    all_urls.append(url)
                
                combined_image = torch.cat(result_images, dim=0)
                combined_response = "\n---\n".join(all_responses)
                combined_urls = "\n".join(all_urls)
                return (combined_image, combined_response, combined_urls)
            
            # === 普通模式 ===
            return self._text2img_single_call(
                provider_instance, prompt, final_aspect_ratio,
                resolution, output_format, model, temperature,
                top_p, seed, max_tokens, error_retry, max_retries,
            )
            
        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            print(f"[APIcaller] {error_msg}")
            return (create_blank_image(), error_msg, "")

    def _text2img_single_call(self, provider_instance, prompt, aspect_ratio,
                              resolution, output_format, model, temperature,
                              top_p, seed, max_tokens, error_retry, max_retries):
        """单次文生图调用（含重试）"""
        attempts = max_retries if error_retry else 1
        last_error = ""
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                print(f"[APIcaller] 第 {attempt}/{attempts} 次重试...")
                time.sleep(2)
            
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(5)
            result_image, response, image_url = provider_instance.nano_banana_text2img(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                output_format=output_format,
                pbar=pbar,
                model=model,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_tokens=max_tokens,
            )
            
            if image_url:
                return (result_image, response, image_url)
            last_error = response
        
        return (create_blank_image(), last_error, "")
