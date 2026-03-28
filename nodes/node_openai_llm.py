"""
OpenAI LLM 通用节点
使用标准 OpenAI Chat Completions 格式调用各种 LLM
通过 Custom Provider 设置 api_key 和 base_url
"""
import json
import requests
from typing import Tuple


class OpenAILLM:
    """
    通用 OpenAI 格式 LLM 节点
    支持任何兼容 OpenAI API 的服务
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_provider": ("CUSTOM_PROVIDER",),
                "model": ("STRING", {"default": "gpt-4o", "placeholder": "模型名称，如 gpt-4o, claude-3-opus 等"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 131072}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "raw_json")
    FUNCTION = "chat"
    CATEGORY = "APIcaller/LLM"

    def chat(
        self,
        custom_provider: dict,
        model: str,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        seed: int = 0,
        image=None,
    ) -> Tuple[str, str]:

        api_key = custom_provider.get("api_key", "")
        base_url = custom_provider.get("base_url", "")

        if not api_key:
            return ("错误: 未设置 API Key，请连接 Custom Provider 节点", "")
        if not base_url:
            return ("错误: 未设置 Base URL，请连接 Custom Provider 节点", "")

        # 构建 messages
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        # 用户消息（支持图像）
        if image is not None:
            user_content = self._build_vision_content(prompt, image)
        else:
            user_content = prompt

        messages.append({"role": "user", "content": user_content})

        # 构建请求
        url = f"{base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if seed > 0:
            payload["seed"] = seed

        print(f"[APIcaller] OpenAI LLM 调用: model={model}, url={url}")

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()

            # 提取回复文本
            content = ""
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")

            raw_json = json.dumps(data, ensure_ascii=False, indent=2)

            if not content:
                return (f"API 返回为空。原始响应: {raw_json[:500]}", raw_json)

            return (content, raw_json)

        except requests.exceptions.HTTPError as e:
            error_body = ""
            try:
                error_body = e.response.text[:500]
            except Exception:
                pass
            error_msg = f"HTTP 错误 {e.response.status_code}: {error_body}"
            print(f"[APIcaller] {error_msg}")
            return (error_msg, "")

        except requests.exceptions.Timeout:
            return ("请求超时 (300s)", "")

        except Exception as e:
            error_msg = f"请求错误: {str(e)}"
            print(f"[APIcaller] {error_msg}")
            return (error_msg, "")

    def _build_vision_content(self, prompt: str, image) -> list:
        """构建带图像的多模态 content，支持批次中多张图像"""
        import torch
        import numpy as np
        from io import BytesIO
        from PIL import Image
        import base64

        content = []
        if prompt.strip():
            content.append({"type": "text", "text": prompt.strip()})

        # 收集所有图像
        images_to_encode = []
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                # 批次张量：逐张拆开
                for i in range(image.shape[0]):
                    img_np = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    images_to_encode.append(Image.fromarray(img_np))
            else:
                img_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                images_to_encode.append(Image.fromarray(img_np))
        else:
            images_to_encode.append(image)

        print(f"[APIcaller] VLM 传入 {len(images_to_encode)} 张图像")

        for pil_img in images_to_encode:
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

        return content
