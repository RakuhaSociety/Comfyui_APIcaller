"""
Midjourney 节点 (Lingke 供应商)
- MJ Imagine: 文生图（可带垫图），自动将 2x2 网格裁切成 4 张
- MJ Action:  执行 U/V/Reroll/Upscale/Pan/Zoom 等按钮动作
"""
import json
from io import BytesIO
from typing import List, Optional, Tuple

import comfy.utils
import torch
from PIL import Image

from ..config import create_provider_instance
from ..utils import (
    base64_to_pil,
    create_blank_image,
    download_image,
    image_to_base64,
    pil2tensor,
)


def _split_grid_to_batch(pil: Image.Image) -> torch.Tensor:
    """将 MJ 2x2 网格图裁切为 4 张并转 IMAGE batch。"""
    w, h = pil.size
    hw, hh = w // 2, h // 2
    tiles: List[Image.Image] = [
        pil.crop((0, 0, hw, hh)),
        pil.crop((hw, 0, w, hh)),
        pil.crop((0, hh, hw, h)),
        pil.crop((hw, hh, w, h)),
    ]
    return pil2tensor(tiles)


def _is_grid_result(buttons: list) -> bool:
    """根据返回的 buttons 判断是否为 4 图网格（Imagine/V/Reroll 结果含 U1-U4）。"""
    for btn in buttons or []:
        label = (btn.get("label") or "").upper()
        if label in ("U1", "U2", "U3", "U4"):
            return True
    return False


def _result_to_images(task_data: dict) -> torch.Tensor:
    """根据任务结果下载图片，必要时裁切为 4 张 batch。"""
    image_url = task_data.get("imageUrl") or ""
    if not image_url:
        return create_blank_image()

    img_bytes, err = download_image(image_url)
    if err or not img_bytes:
        print(f"[MJ] 下载图像失败: {err}")
        return create_blank_image()

    try:
        pil = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print(f"[MJ] 解析图像失败: {e}")
        return create_blank_image()

    buttons = task_data.get("buttons") or []
    if _is_grid_result(buttons):
        return _split_grid_to_batch(pil)
    return pil2tensor(pil)


def _ensure_lingke(custom_provider: dict) -> Tuple[Optional[object], Optional[str]]:
    if not custom_provider.get("api_key") or not custom_provider.get("base_url"):
        return None, "请在 Custom Provider 节点中设置 API Key 和 Base URL"
    if custom_provider.get("provider_type", "lingke") != "lingke":
        return None, "Midjourney 节点目前仅支持 Lingke 供应商"
    return create_provider_instance(custom_provider), None


def _images_to_base64_array(images: list) -> List[str]:
    out: List[str] = []
    for img in images:
        if img is None:
            continue
        b64 = image_to_base64(img[0:1] if img.dim() == 4 else img)
        if b64:
            out.append(f"data:image/png;base64,{b64}")
    return out


class MJImagineNode:
    """Midjourney - Imagine 文生图（含 2x2 自动拆分）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a cute cat"}),
                "custom_provider": ("CUSTOM_PROVIDER",),
            },
            "optional": {
                "bot_type": (["MID_JOURNEY", "NIJI_JOURNEY"], {"default": "MID_JOURNEY"}),
                "version": (
                    ["auto", "8.1", "8", "7", "6.1", "6", "5.2", "5.1", "5", "niji 7", "niji 6", "niji 5"],
                    {"default": "auto"},
                ),
                "aspect_ratio": (
                    ["auto", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"],
                    {"default": "auto"},
                ),
                "quality": (["auto", "0.25", "0.5", "1", "2", "4"], {"default": "auto"}),
                "stylize": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "[范围 0-1000，-1=不指定]  --s：MJ 默认美学风格强度，越高越艺术化/偏离 prompt 字面（默认 100）。与垫图无关"}),
                "chaos": ("INT", {"default": -1, "min": -1, "max": 100, "tooltip": "[范围 0-100，-1=不指定]  --c：4 张图之间的差异度，越高越多样"}),
                "weird": ("INT", {"default": -1, "min": -1, "max": 3000, "tooltip": "[范围 0-3000，-1=不指定]  --weird：越高画风越怪异/小众"}),
                "image_weight": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 3.0, "step": 0.1, "tooltip": "[范围 0-3，-1=不指定]  --iw：垫图权重，越高越像垫图"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295, "tooltip": "[范围 0-4294967295，-1=不指定]  --seed：随机种子"}),
                "tile": ("BOOLEAN", {"default": False, "tooltip": "添加 --tile 无缝平铺"}),
                "raw_mode": ("BOOLEAN", {"default": False, "tooltip": "添加 --raw"}),
                "no_text": ("STRING", {"default": "", "multiline": False, "placeholder": "排除内容，对应 --no"}),
                "extra_params": ("STRING", {"default": "", "multiline": False, "placeholder": "其他原始参数，例如 --weird 100 --sref xxx"}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "state": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "task_id", "buttons_json", "image_url", "final_prompt", "response")
    FUNCTION = "run"
    CATEGORY = "APIcaller/Midjourney"

    @staticmethod
    def _build_prompt(
        prompt: str,
        bot_type: str,
        version: str,
        aspect_ratio: str,
        quality: str,
        stylize: int,
        chaos: int,
        weird: int,
        image_weight: float,
        seed: int,
        tile: bool,
        raw_mode: bool,
        no_text: str,
        extra_params: str,
    ) -> str:
        """将选项拼接为 MJ prompt 后缀参数。如 prompt 中已存在对应参数则跳过。"""
        text = (prompt or "").strip()
        lower = text.lower()

        def has_flag(*flags: str) -> bool:
            return any(f in lower for f in flags)

        parts: list = []
        if aspect_ratio != "auto" and not has_flag(" --ar ", " --aspect "):
            parts.append(f"--ar {aspect_ratio}")
        if version != "auto" and not has_flag(" --v ", " --version ", " --niji"):
            v = version.strip().lower()
            if v.startswith("niji"):
                # niji 6 / niji 5
                niji_ver = v.split()[-1] if " " in v else ""
                parts.append(f"--niji {niji_ver}".strip())
            else:
                parts.append(f"--v {v}")
        if quality != "auto" and not has_flag(" --q ", " --quality "):
            parts.append(f"--q {quality}")
        if stylize >= 0 and not has_flag(" --s ", " --stylize "):
            parts.append(f"--s {stylize}")
        if chaos >= 0 and not has_flag(" --c ", " --chaos "):
            parts.append(f"--c {chaos}")
        if weird >= 0 and not has_flag(" --w ", " --weird "):
            parts.append(f"--weird {weird}")
        if image_weight >= 0 and not has_flag(" --iw "):
            parts.append(f"--iw {image_weight}")
        if seed >= 0 and not has_flag(" --seed "):
            parts.append(f"--seed {seed}")
        if tile and not has_flag(" --tile"):
            parts.append("--tile")
        if raw_mode and not has_flag(" --raw"):
            parts.append("--raw")
        if no_text.strip() and not has_flag(" --no "):
            parts.append(f"--no {no_text.strip()}")
        if extra_params.strip():
            parts.append(extra_params.strip())

        if parts:
            text = f"{text} {' '.join(parts)}".strip()
        return text

    def run(
        self,
        prompt: str,
        custom_provider: dict,
        bot_type: str = "MID_JOURNEY",
        version: str = "auto",
        aspect_ratio: str = "auto",
        quality: str = "auto",
        stylize: int = -1,
        chaos: int = -1,
        weird: int = -1,
        image_weight: float = -1.0,
        seed: int = -1,
        tile: bool = False,
        raw_mode: bool = False,
        no_text: str = "",
        extra_params: str = "",
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        state: str = "",
    ):
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)

        provider, err = _ensure_lingke(custom_provider)
        if err:
            return (create_blank_image(), "", "[]", "", "", json.dumps({"error": err}, ensure_ascii=False))

        base64_array = _images_to_base64_array([image1, image2, image3, image4])

        final_prompt = self._build_prompt(
            prompt, bot_type, version, aspect_ratio, quality,
            stylize, chaos, weird, image_weight, seed, tile, raw_mode, no_text, extra_params,
        )
        print(f"[MJ Imagine] final prompt: {final_prompt}")

        try:
            task_data, error = provider.mj_imagine(
                prompt=final_prompt,
                base64_array=base64_array,
                bot_type=bot_type,
                state=state,
                pbar=pbar,
            )
            if error or not task_data:
                return (create_blank_image(), "", "[]", "", final_prompt, json.dumps({"error": error or "未知错误"}, ensure_ascii=False))

            images_tensor = _result_to_images(task_data)
            task_id = str(task_data.get("id") or "")
            image_url = str(task_data.get("imageUrl") or "")
            buttons = task_data.get("buttons") or []
            buttons_json = json.dumps(buttons, ensure_ascii=False)
            response = json.dumps(
                {
                    "id": task_id,
                    "status": task_data.get("status"),
                    "prompt": task_data.get("prompt"),
                    "finalPrompt": (task_data.get("properties") or {}).get("finalPrompt"),
                    "imageUrl": image_url,
                    "buttons": buttons,
                },
                ensure_ascii=False,
                indent=2,
            )
            pbar.update_absolute(100)
            return (images_tensor, task_id, buttons_json, image_url, final_prompt, response)
        except Exception as e:
            err_msg = f"处理错误: {str(e)}"
            print(f"[MJ Imagine] {err_msg}")
            return (create_blank_image(), "", "[]", "", final_prompt, json.dumps({"error": err_msg}, ensure_ascii=False))


class MJActionNode:
    """Midjourney - Action（U/V/Reroll/Upscale/Pan/Zoom 等按钮动作）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": "", "multiline": False}),
                "custom_id": ("STRING", {"default": "", "multiline": False, "placeholder": "粘贴 buttons 中的 customId"}),
                "custom_provider": ("CUSTOM_PROVIDER",),
            },
            "optional": {
                "buttons_json": ("STRING", {"default": "[]", "multiline": True, "placeholder": "可选: 上游 MJ Imagine 输出的 buttons_json"}),
                "button_label": ("STRING", {"default": "", "multiline": False, "placeholder": "可选: 直接填 U1/U2/V1/Upscale 等 label，自动从 buttons_json 查找"}),
                "choose_same_channel": ("BOOLEAN", {"default": True}),
                "state": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "task_id", "buttons_json", "image_url", "response")
    FUNCTION = "run"
    CATEGORY = "APIcaller/Midjourney"

    @staticmethod
    def _resolve_custom_id(custom_id: str, buttons_json: str, button_label: str) -> str:
        if custom_id.strip():
            return custom_id.strip()
        if not button_label.strip():
            return ""
        try:
            buttons = json.loads(buttons_json or "[]")
        except Exception:
            return ""
        target = button_label.strip().upper()
        for btn in buttons:
            label = (btn.get("label") or "").upper()
            emoji = btn.get("emoji") or ""
            if label == target or (target == "REROLL" and emoji == "🔄"):
                return btn.get("customId") or ""
        return ""

    def run(
        self,
        task_id: str,
        custom_id: str,
        custom_provider: dict,
        buttons_json: str = "[]",
        button_label: str = "",
        choose_same_channel: bool = True,
        state: str = "",
    ):
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)

        provider, err = _ensure_lingke(custom_provider)
        if err:
            return (create_blank_image(), "", "[]", "", json.dumps({"error": err}, ensure_ascii=False))

        resolved_custom_id = self._resolve_custom_id(custom_id, buttons_json, button_label)
        if not resolved_custom_id:
            return (
                create_blank_image(),
                "",
                "[]",
                "",
                json.dumps({"error": "未能确定 customId，请直接填写 custom_id 或提供 buttons_json + button_label"}, ensure_ascii=False),
            )

        try:
            task_data, error = provider.mj_action(
                custom_id=resolved_custom_id,
                task_id=task_id,
                choose_same_channel=choose_same_channel,
                state=state,
                pbar=pbar,
            )
            if error or not task_data:
                return (create_blank_image(), "", "[]", "", json.dumps({"error": error or "未知错误"}, ensure_ascii=False))

            images_tensor = _result_to_images(task_data)
            new_task_id = str(task_data.get("id") or "")
            image_url = str(task_data.get("imageUrl") or "")
            buttons = task_data.get("buttons") or []
            response = json.dumps(
                {
                    "id": new_task_id,
                    "action": task_data.get("action"),
                    "status": task_data.get("status"),
                    "imageUrl": image_url,
                    "buttons": buttons,
                },
                ensure_ascii=False,
                indent=2,
            )
            pbar.update_absolute(100)
            return (images_tensor, new_task_id, json.dumps(buttons, ensure_ascii=False), image_url, response)
        except Exception as e:
            err_msg = f"处理错误: {str(e)}"
            print(f"[MJ Action] {err_msg}")
            return (create_blank_image(), "", "[]", "", json.dumps({"error": err_msg}, ensure_ascii=False))
