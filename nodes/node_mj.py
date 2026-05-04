"""
Midjourney 节点 (Lingke 供应商)
- MJ Imagine: 文生图（可带垫图），自动将 2x2 网格裁切成 4 张
- MJ Action:  执行 U/V/Reroll/Upscale/Pan/Zoom 等按钮动作
"""
import json
import re
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
    tensor2pil,
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
        if hasattr(img, "dim") and img.dim() == 4:
            for i in range(img.shape[0]):
                b64 = image_to_base64(img[i:i + 1])
                if b64:
                    out.append(f"data:image/png;base64,{b64}")
        else:
            b64 = image_to_base64(img)
            if b64:
                out.append(f"data:image/png;base64,{b64}")
    return out


def _image_to_data_uri(image) -> str:
    if image is None:
        return ""
    b64 = image_to_base64(image[0:1] if image.dim() == 4 else image)
    if not b64:
        return ""
    return f"data:image/png;base64,{b64}"


def _mask_to_base64(mask) -> str:
    """将 ComfyUI MASK/IMAGE 输入转为 data:image/png;base64。"""
    if mask is None:
        return ""
    try:
        tensor = mask[0:1] if hasattr(mask, "dim") and mask.dim() >= 3 else mask
        pil = tensor2pil(tensor)[0]
        if pil.mode != "L":
            pil = pil.convert("L")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        import base64
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[MJ Modal] mask 转 base64 失败: {e}")
        return ""


def _extract_descriptions(task_data: dict) -> List[str]:
    """兼容不同 MJ proxy 返回结构，尽量提取 Describe 的 4 条 prompt。"""
    if not task_data:
        return []

    props = task_data.get("properties") or {}
    candidates = [
        props.get("descriptions"),
        props.get("description"),
        props.get("finalPrompt"),
        task_data.get("descriptions"),
        task_data.get("description"),
        task_data.get("prompt"),
        task_data.get("content"),
    ]

    for value in candidates:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()][:4]
        if isinstance(value, str) and value.strip():
            text = value.strip()
            lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n") if line.strip()]
            if len(lines) >= 2:
                cleaned = []
                for line in lines:
                    # 常见格式: 1️⃣ prompt / 1. prompt / 1) prompt
                    line = line.lstrip("1234567890. )、:-️⃣ ").strip()
                    if line:
                        cleaned.append(line)
                if cleaned:
                    return cleaned[:4]
            return [text]
    return []


def _strip_mj_aspect_ratio(text: str) -> str:
    """Describe 常会在末尾附带 --ar；去掉它，避免接 Imagine 时重复比例参数。"""
    return re.sub(r"\s+--(?:ar|aspect)\s+\d+\s*:\s*\d+\s*$", "", text or "", flags=re.IGNORECASE).strip()


def _strip_leading_image_urls(text: str) -> str:
    return re.sub(r"^(?:https?://\S+\s+)+", "", text or "").strip()


def _extract_leading_image_urls(text: str) -> List[str]:
    match = re.match(r"^((?:https?://\S+\s+)+)", text or "")
    if not match:
        return []
    return [part.strip() for part in match.group(1).split() if part.strip()]


def _download_urls_to_batch(urls: List[str]) -> torch.Tensor:
    images: List[Image.Image] = []
    for url in urls:
        img_bytes, err = download_image(url)
        if err or not img_bytes:
            print(f"[MJ] 下载参考图失败: {url} {err}")
            continue
        try:
            images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))
        except Exception as e:
            print(f"[MJ] 解析参考图失败: {url} {e}")
    if not images:
        return create_blank_image()
    target_size = images[0].size
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    images = [img if img.size == target_size else img.resize(target_size, resample) for img in images]
    return pil2tensor(images)


def _strip_mj_params(text: str) -> str:
    """去掉 prompt 尾部 MJ 参数，保留纯提示词。"""
    text = _strip_leading_image_urls(text or "")
    if not text:
        return ""
    flag_with_value = (
        "ar", "aspect", "v", "version", "niji", "q", "quality", "s", "stylize",
        "c", "chaos", "seed", "iw", "weird", "w", "style", "sref", "cref", "cw",
    )
    flag_without_value = ("tile", "raw", "relax", "fast", "turbo", "public")
    pattern = r"\s+--(?:" + "|".join(flag_with_value) + r")\s+\S+|\s+--(?:" + "|".join(flag_without_value) + r")\b"
    previous = None
    while previous != text:
        previous = text
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text


def _first_present(data: dict, keys: List[str], default=""):
    for key in keys:
        if key in data and data.get(key) not in (None, ""):
            return data.get(key)
    return default


def _extract_actual_seed(seed_data: dict, task_data: dict) -> str:
    """从 image-seed 或任务结构里提取服务端返回的真实 seed。"""
    def valid_seed(value) -> str:
        if value in (None, ""):
            return ""
        text = str(value).strip()
        return text if _to_int(text) >= 0 else ""

    for data in (seed_data or {}, task_data or {}):
        props = data.get("properties") or {}
        for key in ("seed", "imageSeed", "image_seed"):
            seed = valid_seed(data.get(key))
            if seed:
                return seed
            seed = valid_seed(props.get(key))
            if seed:
                return seed
        result = data.get("result")
        seed = valid_seed(result) if isinstance(result, (str, int)) else ""
        if seed:
            return seed
        if isinstance(result, dict):
            for key in ("seed", "imageSeed", "image_seed"):
                seed = valid_seed(result.get(key))
                if seed:
                    return seed
    return ""


def _extract_mj_params(task_data: dict) -> dict:
    props = task_data.get("properties") or {}
    prompt = _first_present(task_data, ["prompt", "promptEn", "description"], "")
    final_prompt = _first_present(props, ["finalPrompt", "final_prompt"], "") or _first_present(task_data, ["finalPrompt", "final_prompt"], "")
    source = final_prompt or prompt

    def find(pattern: str) -> str:
        match = re.search(pattern, source or "", flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    niji = find(r"--niji\s+([^\s]+)")
    version = f"niji {niji}" if niji else (_first_present(props, ["version", "v"], "") or find(r"--(?:v|version)\s+([^\s]+)"))
    reference_urls = _extract_leading_image_urls(final_prompt or "") or _extract_leading_image_urls(task_data.get("promptEn") or "")
    bot_type = "NIJI_JOURNEY" if str(version).lower().startswith("niji") else "MID_JOURNEY"
    return {
        "prompt": _strip_mj_params(prompt),
        "raw_prompt": prompt,
        "raw_final_prompt": final_prompt,
        "reference_urls": reference_urls,
        "bot_type": bot_type,
        "aspect_ratio": _first_present(props, ["aspectRatio", "aspect_ratio", "ar"], "") or find(r"--(?:ar|aspect)\s+([^\s]+)"),
        "version": version,
        "quality": _first_present(props, ["quality", "q"], "") or find(r"--(?:q|quality)\s+([^\s]+)"),
        "stylize": _first_present(props, ["stylize", "s"], "") or find(r"--(?:s|stylize)\s+([^\s]+)"),
        "chaos": _first_present(props, ["chaos", "c"], "") or find(r"--(?:c|chaos)\s+([^\s]+)"),
        "weird": _first_present(props, ["weird", "w"], "") or find(r"--(?:weird|w)\s+([^\s]+)"),
        "image_weight": _first_present(props, ["imageWeight", "image_weight", "iw"], "") or find(r"--iw\s+([^\s]+)"),
        "seed": _first_present(props, ["seed"], "") or find(r"--seed\s+([^\s]+)"),
    }


def _to_int(value, default=-1) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value)))
    except Exception:
        return default


def _to_float(value, default=-1.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(str(value))
    except Exception:
        return default


def _normalize_bot_type(value: str, default="MID_JOURNEY") -> str:
    text = str(value or "").strip().upper()
    if "NIJI" in text:
        return "NIJI_JOURNEY"
    if "MID" in text or "MJ" == text:
        return "MID_JOURNEY"
    return default


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
                "params_json": ("STRING", {"default": "", "multiline": True, "placeholder": "可选: 接 MJ Task Info 的 params_json，一次继承 prompt/参数"}),
                "bot_type": ("STRING", {"default": "MID_JOURNEY", "widgetType": "COMBO", "options": ["MID_JOURNEY", "NIJI_JOURNEY"]}),
                "version": ("STRING", {"default": "auto", "widgetType": "COMBO", "options": ["auto", "8.1", "8", "7", "6.1", "6", "5.2", "5.1", "5", "niji 7", "niji 6", "niji 5"]}),
                "aspect_ratio": ("STRING", {"default": "auto", "widgetType": "COMBO", "options": ["auto", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"]}),
                "quality": ("STRING", {"default": "auto", "widgetType": "COMBO", "options": ["auto", "0.25", "0.5", "1", "2", "4"]}),
                "stylize": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "[范围 0-1000，-1=不指定]  --s：MJ 默认美学风格强度，越高越艺术化/偏离 prompt 字面（默认 100）。与垫图无关"}),
                "chaos": ("INT", {"default": -1, "min": -1, "max": 100, "tooltip": "[范围 0-100，-1=不指定]  --c：4 张图之间的差异度，越高越多样"}),
                "weird": ("INT", {"default": -1, "min": -1, "max": 3000, "tooltip": "[范围 0-3000，-1=不指定]  --weird：越高画风越怪异/小众"}),
                "image_weight": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 3.0, "step": 0.1, "tooltip": "[范围 0-3，-1=不指定]  --iw：垫图权重，越高越像垫图"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295, "tooltip": "[范围 0-4294967295，-1=不指定]  --seed：随机种子"}),
                "tile": ("BOOLEAN", {"default": False, "tooltip": "添加 --tile 无缝平铺"}),
                "raw_mode": ("BOOLEAN", {"default": False, "tooltip": "添加 --raw"}),
                "no_text": ("STRING", {"default": "", "multiline": False, "placeholder": "排除内容，对应 --no"}),
                "extra_params": ("STRING", {"default": "", "multiline": False, "placeholder": "其他原始参数，例如 --weird 100 --sref xxx"}),
                "reference_images": ("IMAGE",),
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
        params_json: str = "",
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
        reference_images=None,
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

        if params_json.strip():
            try:
                params = json.loads(params_json)
                if isinstance(params, dict):
                    prompt = str(params.get("prompt") or prompt)
                    bot_type = _normalize_bot_type(params.get("bot_type"), bot_type)
                    version = str(params.get("version") or version)
                    aspect_ratio = str(params.get("aspect_ratio") or aspect_ratio)
                    quality = str(params.get("quality") or quality)
                    stylize = _to_int(params.get("stylize"), stylize)
                    chaos = _to_int(params.get("chaos"), chaos)
                    weird = _to_int(params.get("weird"), weird)
                    seed = _to_int(params.get("seed"), seed)
                    image_weight = _to_float(params.get("image_weight"), image_weight)
            except Exception as e:
                print(f"[MJ Imagine] params_json 解析失败，将忽略: {e}")

        bot_type = _normalize_bot_type(bot_type, "MID_JOURNEY")

        base64_array = _images_to_base64_array([reference_images, image1, image2, image3, image4])

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


class MJDescribeNode:
    """Midjourney - Describe 图生提示词"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "custom_provider": ("CUSTOM_PROVIDER",),
            },
            "optional": {
                "state": ("STRING", {"default": "", "multiline": False, "tooltip": "可选: 自定义跟踪标记"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("description_1", "description_2", "description_3", "description_4", "descriptions_json", "task_id", "response")
    FUNCTION = "run"
    CATEGORY = "APIcaller/Midjourney"

    def run(self, image, custom_provider: dict, state: str = ""):
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)

        provider, err = _ensure_lingke(custom_provider)
        if err:
            response = json.dumps({"error": err}, ensure_ascii=False)
            return ("", "", "", "", "[]", "", response)

        image_base64 = _image_to_data_uri(image)
        if not image_base64:
            response = json.dumps({"error": "image 转 base64 失败"}, ensure_ascii=False)
            return ("", "", "", "", "[]", "", response)

        try:
            task_data, error = provider.mj_describe(
                image_base64=image_base64,
                state=state,
                pbar=pbar,
            )
            if error or not task_data:
                response = json.dumps({"error": error or "未知错误"}, ensure_ascii=False)
                return ("", "", "", "", "[]", "", response)

            descriptions = _extract_descriptions(task_data)
            descriptions = [_strip_mj_aspect_ratio(desc) for desc in descriptions]
            while len(descriptions) < 4:
                descriptions.append("")

            task_id = str(task_data.get("id") or "")
            response = json.dumps(
                {
                    "id": task_id,
                    "status": task_data.get("status"),
                    "descriptions": descriptions[:4],
                    "raw": task_data,
                },
                ensure_ascii=False,
                indent=2,
            )
            pbar.update_absolute(100)
            return (
                descriptions[0],
                descriptions[1],
                descriptions[2],
                descriptions[3],
                json.dumps(descriptions[:4], ensure_ascii=False),
                task_id,
                response,
            )
        except Exception as e:
            err_msg = f"处理错误: {str(e)}"
            print(f"[MJ Describe] {err_msg}")
            response = json.dumps({"error": err_msg}, ensure_ascii=False)
            return ("", "", "", "", "[]", "", response)


class MJBlendNode:
    """Midjourney - Blend 多图融合"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "custom_provider": ("CUSTOM_PROVIDER",),
            },
            "optional": {
                "dimensions": (
                    ["SQUARE", "PORTRAIT", "LANDSCAPE"],
                    {"default": "SQUARE", "tooltip": "输出构图比例：SQUARE=方图，PORTRAIT=竖图，LANDSCAPE=横图"},
                ),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "state": ("STRING", {"default": "", "multiline": False, "tooltip": "可选: 自定义跟踪标记"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "task_id", "buttons_json", "image_url", "response")
    FUNCTION = "run"
    CATEGORY = "APIcaller/Midjourney"

    def run(
        self,
        image1,
        image2,
        custom_provider: dict,
        dimensions: str = "SQUARE",
        image3=None,
        image4=None,
        image5=None,
        state: str = "",
    ):
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)

        provider, err = _ensure_lingke(custom_provider)
        if err:
            return (create_blank_image(), "", "[]", "", json.dumps({"error": err}, ensure_ascii=False))

        base64_array = _images_to_base64_array([image1, image2, image3, image4, image5])
        if len(base64_array) < 2:
            return (create_blank_image(), "", "[]", "", json.dumps({"error": "Blend 至少需要 2 张图片"}, ensure_ascii=False))

        try:
            task_data, error = provider.mj_blend(
                base64_array=base64_array,
                dimensions=dimensions,
                state=state,
                pbar=pbar,
            )
            if error or not task_data:
                return (create_blank_image(), "", "[]", "", json.dumps({"error": error or "未知错误"}, ensure_ascii=False))

            images_tensor = _result_to_images(task_data)
            task_id = str(task_data.get("id") or "")
            image_url = str(task_data.get("imageUrl") or "")
            buttons = task_data.get("buttons") or []
            response = json.dumps(
                {
                    "id": task_id,
                    "status": task_data.get("status"),
                    "imageUrl": image_url,
                    "buttons": buttons,
                    "dimensions": dimensions,
                    "imageCount": len(base64_array),
                },
                ensure_ascii=False,
                indent=2,
            )
            pbar.update_absolute(100)
            return (images_tensor, task_id, json.dumps(buttons, ensure_ascii=False), image_url, response)
        except Exception as e:
            err_msg = f"处理错误: {str(e)}"
            print(f"[MJ Blend] {err_msg}")
            return (create_blank_image(), "", "[]", "", json.dumps({"error": err_msg}, ensure_ascii=False))


class MJTaskInfoNode:
    """Midjourney - Task Info 查询（任务信息 / 参数 / seed / 垫图）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": "", "multiline": False, "tooltip": "MJ 任务 ID。可填 Imagine / Blend / Action 输出的 task_id"}),
                "custom_provider": ("CUSTOM_PROVIDER",),
            },
        }

    RETURN_TYPES = (
        "IMAGE", "IMAGE",
        "STRING", "STRING", "STRING", "STRING", "STRING",
        "INT", "INT", "INT", "FLOAT", "INT",
        "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING"
    )
    RETURN_NAMES = (
        "image",
        "reference_images",
        "prompt",
        "bot_type",
        "version",
        "aspect_ratio",
        "quality",
        "stylize",
        "chaos",
        "weird",
        "image_weight",
        "seed",
        "raw_prompt",
        "raw_final_prompt",
        "status",
        "image_url",
        "params_json",
        "task_json",
        "response",
    )
    FUNCTION = "run"
    CATEGORY = "APIcaller/Midjourney"

    def run(self, task_id: str, custom_provider: dict):
        provider, err = _ensure_lingke(custom_provider)
        if err:
            response = json.dumps({"error": err}, ensure_ascii=False)
            return (create_blank_image(), create_blank_image(), "", "", "", "", "", -1, -1, -1, -1.0, -1, "", "", "", "", "{}", "{}", response)
        if not task_id.strip():
            response = json.dumps({"error": "task_id 不能为空"}, ensure_ascii=False)
            return (create_blank_image(), create_blank_image(), "", "", "", "", "", -1, -1, -1, -1.0, -1, "", "", "", "", "{}", "{}", response)

        task_data, task_error = provider.mj_fetch_task(task_id.strip())
        seed_data, seed_error = provider.mj_image_seed(task_id.strip())

        task_data = task_data or {}
        seed_data = seed_data or {}
        params = _extract_mj_params(task_data)
        submitted_seed = str(params.get("seed") or "")
        actual_seed = _extract_actual_seed(seed_data, task_data)
        seed = actual_seed
        seed_source = "actual" if actual_seed else ""
        if not seed and submitted_seed and _to_int(submitted_seed) >= 0:
            seed = submitted_seed
            seed_source = "submitted"
        if seed:
            params["seed"] = seed
        else:
            params["seed"] = ""
        params["submitted_seed"] = submitted_seed
        params["actual_seed"] = actual_seed
        params["seed_source"] = seed_source or ("submitted_random" if submitted_seed == "-1" else "unavailable")
        image_url = str(task_data.get("imageUrl") or "")
        image_tensor = _result_to_images(task_data) if image_url else create_blank_image()
        reference_images = _download_urls_to_batch(params.get("reference_urls") or [])
        params_json = json.dumps(params, ensure_ascii=False)

        response = json.dumps(
            {
                "id": task_id.strip(),
                "seed": seed,
                "actualSeed": actual_seed,
                "submittedSeed": submitted_seed,
                "seedSource": params["seed_source"],
                "taskError": task_error,
                "seedError": seed_error,
                "status": task_data.get("status") or "",
                "imageUrl": image_url,
                "prompt": params["prompt"],
                "botType": params["bot_type"],
                "rawPrompt": params["raw_prompt"],
                "rawFinalPrompt": params["raw_final_prompt"],
                "referenceUrls": params["reference_urls"],
                "aspectRatio": params["aspect_ratio"],
                "version": params["version"],
                "quality": params["quality"],
                "stylize": params["stylize"],
                "chaos": params["chaos"],
                "weird": params["weird"],
                "imageWeight": params["image_weight"],
                "seedRaw": seed_data,
                "taskRaw": task_data,
            },
            ensure_ascii=False,
            indent=2,
        )

        return (
            image_tensor,
            reference_images,
            str(params["prompt"] or ""),
            str(params["bot_type"] or ""),
            str(params["version"] or ""),
            str(params["aspect_ratio"] or ""),
            str(params["quality"] or ""),
            _to_int(params["stylize"]),
            _to_int(params["chaos"]),
            _to_int(params["weird"]),
            _to_float(params["image_weight"]),
            _to_int(params["seed"]),
            str(params["raw_prompt"] or ""),
            str(params["raw_final_prompt"] or ""),
            str(task_data.get("status") or ""),
            image_url,
            params_json,
            json.dumps(task_data, ensure_ascii=False),
            response,
        )


def _find_button_custom_id(buttons: list, candidates: List[str]) -> Tuple[str, str]:
    """在 buttons 中按候选词列表查找 customId，候选词不区分大小写，支持子串与 emoji。"""
    for cand in candidates:
        cand_low = cand.lower()
        for btn in buttons:
            label = (btn.get("label") or "").lower()
            emoji = btn.get("emoji") or ""
            if label == cand_low or (cand_low in label and label) or cand == emoji:
                return btn.get("customId") or "", f"匹配 '{cand}'"
    return "", f"未匹配到候选: {candidates}"


def _resolve_buttons(
    provider,
    task_id: str,
    buttons_json: str,
    auto_fetch_buttons: bool,
) -> Tuple[list, Optional[str]]:
    """获取 buttons：优先 buttons_json，否则按需 fetch。"""
    if buttons_json.strip():
        try:
            return json.loads(buttons_json) or [], None
        except Exception as e:
            return [], f"buttons_json 解析失败: {e}"
    if auto_fetch_buttons:
        print(f"[MJ Action] 自动 fetch 任务 {task_id} 获取 buttons")
        fetched, ferr = provider.mj_fetch_task(task_id)
        if ferr or not fetched:
            return [], f"自动获取 buttons 失败: {ferr or '空响应'}"
        return fetched.get("buttons") or [], None
    return [], "未获得 buttons 列表（buttons_json 为空且 auto_fetch_buttons=False）"


def _execute_mj_action(
    provider,
    task_id: str,
    custom_id: str,
    match_info: str,
    choose_same_channel: bool,
    state: str,
    pbar,
) -> tuple:
    """执行 MJ action 并组装返回元组。返回 (images, task_id, buttons_json, image_url, response)。"""
    print(f"[MJ Action] -> customId={custom_id} ({match_info})")
    try:
        task_data, error = provider.mj_action(
            custom_id=custom_id,
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
                "matchInfo": match_info,
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


class MJActionGridNode:
    """Midjourney - Action 一阶段（4 图网格操作：U/V/Reroll）

    适用对象：MJ Imagine 的输出，或 Vary/Pan/Zoom 等返回的新 4 图网格。
    输出：单图（U1-U4）或新 4 图网格（V1-V4 / Reroll）。
    要继续做 Upscale 2x/4x、Vary、Pan、Zoom 等精修，请把本节点输出 task_id
    送入「MJ Action Refine」节点。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": "", "multiline": False, "tooltip": "上一个 MJ Imagine 或返回 4 图网格的任务 task_id"}),
                "action": (
                    ["Upscale (U1-U4)", "Variation (V1-V4)", "Reroll"],
                    {"default": "Upscale (U1-U4)", "tooltip": "U=选第N张提取单图；V=基于第N张做变体；Reroll=同 prompt 重抽 4 张"},
                ),
                "index": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "[范围 1-4] 选择 U 或 V 的第几张；Reroll 时忽略"}),
                "custom_provider": ("CUSTOM_PROVIDER",),
            },
            "optional": {
                "auto_fetch_buttons": ("BOOLEAN", {"default": True, "tooltip": "若未提供 buttons_json，自动调 fetch 接口获取"}),
                "buttons_json": ("STRING", {"default": "", "multiline": True, "placeholder": "可选: 直接传上游 buttons_json 跳过 fetch"}),
                "choose_same_channel": ("BOOLEAN", {"default": True, "tooltip": "保持 True：尽量在同一 Discord 频道执行"}),
                "state": ("STRING", {"default": "", "multiline": False, "tooltip": "可选: 自定义跟踪标记"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "task_id", "buttons_json", "image_url", "response")
    FUNCTION = "run"
    CATEGORY = "APIcaller/Midjourney"

    def run(
        self,
        task_id: str,
        action: str,
        index: int,
        custom_provider: dict,
        auto_fetch_buttons: bool = True,
        buttons_json: str = "",
        choose_same_channel: bool = True,
        state: str = "",
    ):
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)

        provider, err = _ensure_lingke(custom_provider)
        if err:
            return (create_blank_image(), "", "[]", "", json.dumps({"error": err}, ensure_ascii=False))
        if not task_id.strip():
            return (create_blank_image(), "", "[]", "", json.dumps({"error": "task_id 不能为空"}, ensure_ascii=False))

        buttons, berr = _resolve_buttons(provider, task_id.strip(), buttons_json, auto_fetch_buttons)
        if berr:
            return (create_blank_image(), "", "[]", "", json.dumps({"error": berr}, ensure_ascii=False))
        pbar.update_absolute(15)

        # 候选词
        if action == "Upscale (U1-U4)":
            candidates = [f"U{index}"]
        elif action == "Variation (V1-V4)":
            candidates = [f"V{index}"]
        elif action == "Reroll":
            candidates = ["reroll", "🔄"]
        else:
            return (create_blank_image(), "", "[]", "", json.dumps({"error": f"未知 action: {action}"}, ensure_ascii=False))

        custom_id, match_info = _find_button_custom_id(buttons, candidates)
        if not custom_id:
            available = [b.get("label") or b.get("emoji") or "?" for b in buttons]
            hint = ""
            # 智能提示：如果只有精修按钮，说明用户接错了 task_id
            if any((b.get("label") or "").lower().startswith(("upscale (", "vary (", "zoom out", "make square")) for b in buttons):
                hint = "  提示：当前 task_id 似乎是单图任务，应使用「MJ Action Refine」节点。"
            return (
                create_blank_image(),
                "",
                json.dumps(buttons, ensure_ascii=False),
                "",
                json.dumps({"error": f"未匹配到 '{action}' index={index} 的按钮{hint}", "available_buttons": available}, ensure_ascii=False),
            )

        return _execute_mj_action(provider, task_id.strip(), custom_id, match_info, choose_same_channel, state, pbar)


class MJActionRefineNode:
    """Midjourney - Action 二阶段（单图精修：Upscale / Vary / Zoom / Pan / Square）

    适用对象：U1-U4 之后的单图任务的 task_id。
    四类分组下拉，每组默认 none；只能选一项执行（按 Upscale > Vary > Zoom > Pan 优先级）。
    """

    UPSCALE_OPTS = {
        "无": None,
        "Subtle": ["upscale (subtle)", "upscale subtle"],
        "Creative": ["upscale (creative)", "upscale creative"],
        # 旧版 MJ (V5) 才有 2x/4x；V6/V7 已废弃
        "2x (V5 only)": ["upscale (2x)", "upscale 2x"],
        "4x (V5 only)": ["upscale (4x)", "upscale 4x"],
    }
    VARY_OPTS = {
        "无": None,
        "Strong": ["vary (strong)", "vary strong"],
        "Subtle": ["vary (subtle)", "vary subtle"],
        "Region": ["vary (region)", "vary region"],
    }
    ZOOM_OPTS = {
        "无": None,
        "Zoom Out 2x": ["zoom out 2x", "zoom out (2x)"],
        "Zoom Out 1.5x": ["zoom out 1.5x", "zoom out (1.5x)"],
        "Custom Zoom": ["custom zoom"],
        "Make Square": ["make square"],
    }
    PAN_OPTS = {
        "无": None,
        "Left": ["⬅️", "pan left"],
        "Right": ["➡️", "pan right"],
        "上": ["⬆️", "pan up"],
        "下": ["⬇️", "pan down"],
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": "", "multiline": False, "tooltip": "U1-U4 之后的单图任务 task_id"}),
                "upscale": (list(cls.UPSCALE_OPTS.keys()), {"default": "无", "tooltip": "高清放大。Subtle=保守放大，尽量保持原图；Creative=创意放大，会补细节且可能轻微改图；2x/4x 仅 MJ V5 支持，V6/V7 通常用 Subtle/Creative"}),
                "vary": (list(cls.VARY_OPTS.keys()), {"default": "无", "tooltip": "单图变体。Strong=强变体，变化更大；Subtle=弱变体，更接近原图；Region=局部重绘，会先返回 WAITING_MODAL，需要再接 MJ Modal Submit 并提供 mask + prompt"}),
                "zoom": (list(cls.ZOOM_OPTS.keys()), {"default": "无", "tooltip": "画布外扩 / 拉方。Zoom Out 2x/1.5x=自动拉远；Custom Zoom=先返回 WAITING_MODAL，需要再接 MJ Modal Submit 填 prompt 和 zoom_ratio；Make Square=转方图"}),
                "pan": (list(cls.PAN_OPTS.keys()), {"default": "无", "tooltip": "朝某方向延伸画布。Left/Right/上/下 分别向左、右、上、下扩展画面"}),
                "custom_provider": ("CUSTOM_PROVIDER",),
            },
            "optional": {
                "auto_fetch_buttons": ("BOOLEAN", {"default": True, "tooltip": "若未提供 buttons_json，自动 fetch 任务详情"}),
                "buttons_json": ("STRING", {"default": "", "multiline": True, "placeholder": "可选: 直接传 buttons_json 跳过 fetch"}),
                "choose_same_channel": ("BOOLEAN", {"default": True, "tooltip": "保持 True：尽量在同一 Discord 频道执行"}),
                "state": ("STRING", {"default": "", "multiline": False, "tooltip": "可选: 自定义跟踪标记"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "task_id", "buttons_json", "image_url", "response")
    FUNCTION = "run"
    CATEGORY = "APIcaller/Midjourney"

    def _pick_active(self, upscale: str, vary: str, zoom: str, pan: str) -> Tuple[str, str, list]:
        """按 Upscale > Vary > Zoom > Pan 顺序选第一个非 none 的项。返回 (类别, 子项名, 候选词)。"""
        for cat, sel, opts in [
            ("Upscale", upscale, self.UPSCALE_OPTS),
            ("Vary", vary, self.VARY_OPTS),
            ("Zoom", zoom, self.ZOOM_OPTS),
            ("Pan", pan, self.PAN_OPTS),
        ]:
            if sel and sel != "无":
                cands = opts.get(sel)
                if cands:
                    return cat, sel, cands
        return "", "", []

    def run(
        self,
        task_id: str,
        upscale: str,
        vary: str,
        zoom: str,
        pan: str,
        custom_provider: dict,
        auto_fetch_buttons: bool = True,
        buttons_json: str = "",
        choose_same_channel: bool = True,
        state: str = "",
    ):
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)

        provider, err = _ensure_lingke(custom_provider)
        if err:
            return (create_blank_image(), "", "[]", "", json.dumps({"error": err}, ensure_ascii=False))
        if not task_id.strip():
            return (create_blank_image(), "", "[]", "", json.dumps({"error": "task_id 不能为空"}, ensure_ascii=False))

        cat, sub, candidates = self._pick_active(upscale, vary, zoom, pan)
        if not candidates:
            return (create_blank_image(), "", "[]", "", json.dumps({"error": "请至少在 upscale/vary/zoom/pan 中选择一项"}, ensure_ascii=False))

        # 多选警告（仅日志）
        chosen = [(n, v) for n, v in [("upscale", upscale), ("vary", vary), ("zoom", zoom), ("pan", pan)] if v and v != "无"]
        if len(chosen) > 1:
            print(f"[MJ Refine] 同时选了多项 {chosen}，按优先级 Upscale>Vary>Zoom>Pan，将执行: {cat}/{sub}")

        buttons, berr = _resolve_buttons(provider, task_id.strip(), buttons_json, auto_fetch_buttons)
        if berr:
            return (create_blank_image(), "", "[]", "", json.dumps({"error": berr}, ensure_ascii=False))
        pbar.update_absolute(15)

        custom_id, match_info = _find_button_custom_id(buttons, candidates)
        if not custom_id:
            available = [b.get("label") or b.get("emoji") or "?" for b in buttons]
            hint = ""
            if any((b.get("label") or "").upper() in ("U1", "U2", "U3", "U4") for b in buttons):
                hint = "  提示：当前 task_id 是 4 图网格任务，请先用「MJ Action Grid」选 U1-U4 提取单图，再做精修。"
            elif sub.startswith(("2x", "4x")):
                hint = "  提示：Upscale 2x/4x 仅 MJ V5 支持；V6/V7 请用 Upscale Subtle 或 Creative。"
            return (
                create_blank_image(),
                "",
                json.dumps(buttons, ensure_ascii=False),
                "",
                json.dumps({"error": f"未匹配到 '{cat} / {sub}' 的按钮{hint}", "available_buttons": available}, ensure_ascii=False),
            )

        return _execute_mj_action(provider, task_id.strip(), custom_id, f"{cat}/{sub} ({match_info})", choose_same_channel, state, pbar)


class MJModalSubmitNode:
    """Midjourney - Modal 二次提交（Vary Region / Custom Zoom 等）

    适用对象：MJ Action Refine 执行 Vary/Region 或 Custom Zoom 后返回的 WAITING_MODAL task_id。
    Vary Region 需要 mask + prompt；Custom Zoom 需要 prompt + zoom_ratio。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": "", "multiline": False, "tooltip": "Action 返回 code=21 / WAITING_MODAL 的 task_id"}),
                "modal_type": (
                    ["Vary Region", "Custom Zoom", "Generic"],
                    {"default": "Vary Region", "tooltip": "Vary Region=局部重绘，需要 mask + prompt；Custom Zoom=自定义拉远/扩图，需要 prompt + zoom_ratio；Generic=通用 modal 提交，字段可用 extra_payload_json 覆盖"},
                ),
                "prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "必填：写入局部重绘或 Custom Zoom 的提示词", "tooltip": "必填：Vary Region 用来描述 mask 区域要变成什么；Custom Zoom 用来描述扩展后的画面"}),
                "custom_provider": ("CUSTOM_PROVIDER",),
            },
            "optional": {
                "mask": ("MASK",),
                "zoom_ratio": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "[范围 1.0-2.0] 仅 Custom Zoom 生效并始终发送；1.0=默认，调大可尝试更强拉远"}),
                "extra_payload_json": ("STRING", {"default": "", "multiline": True, "placeholder": "高级: JSON 字段覆盖/追加，例如 {\"zoomRatio\": 1.8}"}),
                "state": ("STRING", {"default": "", "multiline": False, "tooltip": "可选: 自定义跟踪标记"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "task_id", "buttons_json", "image_url", "response")
    FUNCTION = "run"
    CATEGORY = "APIcaller/Midjourney"

    def run(
        self,
        task_id: str,
        modal_type: str,
        prompt: str,
        custom_provider: dict,
        mask=None,
        zoom_ratio: float = 1.0,
        extra_payload_json: str = "",
        state: str = "",
    ):
        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(5)

        provider, err = _ensure_lingke(custom_provider)
        if err:
            return (create_blank_image(), "", "[]", "", json.dumps({"error": err}, ensure_ascii=False))
        if not task_id.strip():
            return (create_blank_image(), "", "[]", "", json.dumps({"error": "task_id 不能为空"}, ensure_ascii=False))
        if not prompt.strip():
            return (create_blank_image(), "", "[]", "", json.dumps({"error": "prompt 必填：请写入局部重绘或 Custom Zoom 的提示词"}, ensure_ascii=False))

        extra_payload = {}
        if modal_type == "Custom Zoom":
            extra_payload["zoomRatio"] = float(zoom_ratio)

        if extra_payload_json.strip():
            try:
                parsed = json.loads(extra_payload_json)
                if not isinstance(parsed, dict):
                    return (create_blank_image(), "", "[]", "", json.dumps({"error": "extra_payload_json 必须是 JSON object"}, ensure_ascii=False))
                extra_payload.update(parsed)
            except Exception as e:
                return (create_blank_image(), "", "[]", "", json.dumps({"error": f"extra_payload_json 解析失败: {e}"}, ensure_ascii=False))

        mask_base64 = _mask_to_base64(mask) if modal_type in ("Vary Region", "Generic") else ""
        if modal_type == "Vary Region" and not mask_base64:
            return (
                create_blank_image(),
                "",
                "[]",
                "",
                json.dumps({"error": "Vary Region 需要连接 mask 输入，用白色区域表示要重绘的位置"}, ensure_ascii=False),
            )

        try:
            task_data, error = provider.mj_modal(
                task_id=task_id.strip(),
                prompt=prompt,
                mask_base64=mask_base64,
                extra_payload=extra_payload,
                state=state,
                pbar=pbar,
            )
            if error or not task_data:
                if modal_type == "Custom Zoom" and "maskBase64 is blank" in str(error):
                    error = (
                        f"{error}。提示：这个 WAITING_MODAL task_id 很可能来自 Vary Region/Inpaint，"
                        "不是 Custom Zoom。请先在 MJ Action Refine 里选择 zoom=Custom Zoom，"
                        "再把它返回的 task_id 接到本节点；如果来自 Vary Region，则必须连接 mask。"
                    )
                return (create_blank_image(), "", "[]", "", json.dumps({"error": error or "未知错误"}, ensure_ascii=False))

            images_tensor = _result_to_images(task_data)
            new_task_id = str(task_data.get("id") or "")
            image_url = str(task_data.get("imageUrl") or "")
            buttons = task_data.get("buttons") or []
            response = json.dumps(
                {
                    "id": new_task_id,
                    "status": task_data.get("status"),
                    "imageUrl": image_url,
                    "buttons": buttons,
                    "modalType": modal_type,
                    "extraPayload": extra_payload,
                },
                ensure_ascii=False,
                indent=2,
            )
            pbar.update_absolute(100)
            return (images_tensor, new_task_id, json.dumps(buttons, ensure_ascii=False), image_url, response)
        except Exception as e:
            err_msg = f"处理错误: {str(e)}"
            print(f"[MJ Modal] {err_msg}")
            return (create_blank_image(), "", "[]", "", json.dumps({"error": err_msg}, ensure_ascii=False))

