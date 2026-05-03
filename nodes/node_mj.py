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
        "Up": ["⬆️", "pan up"],
        "Down": ["⬇️", "pan down"],
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": "", "multiline": False, "tooltip": "U1-U4 之后的单图任务 task_id"}),
                "upscale": (list(cls.UPSCALE_OPTS.keys()), {"default": "无", "tooltip": "高清放大；2x/4x 仅 MJ V5 支持，V6/V7 请用 Subtle/Creative"}),
                "vary": (list(cls.VARY_OPTS.keys()), {"default": "无", "tooltip": "基于单图做变体"}),
                "zoom": (list(cls.ZOOM_OPTS.keys()), {"default": "无", "tooltip": "画布外扩 / 拉方"}),
                "pan": (list(cls.PAN_OPTS.keys()), {"default": "无", "tooltip": "朝某方向延伸画布"}),
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

