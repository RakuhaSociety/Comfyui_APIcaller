"""
API Key 池节点
支持多个 API Key 轮换使用，避免单一 Key 拥挤
"""
import random
from typing import Tuple


# 用于在 lock 模式下持久化上次选中的索引
_last_selected_index: dict = {}


class APIKeyPoolNode:
    """
    API Key 池节点 - 随机选取 Key，可锁定当前选择
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keys": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "每行一个API Key",
                }),
                "lock": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "note1": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "每行对应一个Key的备注1（可为空）",
                }),
                "note2": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "每行对应一个Key的备注2（可为空）",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("selected_key", "info")
    FUNCTION = "select_key"
    CATEGORY = "APIcaller"

    @classmethod
    def IS_CHANGED(cls, keys, lock, note1="", note2=""):
        if lock:
            return ""
        return float("nan")

    def select_key(
        self,
        keys: str,
        lock: bool = False,
        note1: str = "",
        note2: str = "",
    ) -> Tuple[str, str]:
        key_lines = [line.strip() for line in keys.strip().splitlines() if line.strip()]

        if not key_lines:
            return ("", "错误: 未提供任何API Key")

        key_count = len(key_lines)

        note1_lines = [line.strip() for line in note1.strip().splitlines()] if note1.strip() else []
        note2_lines = [line.strip() for line in note2.strip().splitlines()] if note2.strip() else []

        if note1_lines and len(note1_lines) != key_count:
            return ("", f"错误: 备注1行数({len(note1_lines)})与Key行数({key_count})不一致")
        if note2_lines and len(note2_lines) != key_count:
            return ("", f"错误: 备注2行数({len(note2_lines)})与Key行数({key_count})不一致")

        node_key = id(self)

        if lock and node_key in _last_selected_index:
            idx = _last_selected_index[node_key]
            if idx >= key_count:
                idx = random.randint(0, key_count - 1)
                _last_selected_index[node_key] = idx
        else:
            idx = random.randint(0, key_count - 1)
            _last_selected_index[node_key] = idx

        selected_key = key_lines[idx]
        n1 = note1_lines[idx] if note1_lines else ""
        n2 = note2_lines[idx] if note2_lines else ""

        info_lines = [selected_key, n1, n2]
        info = "\n".join(info_lines)

        print(f"[APIcaller] KeyPool 选中第{idx + 1}个Key (共{key_count}个), lock={lock}")

        return (selected_key, info)
