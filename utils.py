"""
通用工具函数
"""
import numpy as np
import torch
import requests
import base64
import tempfile
import os
try:
    import imageio
except ImportError:
    imageio = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

import folder_paths
import uuid
import os

from PIL import Image
from io import BytesIO
import shutil
import requests
from typing import List, Union, Tuple, Optional, Dict

class VideoAdapter:
    """
    Video Adapter similar to ComflyVideoAdapter
    Allows ComfyUI to handle video preview and saving via save_to method.
    """
    def __init__(self, video_path_or_url):
        if video_path_or_url is None:
             self.is_url = False
             self.video_path = None
             self.video_url = None
             return

        if video_path_or_url.startswith('http'):
            self.is_url = True
            self.video_url = video_path_or_url
            self.video_path = None
        else:
            self.is_url = False
            self.video_path = video_path_or_url
            self.video_url = None
        
    def get_dimensions(self):
        # Default fallback dimensions if we can't probe
        return 1280, 720
            
    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        if hasattr(self, 'video_url') and self.is_url:
            try:
                response = requests.get(self.video_url, stream=True)
                response.raise_for_status()
                
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            except Exception as e:
                print(f"[APIcaller] Error downloading video from URL: {str(e)}")
                return False
        elif hasattr(self, 'video_path') and self.video_path:
            try:
                shutil.copyfile(self.video_path, output_path)
                return True
            except Exception as e:
                print(f"[APIcaller] Error saving video: {str(e)}")
                return False
        return False

class EmptyVideoAdapter:
    """Empty video adapter for error cases"""
    def __init__(self):
        self.is_empty = True
        
    def get_dimensions(self):
        return 1, 1  # Minimal dimensions
    
    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        # Create a minimal black video file or just fail gracefully
        return False


def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    """
    将PIL图像转换为tensor
    
    Args:
        image: 单个PIL Image或PIL Image列表
        
    Returns:
        torch.Tensor: 图像tensor，值归一化到 [0, 1]
    """
    if isinstance(image, list):
        if len(image) == 0:
            return torch.empty(0)
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)[None,]


def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    """
    将tensor转换为PIL图像
    
    Args:
        image: Tensor，形状为 [B, H, W, 3] 或 [H, W, 3]，值范围 [0, 1]
        
    Returns:
        List[Image.Image]: PIL Image列表
    """
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out

    numpy_image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return [Image.fromarray(numpy_image)]


def image_to_base64(image_tensor: torch.Tensor, format: str = "PNG") -> Optional[str]:
    """
    将tensor转换为base64字符串
    
    Args:
        image_tensor: 图像tensor
        format: 图像格式，默认PNG
        
    Returns:
        base64编码的字符串，如果失败返回None
    """
    if image_tensor is None:
        return None
        
    try:
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"[APIcaller] Error converting image to base64: {str(e)}")
        return None


def base64_to_pil(base64_string: str) -> Optional[Image.Image]:
    """
    将base64字符串转换为PIL图像
    
    Args:
        base64_string: base64编码的图像字符串
        
    Returns:
        PIL Image，如果失败返回None
    """
    try:
        # 移除可能的data URI前缀
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data))
    except Exception as e:
        print(f"[APIcaller] Error converting base64 to image: {str(e)}")
        return None


def download_image(url: str, timeout: int = 60) -> Tuple[Optional[bytes], Optional[str]]:
    """
    下载图像
    
    Args:
        url: 图像URL
        timeout: 超时时间（秒）
        
    Returns:
        (图像字节数据, 错误信息)，成功时错误信息为None
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content, None
    except Exception as e:
        return None, str(e)


def create_blank_image(width: int = 1024, height: int = 1024, color: str = 'white') -> torch.Tensor:
    """
    创建空白图像tensor
    
    Args:
        width: 宽度
        height: 高度
        color: 颜色
        
    Returns:
        图像tensor
    """
    blank_image = Image.new('RGB', (width, height), color=color)
    return pil2tensor(blank_image)


def upload_image_to_url(image_tensor: torch.Tensor, upload_url: str, api_key: str) -> Optional[str]:
    """
    上传图像到指定URL并返回可访问的图像URL
    
    Args:
        image_tensor: 图像tensor
        upload_url: 上传API地址
        api_key: API密钥
        
    Returns:
        上传后的图像URL，如果失败返回None
    """
    try:
        pil_image = tensor2pil(image_tensor)[0]
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        buffered.seek(0)
        
        files = {'file': ('image.png', buffered, 'image/png')}
        headers = {'Authorization': f'Bearer {api_key}'}
        
        response = requests.post(upload_url, files=files, headers=headers, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result.get('url') or result.get('data', {}).get('url')
    except Exception as e:
        print(f"[APIcaller] Error uploading image: {str(e)}")
        return None


def save_video_to_temp(url: str, timeout: int = 120) -> Tuple[Optional[str], Optional[str]]:
    """
    下载视频并保存到ComfyUI的temp目录
    
    Args:
        url: 视频URL
        timeout: 下载超时
        
    Returns:
        (Temp Filename, Error Message)
    """
    try:
        content, error = download_image(url, timeout)
        if error:
            return None, error
            
        filename = f"grok_video_{uuid.uuid4().hex[:8]}.mp4"
        temp_dir = folder_paths.get_temp_directory()
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, 'wb') as f:
            f.write(content)
            
        return filename, None
    except Exception as e:
        return None, f"Error saving video: {str(e)}"


def load_video_from_url(url: str, timeout: int = 120) -> Tuple[Optional[torch.Tensor], Optional[Dict], Optional[str]]:
    """
    从URL加载视频并转换为Tensor
    
    Args:
        url: 视频URL
        timeout: 下载超时
        
    Returns:
        (Frame Tensor [B, H, W, 3], Audio Dict, Error Message)
    """
    if imageio is None:
        return None, None, "imageio module not found, cannot process video"

    try:
        # Save to temp file using our helper (or reuse logic)
        # We use a persistent temp file in ComfyUI temp dir now for consistency
        filename, error = save_video_to_temp(url, timeout)
        if error:
            return None, None, error
            
        file_path = os.path.join(folder_paths.get_temp_directory(), filename)
        
        try:
            # Read frames
            frames = []
            reader = imageio.get_reader(file_path)
            for frame in reader:
                # Convert numpy array to PIL Image
                frames.append(Image.fromarray(frame))
            reader.close()
            
            if not frames:
                return None, None, "No frames decoded from video"

            video_frames = pil2tensor(frames)
            
            # Read Audio
            audio_data = None
            if torchaudio is not None:
                try:
                    # torchaudio.load returns (waveform, sample_rate)
                    # waveform is (channels, time)
                    waveform, sample_rate = torchaudio.load(file_path)
                    # ComfyUI audio format: {"waveform": (batch, channels, time), "sample_rate": int}
                    audio_data = {
                        "waveform": waveform.unsqueeze(0),
                        "sample_rate": sample_rate
                    }
                except Exception as e:
                    print(f"[APIcaller] Failed to load audio: {e}")
                    pass
            
            return video_frames, audio_data, None
            
        except Exception as e:
            return None, None, f"Video processing error: {str(e)}"
         # Note: We do NOT delete the file here anymore, as we might need it for VIDEO output
         # ComfyUI cleans up temp dir on restart usually
                    
    except Exception as e:
        return None, None, f"Video processing error: {str(e)}"

