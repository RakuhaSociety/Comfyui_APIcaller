# Comfyui_APIcaller

[中文](#中文) | [English](#english)

---

## 中文

一个支持多API供应商的ComfyUI自定义节点插件，整合视频生成、图像生成/编辑、LLM对话等功能。

### ✨ 功能节点

| 节点 | 说明 | 供应商 |
|------|------|--------|
| 🍌 Nano Banana Edit | 图像编辑（图生图） | 通过 Custom Provider 接入 |
| 🍌 Nano Banana Text2Img | 文生图 | 通过 Custom Provider 接入 |
| 🎬 Grok Video Generator | Grok 视频生成（支持多图） | Lingke / Kie |
| 🎬 Sora 2 Video Generator | Sora 2 视频生成 | Lingke / Kie |
| 🎬 Veo 3.1 Video Generator | Veo 3.1 视频生成 | Lingke / Kie |
| 🎬 Hailuo Video Generator | 海螺视频生成 | Lingke / Kie |
| 🤖 OpenAI LLM | 通用 OpenAI 格式 LLM 对话（支持 Vision） | 任意兼容 OpenAI API 的服务 |
| 🔧 Custom Provider | 自定义 API Key + Base URL | — |
| 🔑 API Key Pool | API 密钥池，随机选取 | — |

### 🔑 特性

- **Custom Provider** — 输入自定义 API Key + Base URL，接入任何兼容的第三方 API（Nano Banana / OpenAI LLM 必须连接）
- **API Key Pool** — 多 Key 轮盘，随机选取，分散调用压力
- **批次模式** — Nano Banana 节点支持批量处理多张图像 / 多行提示词
- **错误重试** — 调用失败自动重试，减少白图

### 📦 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RakuhaSociety/Comfyui_APIcaller.git
cd Comfyui_APIcaller
pip install -r requirements.txt
```
重启 ComfyUI 即可使用。

### ⚙️ 配置

在工作流中添加 `🔧 Custom Provider` 节点，输入你的 API Key 和 Base URL，连接到其他节点即可。

视频节点（Grok / Sora2 / Veo3.1 / Hailuo）也支持节点内直接输入 `api_key`。

#### 常用供应商 Base URL

| 供应商 | Base URL |
|--------|----------|
| Lingke (灵客) | `https://lingkeapi.com` |
| Kie | `https://api.kie.ai` |
| WaveSpeed | `https://api.wavespeed.ai` |

### 📖 使用说明

#### Nano Banana 批次模式
开启 `batch_mode` 后：
- **image1-4** 支持图像批次输入
- **prompt** 多行时，每行对应一个批次
- 所有多批次输入（图像批次数 / prompt行数）必须一致
- 单张图像和单行 prompt 会自动广播到所有批次

#### Nano Banana 错误重试
开启 `error_retry` 后，API 调用失败（返回空图/报错/下载失败等）会自动重试：
- **max_retries** 设置最大重试次数（默认 3，范围 1-10）
- 每次重试间隔 2 秒
- 批次模式下每个批次独立重试
- 关闭时不重试，失败直接返回白图 + 错误信息

#### API Key Pool
`🔑 API Key Pool` 节点用于管理多个 API 密钥，每次执行随机选取一个：
- 填入多个 key（key1 ~ key5），留空的会被跳过
- 每次运行随机选取一个可用 key 输出
- 可附带 note1 / note2 备注信息
- 输出的 key 可连接到各生成节点的 `api_key` 输入
- 适合多 key 轮换，分散调用频率限制

#### OpenAI LLM
- 必须连接 `Custom Provider` 节点提供 API Key 和 Base URL
- 支持图像输入（Vision），自动处理图像批次
- 兼容任何 OpenAI Chat Completions 格式的 API

### 🏗️ 扩展开发

添加新的 API 供应商：
1. 在 `providers/` 目录下创建 `provider_xxx.py`
2. 继承 `BaseProvider` 类并实现所需方法
3. 在 `providers/__init__.py` 中注册

### 📄 许可证

MIT License

---

## English

A multi-provider ComfyUI custom node plugin integrating video generation, image generation/editing, and LLM chat.

### ✨ Nodes

| Node | Description | Providers |
|------|-------------|-----------|
| 🍌 Nano Banana Edit | Image editing (img2img) | Via Custom Provider |
| 🍌 Nano Banana Text2Img | Text to image | Via Custom Provider |
| 🎬 Grok Video Generator | Grok video generation (multi-image) | Lingke / Kie |
| 🎬 Sora 2 Video Generator | Sora 2 video generation | Lingke / Kie |
| 🎬 Veo 3.1 Video Generator | Veo 3.1 video generation | Lingke / Kie |
| 🎬 Hailuo Video Generator | Hailuo video generation | Lingke / Kie |
| 🤖 OpenAI LLM | Universal OpenAI-format LLM chat (Vision support) | Any OpenAI-compatible API |
| 🔧 Custom Provider | Custom API Key + Base URL | — |
| 🔑 API Key Pool | API key pool with random selection | — |

### 🔑 Features

- **Custom Provider** — Input custom API Key + Base URL to connect any compatible third-party API (required for Nano Banana / OpenAI LLM)
- **API Key Pool** — Multiple keys with random selection to distribute API call load
- **Batch mode** — Nano Banana nodes support batch processing of multiple images / multi-line prompts
- **Error retry** — Automatic retry on failure to reduce blank outputs

### 📦 Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RakuhaSociety/Comfyui_APIcaller.git
cd Comfyui_APIcaller
pip install -r requirements.txt
```
Restart ComfyUI to use.

### ⚙️ Configuration

Add the `🔧 Custom Provider` node to your workflow, enter your API Key and Base URL, and connect it to other nodes.

Video nodes (Grok / Sora2 / Veo3.1 / Hailuo) also support entering `api_key` directly in the node.

#### Common Provider Base URLs

| Provider | Base URL |
|----------|----------|
| Lingke | `https://lingkeapi.com` |
| Kie | `https://api.kie.ai` |
| WaveSpeed | `https://api.wavespeed.ai` |

### 📖 Usage

#### Nano Banana Batch Mode
When `batch_mode` is enabled:
- **image1-4** accept image batch inputs
- **prompt** with multiple lines: each line corresponds to a batch
- All multi-batch inputs (image batch count / prompt line count) must match
- Single images and single-line prompts are automatically broadcast to all batches

#### Nano Banana Error Retry
When `error_retry` is enabled, failed API calls (empty result / errors / download failures) are automatically retried:
- **max_retries** sets the maximum retry count (default 3, range 1-10)
- 2-second delay between retries
- In batch mode, each batch retries independently
- When disabled, failures return a blank white image + error message immediately

#### API Key Pool
The `🔑 API Key Pool` node manages multiple API keys with random selection:
- Enter multiple keys (key1 ~ key5); empty slots are skipped
- A random available key is selected on each run
- Optional note1 / note2 fields for labeling
- Output key connects to the `api_key` input of any generation node
- Useful for key rotation to distribute rate limits

#### OpenAI LLM
- Requires a `Custom Provider` node for API Key and Base URL
- Supports image input (Vision) with automatic batch handling
- Compatible with any OpenAI Chat Completions format API

### 🏗️ Development

To add a new API provider:
1. Create `provider_xxx.py` in the `providers/` directory
2. Inherit from `BaseProvider` and implement required methods
3. Register in `providers/__init__.py`

### 📄 License

MIT License
