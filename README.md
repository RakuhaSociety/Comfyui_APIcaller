# Comfyui_APIcaller

[中文](#中文) | [English](#english)

---

## 中文

一个支持多API供应商的ComfyUI自定义节点插件，整合视频生成、图像生成/编辑、LLM对话等功能。

### ✨ 功能节点

| 节点 | 说明 | 供应商 |
|------|------|--------|
| 🍌 Nano Banana Edit | 图像编辑（图生图） | Lingke / Kie / WaveSpeed |
| 🍌 Nano Banana Text2Img | 文生图 | Lingke / Kie / WaveSpeed |
| 🎬 Grok Video Generator | Grok 视频生成（支持多图） | Lingke / Kie |
| 🎬 Sora 2 Video Generator | Sora 2 视频生成 | Lingke / Kie |
| 🎬 Veo 3.1 Video Generator | Veo 3.1 视频生成 | Lingke / Kie |
| 🎬 Hailuo Video Generator | 海螺视频生成 | Lingke / Kie |
| 🤖 OpenAI LLM | 通用 OpenAI 格式 LLM 对话（支持 Vision） | 任意兼容 OpenAI API 的服务 |
| 🔧 Custom Provider | 自定义 API Key + Base URL | — |
| 🔑 API Key Pool | API 密钥池，随机选取 | — |

### 🔑 特性

- **多供应商切换** — 同一节点通过下拉菜单选择不同供应商
- **Custom Provider** — 输入自定义 API Key + Base URL，接入任何兼容的第三方 API
- **API Key Pool** — 多 Key 轮盘，随机选取，分散调用压力
- **批次模式** — Nano Banana 节点支持批量处理多张图像 / 多行提示词
- **错误重试** — 调用失败自动重试，减少白图

### 📦 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/你的用户名/Comfyui_APIcaller.git
cd Comfyui_APIcaller
pip install -r requirements.txt
```
重启 ComfyUI 即可使用。

### ⚙️ 配置

#### 方式一：使用 Custom Provider 节点（推荐）
直接在工作流中添加 `🔧 Custom Provider` 节点，输入你的 API Key 和 Base URL，连接到其他节点即可。

#### 方式二：节点内输入
每个节点都有 `api_key` 输入框，可直接填写密钥。

#### 供应商默认 Base URL

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
| 🍌 Nano Banana Edit | Image editing (img2img) | Lingke / Kie / WaveSpeed |
| 🍌 Nano Banana Text2Img | Text to image | Lingke / Kie / WaveSpeed |
| 🎬 Grok Video Generator | Grok video generation (multi-image) | Lingke / Kie |
| 🎬 Sora 2 Video Generator | Sora 2 video generation | Lingke / Kie |
| 🎬 Veo 3.1 Video Generator | Veo 3.1 video generation | Lingke / Kie |
| 🎬 Hailuo Video Generator | Hailuo video generation | Lingke / Kie |
| 🤖 OpenAI LLM | Universal OpenAI-format LLM chat (Vision support) | Any OpenAI-compatible API |
| 🔧 Custom Provider | Custom API Key + Base URL | — |
| 🔑 API Key Pool | API key pool with random selection | — |

### 🔑 Features

- **Multi-provider switching** — Select different providers from a dropdown within the same node
- **Custom Provider** — Input custom API Key + Base URL to connect any compatible third-party API
- **API Key Pool** — Multiple keys with random selection to distribute API call load
- **Batch mode** — Nano Banana nodes support batch processing of multiple images / multi-line prompts
- **Error retry** — Automatic retry on failure to reduce blank outputs

### 📦 Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/Comfyui_APIcaller.git
cd Comfyui_APIcaller
pip install -r requirements.txt
```
Restart ComfyUI to use.

### ⚙️ Configuration

#### Option 1: Custom Provider node (Recommended)
Add the `🔧 Custom Provider` node to your workflow, enter your API Key and Base URL, and connect it to other nodes.

#### Option 2: Direct input
Each node has an `api_key` input field where you can enter your key directly.

#### Default Provider Base URLs

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
