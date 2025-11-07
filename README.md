# CV + LLM 图像分析系统

一个结合计算机视觉（CV）和大语言模型（LLM）的图像分析应用，能够自动检测图像中的目标并生成自然语言描述。

## 项目概述

该系统通过以下流程实现图像分析：

1. 使用 YOLOv8 模型检测图像中的目标物体，获取类别、置信度和位置信息
2. 将检测结果通过 Kimi API 处理
3. 生成自然、流畅的语言描述，说明图像中的内容

## 核心功能

- 图像上传与显示
- 目标检测（支持多种常见物体类别）
- 检测结果可视化（带边界框和标签）
- 基于检测结果的自然语言描述生成
- 自动判断 GPU/CPU 运行环境

## 技术栈

- 目标检测：YOLOv8n（轻量级模型）
- 语言生成：Kimi API (Moonshot)
- Web 框架：Streamlit
- 图像处理：OpenCV & Pillow

## 安装说明

### 前置要求

- Python 3.8+
- （可选）NVIDIA GPU 及 CUDA 环境（推荐，可加速目标检测）
- Kimi API 密钥（必需）

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/kafuuchino123/CV_LLM_PRO.git
cd CV_LLM_PRO
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置 API 密钥：
   - 打开 `config.py`
   - 将您的 Kimi API 密钥填入 `API_KEY` 字段

### 运行应用

```bash
streamlit run app.py
```

## 项目结构

```
.
├── app.py           # 主应用程序（UI和流程控制）
├── cv_model.py      # 计算机视觉模型（YOLOv8）
├── llm_model.py     # 语言模型（Kimi API 接口）
├── config.py        # 配置文件
├── requirements.txt # 项目依赖
└── README.md        # 项目文档
```

## 配置说明

在 `config.py` 中可以修改以下配置：

- `API_KEY`：Kimi API 密钥（必需配置）
- `CV_MODEL_PATH`：YOLOv8 模型路径（默认 "yolov8n.pt"）
- `CONF_THRESHOLD`：目标检测置信度阈值（默认 0.3）

## 使用流程

1. 启动应用后，在浏览器中打开显示的本地地址
2. 上传图片（支持 jpg、jpeg、png 格式）
3. 系统自动进行目标检测
4. 显示检测结果和边界框标注
5. 自动生成图像内容的文字描述

## 注意事项

- 首次运行时会自动下载 YOLOv8 模型
- 确保已正确配置 Kimi API 密钥
- 大尺寸图片可能需要更长处理时间
- 检测结果的准确性受图像质量影响

## 系统要求

- 内存：最低 4GB，推荐 8GB 以上
- 存储：至少 2GB 可用空间
- GPU：可选，但推荐使用 NVIDIA GPU 以获得更好性能

## 常见问题

1. 如果遇到模型下载失败，请检查网络连接
2. 如果显示 API 调用失败，请检查 API 密钥是否正确配置
3. 如果检测结果不理想，可以在 config.py 中调整置信度阈值