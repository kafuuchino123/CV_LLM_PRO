项目概述
该系统通过以下流程实现图像分析：

    使用 YOLOv8 模型检测图像中的目标物体，获取类别、置信度和位置信息
    将检测结果输入到 phi-2 语言模型
    生成结构化的自然语言描述，说明图像中的内容

界面采用 Streamlit 构建，支持图片上传、检测结果可视化和描述生成功能。
核心功能

    图像上传与显示
    目标检测（支持多种常见物体类别）
    检测结果可视化（带边界框和标签）
    基于检测结果的自然语言描述生成
    自动判断 GPU/CPU 运行环境

安装说明
前置要求

    Python 3.8+
    （可选）NVIDIA GPU 及 CUDA 环境（推荐，可加速模型运行）

依赖安装
bash

pip install streamlit opencv-python numpy pillow ultralytics transformers torch torchvision torchaudio bitsandbytes

模型准备

    YOLOv8n 模型：会在首次运行时自动下载（yolov8n.pt）
    LLM 模型：首次运行会自动下载模型并进行运行

使用方法

    克隆或下载项目代码
    准备好 phi-2 模型并配置路径
    运行应用：
    bash

    streamlit run app.py

    在浏览器中打开显示的本地地址
    上传图片（支持 jpg、jpeg、png 格式）
    查看检测结果和生成的描述

项目结构
plaintext

.
├── app.py           # 主应用程序，包含UI和流程控制
├── cv_model.py      # 计算机视觉模型封装（YOLOv8）
├── llm_model.py     # 语言模型封装（phi-2）
└── README.md        # 项目说明文档

文件说明

    app.py：Streamlit 应用主文件，负责界面渲染、图像处理流程和结果展示
    cv_model.py：基于 YOLOv8 的目标检测模型封装，处理图像并返回检测结果
    llm_model.py：基于 phi-2 的语言模型封装，将检测结果转换为自然语言描述

技术细节

    目标检测：使用 YOLOv8n 轻量级模型，平衡速度和精度
    语言模型：采用 phi-2 模型，通过 4-bit 量化减少内存占用
    性能优化：使用 Streamlit 缓存机制减少重复计算
    兼容性：自动适配 CPU/GPU 环境，优先使用 GPU 加速

注意事项

    首次运行时模型加载可能需要较长时间
    大尺寸图片可能需要更长处理时间
    LLM 模型需要较大内存，建议在有足够资源的环境中运行
    检测结果的准确性受光照、角度等因素影响

示例效果

    上传图片后，系统会显示原始图片
    展示检测到的目标列表（类别、置信度、位置）
    显示带有边界框的图像
    生成并展示图像内容的自然语言描述

该项目结合了计算机视觉和自然语言处理技术，可用于图像内容分析、辅助描述生成等场景。# CV + LLM 图像分析系统
一个结合计算机视觉（CV）和大语言模型（LLM）的图像分析应用，能够自动检测图像中的目标并生成自然语言描述。
项目概述
该系统通过以下流程实现图像分析：

    使用 YOLOv8 模型检测图像中的目标物体，获取类别、置信度和位置信息
    将检测结果输入到 phi-2 语言模型
    生成结构化的自然语言描述，说明图像中的内容

界面采用 Streamlit 构建，支持图片上传、检测结果可视化和描述生成功能。
核心功能

    图像上传与显示
    目标检测（支持多种常见物体类别）
    检测结果可视化（带边界框和标签）
    基于检测结果的自然语言描述生成
    自动判断 GPU/CPU 运行环境

安装说明
前置要求

    Python 3.8+
    （可选）NVIDIA GPU 及 CUDA 环境（推荐，可加速模型运行）

依赖安装
bash

pip install streamlit opencv-python numpy pillow ultralytics transformers torch torchvision torchaudio bitsandbytes

模型准备

    YOLOv8n 模型：会在首次运行时自动下载（yolov8n.pt）
    LLM 模型：需要手动准备 phi-2 模型文件，并在llm_model.py中修改model_name路径指向模型所在目录

使用方法

    克隆或下载项目代码
    准备好 phi-2 模型并配置路径
    运行应用：
    bash

    streamlit run app.py

    在浏览器中打开显示的本地地址
    上传图片（支持 jpg、jpeg、png 格式）
    查看检测结果和生成的描述

项目结构
plaintext

.
├── app.py           # 主应用程序，包含UI和流程控制
├── cv_model.py      # 计算机视觉模型封装（YOLOv8）
├── llm_model.py     # 语言模型封装（phi-2）
└── README.md        # 项目说明文档

文件说明

    app.py：Streamlit 应用主文件，负责界面渲染、图像处理流程和结果展示
    cv_model.py：基于 YOLOv8 的目标检测模型封装，处理图像并返回检测结果
    llm_model.py：基于 phi-2 的语言模型封装，将检测结果转换为自然语言描述

技术细节

    目标检测：使用 YOLOv8n 轻量级模型，平衡速度和精度
    语言模型：采用 phi-2 模型，通过 4-bit 量化减少内存占用
    性能优化：使用 Streamlit 缓存机制减少重复计算
    兼容性：自动适配 CPU/GPU 环境，优先使用 GPU 加速

注意事项

    首次运行时模型加载可能需要较长时间
    大尺寸图片可能需要更长处理时间
    LLM 模型需要较大内存，建议在有足够资源的环境中运行
    检测结果的准确性受光照、角度等因素影响

示例效果

    上传图片后，系统会显示原始图片
    展示检测到的目标列表（类别、置信度、位置）
    显示带有边界框的图像
    生成并展示图像内容的自然语言描述

该项目结合了计算机视觉和自然语言处理技术，可用于图像内容分析、辅助描述生成等场景。