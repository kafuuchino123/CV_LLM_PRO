class Config:
    """配置类"""
    # LLM配置
    API_KEY = "你的api密钥"  # 替换为您的 API 密钥
    
    # CV模型配置
    CV_MODEL_PATH = "yolov8n.pt"  # YOLOv8模型路径
    CONF_THRESHOLD = 0.3  # 目标检测置信度阈值

    # 以下是固定配置，通常不需要修改
    MAX_RETRIES = 3  # API调用失败时的重试次数
    TEMPERATURE = 0.3  # 生成文本的温度值（越低越稳定）