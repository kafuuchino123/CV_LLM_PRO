import time
from typing import List, Dict
from config import Config
from openai import OpenAI

class LLMModel:
    def __init__(self):
        if not Config.API_KEY or Config.API_KEY == "在这里填入你的QWEN API密钥":
            raise ValueError("请在 config.py 中设置您的 QWEN API 密钥")
        
        self.api_key = Config.API_KEY
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.loaded = True
        
    def _make_api_request(self, prompt: str, max_retries: int = None) -> str:
        max_retries = max_retries or Config.MAX_RETRIES
        
        # 构建完整的提示，包含系统指令
        full_prompt = f"你是一个专业的图像分析助手，负责将图像检测结果转化为自然、准确的描述。\n\n{prompt}"
        
        for attempt in range(max_retries):
            try:
                response = self.client.responses.create(
                    model="qwen3.6-plus",
                    input=full_prompt
                )
                return response.output_text
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"❌ API调用失败: {str(e)}"
                time.sleep(1)  # 失败后等待1秒再重试
                
        return "❌ 超过最大重试次数"

    def _format_detections(self, predictions: List[Dict]) -> str:
        detection_items = []
        # 按置信度排序
        sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        for p in sorted_predictions:
            item = f"- {p['label']}，置信度：{p['confidence']:.2f}，位置：{p['position']}"
            detection_items.append(item)
        
        return "\n".join(detection_items)

    def generate_description(self, predictions: List[Dict]) -> str:
        if not self.loaded:
            return "❌ API未正确初始化"
            
        if not predictions:
            return "未检测到任何物体。"
            
        # 优化后的prompt模板
        prompt_template = """请根据以下图像检测结果生成一段自然、流畅的描述：
        
检测结果：
{DETECTION_TEXT}

要求：
1. 使用自然的语言描述检测到的物体，包括它们的位置和置信度
2. 将多个检测结果整合成连贯的句子
3. 优先描述置信度较高的物体
4. 使用恰当的连接词使描述流畅自然

请生成描述："""
        
        # 格式化检测结果并代入模板
        detection_text = self._format_detections(predictions)
        prompt = prompt_template.replace("{DETECTION_TEXT}", detection_text)
        
        # 调用API生成描述
        return self._make_api_request(prompt)