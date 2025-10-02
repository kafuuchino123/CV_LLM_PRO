from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMModel:
    def __init__(self, model_name="Qwen/Qwen-7B-Chat"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 检查是否有GPU可用
        device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True
        )
        self.model.eval()

    def generate_description(self, predictions):
        # 构建 prompt（字符串格式）
        prompt = "你是一个图像描述助手。\n"
        prompt += "请根据以下检测结果生成一段自然语言描述：\n"
        for pred in predictions:
            prompt += f"- {pred['label']} (置信度: {pred['confidence']:.2f})\n"
        prompt += "请用简洁、生动的语言描述这张图片的内容。\n"

        # 🔑 使用 Qwen 自带的 chat 方法（传入字符串，不是 messages 列表）
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)
        return response
