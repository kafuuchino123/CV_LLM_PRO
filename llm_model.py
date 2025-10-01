from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMModel:
    def __init__(self, model_name="Qwen/Qwen-7B-Chat"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
    
    def generate_description(self, predictions):
        # 构建提示词
        prompt = f"请根据以下检测结果生成一段自然语言描述："
        for i, pred in enumerate(predictions):
            prompt += f"第{i+1}个目标：{pred['label']}，置信度{pred['confidence']:.2f}；"
        
        prompt += "请用简洁、生动的语言描述这张图片的内容，包括主要物体、场景和可能的活动。"
        
        # 生成描述
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return description