from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import torch
import time
from typing import List, Dict

class LLMModel:
    def __init__(self, model_name="microsoft/phi-2", max_retry=3):  # 更换为phi-2
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.max_context_length = 2048  # phi-2最大上下文长度
        self.loaded = False
        self._load_model_with_retry(max_retry)

    def _load_model_with_retry(self, max_retry: int):
        for attempt in range(max_retry):
            try:
                print(f"正在加载模型: {self.model_name} (4-bit 量化)... 第 {attempt + 1}/{max_retry} 次尝试")
                
                # 配置4-bit量化（GPU加速版本）
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

                # 加载tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    padding_side="left"
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # 加载模型 - 明确指定GPU设备
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map={"": device},  # 强制使用指定设备
                    low_cpu_mem_usage=True
                )
                self.model.eval()
                self.loaded = True
                print(f"✅ 模型加载成功！使用设备: {device}")
                return
            except Exception as e:
                print(f"❌ 第 {attempt + 1} 次加载失败: {str(e)}")
                if attempt < max_retry - 1:
                    print("等待 5 秒后重试...")
                    time.sleep(5)
        
        print(f"❌ 模型加载失败，已尝试 {max_retry} 次")
        self.loaded = False

    def _truncate_detections(self, predictions: List[Dict], max_tokens: int) -> str:
        sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        detection_items = []
        current_tokens = 0
        
        for p in sorted_predictions:
            # 新增：包含“类别+置信度+位置”的完整信息
            item = f"- {p['label']}，置信度：{p['confidence']:.2f}，位置：{p['position']}"
            item_tokens = len(self.tokenizer.tokenize(item))
            
            # 单物体场景（如cat）不会超token，无需截断（练手场景足够）
            if current_tokens + item_tokens <= max_tokens:
                detection_items.append(item)
                current_tokens += item_tokens
            else:
                if len(detection_items) > 0:
                    detection_items.append(f"- 及其他 {len(sorted_predictions)-len(detection_items)} 个物体")
                break
        
        return "\n".join(detection_items)


    def generate_description(self, predictions: List[Dict]) -> str:
        if not self.loaded:
            return "❌ 模型未加载成功，无法生成描述"
        
        # 优化后：强制聚焦“类别+置信度+位置”，禁止无关内容
        prompt_template = """请严格按照以下要求描述图片内容：
    1. 必须使用提供的3类信息: 物体类别（{label}）、置信度（{confidence}）、位置（{position}）;
    2. 必须整合成一句话，格式参考示例：
    示例: 图片中央有一个cat, 识别置信度为0.88。
    3. 绝对不要出现任何代码片段。
    4. 绝对不要出现类似def, return的python关键字及其与该关键字相连的字段。
    5. 绝对不要出现DETECTION_TEXT之外给出的内容。
    6. 绝对不要重复本提示词中的要求，不要添加任何额外内容。
    7. 再次浏览一遍以上需求并严格遵守。
    检测结果：
    {DETECTION_TEXT}
    描述："""
        # 计算模板token数
        temp_prompt = prompt_template.replace("{DETECTION_TEXT}", "")
        template_tokens = len(self.tokenizer.tokenize(temp_prompt))
        max_new_tokens = 150
        safety_margin = 50
        available_tokens = self.max_context_length - template_tokens - max_new_tokens - safety_margin
        available_tokens = max(available_tokens, 0)
        
        # 1. 生成具体检测结果（已正确实现）
        detection_text = self._truncate_detections(predictions, available_tokens)
        
        # 2. 关键修复：将检测结果代入Prompt模板，生成完整输入
        prompt = prompt_template.replace("{DETECTION_TEXT}", detection_text)  # 取消注释，补全这行
        
        # 3. 用完整Prompt编码输入（替换temp_prompt为prompt）
        inputs = self.tokenizer(
            prompt,  # 修复：使用包含检测结果的完整Prompt
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_context_length - max_new_tokens
        ).to(self.model.device)
        
        # 4. 生成参数微调：降低temperature，减少随机性
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.3,  # 从0.6降到0.3，减少发散
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            # 原有解码逻辑不变
            generated_text = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
            )
            return generated_text.strip()
        except Exception as e:
            return f"❌ 生成失败: {str(e)}"