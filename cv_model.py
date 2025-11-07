import cv2
import numpy as np
from ultralytics import YOLO
from config import Config

class CVModel:
    def __init__(self, conf_threshold=None):
        # 使用配置中的模型路径
        self.model = YOLO(Config.CV_MODEL_PATH)
        self.class_names = self.model.names
        # 使用配置中的置信度阈值，如果没有传入自定义值的话
        self.conf_threshold = conf_threshold or Config.CONF_THRESHOLD
        
    def predict(self, image):
        # 输入有效性校验
        if image is None:
            raise ValueError("Input image cannot be None")
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")
        if len(image.shape) not in (2, 3):
            raise ValueError("Input image must be 2D (grayscale) or 3D (RGB/BGR/RGBA)")
        if len(image.shape) == 3 and image.shape[2] not in (3, 4):
            raise ValueError("3D image must have 3 (RGB/BGR) or 4 (RGBA) channels")
        
        # 图像格式处理：确保3通道RGB格式
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # BGR转RGB（OpenCV默认读取为BGR）
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                # RGBA转RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:  # 单通道灰度图
            # 灰度图转RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 进行推理
        results = self.model(image, verbose=False)
        
        # 解析结果（包含置信度过滤）
        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                # 过滤低置信度结果
                if confidence < self.conf_threshold:
                    continue
                class_name = self.class_names[cls_id]
                predictions.append({
                    "label": class_name,
                    "confidence": confidence,
                    "bbox": box.xyxy[0].tolist()
                })
        
        return predictions