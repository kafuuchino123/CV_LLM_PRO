import cv2
import numpy as np
from ultralytics import YOLO

class CVModel:
    def __init__(self):
        # 使用YOLOv8n作为基础模型（轻量级且效果好）
        self.model = YOLO('yolov8n.pt')
        self.class_names = self.model.names
        
    def predict(self, image):
        # 确保是RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 进行推理
        results = self.model(image, verbose=False)
        
        # 解析结果
        predictions = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.class_names[cls_id]
                predictions.append({
                    "label": class_name,
                    "confidence": confidence,
                    "bbox": box.xyxy[0].tolist()
                })
        
        return predictions