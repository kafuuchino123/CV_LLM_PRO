import streamlit as st
import cv2
import numpy as np
from PIL import Image
from cv_model import CVModel
from llm_model import LLMModel

# 初始化模型
cv_model = CVModel()
llm_model = LLMModel()

st.set_page_config(page_title="CV + LLM Image Analysis", layout="wide")

st.title("CV + LLM 图像分析系统")
st.markdown("上传一张图片，系统将识别图像内容并生成自然语言描述")

# 上传图片
uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示上传的图片
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图片", use_column_width=True)
    
    # 转换为OpenCV格式
    image_cv = np.array(image)
    
    # 使用CV模型进行推理
    predictions = cv_model.predict(image_cv)
    
    # 显示CV结果
    st.subheader("CV模型预测结果")
    if predictions:
        st.write("检测到以下目标：")
        for i, pred in enumerate(predictions):
            st.write(f"目标 {i+1}: {pred['label']} (置信度: {pred['confidence']:.2f})")
    else:
        st.write("未检测到任何目标")
    
    # 生成LLM描述
    if predictions:
        description = llm_model.generate_description(predictions)
        st.subheader("LLM生成的描述")
        st.write(description)
    else:
        st.write("没有检测到目标，无法生成描述")
    
    # 显示检测框
    if predictions:
        st.subheader("检测结果可视化")
        # 在原始图像上绘制边界框
        image_with_boxes = image_cv.copy()
        for pred in predictions:
            x1, y1, x2, y2 = map(int, pred['bbox'])
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, f"{pred['label']} ({pred['confidence']:.2f})", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        st.image(image_with_boxes, caption="检测结果", use_column_width=True)