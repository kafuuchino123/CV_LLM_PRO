import streamlit as st
import cv2
import numpy as np
from PIL import Image
from cv_model import CVModel
from llm_model import LLMModel
import hashlib
import torch

# 模型初始化缓存，避免重复加载
@st.cache_resource
def get_cv_model():
    return CVModel()

@st.cache_resource
def get_llm_model():
    return LLMModel()

# 初始化模型（使用缓存的模型实例）
cv_model = get_cv_model()
llm_model = get_llm_model()

st.set_page_config(page_title="CV + LLM Image Analysis", layout="wide")

# 在模型初始化后添加
st.sidebar.subheader("系统信息")
if torch.cuda.is_available():
    st.sidebar.success(f"GPU可用: {torch.cuda.get_device_name(0)}")
    st.sidebar.write(f"CUDA版本: {torch.version.cuda}")
else:
    st.sidebar.warning("未检测到可用GPU，将使用CPU运行")

st.title("CV + LLM 图像分析系统")
st.markdown("上传一张图片，系统将识别图像内容并生成自然语言描述")

# 上传图片
uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

# 缓存模型推理结果，优化性能
@st.cache_data(hash_funcs={np.ndarray: lambda x: hashlib.md5(x).hexdigest()})
def run_cv_inference(image_array, _model):  # 参数名前添加下划线
    return _model.predict(image_array)  # 同时修改内部使用

@st.cache_data
def run_llm_generation(predictions, _model):
    return _model.generate_description(predictions)

def get_simple_position(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox  # YOLO输出的bbox是「左上角x,y → 右下角x,y」
    # 计算bbox中心点（判断物体在图像中的大致区域）
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # 横向位置（左/中/右）
    if center_x < img_width / 3:
        hori_pos = "左侧"
    elif center_x < img_width * 2 / 3:
        hori_pos = "中间"
    else:
        hori_pos = "右侧"
    
    # 纵向位置（上/中/下）
    if center_y < img_height / 3:
        vert_pos = "上方"
    elif center_y < img_height * 2 / 3:
        vert_pos = "中间"
    else:
        vert_pos = "下方"
    
    # 组合成自然位置（如“中间中间”→“中央”，避免重复）
    if hori_pos == "中间" and vert_pos == "中间":
        return "中央"
    return f"{vert_pos}{hori_pos}"

if uploaded_file is not None:
    try:
        # 显示上传的图片
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", use_container_width=True)
        
        # 转换为OpenCV格式并处理通道问题
        image_cv = np.array(image)
        # 处理4通道(含alpha通道)图像，转换为3通道
        if image_cv.ndim == 3 and image_cv.shape[-1] == 4:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGBA2RGB)
        img_height, img_width = image_cv.shape[:2]  # 新增：获取图像宽高
        
        # 使用CV模型进行推理（带缓存）
        try:
            predictions = run_cv_inference(image_cv, cv_model)
            for pred in predictions:
                pred["position"] = get_simple_position(pred["bbox"], img_width, img_height)
        except Exception as e:
            st.error(f"CV模型推理失败: {str(e)}")
            predictions = None
        
        # 显示CV结果
        st.subheader("CV模型预测结果")
        if predictions:
            st.write("检测到以下目标：")
            for i, pred in enumerate(predictions):
                st.write(f"目标 {i+1}: {pred['label']} (置信度: {pred['confidence']:.2f})")
        else:
            st.write("未检测到任何目标")
        
        # 生成LLM描述（带缓存）
        if predictions:
            try:
                description = run_llm_generation(predictions, llm_model)
                st.subheader("LLM生成的描述")
                st.write(description)
            except Exception as e:
                st.error(f"LLM生成描述失败: {str(e)}")
        else:
            st.write("没有检测到目标，无法生成描述")
            
        # 显示检测框
        if predictions:
            st.subheader("检测结果可视化")
            try:
                # 在原始图像上绘制边界框
                image_with_boxes = image_cv.copy()
                h, w = image_with_boxes.shape[:2]  # 获取图像尺寸
                
                for pred in predictions:
                    # 解析边界框并确保坐标有效
                    x1, y1, x2, y2 = map(int, pred['bbox'])
                    # 边界检查，防止坐标超出图像范围
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image_with_boxes, 
                        f"{pred['label']} ({pred['confidence']:.2f})", 
                        (x1, max(10, y1-10)),  # 防止文本超出顶部边界
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        2
                    )
            
                st.image(image_with_boxes, caption="检测结果", use_container_width=True)
            except Exception as e:
                st.error(f"绘制检测框失败: {str(e)}")
                
    except Exception as e:
        st.error(f"处理图像时发生错误: {str(e)}")