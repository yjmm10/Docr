import platform
import time

import cv2
import numpy as np
import psutil
import streamlit as st

from docly import (CRNN, OCR, DBNet, DetFormula, LatexOCR, Layout,
                   ReadingOrder, Table_TSR, YOLOv8)

st.set_page_config(layout="wide")
st.title("docly 演示")


@st.cache_data
def load_image(image_file):
    img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_hardware_info():
    cpu_info = f"CPU: {platform.processor()} ({psutil.cpu_count(logical=False)} 核心, {psutil.cpu_count()} 线程)"
    memory_info = f"内存: {psutil.virtual_memory().total / (1024**3):.2f} GB"
    return f"{cpu_info}\n{memory_info}"


col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])

with col2:
    st.sidebar.header("硬件信息")
    st.sidebar.text(get_hardware_info())

    st.sidebar.markdown("---")
    st.sidebar.header("模型选择")
    selected_model = st.sidebar.selectbox(
        "选择一个模型",
        [
            "YOLOv8",
            "Layout",
            "DetFormula",
            "LatexOCR",
            "DBNet",
            "CRNN",
            "OCR",
            "ReadingOrder",
            "Table_TSR",
        ],
    )

    # 添加模型说明
    model_descriptions = {
        "YOLOv8": "用于目标检测的快速准确模型",
        "Layout": "文档布局分析模型",
        "DetFormula": "数学公式检测模型",
        "LatexOCR": "LaTeX公式识别模型",
        "DBNet": "文本检测模型",
        "CRNN": "文本识别模型",
        "OCR": "光学字符识别模型",
        "ReadingOrder": "文档阅读顺序分析模型",
        "Table_TSR": "表格结构识别模型",
    }

    st.sidebar.markdown(f"**模型说明：** {model_descriptions[selected_model]}")

    st.sidebar.markdown("---")
    st.sidebar.header("模型参数")

    # 定义每个模型的参数
    model_params = {
        "YOLOv8": ["conf_thres", "iou_thres"],
        "Layout": ["conf_thres", "iou_thres"],
        "DetFormula": ["conf_thres", "iou_thres"],
        "LatexOCR": ["max_length"],
        "DBNet": ["conf_thres", "iou_thres"],
        "CRNN": ["batch_size"],
        "OCR": ["conf_thres", "iou_thres"],
        "ReadingOrder": ["min_area"],
        "Table_TSR": ["max_rows", "max_cols"],
    }

    # 只有当模型有参数时才显示参数滑块
    if selected_model in model_params:
        for param in model_params[selected_model]:
            if param == "conf_thres" or param == "iou_thres":
                globals()[param] = st.sidebar.slider(
                    f"{param.replace('_', ' ').title()}",
                    0.0,
                    1.0,
                    0.3,
                    key=f"{selected_model}_{param}",
                )
            elif param == "max_length":
                globals()[param] = st.sidebar.slider(
                    "最大长度", 50, 300, 150, key="LatexOCR_max_length"
                )
            elif param == "batch_size":
                globals()[param] = st.sidebar.slider(
                    "批处理大小", 1, 32, 8, key="CRNN_batch_size"
                )
            elif param == "min_area":
                globals()[param] = st.sidebar.slider(
                    "最小区域", 0, 1000, 100, key="ReadingOrder_min_area"
                )
            elif param == "max_rows" or param == "max_cols":
                globals()[param] = st.sidebar.slider(
                    f"最大{'行' if param == 'max_rows' else '列'}数",
                    10,
                    100,
                    50,
                    key=f"Table_TSR_{param}",
                )

    # 添加推理时间显示
    inference_time_placeholder = st.sidebar.empty()

if uploaded_file is not None:
    image = load_image(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="原图", use_column_width=True)

    with col2:
        start_time = time.time()

        if selected_model == "YOLOv8":
            model_path = "detection/yolov8n_cdla.onnx"
            labels = [
                "Header",
                "Text",
                "Reference",
                "Figure caption",
                "Figure",
                "Table caption",
                "Table",
                "Title",
                "Footer",
                "Equation",
            ]
            model = YOLOv8(
                model_path, labels=labels, conf_thres=conf_thres, iou_thres=iou_thres
            )
            result = model(image)
            result_image = model.draw_detections(image, mask_alpha=0.2)

        elif selected_model == "Layout":
            model = Layout(conf_thres=conf_thres, iou_thres=iou_thres)
            result = model(image)
            result_image = model.draw_detections(image, mask_alpha=0.2)

        elif selected_model == "DetFormula":
            model = DetFormula(conf_thres=conf_thres, iou_thres=iou_thres)
            result = model(image)
            result_image = model.draw_detections(image, mask_alpha=0.2)

        elif selected_model == "LatexOCR":
            model = LatexOCR(
                model_path="docly/models/recognition/rec_formula", max_length=max_length
            )
            result = model(image)
            result_image = image  # LatexOCR doesn't modify the image

        elif selected_model == "DBNet":
            model_path = "detection/det_text.onnx"
            model = DBNet(
                model_path, labels=["text"], conf_thres=conf_thres, iou_thres=iou_thres
            )
            result = model(image)
            result_image = model.draw_detections(image, mask_alpha=0.2)

        elif selected_model == "CRNN":
            model = CRNN(model_path="recognition/rec_text", batch_size=batch_size)
            result = model(image)
            result_image = image  # CRNN doesn't modify the image

        elif selected_model == "OCR":
            model = OCR(conf_thres=conf_thres, iou_thres=iou_thres)
            result = model(image)
            result_image = image  # OCR doesn't modify the image

        elif selected_model == "ReadingOrder":
            det = Layout(conf_thres=conf_thres, iou_thres=iou_thres)
            bboxs, scores, class_id = det(image)
            model = ReadingOrder(min_area=min_area)
            result = model(image, bboxs)
            result_image = image  # ReadingOrder doesn't modify the image

        elif selected_model == "Table_TSR":
            model = Table_TSR(max_rows=max_rows, max_cols=max_cols)
            result = model(image)
            result_image = image  # Table_TSR doesn't modify the image

        end_time = time.time()
        inference_time = end_time - start_time

        st.image(result_image, caption="处理后的图片", use_column_width=True)

        # 更新侧边栏中的推理时间
        inference_time_placeholder.markdown(f"**推理时间：** {inference_time:.4f} 秒")

    st.markdown("---")
    st.subheader("处理结果")

    result_container = st.container()
    with result_container:
        if selected_model in ["LatexOCR", "CRNN", "OCR"]:
            st.text(f"识别结果: {result}")
        elif selected_model == "ReadingOrder":
            st.text(f"阅读顺序: {result}")
        elif selected_model == "Table_TSR":
            st.text("表格结构识别结果:")
            st.code(result, language="html")
        elif isinstance(result, (list, tuple, dict)):
            st.json(result)

    # 添加滚动条
    st.markdown(
        """
        <style>
            .stContainer {
                max-height: 300px;
                overflow-y: auto;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
