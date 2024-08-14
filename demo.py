import gradio as gr
import numpy as np

from docly import *


def dla(img, conf, iou, model):
    import cv2

    model_path = "detection/yolov8n_cdla.onnx"

    # Initialize YOLOv8 object detector
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
    model = YOLOv8(model_path, labels=labels, conf_thres=conf, iou_thres=iou)
    # Detect Objects
    res = model(img)

    # Draw detections
    combined_img = model.draw_detections(img, mask_alpha=0.2)
    # print(res)
    # tb_dla_res.change(res)
    return combined_img


def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)


with gr.Blocks() as demo:
    gr.Markdown("Flip text or image files using this demo.")
    with gr.Tab("版式分析"):
        with gr.Row():
            with gr.Column():
                conf = gr.Slider(0, 1, 0.3, label="Conf")
                iou = gr.Slider(0, 1, 0.5, label="Iou")
                btn_dla = gr.Button("版式分析")
            model_name = gr.Dropdown(["cat", "dog", "bird"], label="版式模型")
            # tb_dla_res = gr.Textbox()
    with gr.Row():
        img_dla_in = gr.Image()
        img_dla_out = gr.Image()

    with gr.Accordion("Open for More!", open=False):
        gr.Markdown("Look at me...")
        temp_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.1,
            step=0.1,
            interactive=True,
            label="Slide me",
        )
        temp_slider.change(lambda x: x, [temp_slider])

    btn_dla.click(dla, inputs=[img_dla_in, conf, iou, model_name], outputs=img_dla_out)

if __name__ == "__main__":
    demo.launch()
