from pathlib import Path

import cv2
import pytest


def test_dbnet():    

    from telos import DBNet

    model_path = "detection/det_text.onnx"

    # Initialize YOLOv8 object detector
    model = DBNet(model_path,labels=["text"])
    img = cv2.imread("./tests/test_img/test_dbnet.png")
    result = model(img)
    # assert result == None,print(result)

    # Draw detections
    combined_img = model.draw_detections(img,mask_alpha=0.2)
    cv2.imwrite("output-dbnet.jpg", combined_img)



def test_yolov8():    
    from telos import YOLOv8
    
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
    # Initialize YOLOv8 object detector
    model = YOLOv8(model_path, labels=labels,conf_thres=0.3, iou_thres=0.5)

    # img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
    # img = imread_from_url(img_url)
    img = cv2.imread("./tests/test_img/page_p6.png")

    # Detect Objects
    model(img)

    # Draw detections
    combined_img = model.draw_detections(img)
    cv2.imwrite("output-yolov8.jpg", combined_img)

def test_lore():    

    from telos import Lore

    model_path = "detection"

    model = Lore(model_path)
    img = cv2.imread("./tests/test_img/test_lore.jpg")
    result = model(img)
    show_img = model.visual(img)
    
    cv2.imwrite("output-lore.jpg", show_img)    

def test_lore_ocr():    

    from telos import Lore,OCR
    
    ocr = OCR()
    model_path = "detection"

    model = Lore(model_path)
    img = cv2.imread("./tests/test_img/test_lore.jpg")
    result = model(img)

    res_table = model.post_process_4ocr(img, result,ocr)
    with open(f"output-lore-ocr.html", "w", encoding="utf-8") as f:
        f.write(res_table)