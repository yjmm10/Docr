import cv2
import pytest


def test_formula_det():

    from docr import DetFormula

    model = DetFormula(conf_thres=0.3, iou_thres=0.5)
    img = cv2.imread("tests/test_img/formula_page0.jpg")

    # Detect Objects
    result = model(img)
    # print(result)
    result_T = model._docr()
    print(result_T)

    # Draw detections
    combined_img = model.draw_detections(img, mask_alpha=0.2)
    cv2.imwrite("tests/output/output-formula-det.jpg", combined_img)


def test_latexocr():
    import os

    from docr import LatexOCR

    # assert os.getcwd()==""

    engine = LatexOCR(
        model_path="docr/models/recognition/rec_formula",
    )
    img = cv2.imread("tests/test_img/formula01.png")
    result = engine(img)
    assert result == "x^{2}+y^{2}=1"
