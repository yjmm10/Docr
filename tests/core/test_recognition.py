import cv2
import pytest


def test_crnnnet():
    import cv2

    from telos import CRNN

    model_path = "recognition/rec_text"

    model = CRNN(model_path)
    img = cv2.imread("./tests/test_img/test_crnnnet.png")
    result = model([img])
    assert list(result[0]) == ["CodeGeeX 智能编程助手"]

    result = model(img)
    assert result[0] == ["CodeGeeX 智能编程助手"]
