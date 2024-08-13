import cv2
import pytest


def test_xy_cut():
    from telos import ReadingOrder

    model = ReadingOrder()
    bboxes = [
        [847, 1155, 1460, 1558],
        [848, 1640, 1460, 1966],
        [194, 205, 804, 911],
        [849, 204, 1460, 1058],
        [849, 2042, 1460, 2144],
        [193, 953, 806, 2000],
        [850, 1592, 1231, 1620],
        [850, 1094, 1081, 1124],
        [851, 2000, 1430, 2026],
        [883, 354, 1405, 381],
        [195, 2034, 806, 2145],
    ]
    img = cv2.imread("tests/test_img/layout3.jpg")
    result = model(img, bboxes)
    model.draw_result(img)

    assert result == [2, 5, 3, 9, 7, 0, 6, 1, 8, 10, 4]
