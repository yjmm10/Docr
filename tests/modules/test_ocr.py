def test_OCR():
    import cv2

    from telos import OCR

    # Initialize YOLOv8 object detector
    model = OCR()
    img = cv2.imread("tests/test_img/test_ocr.png")
    result = model(img)
    
    # TODO:可视化
    # assert result ==""
    res_telos = model._telos()
    print(model._json())

def test_OCR_wo_det():
    import cv2

    from telos import OCR

    # Initialize YOLOv8 object detector
    model = OCR()
    img = cv2.imread("./tests/test_img/test_crnnnet.png")
    result = model(img,use_det=False)
    assert result[0]==['CodeGeeX 智能编程助手']