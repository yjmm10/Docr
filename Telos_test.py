import telos
from telos import YOLOv8

print(telos.__version__)
# print(telos.check_source())
# result = telos.MetaFile(file_path="demo_1.png")
# print(result)
# result = telos.MetaFile(file_path="test_img/PDF.pdf")
# print(result)
# result = telos.MetaFile(file_path="test_img/SCAN.pdf")
# print(result)

# result = telos.CVModel("telos")
# print(result)

def test_yolo():
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
    model = YOLOv8(model_path, labels=labels,conf_thres=0.3, iou_thres=0.5)

    # img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
    # img = imread_from_url(img_url)
    img = cv2.imread("test_img/page_p6.png")

    # Detect Objects
    model(img)

    # Draw detections
    combined_img = model.draw_detections(img,mask_alpha=0.2)
    cv2.imwrite("output1.jpg", combined_img)



def test_layout():
    import cv2

    from telos import Layout
   
    model = Layout(conf_thres=0.3, iou_thres=0.5)
    img = cv2.imread("test_img/page_p6.png")

    # Detect Objects
    result = model(img)
    result_T = model._telos()
    # print(result_T)

    # Draw detections
    combined_img = model.draw_detections(img,mask_alpha=0.2)
    cv2.imwrite("output-layout.jpg", combined_img)

def test_formula():
    import cv2

    from telos import DetFormula
   
    model = DetFormula(conf_thres=0.3, iou_thres=0.5)
    img = cv2.imread("test_img/formula_page0.jpg")

    # Detect Objects
    result= model(img)
    # print(result)
    result_T = model._telos()
    print(result_T)

    # Draw detections
    combined_img = model.draw_detections(img,mask_alpha=0.2)
    cv2.imwrite("output-formula.jpg", combined_img)

def test_latexocr():
    import cv2

    from telos import LatexOCR

    engine = LatexOCR(
        model_path="telos/models/recognition/rec_formula",
    )
    img = cv2.imread("/home/zyj/project/MOP/test_img/formula01.png")
    result = engine(img)
    # print(result)
    # model = LatexOCR(model_path=)
    # img = cv2.imread("test_img/formula_page0.jpg")

    # # Detect Objects
    # result= model(img)
    # result_T = model._telos()
    # print(result_T)

    # # Draw detections
    # combined_img = model.draw_detections(img,mask_alpha=0.2)
    # cv2.imwrite("output-formula.jpg", combined_img)
def test_dbnet():
    import cv2

    from telos import DBNet
    model_path = "detection/det_text.onnx"

    # Initialize YOLOv8 object detector
    model = DBNet(model_path,labels=["text"])
    img = cv2.imread("/home/zyj/project/MOP/test_img/page_p0.png")
    result = model(img)
    # print(result)

    # Draw detections
    combined_img = model.draw_detections(img,mask_alpha=0.2)
    cv2.imwrite("output-text.jpg", combined_img)

def test_crnnnet():
    import cv2

    from telos import CRNN
    model_path = "/home/zyj/project/MOP/telos/models/recognition/rec_text"

    # Initialize YOLOv8 object detector
    model = CRNN(model_path="recognition/rec_text")
    img = cv2.imread("/home/zyj/project/MOP/test_img/page_p0.png")
    result = model([img]*3)
    print(result)

    result = model(img)
    print(result)
    # # Draw detections
    # combined_img = model.draw_detections(img,mask_alpha=0.2)
    # cv2.imwrite("output-text.jpg", combined_img)

def test_OCR():
    import cv2

    from telos import OCR

    # Initialize YOLOv8 object detector
    model = OCR()
    img = cv2.imread("/nas/projects/Github/Telos/tests/test_img/test_ocr.png")
    result = model(img)
    print(result)
    # res_telos = model._telos()
    # print(model._json())
    # print(result)
    # print(res_telos)

    # # Draw detections
    # combined_img = model.draw_detections(img,mask_alpha=0.2)
    # cv2.imwrite("output-text.jpg", combined_img)

def test_OCR_wo_det():
    import cv2

    from telos import OCR

    # Initialize YOLOv8 object detector
    model = OCR()
    img = cv2.imread("/nas/projects/Github/Telos/tests/test_img/test_crnnnet.png")
    result = model(img,use_det=False)
    
    # TODO:可视化
    res_telos = model._telos()
    print(model._json())

def test_reading_order():
    import cv2

    from telos import OCR, DBNet, Layout, ReadingOrder

    # Initialize YOLOv8 object detector
    det = Layout(conf_thres=0.3, iou_thres=0.5)
    # det = OCR()
    img = cv2.imread("tests/test_img/layout3.jpg")

    # Detect Objects
    result = det(img)
    bboxs,scores,class_id = result
    # print(result_T)

    # # Draw detections
    # combined_img = model.draw_detections(img,mask_alpha=0.2)
    # cv2.imwrite("output-layout.jpg", combined_img)

    model = ReadingOrder()
    # img = cv2.imread("/home/zyj/project/MOP/test_img/page_p0.png")
    result = model(img,bboxs)
    # print(model._json())
    print(result)
    # print(res_telos)

    # # Draw detections
    # combined_img = model.draw_detections(img,mask_alpha=0.2)
    # cv2.imwrite("output-text.jpg", combined_img)

def test_Table():
    import cv2

    from telos import Table_TSR

    # Initialize YOLOv8 object detector
    model = Table_TSR()
    img = cv2.imread("/nas/projects/Github/Telos/tests/test_img/test_lore.jpg")
    result = model(img)
    print(result)
    # res_telos = model._telos()
    # print(model._json())
    # print(result)
    # print(res_telos)

    # # Draw detections
    # combined_img = model.draw_detections(img,mask_alpha=0.2)
    # cv2.imwrite("output-text.jpg", combined_img)


if __name__ == "__main__":
    # test_yolo()
    # test_layout()
    # test_formula()
    # test_latexocr()
    # test_dbnet()
    # test_crnnnet()
    # test_OCR()
    # test_OCR_wo_det()

    # test_reading_order()
    test_Table()
