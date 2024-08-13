import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from telos import (CRNN, OCR, DBNet, DetFormula, LatexOCR, Layout,
                   ReadingOrder, Table_TSR, YOLOv8)

app = FastAPI()


@app.post("/yolov8")
async def yolov8_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
    model = YOLOv8(model_path, labels=labels, conf_thres=0.3, iou_thres=0.5)
    boxes, scores, class_ids = model(img)

    result = {
        "boxes": boxes.tolist(),
        "scores": scores.tolist(),
        "class_ids": class_ids.tolist(),
    }
    return JSONResponse(content=result)


@app.post("/layout")
async def layout_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model = Layout(conf_thres=0.3, iou_thres=0.5)
    result = model(img)
    result_T = model._telos()

    return JSONResponse(content={"result": result, "result_T": result_T})


@app.post("/formula_detection")
async def formula_detection_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model = DetFormula(conf_thres=0.3, iou_thres=0.5)
    result = model(img)
    result_T = model._telos()

    return JSONResponse(content={"result": result, "result_T": result_T})


@app.post("/latex_ocr")
async def latex_ocr_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    engine = LatexOCR(model_path="telos/models/recognition/rec_formula")
    result = engine(img)

    return JSONResponse(content={"result": result})


@app.post("/dbnet")
async def dbnet_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model_path = "detection/det_text.onnx"
    model = DBNet(model_path, labels=["text"])
    result = model(img)

    return JSONResponse(content={"result": result})


@app.post("/crnn")
async def crnn_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model = CRNN(model_path="recognition/rec_text")
    result = model([img])

    return JSONResponse(content={"result": result})


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model = OCR()
    result = model(img)

    return JSONResponse(content={"result": result})


@app.post("/reading_order")
async def reading_order_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    det = Layout(conf_thres=0.3, iou_thres=0.5)
    bboxs, scores, class_id = det(img)

    model = ReadingOrder()
    result = model(img, bboxs)

    return JSONResponse(content={"result": result})


@app.post("/table_tsr")
async def table_tsr_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model = Table_TSR()
    result = model(img)

    return JSONResponse(content={"result": result})


if __name__ == "__main__":
    import uvicorn

    # 定义你的模型路径和字典路径
    uvicorn.run(
        "telos_api:app", host="0.0.0.0", port=8000, reload=True, timeout_keep_alive=30
    )
