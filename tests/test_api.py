from fastapi.testclient import TestClient
from api.docly_api import app
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2

client = TestClient(app)

@pytest.fixture
def mock_cv2_imread():
    with patch('cv2.imdecode') as mock:
        mock.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        yield mock

@pytest.fixture
def mock_ocr():
    with patch('api.docly_api.OCR') as mock:
        yield mock

@pytest.fixture
def mock_layout():
    with patch('api.docly_api.Layout') as mock:
        yield mock

@pytest.fixture
def mock_reading_order():
    with patch('api.docly_api.ReadingOrder') as mock:
        yield mock

@pytest.fixture
def mock_table_tsr():
    with patch('api.docly_api.Table_TSR') as mock:
        yield mock

@pytest.fixture
def mock_yolov8():
    with patch('api.docly_api.YOLOv8') as mock:
        yield mock

@pytest.fixture
def mock_detformula():
    with patch('api.docly_api.DetFormula') as mock:
        yield mock

@pytest.fixture
def mock_latexocr():
    with patch('api.docly_api.LatexOCR') as mock:
        yield mock

@pytest.fixture
def mock_dbnet():
    with patch('api.docly_api.DBNet') as mock:
        yield mock

@pytest.fixture
def mock_crnn():
    with patch('api.docly_api.CRNN') as mock:
        yield mock

def test_ocr_endpoint(mock_ocr, mock_cv2_imread):
    mock_ocr.return_value.return_value = "Mocked OCR result"
    response = client.post("/ocr", files={"file": ("filename", b"file content", "image/jpeg")})
    assert response.status_code == 200
    assert response.json() == {"result": "Mocked OCR result"}

def test_reading_order_endpoint(mock_layout, mock_reading_order, mock_cv2_imread):
    mock_layout.return_value.return_value = ([], [], [])
    mock_reading_order.return_value.return_value = "Mocked reading order result"
    response = client.post("/reading_order", files={"file": ("filename", b"file content", "image/jpeg")})
    assert response.status_code == 200
    assert response.json() == {"result": "Mocked reading order result"}

def test_table_tsr_endpoint(mock_table_tsr, mock_cv2_imread):
    mock_table_tsr.return_value.return_value = "Mocked table TSR result"
    response = client.post("/table_tsr", files={"file": ("filename", b"file content", "image/jpeg")})
    assert response.status_code == 200
    assert response.json() == {"result": "Mocked table TSR result"}

def test_yolov8_endpoint(mock_yolov8, mock_cv2_imread):
    mock_yolov8.return_value.return_value = "Mocked YOLOv8 result"
# Add more tests for other endpoints
