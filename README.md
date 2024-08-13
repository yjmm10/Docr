English | [中文](README_zh.md) | [日语]()
# Telos 🚀

## 1. Overview 🌟

🛠️ Component design with module-based functionality, allowing for on-demand feature acquisition, 🚀 easy to expand, and flexible to use, just like playing with building blocks!

Telos is a modular component-based toolkit for document analysis and processing. It's designed with flexibility and extensibility in mind, making it easy to expand and use various document processing functionalities as needed.

## 2. Features 🛠️

- 📄 Layout Analysis
- 🔢 Formula Detection and Recognition
- 📝 Optical Character Recognition (OCR)
- 📊 Table Structure Recognition
- 📚 Reading Order Analysis
- 🖼️ Image Processing Utilities

## 3. Installation and Usage 📦

### 3.1 Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)

### 3.2 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yjmm10/telos.git
   cd telos
   git clone https://huggingface.co/liferecords/Telos.git telos/models
   ```

2. Install dependencies:
   ```bash
   poetry install -v
   ```

### 3.3 Usage

Here's a quick example of how to use Telos for OCR:

```python
from telos import OCR
import cv2

# Initialize the OCR model
ocr_model = OCR()

# Read an image
image = cv2.imread("path/to/your/image.png")

# Perform OCR
result = ocr_model(image)

print(result)
```

Telos comes with a Streamlit-based web UI for easy demonstration of its capabilities:

1. Run the demo:
   ```bash
   streamlit run webui/demo.py
   ```

2. Open your browser and navigate to the provided URL (usually http://localhost:8501)

3. Upload an image and select the model you want to use for processing

Telos also provides a FastAPI-based API service for integration into other applications:

1. Start the API server:
   ```bash
   uvicorn api.telos_api:app --host 0.0.0.0 --port 8000
   ```

2. The API documentation will be available at http://localhost:8000/docs

## 4. Development 🔬

For detailed information on development, please refer to the [development guide](./docs/development.md). This guide will help you set up your IDE for working with Telos, including SRC Layout configuration.

## 5. Contributing 🤝

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 6. License 📄

Telos is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 7. Contact 📧

For any questions or feedback, please contact the project maintainer:
liferecords <yjmm10@yeah.net>