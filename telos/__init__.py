"""telos"""
__version__ = '0.0.1'
__project__ = "telos"

from telos.data import IMG_FORMATS,MetaFile
# from telos.utils import check_source
from telos.core import CVModel,YOLOv8,LatexOCR,DBNet,CRNN,ReadingOrder
from telos.config import *
from telos.modules import Layout,DetFormula,OCR

__all__ = (
    "__version__",
    "__project__",
    "IMG_FORMATS",
    "MetaFile",
    "CVModel",
    "YOLOv8",
    "Layout",
    "DetFormula",
    "LatexOCR",
    "DBNet",
    "CRNN",
    "OCR",
    "ReadingOrder",
)