"""docly"""

__version__ = "0.0.3"
__project__ = "docly"

from docly.config import *

# from docly.utils import check_source
from docly.core import CRNN, CVModel, DBNet, LatexOCR, Lore, ReadingOrder, YOLOv8
from docly.data import IMG_FORMATS, MetaFile
from docly.modules import OCR, DetFormula, Layout, Table_TSR

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
    "Lore",
    "CRNN",
    "OCR",
    "ReadingOrder",
    "Table_TSR",
)
