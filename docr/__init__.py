"""docr"""

__version__ = "0.0.3"
__project__ = "docr"

from docr.config import *

# from docr.utils import check_source
from docr.core import CRNN, CVModel, DBNet, LatexOCR, Lore, ReadingOrder, YOLOv8
from docr.data import IMG_FORMATS, MetaFile
from docr.modules import OCR, DetFormula, Layout, Table_TSR

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
