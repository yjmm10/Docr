"""telos"""
__version__ = '0.0.1'
__project__ = "telos"

from telos.config import *
# from telos.utils import check_source
from telos.core import CRNN, CVModel, DBNet, LatexOCR, ReadingOrder, YOLOv8,Lore
from telos.data import IMG_FORMATS, MetaFile
from telos.modules import OCR, DetFormula, Layout

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
    
)