from docr.config import det_formula
from docr.core import YOLOv8

# from .base import DataWrap
# from docrdata import MetaFile,MetaArea,MetaBbox,MetaLayout,MetaText


class DetFormula(YOLOv8):
    def __init__(self, **params):
        super().__init__(
            model_path=det_formula["model_path"], labels=det_formula["labels"], **params
        )
