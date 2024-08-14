from docr.config import layout
from docr.core import YOLOv8

from .base import DataWrap


class Layout(DataWrap, YOLOv8):
    def __init__(self, **params):
        super().__init__(
            model_path=layout["model_path"], labels=layout["labels"], **params
        )
