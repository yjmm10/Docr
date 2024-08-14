from docly.config import det_ts
from docly.core import Lore
from docly.modules import OCR

# from .base import DataWrap
# from doclydata import MetaFile,MetaArea,MetaBbox,MetaLayout,MetaText


# 表格结构检测
class Table_TSR(Lore):
    def __init__(self, type="wireless", **params):
        super().__init__(model_path=det_ts["model_path"], **params)
        self.ocr = OCR()

    def __call__(self, img):
        sorted_polygons = super().__call__(img)
        table_str = self.post_process_4ocr(img, sorted_polygons, self.ocr)
        return table_str
