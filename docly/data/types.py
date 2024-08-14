import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import fitz
import numpy as np
from PIL import Image
from pydantic import BaseModel

IMG_FORMATS = {
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
}  # image suffixes
PDF_FORMATS = {"pdf"}  # pdf suffixes
DOC_FORMATS = {"IMAGE", "SCAN", "PDF"}  # document types


class MetaFile:
    def __init__(self, file_path: str = ""):
        self.file_path = Path(file_path).resolve()
        assert self.file_path.exists(), f"file {file_path} not exist"

        self.file_name = self.file_path.stem
        self.file_extension = self.file_path.suffix

        file_stat = self.file_path.stat()
        self.file_size = file_stat.st_size
        self.file_created_time = datetime.datetime.fromtimestamp(
            file_stat.st_ctime
        ).strftime("%Y.%m.%d %H:%M:%S")
        self.file_modified_time = datetime.datetime.fromtimestamp(
            file_stat.st_mtime
        ).strftime("%Y.%m.%d %H:%M:%S")
        # 获取文件的类型以及原数据
        self.file_types = []
        self.pages = []

        # 文件处理
        if self.file_extension[1:].lower() in IMG_FORMATS:
            self.file_types.append("IMAGE")
            self.pages.append(cv2.imread(file_path))
        elif self.file_extension[1:].lower() in PDF_FORMATS:
            doc = fitz.open(self.file_path)
            for page in doc:
                if page.get_text():
                    self.file_types.append("PDF")
                    self.pages.append(page)
                else:
                    self.file_types.append("SCAN")
                    self.pages.append(cv2.imread(file_path))

        else:
            print(f"not support this type: {self.file_extension[1:]}")


# 文本内容属性 -> 识别模块输出
class MetaText(BaseModel):
    text: Union[str, None] = None
    score: float = -1.0


# 文本框属性 -> 文本检测模块
class MetaBbox(BaseModel):
    id: int
    bbox: List[Union[float, int]]  # [x1,y1,x2,y2]
    score: float = 1.0  # 置信度
    label_id: int = -1  # 标签id，-1为文本
    label: str = "text"  # 标签名
    order: int = -1  # 排序
    content: Union[MetaText, None] = None  # 段、句文本
    # detail: Union[Any,List[Any],None] = None


# 版面区域属性 -> 版面区域检测模块
class MetaArea(MetaBbox):
    detail: Union[List[MetaBbox], None] = None

    def get_content(self):
        # 处理行之间的逻辑
        for part in self.detail:
            if isinstance(part, MetaBbox):
                if self.content is None:
                    self.content = ""
                if part.content.text[-1] in ["-", "—"]:
                    self.content += part.content.text
                else:
                    self.content += " " + part.content.text


# 单个页面属性 -> 版面分析结果
class MetaLayout(BaseModel):
    page: int = -1  # 页码
    info: Any = None
    layout: List[Union[MetaArea, None]] = None

    # 筛选指定标签的meta
    def filter_label(self, label: Union[int, str] = -1) -> List[MetaArea]:
        filtered_metas = []
        for meta in self.layout:
            if isinstance(label, int) and meta.label_id == label:
                filtered_metas.append(meta)
            elif isinstance(label, str) and meta.label == label:
                filtered_metas.append(meta)
        return filtered_metas

    # 按顺序获取bbox
    def get_bbox(self):
        bboxs = []
        for metabbox in self.layout:
            bboxs.append(metabbox.bbox)
        return bboxs
