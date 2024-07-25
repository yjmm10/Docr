from pathlib import Path
from typing import Union,List
import numpy as np
from abc import ABC, abstractmethod

from telos import __project__

from telos.config import __model_path__
from telos.data import MetaFile,MetaArea,MetaBbox,MetaLayout,MetaText

class DataWrap:
    def _telos(self,):
        res_nums = len(self.result)
        if res_nums == 3:
            boxes, scores, class_ids = self.result
            result = []
            ids = list(range(len(boxes)))
            for id,box,score,class_id in zip(ids,boxes, scores, class_ids):
                result.append(MetaArea(id=id,bbox=box,score=score, label = self.class_names[class_id],label_id=class_id))
            return result
        
        elif res_nums == 2:
            texts, scores = self.result
            result = []
            ids = list(range(len(texts)))
            for text, score in zip(texts, scores):
                result.append(MetaText(text=text, score=score))
            return result

        

    