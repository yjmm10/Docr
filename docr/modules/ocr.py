from typing import List, Tuple, Union

import cv2
import numpy as np

from docr.config import det_text, rec_text
from docr.core import CRNN, DBNet

from .base import DataWrap


class OCR(DBNet, CRNN):
    def __init__(self, **params):
        self.class_names = params.get("labels", ["text"])
        self.dbnet = DBNet(
            model_path=det_text["model_path"], labels=self.class_names, **params
        )
        self.crnn = CRNN(model_path=rec_text["model_path"], **params)

        self.drop_score = params.get("drop_score", 0.5)

    def __call__(
        self, image: np.ndarray, use_det=True, use_cls=True, use_rec=True
    ) -> List[Union[List, Tuple]]:
        if not use_det:
            rec_result = self.crnn(image)
            if rec_result is not None:
                self.result = rec_result
                return rec_result
            else:
                return None

        res_bbox = self.dbnet(image)
        assert res_bbox is not None, "detect text failed"
        # filter_boxes, filter_rec_res = [], []
        res_texts = []
        # res_scores = []
        boxes, scores, class_ids = res_bbox
        for bbox, score, label in zip(boxes, scores, class_ids):
            # for bbox,score,label in zip(res_bbox[0],res_bbox[1],res_bbox[2]):
            crop_img = self.get_rotate_crop_image(image, bbox)
            rec_result = self.crnn(crop_img)
            if rec_result is not None:
                text, score = rec_result
                for t, s in zip(text, score):
                    if s >= self.drop_score:
                        res_texts.append([t, s])
                    else:
                        res_texts.append(["", s])
        self.result = [res_bbox[0], res_bbox[1], res_bbox[2], res_texts]
        return self.result

    def get_rotate_crop_image(self, img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        # assert len(points) == 4, "shape of points must be 4*2"
        if len(np.array(points).shape) == 1:
            x0, y0, x1, y1 = points
            points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
