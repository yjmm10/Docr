from typing import List, Union

import cv2
import numpy as np
import pyclipper
from numpy.core.multiarray import array as array
from shapely.geometry import Polygon

from telos.core import CVModel
from telos.utils import visual


class DBNet(CVModel):
    def __init__(self, model_path, labels=["text"], **params):
        super().__init__(model_path, labels, **params)

        self.config = params.get(
            "config",
            {
                "limit_side_len": 960,
                "thresh": 0.3,
                "box_thresh": 0.5,
                "max_candidates": 1000,
                "unclip_ratio": 1.5,
                "use_dilation": False,
                "score_mode": "fast",
                "box_type": "quad",
            },
        )

        self.limit_side_len = self.config.get("limit_side_len", 960)
        post_params = {
            "thresh": self.config.get("thresh", 0.3),
            "box_thresh": self.config.get("thresh", 0.5),
            "max_candidates": self.config.get("max_candidates", 1000),
            "unclip_ratio": self.config.get("unclip_ratio", 1.5),
            "use_dilation": self.config.get("use_dilation", False),
            "score_mode": self.config.get("score_mode", "fast"),
            "box_type": self.config.get("box_type", "quad"),
        }
        self.postprocess_op = DBPostProcess(**post_params)

    def pre_process(self, image: np.array):
        self.image_shape = image.shape
        resize_img = self.resize_image(image)
        if resize_img is None:
            return None
        resize_img = self.normalize_image(resize_img)
        resize_img = self.tochwImage(resize_img)

        img = np.expand_dims(resize_img, axis=0)

        self.shape_list = np.expand_dims(self.shape_list, axis=0)
        return img

    def post_process(self, output: np.ndarray) -> List[Union[np.ndarray, List]]:

        post_result = self.postprocess_op(output, self.shape_list)
        dt_boxes = post_result[0]["points"]
        dt_boxes = self.filter_tag_det_res(dt_boxes, self.image_shape)
        scores = post_result[0]["scores"]
        # 排序
        sorted_boxes = self.sorted_boxes(dt_boxes)
        order_index = self.get_sorted_index(dt_boxes, sorted_boxes)
        sorted_scores = [scores[i] for i in order_index]
        # 转左上右下坐标
        sorted_boxes = [self.point2box(points) for points in sorted_boxes]
        self.result = [sorted_boxes, sorted_scores, [0] * len(sorted_boxes)]
        return (
            self.result
        )  # np.array(sorted_boxes),  np.array(sorted_scores),  np.array([0]*len(sorted_boxes))

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    @staticmethod
    def get_sorted_index(a, b):
        b = np.array(b)
        a = np.array(a)
        # 初始化索引数组
        indexes = np.zeros(len(b), dtype=int)

        # 寻找b在a中的索引
        for i, item in enumerate(b):
            # np.argwhere会返回数组中满足条件的所有索引，这里我们是查找完全匹配的子数组
            # 我们在这里只取第一个匹配的索引，因为我们假设a的子数组是唯一的
            index = np.argwhere(np.all(a == item, axis=(1, 2)))[0][0]
            indexes[i] = index
        return indexes.tolist()

    def point2box(self, points):
        """
        input:
            points: [N,4]
        output:
            box: [N,2]
        """
        x = points[:, 0]
        y = points[:, 1]
        x0, x1 = x.min(), x.max()
        y0, y1 = y.min(), y.max()
        return [x0, y0, x1, y1]

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        boxes, scores, class_ids = self.result
        return visual(
            image,
            boxes,
            class_ids,
            class_names=self.class_names,
            scores=scores,
            mask_alpha=mask_alpha,
        )

    def resize_image(self, img: np.ndarray):
        """max"""
        limit_side_len = self.limit_side_len
        src_h, src_w, _ = img.shape
        h, w, c = img.shape
        if max(h, w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        else:
            ratio = 1.0
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            resize_img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except BaseException:
            print(img.shape, resize_w, resize_h)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        self.shape_list = [src_h, src_w, ratio_h, ratio_w]
        return resize_img

    def normalize_image(self, img: np.ndarray):
        scale = 1.0 / 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        shape = (1, 1, 3)
        mean = np.array(mean).reshape(shape).astype("float32")
        std = np.array(std).reshape(shape).astype("float32")
        nor_img = (img.astype("float32") * scale - mean) / std
        return nor_img

    def tochwImage(self, img: np.ndarray):
        return img.transpose((2, 0, 1))

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        if len(image_shape) == 2:
            img_height, img_width = image_shape
        else:
            img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(
        self,
        thresh=0.3,
        box_thresh=0.7,
        max_candidates=1000,
        unclip_ratio=2.0,
        use_dilation=False,
        score_mode="fast",
        box_type="quad",
        **kwargs
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        assert score_mode in [
            "slow",
            "fast",
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        """

        bitmap = _bitmap
        height, width = bitmap.shape

        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours[: self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)

            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        """

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        """
        box_score_fast: use bbox mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        """
        box_score_slow: use polyon mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def __call__(self, pred, shape_list):
        # pred = outs_dict['maps']
        if not isinstance(pred, np.ndarray):
            pred = pred.numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel,
                )
            else:
                mask = segmentation[batch_index]
            if self.box_type == "poly":
                boxes, scores = self.polygons_from_bitmap(
                    pred[batch_index], mask, src_w, src_h
                )
            elif self.box_type == "quad":
                boxes, scores = self.boxes_from_bitmap(
                    pred[batch_index], mask, src_w, src_h
                )
            else:
                raise ValueError("box_type can only be one of ['quad', 'poly']")

            boxes_batch.append({"points": boxes, "scores": scores})
        return boxes_batch


if __name__ == "__main__":
    # from imread_from_url import imread_from_url

    model_path = "/home/zyj/project/MOP/telos/core/detection/yolov8n_cdla.onnx"

    # Initialize YOLOv8 object detector
    dbnet = DBNet(model_path)

    # img = cv2.imread("/home/zyj/project/MOP/test_img/page_p6.png")

    # # Detect Objects
    # yolov8_detector(img)

    # # Draw detections
    # combined_img = yolov8_detector.draw_detections(img)
    # cv2.imwrite("output.jpg", combined_img)
