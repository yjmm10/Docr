import argparse
import random
import logging
import time
import warnings
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from telos.core import CVModel,OrtInferSession

import cv2
import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon

from telos.config import __model_path__

# suppress warnings
warnings.filterwarnings("ignore")

class Lore:
    def __init__(
        self,
        model_path, **params
    ):

        self.config = params.get("config",
                        {  "det_name":"lore_detect.onnx",
                            "process_name": "lore_process.onnx", 
                            "input_shape": [768,768],
                            })
        model_path = Path(__model_path__) / model_path

        self.mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)

        self.inp_h, self.inp_w = self.config["input_shape"]
        detect_model_path = Path(model_path) / self.config["det_name"]
        process_model_path = Path(model_path) / self.config["process_name"]

        self.det_session = OrtInferSession(detect_model_path)
        self.process_session = OrtInferSession(process_model_path)

        self.det_process = DetProcess()

    def __call__(self, img) -> str:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        input_info = self.pre_process(img)
        try:
            self.polygons, self.slct_logi = self.infer(input_info)
            sorted_polygons = sorted_boxes(self.polygons)
            self.sorted_polygons = sorted_polygons
            return sorted_polygons

        except Exception:
            logging.warning(traceback.format_exc())
            return "", 0.0

    def post_process_4ocr(self, img, sorted_polygons,ocr) -> List[Dict[str, Any]]:
        
        # 使用ocr引擎
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ocr_res, _ = ocr(img)
        logi_points = self.filter_logi_points(self.slct_logi)
        
        cell_box_map = match_ocr_cell(sorted_polygons, ocr_res)
        cell_box_map = self.re_rec(img, sorted_polygons, cell_box_map,ocr)

        logi_points = self.sort_logi_by_polygons(
            sorted_polygons, self.polygons, logi_points
        )

        table_str = plot_html_table(logi_points, cell_box_map)
        return table_str

    def pre_process(self, img: np.ndarray) -> Dict[str, Any]:
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]
        resized_image = cv2.resize(img, (width, height))

        c = np.array([0, 0], dtype=np.float32)
        s = max(height, width) * 1.0
        trans_input = get_affine_transform_upper_left(c, s, [self.inp_w, self.inp_h])

        inp_image = cv2.warpAffine(
            resized_image, trans_input, (self.inp_w, self.inp_h), flags=cv2.INTER_LINEAR
        )
        inp_image = ((inp_image / 255.0 - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, self.inp_h, self.inp_w)
        meta = {
            "c": c,
            "s": s,
            "out_height": self.inp_h // 4,
            "out_width": self.inp_w // 4,
        }
        return {"img": images, "meta": meta}

    def infer(self, input_content: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        hm, st, wh, ax, cr, reg = self.det_session([input_content["img"]])
        output = {
            "hm": hm,
            "st": st,
            "wh": wh,
            "ax": ax,
            "cr": cr,
            "reg": reg,
        }
        slct_logi_feat, slct_dets_feat, slct_output_dets = self.det_process(
            output, input_content["meta"]
        )

        slct_output_dets = slct_output_dets.reshape(-1, 4, 2)

        _, slct_logi = self.process_session(
            [slct_logi_feat, slct_dets_feat.astype(np.int64)]
        )
        return slct_output_dets, slct_logi

    def filter_logi_points(self, slct_logi: np.ndarray) -> Dict[str, Any]:
        logi_floor = np.floor(slct_logi)
        dev = slct_logi - logi_floor
        slct_logi = np.where(dev > 0.5, logi_floor + 1, logi_floor)
        return slct_logi[0]

    @staticmethod
    def sort_logi_by_polygons(
        sorted_polygons: np.ndarray, polygons: np.ndarray, logi_points: np.ndarray
    ) -> np.ndarray:
        sorted_idx = []
        for v in sorted_polygons:
            loc_idx = np.argwhere(v[0, 0] == polygons[:, 0, 0]).squeeze()
            sorted_idx.append(int(loc_idx))
        logi_points = logi_points[sorted_idx]
        return logi_points

    def re_rec(
        self,
        img: np.ndarray,
        sorted_polygons: np.ndarray,
        cell_box_map: Dict[int, List[str]],
        ocr=None
    ) -> Dict[int, List[str]]:
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        """找到poly对应为空的框，尝试将直接将poly框直接送到识别中"""
        for k, v in cell_box_map.items():
            if v[0]:
                continue

            crop_img = get_rotate_crop_image(img, sorted_polygons[k])
            pad_img = cv2.copyMakeBorder(
                crop_img, 2, 2, 100, 100, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            # if ocr:
            rec_res, _ = ocr(pad_img, use_det=False, use_cls=True, use_rec=True)
            # else:
            #     rec_res, _ = self.ocr(pad_img, use_det=False, use_cls=True, use_rec=True)
            cell_box_map[k] = [rec_res[0][0]]
        return cell_box_map

    def visual(self, img: np.ndarray, polygons: np.ndarray=None) -> np.ndarray:
        if polygons is None:
            polygons = self.sorted_polygons
        for i, poly in enumerate(polygons):
            poly = np.round(poly).astype(np.int32).reshape(4, 2)

            random_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            cv2.polylines(img, [poly], 3, random_color)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(i), poly[0], font, 1, (0, 0, 255), 1)
        return img
  



def sorted_boxes(dt_boxes: np.ndarray) -> np.ndarray:
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape (N, 4, 2)
    return:
        sorted boxes(array) with shape (N, 4, 2)
    """
    num_boxes = dt_boxes.shape[0]
    dt_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(dt_boxes)

    # 解决相邻框，后边比前面y轴小，则会被排到前面去的问题
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if (
                abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10
                and _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                _boxes[j], _boxes[j + 1] = _boxes[j + 1], _boxes[j]
            else:
                break
    return np.array(_boxes)


def compute_poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个多边形的IOU

    Args:
        poly1 (np.ndarray): (4, 2)
        poly2 (np.ndarray): (4, 2)

    Returns:
        float: iou
    """
    poly1 = Polygon(a).convex_hull
    poly2 = Polygon(b).convex_hull

    union_poly = np.concatenate((a, b))

    if not poly1.intersects(poly2):
        return 0.0

    try:
        inter_area = poly1.intersection(poly2).area
        union_area = MultiPoint(union_poly).convex_hull.area
    except shapely.geos.TopologicalError:
        print("shapely.geos.TopologicalError occured, iou set to 0")
        return 0.0

    if union_area == 0:
        return 0.0

    return float(inter_area) / union_area


def merge_adjacent_polys(polygons: np.ndarray) -> np.ndarray:
    """合并相邻iou大于阈值的框"""
    combine_iou_thresh = 0.1
    pair_polygons = list(zip(polygons, polygons[1:, ...]))
    pair_ious = np.array([compute_poly_iou(p1, p2) for p1, p2 in pair_polygons])
    idxs = np.argwhere(pair_ious >= combine_iou_thresh)

    if idxs.size <= 0:
        return polygons

    polygons = combine_two_poly(polygons, idxs)

    # 注意：递归调用
    polygons = merge_adjacent_polys(polygons)
    return polygons


def combine_two_poly(polygons: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    del_idxs, insert_boxes = [], []
    idxs = idxs.squeeze(0)
    for idx in idxs:
        # idx 和 idx + 1 是重合度过高的
        # 合并，取两者各个点的最大值
        new_poly = []
        pre_poly, pos_poly = polygons[idx], polygons[idx + 1]

        # 四个点，每个点逐一比较
        new_poly.append(np.minimum(pre_poly[0], pos_poly[0]))

        x_2 = min(pre_poly[1][0], pos_poly[1][0])
        y_2 = max(pre_poly[1][1], pos_poly[1][1])
        new_poly.append([x_2, y_2])

        # 第3个点
        new_poly.append(np.maximum(pre_poly[2], pos_poly[2]))

        # 第4个点
        x_4 = max(pre_poly[3][0], pos_poly[3][0])
        y_4 = min(pre_poly[3][1], pos_poly[3][1])
        new_poly.append([x_4, y_4])

        new_poly = np.array(new_poly)

        # 删除已经合并的两个框，插入新的框
        del_idxs.extend([idx, idx + 1])
        insert_boxes.append(new_poly)

    # 整合合并后的框
    polygons = np.delete(polygons, del_idxs, axis=0)

    insert_boxes = np.array(insert_boxes)
    polygons = np.append(polygons, insert_boxes, axis=0)
    polygons = sorted_boxes(polygons)
    return polygons


def match_ocr_cell(
    polygons: np.ndarray, ocr_res: List[Tuple[np.ndarray, str, str]]
) -> Dict[int, List]:
    cell_box_map = {}
    dt_boxes, rec_res, _ = list(zip(*ocr_res))
    dt_boxes = np.array(dt_boxes)
    iou_thresh = 0.009
    for i, cell_box in enumerate(polygons):
        ious = [compute_poly_iou(dt_box, cell_box) for dt_box in dt_boxes]

        # 对有iou的值，计算是否存在包含关系。如存在→iou=1
        have_iou_idxs = np.argwhere(ious)
        if have_iou_idxs.size > 0:
            have_iou_idxs = have_iou_idxs.squeeze(1)
            for idx in have_iou_idxs:
                if is_inclusive_each_other(cell_box, dt_boxes[idx]):
                    ious[idx] = 1.0

        if all(x <= iou_thresh for x in ious):
            # 说明这个cell中没有文本
            cell_box_map.setdefault(i, []).append("")
            continue

        same_cell_idxs = np.argwhere(np.array(ious) >= iou_thresh).squeeze(1)
        one_cell_txts = "\n".join([rec_res[idx] for idx in same_cell_idxs])
        cell_box_map.setdefault(i, []).append(one_cell_txts)
    return cell_box_map


def is_inclusive_each_other(box1: np.ndarray, box2: np.ndarray):
    """判断两个多边形框是否存在包含关系

    Args:
        box1 (np.ndarray): (4, 2)
        box2 (np.ndarray): (4, 2)

    Returns:
        bool: 是否存在包含关系
    """
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)

    poly1_area = poly1.convex_hull.area
    poly2_area = poly2.convex_hull.area

    if poly1_area > poly2_area:
        box_max = box1
        box_min = box2
    else:
        box_max = box2
        box_min = box1

    x0, y0 = np.min(box_min[:, 0]), np.min(box_min[:, 1])
    x1, y1 = np.max(box_min[:, 0]), np.max(box_min[:, 1])

    edge_x0, edge_y0 = np.min(box_max[:, 0]), np.min(box_max[:, 1])
    edge_x1, edge_y1 = np.max(box_max[:, 0]), np.max(box_max[:, 1])

    if x0 >= edge_x0 and y0 >= edge_y0 and x1 <= edge_x1 and y1 <= edge_y1:
        return True
    return False


def plot_html_table(logi_points: np.ndarray, cell_box_map: Dict[int, List[str]]) -> str:
    logi_points = logi_points.astype(np.int32)
    table_dict = {}
    for cell_idx, v in enumerate(logi_points):
        cur_row = v[0]
        cur_txt = "\n".join(cell_box_map.get(cell_idx))
        sr, er, sc, ec = v.tolist()
        rowspan, colspan = er - sr + 1, ec - sc + 1
        table_str = f'<td rowspan="{rowspan}" colspan="{colspan}">{cur_txt}</td>'
        # table_str = f'<td rowspan="{rowspan}" colspan="{colspan}"><div style="line-height: 18px;">{cur_txt}</div></td>'
        table_dict.setdefault(cur_row, []).append(table_str)

    new_table_dict = {}
    for k, v in table_dict.items():
        new_table_dict[k] = ["<tr>"] + v + ["</tr>"]

    html_start = """<html><body><table><tbody>"""
    # html_start = """<html><style type="text/css">td {border-left: 1px solid;border-bottom:1px solid;}table, th {border-top:1px solid;font-size: 10px;border-collapse: collapse;border-right: 1px solid;}</style><body><table style="border-bottom:1px solid;border-top:1px solid;"><tbody>"""
    html_end = "</tbody></table></body></html>"
    html_middle = "".join([vv for v in new_table_dict.values() for vv in v])
    table_str = f"{html_start}{html_middle}{html_end}"
    return table_str

def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
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


# ------------------------------------------------------------------------------
# Part of implementation is adopted from CenterNet,
# made publicly available under the MIT License at https://github.com/xingyizhou/CenterNet.git
# ------------------------------------------------------------------------------



class DetProcess:
    def __init__(self, K: int = 3000, num_classes: int = 2, scale: float = 1.0):
        self.K = K
        self.num_classes = num_classes
        self.scale = scale
        self.max_per_image = 3000

    def __call__(
        self, det_out: Dict[str, np.ndarray], meta: Dict[str, Union[int, np.ndarray]]
    ):
        hm = self.sigmoid(det_out["hm"])
        dets, keep, logi, cr = ctdet_4ps_decode(
            hm[:, 0:1, :, :],
            det_out["wh"],
            det_out["ax"],
            det_out["cr"],
            reg=det_out["reg"],
            K=self.K,
        )

        raw_dets = dets
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_4ps_post_process_upper_left(
            dets.copy(),
            [meta["c"]],
            [meta["s"]],
            meta["out_height"],
            meta["out_width"],
            2,
        )
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 9)
            dets[0][j][:, :8] /= self.scale
        dets = dets[0]
        detections = [dets]

        logi += cr
        results = self.merge_outputs(detections)
        slct_logi_feat, slct_dets_feat = self.filter(results, logi, raw_dets[:, :, :8])
        slct_output_dets = results[1][: slct_logi_feat.shape[1], :8]
        return slct_logi_feat, slct_dets_feat, slct_output_dets

    @staticmethod
    def sigmoid(data: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-data))

    def merge_outputs(self, detections: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        # thresh_conf, thresh_min, thresh_max = 0.1, 0.5, 0.7
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0
            ).astype(np.float32)

        scores = np.hstack([results[j][:, 8] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = results[j][:, 8] >= thresh
                results[j] = results[j][keep_inds]
        return results

    @staticmethod
    def filter(
        results: Dict[int, np.ndarray], logi: np.ndarray, ps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # this function select boxes
        batch_size, feat_dim = logi.shape[0], logi.shape[2]
        num_valid = sum(results[1][:, 8] >= 0.15)

        slct_logi = np.zeros((batch_size, num_valid, feat_dim), dtype=np.float32)
        slct_dets = np.zeros((batch_size, num_valid, 8), dtype=np.int32)
        for i in range(batch_size):
            for j in range(num_valid):
                slct_logi[i, j, :] = logi[i, j, :]
                slct_dets[i, j, :] = ps[i, j, :]

        return slct_logi, slct_dets


def ctdet_4ps_decode(
    heat: np.ndarray,
    wh: np.ndarray,
    ax: np.ndarray,
    cr: np.ndarray,
    reg: np.ndarray = None,
    cat_spec_wh: bool = False,
    K: int = 100,
):
    batch, cat, _, width = heat.shape
    heat, keep = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.reshape(batch, K, 2)
        xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.reshape(batch, K, 1) + 0.5
        ys = ys.reshape(batch, K, 1) + 0.5

    wh = _tranpose_and_gather_feat(wh, inds)
    ax = _tranpose_and_gather_feat(ax, inds)

    if cat_spec_wh:
        wh = wh.reshape(batch, K, cat, 8)
        clses_ind = clses.reshape(batch, K, 1, 1).expand(batch, K, 1, 8)
        wh = wh.gather(2, clses_ind).reshape(batch, K, 8)
    else:
        wh = wh.reshape(batch, K, 8)

    clses = clses.reshape(batch, K, 1)
    scores = scores.reshape(batch, K, 1)

    bboxes_vec = [
        xs - wh[..., 0:1],
        ys - wh[..., 1:2],
        xs - wh[..., 2:3],
        ys - wh[..., 3:4],
        xs - wh[..., 4:5],
        ys - wh[..., 5:6],
        xs - wh[..., 6:7],
        ys - wh[..., 7:8],
    ]
    bboxes = np.concatenate(bboxes_vec, axis=2)

    cc_match = np.concatenate(
        [
            (xs - wh[..., 0:1]) + width * np.round(ys - wh[..., 1:2]),
            (xs - wh[..., 2:3]) + width * np.round(ys - wh[..., 3:4]),
            (xs - wh[..., 4:5]) + width * np.round(ys - wh[..., 5:6]),
            (xs - wh[..., 6:7]) + width * np.round(ys - wh[..., 7:8]),
        ],
        axis=2,
    )
    cc_match = np.round(cc_match).astype(np.int64)
    cr_feat = _get_4ps_feat(cc_match, cr)
    cr_feat = cr_feat.sum(axis=3)

    detections = np.concatenate([bboxes, scores, clses], axis=2)
    return detections, keep, ax, cr_feat


def _nms(heat: np.ndarray, kernel: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    pad = (kernel - 1) // 2
    hmax = max_pool(heat, kernel_size=kernel, stride=1, padding=pad)
    keep = hmax == heat
    return heat * keep, keep


def max_pool(
    img: np.ndarray, kernel_size: int, stride: int, padding: int
) -> np.ndarray:
    h, w = img.shape[2:]
    img = np.pad(
        img,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        "constant",
        constant_values=0,
    )

    res_h = ((h + 2 - kernel_size) // stride) + 1
    res_w = ((w + 2 - kernel_size) // stride) + 1
    res = np.zeros((img.shape[0], img.shape[1], res_h, res_w))
    for i in range(res_h):
        for j in range(res_w):
            temp = img[
                :,
                :,
                i * stride : i * stride + kernel_size,
                j * stride : j * stride + kernel_size,
            ]
            res[:, :, i, j] = temp.max()
    return res


def _topk(
    scores: np.ndarray, K: int = 40
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch, cat, height, width = scores.shape

    topk_scores, topk_inds = find_topk(scores.reshape(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds / width
    topk_xs = np.float32(np.int32(topk_inds % width))

    topk_score, topk_ind = find_topk(topk_scores.reshape(batch, -1), K)
    topk_clses = np.int32(topk_ind / K)
    topk_inds = _gather_feat(topk_inds.reshape(batch, -1, 1), topk_ind).reshape(
        batch, K
    )
    topk_ys = _gather_feat(topk_ys.reshape(batch, -1, 1), topk_ind).reshape(batch, K)
    topk_xs = _gather_feat(topk_xs.reshape(batch, -1, 1), topk_ind).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def find_topk(
    a: np.ndarray, k: int, axis: int = -1, largest: bool = True, sorted: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size - k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k) - 1, axis=axis)
    else:
        index_array = np.argpartition(a, k - 1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)

    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)

        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis
        )
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis
        )
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def _gather_feat(feat: np.ndarray, ind: np.ndarray) -> np.ndarray:
    dim = feat.shape[2]
    ind = np.broadcast_to(ind[:, :, None], (ind.shape[0], ind.shape[1], dim))
    feat = _gather(feat, 1, ind)
    return feat


def _gather(data: np.ndarray, dim: int, index: np.ndarray) -> np.ndarray:
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1 :]
    data_xsection_shape = data.shape[:dim] + data.shape[dim + 1 :]
    if idx_xsection_shape != data_xsection_shape:
        raise ValueError(
            "Except for dimension "
            + str(dim)
            + ", all dimensions of index and data should be the same size"
        )

    if index.dtype != np.int64:
        raise TypeError("The values of index must be integers")

    data_swaped = np.swapaxes(data, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.take_along_axis(data_swaped, index_swaped, axis=0)
    return np.swapaxes(gathered, 0, dim)


def _tranpose_and_gather_feat(feat: np.ndarray, ind: np.ndarray) -> np.ndarray:
    feat = np.ascontiguousarray(np.transpose(feat, [0, 2, 3, 1]))
    feat = feat.reshape(feat.shape[0], -1, feat.shape[3])
    feat = _gather_feat(feat, ind)
    return feat


def _get_4ps_feat(cc_match: np.ndarray, output: np.ndarray) -> np.ndarray:
    if isinstance(output, dict):
        feat = output["cr"]
    else:
        feat = output

    feat = np.ascontiguousarray(feat.transpose(0, 2, 3, 1))
    feat = feat.reshape(feat.shape[0], -1, feat.shape[3])
    feat = feat[..., None]
    feat = np.concatenate([feat] * 4, axis=-1)

    dim = feat.shape[2]
    cc_match = cc_match[..., None, :]
    cc_match = np.concatenate([cc_match] * dim, axis=2)
    if not (isinstance(output, dict)):
        cc_match = np.where(
            cc_match < feat.shape[1],
            cc_match,
            (feat.shape[0] - 1) * np.ones(cc_match.shape).astype(np.int64),
        )

        cc_match = np.where(
            cc_match >= 0, cc_match, np.zeros(cc_match.shape).astype(np.int64)
        )
    feat = np.take_along_axis(feat, cc_match, axis=1)
    return feat


def ctdet_4ps_post_process_upper_left(
    dets: np.ndarray,
    c: List[np.ndarray],
    s: List[float],
    h: int,
    w: int,
    num_classes: int,
) -> np.ndarray:
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, 0:2] = transform_preds_upper_left(
            dets[i, :, 0:2], c[i], s[i], (w, h)
        )
        dets[i, :, 2:4] = transform_preds_upper_left(
            dets[i, :, 2:4], c[i], s[i], (w, h)
        )
        dets[i, :, 4:6] = transform_preds_upper_left(
            dets[i, :, 4:6], c[i], s[i], (w, h)
        )
        dets[i, :, 6:8] = transform_preds_upper_left(
            dets[i, :, 6:8], c[i], s[i], (w, h)
        )
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = classes == j
            tmp_top_pred = [
                dets[i, inds, :8].astype(np.float32),
                dets[i, inds, 8:9].astype(np.float32),
            ]
            top_preds[j + 1] = np.concatenate(tmp_top_pred, axis=1).tolist()
        ret.append(top_preds)
    return ret


def transform_preds_upper_left(
    coords: np.ndarray,
    center: np.ndarray,
    scale: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    target_coords = np.zeros(coords.shape)

    trans = get_affine_transform_upper_left(center, scale, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform_upper_left(
    center: np.ndarray,
    scale: float,
    output_size: List[Tuple[int, int]],
    inv: int = 0,
) -> np.ndarray:
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    dst[0, :] = [0, 0]
    if center[0] < center[1]:
        src[1, :] = [scale[0], center[1]]
        dst[1, :] = [output_size[0], 0]
    else:
        src[1, :] = [center[0], scale[0]]
        dst[1, :] = [0, output_size[0]]
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def affine_transform(pt: np.ndarray, t: np.ndarray) -> np.ndarray:
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]





