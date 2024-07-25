import numpy as np
from typing import Union,List
import cv2

colors = [
    (255, 0, 0),    # 红色
    (255, 128, 0),  # 橙色
    (255, 255, 0),  # 黄色
    (128, 255, 0),  # 黄绿色
    (0, 255, 0),    # 绿色
    (0, 255, 128),  # 春绿色
    (0, 255, 255),  # 青色
    (0, 128, 255),  # 天蓝色
    (0, 0, 255),    # 蓝色
    (128, 0, 255),  # 紫色
    (255, 0, 255),  # 玫瑰红
    (255, 0, 128),  # 洋红
    (64, 0, 0),     # 暗红
    (64, 32, 0),    # 棕色
    (64, 64, 0),    # 深黄
    (32, 64, 0),    # 橄榄绿
    (0, 64, 0),     # 深绿
    (0, 64, 32),    # 深青
    (0, 64, 64),    # 暗青
    (0, 32, 64),    # 深蓝
    (0, 0, 64),     # 深夜蓝
    (32, 0, 64),    # 深紫
    (64, 0, 64),    # 暗紫
    (64, 0, 32),    # 深洋红
    (191, 191, 191),# 灰色
    (128, 128, 128),# 中灰
    (64, 64, 64),   # 暗灰
    (255, 175, 175),# 浅粉红
    (175, 255, 175),# 浅绿
    (175, 175, 255), # 浅蓝
]

def visual(image: np.array, boxes: Union[List, np.ndarray], class_ids: Union[List, np.ndarray], class_names:List=None, scores="", mask_alpha=0.2):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        # try:
            
        color = colors[class_id]
        # except IndexError: # 越界则自动生成
        #     rng = np.random.default_rng(3)
        #     global colors
        #     colors = rng.uniform(0, 255, size=(10, 3))
        #     color = colors[class_id]

        draw_box(det_img, box, color)
        if class_names is None:
            label = class_id
        else:
            label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box( image: np.ndarray, box:  Union[List, np.ndarray], color: tuple[int, int, int] = (0, 0, 255),
             thickness: int = 1) -> np.ndarray:
    x1, y1, x2, y2 = np.array(box).astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: Union[List, np.ndarray], color: tuple[int, int, int] = (0, 0, 255),
              font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = np.array(box).astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

def draw_masks(image: np.ndarray, boxes: Union[List, np.ndarray], classes: Union[List, np.ndarray], mask_alpha: float = 0.3) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(np.array(boxes), np.array(classes)):
        # try:
            # global colors
            # print(colors)
        color = colors[class_id]
        # except IndexError: # 越界则自动生成
            # global colors
            # rng = np.random.default_rng(3)
            # colors = rng.uniform(0, 255, size=(50, 3))
            # color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)
