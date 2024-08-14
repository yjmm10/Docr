import copy
import math
import os
import re
import time
from typing import List, Union

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download

from docr.config import __model_path__
from docr.core import CVModel


class CRNN(CVModel):
    def __init__(self, model_path, **params):
        names = params.get("names", {"model": "rec_text.onnx", "label": "rec_text.res"})
        label_path = os.path.join(__model_path__, model_path, names["label"])
        model_path = os.path.join(model_path, names["model"])
        labels: List = self.get_labels(label_path)

        super().__init__(model_path=model_path, labels=labels, **params)
        self.rec_image_shape = [3, 48, 320]
        self.rec_batch_num = 16
        self.ctc_decode = CTCLabelDecode(char_list=labels)

    def get_labels(self, label_path):
        character_str = []
        with open(label_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode("utf-8").strip("\n").strip("\r\n")
                character_str.append(line)

        dict_character = ["blank"] + list(character_str) + [" "]

        return dict_character

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))

        w = self.input_shape[3:][0]
        if isinstance(w, str):
            pass
        elif w is not None and w > 0:
            imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def pre_process(self, image: np.ndarray, max_wh_ratio=None):
        imgC, imgH, imgW = self.rec_image_shape[:3]
        if max_wh_ratio is None:
            max_wh_ratio = imgW / imgH
        norm_img = self.resize_norm_img(image, max_wh_ratio)
        norm_img = norm_img[np.newaxis, :]
        return norm_img

    def batch_inference(self, images):
        start = time.perf_counter()

        img_num = len(images)
        rec_res = [["", 0.0]] * img_num
        # Calculate the aspect ratio of all text bars
        width_list = [img.shape[1] / float(img.shape[0]) for img in images]
        # order the image from the longest width to the shortest width
        indices = np.argsort(np.array(width_list))
        # batch_input = []
        for beg_img_no in range(0, img_num, self.rec_batch_num):
            end_img_no = min(img_num, beg_img_no + self.rec_batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH

            # Find the image with the largest aspect ratio in the batch , w/h
            max_wh_ratio = max(
                max(
                    [
                        images[i].shape[1] * 1.0 / images[i].shape[0]
                        for i in indices[beg_img_no:end_img_no]
                    ]
                ),
                max_wh_ratio,
            )

            for ino in range(beg_img_no, end_img_no):
                norm_img = self.pre_process(
                    images[indices[ino]], max_wh_ratio=max_wh_ratio
                )
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            # batch_input.append(norm_img_batch)

            preds = self.ort_infer([norm_img_batch])
            rec_result = self.ctc_decode(preds)

            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        texts = []
        scores = []
        for text, score in rec_res:
            texts.append(text)
            scores.append(score)
        self.result = [np.array(texts), np.array(scores)]
        return self.result

    def post_process(self, output):
        result = self.ctc_decode(output)
        texts = []
        scores = []
        for res in result:
            text, score = res
            texts.append(text)
            scores.append(score)
        return [texts, scores]


class CTCLabelDecode(object):
    """Convert between text-label and text-index"""

    def __init__(self, char_list: List = None, use_space_char=False, **kwargs):

        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        # self.character_str = []

        self.character = char_list
        pass

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        if not isinstance(preds, np.ndarray):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)

        return "".join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id] for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank
