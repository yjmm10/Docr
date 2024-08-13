import os
import re
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from PIL import Image, UnidentifiedImageError
from tokenizers import Tokenizer
from tokenizers.models import BPE

from telos.core import CVModel, OrtInferSession

InputType = Union[str, np.ndarray, bytes, Path]


class LatexOCR(CVModel):
    def __init__(self, model_path: Union[str, Path] = None, **params):
        models = params.get(
            "models",
            {
                "decoder": "decoder.onnx",
                "encoder": "encoder.onnx",
                "image_resizer": "image_resizer.onnx",
                "tokenizer": "tokenizer.json",
            },
        )
        config = params.get(
            "config",
            {
                "max_width": 672,
                "max_height": 192,
                "min_height": 32,
                "min_width": 32,
                "bos_token": 1,
                "max_seq_len": 512,
                "eos_token": 2,
                "temperature": 0.00001,
            },
        )

        self.decoder_path = Path(model_path) / models.get("decoder")
        self.encoder_path = Path(model_path) / models.get("encoder")
        self.image_resizer_path = Path(model_path) / models.get("image_resizer")
        self.tokenizer_json = Path(model_path) / models.get("tokenizer")

        self.max_dims = [config.get("max_width"), config.get("max_height")]
        self.min_dims = [config.get("min_width", 32), config.get("min_height", 32)]
        self.temperature = config.get("temperature", 0.00001)

        # self.load_img = LoadImage()

        self.pre_pro = PreProcess(max_dims=self.max_dims, min_dims=self.min_dims)

        self.image_resizer = OrtInferSession(self.image_resizer_path)

        self.encoder_decoder = EncoderDecoder(
            encoder_path=self.encoder_path,
            decoder_path=self.decoder_path,
            bos_token=config["bos_token"],
            eos_token=config["eos_token"],
            max_seq_len=config["max_seq_len"],
        )
        self.tokenizer = TokenizerCls(self.tokenizer_json)

    def pre_process(self, image: InputType) -> np.ndarray:
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # self.load_img(image)
        except LoadImageError as exc:
            error_info = traceback.format_exc()
            raise LoadImageError(
                f"Load the image meets error. Error info is {error_info}"
            ) from exc

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            resizered_img = self.loop_image_resizer(image)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ValueError(
                f"image resizer meets error. Error info is {error_info}"
            ) from e
        return resizered_img

    def inference(self, input_tensor: np.ndarray):
        try:
            dec = self.encoder_decoder(input_tensor, temperature=self.temperature)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ValueError(
                f"EncoderDecoder meets error. Error info is {error_info}"
            ) from e

        decode = self.tokenizer.token2str(dec)[0]

        return decode  # , elapse

    def __call__(self, image: np.array):
        start = time.perf_counter()
        input_tensor = self.pre_process(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        result = self.post_process(outputs)
        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return result

    def loop_image_resizer(self, img: np.ndarray) -> np.ndarray:
        pillow_img = Image.fromarray(img)
        pad_img = self.pre_pro.pad(pillow_img)
        input_image = self.pre_pro.minmax_size(pad_img).convert("RGB")
        r, w, h = 1, input_image.size[0], input_image.size[1]
        for _ in range(10):
            h = int(h * r)
            final_img, pad_img = self.image_process(input_image, r, w, h)

            resizer_res = self.image_resizer([final_img.astype(np.float32)])[0]

            # argmax_idx = int(np.argmax(resizer_res, axis=-1))
            argmax_idx = int(np.argmax(resizer_res))
            w = (argmax_idx + 1) * 32
            if w == pad_img.size[0]:
                break

            r = w / pad_img.size[0]
        return final_img

    # 将数据处理为模型的格式
    def image_process(
        self, input_image: Image.Image, r, w, h
    ) -> Tuple[np.ndarray, Image.Image]:
        if r > 1:
            resize_func = Image.Resampling.BILINEAR
        else:
            resize_func = Image.Resampling.LANCZOS

        resize_img = input_image.resize((w, h), resize_func)
        pad_img = self.pre_pro.pad(self.pre_pro.minmax_size(resize_img))
        cvt_img = np.array(pad_img.convert("RGB"))

        gray_img = self.pre_pro.to_gray(cvt_img)
        normal_img = self.pre_pro.normalize(gray_img)
        final_img = self.pre_pro.transpose_and_four_dim(normal_img)
        return final_img, pad_img

    @staticmethod
    def post_process(s: str) -> str:
        """Remove unnecessary whitespace from LaTeX code.

        Args:
            s (str): Input string

        Returns:
            str: Processed image
        """
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = r"[\W_^\d]"
        names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s


########################################################
#
#                    模型结构
#
########################################################
class EncoderDecoder:
    def __init__(
        self,
        encoder_path: Union[Path, str],
        decoder_path: Union[Path, str],
        bos_token: int,
        eos_token: int,
        max_seq_len: int,
    ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.max_seq_len = max_seq_len

        self.encoder = OrtInferSession(encoder_path)
        self.decoder = Decoder(decoder_path)

    def __call__(self, x: np.ndarray, temperature: float = 0.25):
        ort_input_data = np.array([self.bos_token] * len(x))[:, None]
        context = self.encoder([x])[0]
        output = self.decoder(
            ort_input_data,
            self.max_seq_len,
            eos_token=self.eos_token,
            context=context,
            temperature=temperature,
        )
        self.result = output
        return output


class Decoder:
    def __init__(self, decoder_path: Union[Path, str]):
        self.max_seq_len = 512
        self.session = OrtInferSession(decoder_path)

    def __call__(
        self,
        start_tokens,
        seq_len=256,
        eos_token=None,
        temperature=1.0,
        filter_thres=0.9,
        context=None,
    ):
        num_dims = len(start_tokens.shape)

        b, t = start_tokens.shape

        out = start_tokens
        mask = np.full_like(start_tokens, True, dtype=bool)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]
            mask = mask[:, -self.max_seq_len :]

            ort_outs = self.session([x.astype(np.int64), mask, context])[0]
            np_preds = ort_outs
            np_logits = np_preds[:, -1, :]

            np_filtered_logits = self.npp_top_k(np_logits, thres=filter_thres)
            np_probs = self.softmax(np_filtered_logits / temperature, axis=-1)

            sample = self.multinomial(np_probs.squeeze(), 1)[None, ...]

            out = np.concatenate([out, sample], axis=-1)
            mask = np.pad(mask, [(0, 0), (0, 1)], "constant", constant_values=True)

            if (
                eos_token is not None
                and (np.cumsum(out == eos_token, axis=1)[:, -1] >= 1).all()
            ):
                break

        out = out[:, t:]
        if num_dims == 1:
            out = out.squeeze(0)
        return out

    @staticmethod
    def softmax(x, axis=None) -> float:
        def logsumexp(a, axis=None, b=None, keepdims=False):
            a_max = np.amax(a, axis=axis, keepdims=True)

            if a_max.ndim > 0:
                a_max[~np.isfinite(a_max)] = 0
            elif not np.isfinite(a_max):
                a_max = 0

            tmp = np.exp(a - a_max)

            # suppress warnings about log of zero
            with np.errstate(divide="ignore"):
                s = np.sum(tmp, axis=axis, keepdims=keepdims)
                out = np.log(s)

            if not keepdims:
                a_max = np.squeeze(a_max, axis=axis)
            out += a_max
            return out

        return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

    def npp_top_k(self, logits, thres=0.9):
        k = int((1 - thres) * logits.shape[-1])
        val, ind = self.np_top_k(logits, k)
        probs = np.full_like(logits, float("-inf"))
        np.put_along_axis(probs, ind, val, axis=1)
        return probs

    @staticmethod
    def np_top_k(
        a: np.ndarray, k: int, axis=-1, largest=True, sorted=True
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

    @staticmethod
    def multinomial(weights, num_samples, replacement=True):
        weights = np.asarray(weights)
        weights /= np.sum(weights)  # 确保权重之和为1
        indices = np.arange(len(weights))
        samples = np.random.choice(
            indices, size=num_samples, replace=replacement, p=weights
        )
        return samples


class LoadImageError(Exception):
    pass


class PreProcess:
    def __init__(self, max_dims: List[int], min_dims: List[int]):
        self.max_dims, self.min_dims = max_dims, min_dims
        self.mean = np.array([0.7931, 0.7931, 0.7931]).astype(np.float32)
        self.std = np.array([0.1738, 0.1738, 0.1738]).astype(np.float32)

    @staticmethod
    def pad(img: Image.Image, divable: int = 32) -> Image.Image:
        """Pad an Image to the next full divisible value of `divable`. Also normalizes the image and invert if needed.

        Args:
            img (PIL.Image): input image
            divable (int, optional): . Defaults to 32.

        Returns:
            PIL.Image
        """
        threshold = 128
        data = np.array(img.convert("LA"))
        if data[..., -1].var() == 0:
            data = (data[..., 0]).astype(np.uint8)
        else:
            data = (255 - data[..., -1]).astype(np.uint8)

        data = (data - data.min()) / (data.max() - data.min()) * 255
        if data.mean() > threshold:
            # To invert the text to white
            gray = 255 * (data < threshold).astype(np.uint8)
        else:
            gray = 255 * (data > threshold).astype(np.uint8)
            data = 255 - data

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        rect = data[b : b + h, a : a + w]
        im = Image.fromarray(rect).convert("L")
        dims: List[Union[int, int]] = []
        for x in [w, h]:
            div, mod = divmod(x, divable)
            dims.append(divable * (div + (1 if mod > 0 else 0)))

        padded = Image.new("L", tuple(dims), 255)
        padded.paste(im, (0, 0, im.size[0], im.size[1]))
        return padded

    def minmax_size(
        self,
        img: Image.Image,
    ) -> Image.Image:
        """Resize or pad an image to fit into given dimensions

        Args:
            img (Image): Image to scale up/down.

        Returns:
            Image: Image with correct dimensionality
        """
        if self.max_dims is not None:
            ratios = [a / b for a, b in zip(img.size, self.max_dims)]
            if any([r > 1 for r in ratios]):
                size = np.array(img.size) // max(ratios)
                img = img.resize(size.astype(int), Image.BILINEAR)

        if self.min_dims is not None:
            padded_size: List[Union[int, int]] = [
                max(img_dim, min_dim)
                for img_dim, min_dim in zip(img.size, self.min_dims)
            ]

            new_pad_size = tuple(padded_size)
            if new_pad_size != img.size:  # assert hypothesis
                padded_im = Image.new("L", new_pad_size, 255)
                padded_im.paste(img, img.getbbox())
                img = padded_im
        return img

    def normalize(self, img: np.ndarray, max_pixel_value=255.0) -> np.ndarray:
        mean = self.mean * max_pixel_value
        std = self.std * max_pixel_value
        denominator = np.reciprocal(std, dtype=np.float32)
        img = img.astype(np.float32)
        img -= mean
        img *= denominator
        return img

    @staticmethod
    def to_gray(img) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def transpose_and_four_dim(img: np.ndarray) -> np.ndarray:
        return img.transpose(2, 0, 1)[:1][None, ...]


class TokenizerCls:
    def __init__(self, json_file: Union[Path, str]):
        self.tokenizer = Tokenizer(BPE()).from_file(str(json_file))

    def token2str(self, tokens) -> List[str]:
        if len(tokens.shape) == 1:
            tokens = tokens[None, :]

        dec = [self.tokenizer.decode(tok.tolist()) for tok in tokens]
        return [
            "".join(detok.split(" "))
            .replace("Ġ", " ")
            .replace("[EOS]", "")
            .replace("[BOS]", "")
            .replace("[PAD]", "")
            .strip()
            for detok in dec
        ]


if __name__ == "__main__":
    # downloader = DownloadModel()
    # downloader("decoder.onnx")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model",
        "--model_path",
        type=str,
        default="/home/zyj/project/MOP/telos/models/recognition/rec_formula",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        help="Only img path of the formula.",
        default="/home/zyj/project/MOP/test_img/formula01.png",
    )
    args = parser.parse_args()

    engine = LatexOCR(
        model_path=args.model_path,
    )

    result = engine(args.img_path)
    print(result)
