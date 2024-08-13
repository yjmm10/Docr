import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from telos import __project__
from telos.config import __model_path__
from telos.data import *
from telos.utils import visual

# version 0.1.0
# class CVModel(ABC):
#     def __init__(self, model_path:Union[str,Path], labels: List, **params):
#         self.enable_cpu_mem_arena = params.get('enable_cpu_mem_arena',False)
#         self.execution_mode = params.get('execution_mode',ort.ExecutionMode.ORT_SEQUENTIAL)
#         self.intra_op_num_threads = params.get('intra_op_num_threads',2)
#         self.inter_op_num_threads = params.get('inter_op_num_threads',2)

#         self.class_names = labels

#         # Initialize model
#         model_path = Path(__model_path__) / model_path
#         self.load_model(model_path)

#     def __call__(self, image: np.array):
#         start = time.perf_counter()
#         input_tensor = self.pre_process(image)

#         # Perform inference on the image
#         outputs = self.inference(input_tensor)

#         self.boxes, self.scores, self.class_ids = self.post_process(outputs)
#         print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
#         return self.boxes, self.scores, self.class_ids

#     def inference(self, input_tensor):
#         # start = time.perf_counter()
#         outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

#         # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
#         return outputs

#     def load_model(self, model_path):
#         assert Path(model_path).exists(), f"Model path {model_path} does not exist."
#         options = ort.SessionOptions()
#         options.enable_cpu_mem_arena = self.enable_cpu_mem_arena
#         options.execution_mode = self.execution_mode
#         options.intra_op_num_threads = self.intra_op_num_threads
#         options.inter_op_num_threads = self.inter_op_num_threads

#         self.session = ort.InferenceSession(model_path, options=options, providers=ort.get_available_providers())
#         # Get model info
#         self.get_model_input()
#         self.get_model_output()


#     def get_model_input(self):
#         model_inputs = self.session.get_inputs()
#         self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

#         self.input_shape = model_inputs[0].shape
#         self.input_height = self.input_shape[2]
#         self.input_width = self.input_shape[3]

#     def get_model_output(self):
#         model_outputs = self.session.get_outputs()
#         self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

#     def pre_process(self, image: np.array):
#         self.img_height, self.img_width = image.shape[:2]

#         input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Resize input image
#         input_img = cv2.resize(input_img, (self.input_width, self.input_height))

#         # Scale input pixel values to 0 to 1
#         input_img = input_img / 255.0
#         input_img = input_img.transpose(2, 0, 1)
#         input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

#         return input_tensor


#     @abstractmethod
#     def post_process(self, output):
#         pass
#         # return None,None,None

#     def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
#         return visual(image,
#                       self.boxes,
#                       self.class_ids,
#                       class_names=self.class_names,
#                       scores=self.scores,
#                       mask_alpha=mask_alpha)


# version 0.1.0
class CVModel(ABC):
    result = None
    class_names = None
    ort_infer = None
    input_shape = None
    cost_time = 0

    def __init__(self, model_path: Union[str, Path], labels: List, **params):
        self.class_names = labels

        # Initialize model
        model_path = Path(__model_path__) / model_path

        self.ort_infer = OrtInferSession(model_path, **params)
        self.input_shape = self.ort_infer.input_shape

    # 单个图像的前处理
    @abstractmethod
    def pre_process(self, image: np.ndarray) -> np.ndarray:
        pass

    # 单个图像的后处理
    @abstractmethod
    def post_process(self, output: np.ndarray) -> List[Union[np.ndarray, List, str]]:
        pass

    # 自动判断批处理与单个处理
    def __call__(
        self, image: Union[np.array, List], batch=False
    ) -> List[Union[np.ndarray, List, str, float]]:
        start = time.perf_counter()
        if isinstance(image, np.ndarray):
            self.result = self.inference(image)
        if isinstance(image, List):
            self.result = self.batch_inference(image)
        self.cost_time = (time.perf_counter() - start) * 1000
        print(f"Inference time: {self.cost_time:.2f} ms")
        return self.result

    # 单个图片的处理
    def inference(self, image: np.array) -> List[Union[np.ndarray, List, str]]:

        input_tensor = self.pre_process(image)  # [1,64,224,3]

        # Perform inference on the image
        outputs = self.ort_infer([input_tensor])[0]  # [1,1,64,224]

        # 获取结果
        self.result = self.post_process(outputs)
        # self.result = [boxes, scores, class_ids]
        return self.result

    def batch_inference(
        self, image: List[np.ndarray]
    ) -> List[Union[np.ndarray, List, str]]:
        print("批量测试接口")
        pass

    # 绘制图像
    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return visual(
            image,
            self.boxes,
            self.class_ids,
            class_names=self.class_names,
            scores=self.scores,
            mask_alpha=mask_alpha,
        )

    ######################################
    #
    #       格式化内容
    #
    ######################################

    def _json(
        self,
    ):
        return self.__telos(type="Json")

    def _telos(
        self,
    ):
        return self.__telos(type="telos")

    # 转化数据格式
    def __telos(self, type="telos"):
        assert self.result is not None, print("check the input")
        res_nums = len(self.result)
        # 识别结果
        if res_nums == 2:
            texts, scores = self.result
            result = []
            ids = list(range(len(texts)))
            for text, score in zip(texts, scores):
                if type == "telos":
                    result.append(MetaText(text=text, score=score))
                if type == "Json":
                    result.append({"text": text, "score": score})
            return result
        # 检测结果
        elif res_nums == 3:
            boxes, scores, class_ids = self.result
            result = []
            ids = list(range(len(boxes)))
            for id, box, score, class_id in zip(ids, boxes, scores, class_ids):
                if type == "telos":
                    result.append(
                        MetaBbox(
                            id=id,
                            bbox=box,
                            score=score,
                            label=self.class_names[class_id],
                            label_id=class_id,
                        )
                    )
                if type == "Json":
                    result.append(
                        {
                            "id": id,
                            "bbox": box,
                            "score": score,
                            "label": self.class_names[class_id],
                            "label_id": class_id,
                        }
                    )
            return result

        elif res_nums == 4:
            # pass
            boxes, scores, class_ids, text_res = self.result
            result = []
            ids = list(range(len(boxes)))
            for id, box, score, class_id, text_res in zip(
                ids, boxes, scores, class_ids, text_res
            ):

                if type == "telos":
                    content = MetaText(text=text_res[0], score=text_res[1])
                    result.append(
                        MetaBbox(
                            id=id,
                            bbox=box,
                            score=score,
                            label=self.class_names[class_id],
                            label_id=class_id,
                            content=content,
                        )
                    )
                if type == "Json":
                    result.append(
                        {
                            "id": id,
                            "bbox": box,
                            "score": score,
                            "label": self.class_names[class_id],
                            "label_id": class_id,
                            "content": {"text": text_res[0], "score": text_res[1]},
                        }
                    )
            return result


class OrtInferSession:
    def __init__(self, model_path: Union[str, Path], **params):
        self.verify_exist(model_path)

        self._init_sess_opt(**params)

        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }
        EP_list = [(cpu_ep, cpu_provider_options)]
        try:
            self.session = InferenceSession(
                str(model_path), sess_options=self.sess_opt, providers=EP_list
            )
        except TypeError:
            # compatible with onnxruntime 1.5.2
            self.session = InferenceSession(str(model_path), sess_options=self.sess_opt)

        # 获取模型输入
        self.input_shape = self.session.get_inputs()[0].shape

    def _init_sess_opt(self, **params):
        self.sess_opt = SessionOptions()
        self.sess_opt.enable_cpu_mem_arena = params.get("enable_cpu_mem_arena", False)
        self.sess_opt.execution_mode = params.get(
            "execution_mode", ort.ExecutionMode.ORT_SEQUENTIAL
        )
        self.sess_opt.intra_op_num_threads = params.get("intra_op_num_threads", 2)
        self.sess_opt.inter_op_num_threads = params.get("inter_op_num_threads", 2)
        self.sess_opt.log_severity_level = params.get("log_severity_level", 4)
        self.sess_opt.graph_optimization_level = params.get(
            "graph_optimization_level", GraphOptimizationLevel.ORT_ENABLE_ALL
        )

    def __call__(self, input_content: List[np.ndarray]) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(None, input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_name(self, output_idx=0):
        return self.session.get_outputs()[output_idx].name

    def get_metadata(self):
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict

    @staticmethod
    def verify_exist(model_path: Union[Path, str]):
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exist!")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} must be a file")


class ONNXRuntimeError(Exception):
    pass
