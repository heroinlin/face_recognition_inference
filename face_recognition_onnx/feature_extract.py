# -*- coding: utf-8 -*-

import numpy as np
import onnxruntime

import os
import cv2

from .defaults import _C as cfg
working_root = os.path.split(os.path.realpath(__file__))[0]


class ONNXInference(object):
    def __init__(self, onnx_file_path=None):
        """
        对ONNXInference进行初始化

        Parameters
        ----------
        onnx_file_path : str
            onnx模型的路径，推荐使用绝对路径
        """
        super().__init__()
        self.onnx_file_path = onnx_file_path
        if self.onnx_file_path is None:
            print("please set onnx model path!\n")
            exit(-1)
        self.session = onnxruntime.InferenceSession(self.onnx_file_path)

    def inference(self, x: np.ndarray):
        """
        onnx的推理
        Parameters
        ----------
        x : np.ndarray
            onnx模型输入

        Returns
        -------
        np.ndarray
            onnx模型推理结果
        """
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        outputs = self.session.run(output_names=[output_name],
                                   input_feed={input_name: x.astype(np.float32)})
        return outputs


class FeatureExtract(ONNXInference):
    def __init__(self, onnx_file_path=None):
        """对FeatureExtract进行初始化

        Parameters
        ----------
        onnx_file_path : str
            onnx模型的路径，推荐使用绝对路径
        """
        if onnx_file_path is None:
            onnx_file_path = os.path.join(working_root,
                                          'onnx_model',
                                          "mobilenet_v2_184_0.1701-sim.onnx")
        super(FeatureExtract, self).__init__(onnx_file_path)
        self.cfg = cfg.clone()
        self.cfg.freeze()

    def _pre_process(self, image: np.ndarray) -> np.ndarray:
        """对图像进行预处理

        Parameters
        ----------
        image : np.ndarray
            输入的原始图像，BGR格式，通常使用cv2.imread读取得到

        Returns
        -------
        np.ndarray
            原始图像经过预处理后得到的数组
        """
        if self.cfg.INPUT.FORMAT == "RGB":
            image = image[:, :, ::-1]
        image = cv2.resize(image, (cfg.INPUT.WIDTH, cfg.INPUT.HEIGHT))
        input_image = (np.array(image, dtype=np.float32) / 255 - cfg.INPUT.PIXEL_MEAN) / cfg.INPUT.PIXEL_STD
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, 0)
        return input_image

    def feature_extract(self, image: np.ndarray) -> np.ndarray:
        """对输入图像提取特征

        Parameters
        ----------
        image : np.ndarray
            输入图片，BGR格式，通常使用cv2.imread获取得到

        Returns
        -------
        np.ndarray
            返回特征
        """
        image = self._pre_process(image)
        feature = self.inference(image)
        return np.array(feature)
