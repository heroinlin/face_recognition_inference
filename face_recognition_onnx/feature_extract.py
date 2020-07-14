# -*- coding: utf-8 -*-

import numpy as np
import onnxruntime

import os
import cv2

working_root = os.path.split(os.path.realpath(__file__))[0]


class ONNXInference(object):
    def __init__(self, model_path=None):
        """
        对ONNXInference进行初始化

        Parameters
        ----------
        model_path : str
            onnx模型的路径，推荐使用绝对路径
        """
        super().__init__()
        self.model_path = model_path
        if self.model_path is None:
            print("please set onnx model path!\n")
            exit(-1)
        self.session = onnxruntime.InferenceSession(self.model_path)

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
    def __init__(self, model_path=None):
        """对FeatureExtract进行初始化

        Parameters
        ----------
        model_path : str
            onnx模型的路径，推荐使用绝对路径
        """
        if model_path is None:
            model_path = os.path.join(working_root,
                                      'onnx_model',
                                      "mobilenet_v2_184_0.1701-sim.onnx")
        super(FeatureExtract, self).__init__(model_path)
        self.config = {
            'width': 112,
            'height': 112,
            'color_format': 'RGB',
            'mean': [0.4914, 0.4822, 0.4465],
            'stddev': [0.247, 0.243, 0.261],
            'divisor': 255.0,
        }

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
        if self.config['color_format'] == "RGB":
            image = image[:, :, ::-1]
        if self.config['width'] > 0 and self.config['height'] > 0:
            image = cv2.resize(image, (self.config['width'], self.config['height']))
        input_image = (np.array(image, dtype=np.float32) / self.config['divisor'] - self.config['mean']) / self.config['stddev']
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
        src_image = self._pre_process(image)
        flip_image = self._pre_process(cv2.flip(image, 1))
        feature = self.inference(src_image)[0] + self.inference(flip_image)[0]
        return np.array(feature)
