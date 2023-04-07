"""
aioz.aiar.truonbgle - Dec 07, 2021
Face Gan - Face restoration
@ref: https://github.com/yangxy/GPEN
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
"""
import os
import cv2
import time
import torch
import enum
import logging
import collections
import numpy as np
import onnxruntime
from inference.wrapper import base_wrapper


logger = logging.getLogger('inf.wrp.faceGan')

HParams = collections.namedtuple(
    'HParams',
    ['resolution', 'is_norm']
)


class FaceGANWrapper(base_wrapper.BaseWrapper):
    def __init__(self, cfg):
        super().__init__()
        self._model_path = cfg.model_path
        self._mode = cfg.mode
        self._resolution = cfg.hparams.resolution
        self._is_norm = cfg.hparams.is_norm
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __del__(self):
        pass

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, val):
        if val == "cuda" or val == "gpu":
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        self._resolution = value

    def init(self):
        self._load_model()

    def _load_model(self):
        logger.info("Load model at {}".format(self._model_path))
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self._device == torch.device("cuda"):
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        self._model = onnxruntime.InferenceSession(self._model_path, options, providers=providers)
        self._input_name = self._model.get_inputs()[0].name
        self._output_name = self._model.get_outputs()[0].name
        logger.info("Load model is DONE.")

    def _pre_process(self, image):
        """image to tensor"""
        tic = time.time()
        img = cv2.resize(image, (self._resolution, self._resolution))
        img = img[:, :, ::-1]
        img = img / 255.0
        if self._is_norm:
            img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        logger.debug("pre-process time: {:.04f}".format(time.time() - tic))
        return img

    def _post_process(self, img_np, im_type=np.uint8):
        tic = time.time()
        if self._is_norm:
            img_np = img_np * 0.5 + 0.5
        img_np = np.squeeze(img_np, 0)
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        img_np = np.clip(img_np, 0, 1) * 255.0
        logger.debug("post-process time: {:.04f}".format(time.time() - tic))
        return img_np.astype(im_type)

    def process_prediction(self, image):
        """ face gan model
        input: image
        """
        start_time = time.time()
        super(FaceGANWrapper, self)._check_model_init(logger=logger)

        img = self._pre_process(image)
        res = self._model.run([self._output_name], {self._input_name: img})
        res = self._post_process(res[0])
        elapsed = round(time.time() - start_time, 4)
        logger.debug("Process time: {:.04f}".format(elapsed))
        return res, elapsed


# test class
if __name__ == '__main__':
    import gc
    import sys
    import config
    import torch
    from inference.utils import utils

    gc.collect()
    torch.cuda.empty_cache()

    im = cv2.imread(sys.argv[1])
    fg = FaceGANWrapper(cfg=config.CONFIG_MAP.GPEN_512_ONNX)
    for _ in range(5):
        out, ti = fg.process_prediction(im)
        gpu_usage = utils.get_gpu_memory()
        logger.info("gpu usage: use / total / percent: {} / {} / {}".format(*gpu_usage))
    cv2.imshow("out", cv2.resize(out, im.shape[:2][::-1]))
    cv2.imshow("out", out)
    cv2.waitKey()
    del fg

# 2021:12:14-17:26:26-+07+0700 inf.wrp.faceGan    DEBUG    :pre-process time: 0.0085
# 2021:12:14-17:26:26-+07+0700 inf.wrp.faceGan    DEBUG    :post-process time: 0.0022
# 2021:12:14-17:26:26-+07+0700 inf.wrp.faceGan    DEBUG    :Process time: 0.1836
# 2021:12:14-17:26:26-+07+0700 inf.wrp.faceGan    INFO     :gpu usage: use / total / percent: 2708 / 12065 / 22.45
# 2021:12:14-17:26:27-+07+0700 inf.wrp.faceGan    INFO     :Del object
