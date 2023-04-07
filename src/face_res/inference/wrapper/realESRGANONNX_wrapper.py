"""
aioz.aiar.truonbgle - Dec 07, 2021
Super resolution
@ref: https://github.com/xinntao/Real-ESRGAN
"""
import os
import cv2
import time
import torch
import logging
import collections
import numpy as np
import onnxruntime
from torch.nn import functional as F
from inference.wrapper import base_wrapper


logger = logging.getLogger('inf.wrp.ESRGan')

HParams = collections.namedtuple(
    'HParams',
    ['net_scale']
)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class RealESRGANWrapper(base_wrapper.BaseWrapper):
    def __init__(self, cfg):
        super().__init__()
        self._model_path = cfg.model_path
        self._mode = cfg.mode
        self._net_scale = cfg.hparams.net_scale
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
    def net_scale(self):
        return self._net_scale

    @net_scale.setter
    def net_scale(self, value):
        if isinstance(value, int):
            self._net_scale = value
        else:
            logger.error("net_scale must be an integer")

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
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self._device)
        if self._net_scale == 2:
            # mod pad for divisible borders
            self._mod_scale = 2
            self._mod_pad_h, self._mod_pad_w = 0, 0
            _, _, h, w = img.size()
            if h % self._mod_scale != 0:
                self._mod_pad_h = (self._mod_scale - h % self._mod_scale)
            if w % self._mod_scale != 0:
                self._mod_pad_w = (self._mod_scale - w % self._mod_scale)
            img = F.pad(img, (0, self._mod_pad_w, 0, self._mod_pad_h), 'reflect')

        logger.debug("pre-process time: {:.04f}".format(time.time() - tic))
        return img

    def _post_process(self, res):
        tic = time.time()
        if self._net_scale == 2:
            _, _, h, w = res.shape
            res = res[:, :, 0:h - self._mod_pad_h, 0:w - self._mod_pad_w]
        res = np.squeeze(res, axis=0)
        res = np.transpose(res[[2, 1, 0], :, :], (1, 2, 0))
        res = np.clip(res, 0, 1)
        res = (res * 255.0).round().astype(np.uint8)

        logger.debug("post-process time: {:.04f}".format(time.time() - tic))
        return res

    def process_prediction(self, image):
        """ face gan model
        input: image
        """
        start_time = time.time()
        super(RealESRGANWrapper, self)._check_model_init(logger=logger)

        img_t = self._pre_process(image)
        with torch.no_grad():
            res = self._model.run([self._output_name], {self._input_name: to_numpy(img_t)})
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
    sr = RealESRGANWrapper(cfg=config.CONFIG_MAP.real_ESRGANx2_ONNX)
    for _ in range(5):
        out, ti = sr.process_prediction(im)
        gpu_usage = utils.get_gpu_memory()
        logger.info("gpu usage: use / total / percent: {} / {} / {}".format(*gpu_usage))
    cv2.imshow("out", cv2.resize(out, im.shape[:2][::-1]))
    cv2.imshow("out", out)
    cv2.waitKey()
    del sr
