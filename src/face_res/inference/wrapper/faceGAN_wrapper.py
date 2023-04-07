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
from inference.wrapper import base_wrapper


logger = logging.getLogger('inf.wrp.faceGan')

HParams = collections.namedtuple(
    'HParams',
    ['resolution', 'is_norm', 'n_mlp', 'channel_multiplier', 'narrow']
)


class MODE(enum.Enum):
    ORIGIN = "torch.load"
    TRACE = "torch.jit.trace"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class FaceGANWrapper(base_wrapper.BaseWrapper):
    def __init__(self, cfg):
        super().__init__()
        self._model_path = cfg.model_path
        self._mode = cfg.mode
        self._resolution = cfg.hparams.resolution
        self._is_norm = cfg.hparams.is_norm
        self._n_mlp = cfg.hparams.n_mlp
        self._channel_multiplier = cfg.hparams.channel_multiplier
        self._narrow = cfg.hparams.narrow
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
        assert MODE.has_value(self._mode), logger.error("Mode {} invalid..".format(self._mode))
        logger.info("Load model at {}".format(self._model_path))
        if self._mode == MODE.ORIGIN.value:
            from inference.utils.face_model.model import FullGenerator
            self._model = FullGenerator(
                self._resolution, 512,
                self._n_mlp, self._channel_multiplier, narrow=self._narrow)
            pretrained_dict = torch.load(self._model_path)
            self._model.load_state_dict(pretrained_dict)
            self._model.eval()
            self._model.to(self._device)
        else:  # use trace
            logger.info("Using {}".format(self._mode))
            self._model = torch.jit.load(self._model_path, map_location=self._device)

        logger.info("Load model is DONE.")

    def _pre_process(self, image):
        """image to tensor"""
        tic = time.time()
        image = cv2.resize(image, (self._resolution, self._resolution))
        img_t = torch.from_numpy(image).to(self._device)
        img_t = img_t / 255.0
        if self._is_norm:
            img_t = (img_t - 0.5) / 0.5
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).flip(1)  # BGR->RGB
        logger.debug("pre-process time: {:.04f}".format(time.time() - tic))
        return img_t

    def _post_process(self, img_t, im_type=np.uint8, mode="tensor"):
        tic = time.time()
        if mode == "tensor":
            if self._is_norm:
                img_t = img_t * 0.5 + 0.5
            img_t = img_t.squeeze(0).permute(1, 2, 0).flip(2)  # RGB->BGR
            img_np = np.clip(img_t.float().cpu().numpy(), 0, 1) * 255.0
        else:
            ti = time.time()
            img_np = img_t.detach().cpu().numpy()
            print(time.time() - ti)
            if self._is_norm:
                img_np = img_np * 0.5 + 0.5
            img_np = np.squeeze(img_np, 0)
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = np.clip(np.flip(img_np, 2), 0, 1) * 255.0
        logger.debug("post-process time: {:.04f}".format(time.time() - tic))
        return img_np.astype(im_type)

    def process_prediction(self, image):
        """ face gan model
        input: image
        """
        start_time = time.time()
        super(FaceGANWrapper, self)._check_model_init(logger=logger)

        img_t = self._pre_process(image)
        with torch.no_grad():
            res = self._model(img_t)
        res = self._post_process(res)

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
    fg = FaceGANWrapper(cfg=config.CONFIG_MAP.GPEN_512_Trace)
    for _ in range(5):
        out, ti = fg.process_prediction(im)
        gpu_usage = utils.get_gpu_memory()
        logger.info("gpu usage: use / total / percent: {} / {} / {}".format(*gpu_usage))
    cv2.imshow("out", cv2.resize(out, im.shape[:2][::-1]))
    cv2.imshow("out", out)
    cv2.waitKey()
    del fg
