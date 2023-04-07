"""
aioz.aiar.truonbgle - Dec 07, 2021
ERS GAN
@ref: https://github.com/yangxy/GPEN
"""
import os
import time
import torch
import logging
import collections
import numpy as np
from torch.nn import functional as F
from inference.wrapper import base_wrapper

logger = logging.getLogger('inf.wrp.ESRNet')

HParams = collections.namedtuple(
    'HParams',
    ['net_scale']
)


class RealESRNetWrapper(base_wrapper.BaseWrapper):
    def __init__(self, cfg):
        super().__init__()
        self._model_path = cfg.model_path
        self._net_scale = cfg.hparams.scale

    def __del__(self):
        logger.info("Delete object")

    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self._model = torch.jit.load(self._model_path, map_location=self.device)
        logger.info("Load model is DONE.")

    def _pre_process(self, image):
        img = image.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).cuda() if self.device == torch.device('cuda') else img.unsqueeze(0).cpu()

        if self._net_scale == 2:
            mod_scale = 2
        else:  # scale = 1
            mod_scale = 4

        h_pad, w_pad = 0, 0
        _, _, h, w = img.size()
        if h % mod_scale != 0:
            h_pad = (mod_scale - h % mod_scale)
        if w % mod_scale != 0:
            w_pad = (mod_scale - w % mod_scale)
        img = F.pad(img, (0, w_pad, 0, h_pad), 'reflect')

        return img, h_pad, w_pad

    def _post_process(self, res, h_pad, w_pad):
        # remove extra pad
        _, _, h, w = res.size()
        res = res[:, :, 0:h - h_pad, 0:w - w_pad]
        res = res.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        res = np.transpose(res[[2, 1, 0], :, :], (1, 2, 0))
        res = (res * 255.0).round().astype(np.uint8)
        return res

    def process_prediction(self, image):
        """
        ESRGAN model
        """
        res = None
        start_time = time.time()
        super(RealESRNetWrapper, self)._check_model_init(logger=logger)
        assert self._net_scale == 2, logger.error("Only scale 2 are supported")
        tic = time.time()
        img, h_pad, w_pad = self._pre_process(image)
        logger.debug("Pre-process time: {:.4f}".format(time.time() - tic))

        try:
            with torch.no_grad():
                tic = time.time()
                res = self._model(img)
                logger.debug("AI process time: {:.4f}".format(time.time() - tic))

            tic = time.time()
            res = self._post_process(res, h_pad, w_pad)
            logger.debug("Post-process time: {:.4f}".format(time.time() - tic))
        except Exception as e:
            logger.error(e)
            pass

        elapsed = round(time.time() - start_time, 4)
        logger.debug("Process time: {}".format(elapsed))
        return res, elapsed


# test class
if __name__ == '__main__':
    import cv2
    import sys
    import config
    im = cv2.imread(sys.argv[1])
    sr_model = RealESRNetWrapper(cfg=config.CONFIG_MAP.real_ESRNet)
    for _ in range(3):
        print("\n")
        out, _ = sr_model.process_prediction(im)

    del sr_model

    # cv2.imwrite("tmp_sr.jpg", out)
    cv2.imshow("out", out)
    cv2.waitKey()
