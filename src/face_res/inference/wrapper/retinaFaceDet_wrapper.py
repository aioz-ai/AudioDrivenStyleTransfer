"""
aioz.aiar.truonbgle - Dec 07, 2021
Face Detection
@ref: https://github.com/yangxy/GPEN
"""
import os
import cv2
import time
import enum
import torch
import logging
import collections
import numpy as np
import torch.backends.cudnn as cudnn
from inference.wrapper import base_wrapper
from inference.utils.prior_box import PriorBox
from inference.utils.py_cpu_nms import py_cpu_nms
from inference.utils.box_utils import decode, decode_landm

logger = logging.getLogger('inf.wrp.retDet')
# logger = logging.getLogger(__name__)

HParams = collections.namedtuple(
    'HParams',
    ['resize', 'threshold', 'nms_threshold', 'top_k', 'keep_top_k', 'max_size']
)


class MODE(enum.Enum):
    ORIGIN = "torch.load"
    TRACE = "torch.jit.trace"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class RetinaFaceDetWrapper(base_wrapper.BaseWrapper):
    def __init__(self, cfg):
        super().__init__()
        self._model_path = cfg.model_path
        self._mode = cfg.mode
        self._threshold = cfg.hparams.threshold
        self._nms_threshold = cfg.hparams.nms_threshold
        self._top_k = cfg.hparams.top_k
        self._keep_top_k = cfg.hparams.keep_top_k
        self._resize = cfg.hparams.resize
        self._max_size = cfg.hparams.max_size
        torch.set_grad_enabled(False)
        cudnn.benchmark = True
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

    def init(self):
        self._load_model()

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, val):
        self._threshold = val

    def _load_model(self):
        assert MODE.has_value(self._mode), logger.error("Mode {} invalid..".format(self._mode))
        logger.info("Load model at {}".format(self._model_path))
        if self._mode == MODE.ORIGIN.value:
            logger.warning("{} not support, using trace for inference".format(self._mode))
        self._model = torch.jit.load(self._model_path, map_location=self._device)
        logger.info("Load model is DONE.")

    def _pre_process(self, image):
        """image to tensor"""
        tic = time.time()
        img = np.float32(image)

        im_height, im_width = img.shape[:2]
        ss = self._max_size / max(im_height, im_width)
        img = cv2.resize(img, (0, 0), fx=ss, fy=ss)
        im_height, im_width = img.shape[:2]

        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self._device)
        scale = scale.to(self._device)

        logger.debug("pre-process time: {:.04f}".format(time.time() - tic))
        return img, scale, im_width, im_height, ss

    def _post_process(self, loc, conf, landms, img, scale, im_width, im_height, ss):
        tic = time.time()
        priorbox = PriorBox(cfg={'min_sizes': [[16, 32], [64, 128], [256, 512]],
                                 'steps': [8, 16, 32],
                                 'clip': False},
                            image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self._device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, variances=[0.1, 0.2])
        boxes = boxes * scale / self._resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, variances=[0.1, 0.2])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self._device)
        landms = landms * scale1 / self._resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self._threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self._top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self._nms_threshold)
        # keep = nms(dets, nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self._keep_top_k, :]
        landms = landms[:self._keep_top_k, :]

        # sort faces(delete)
        '''
        fscores = [det[4] for det in dets]
        sorted_idx = sorted(range(len(fscores)), key=lambda k:fscores[k], reverse=False) # sort index
        tmp = [landms[idx] for idx in sorted_idx]
        landms = np.asarray(tmp)
        '''

        landms = landms.reshape((-1, 5, 2))
        landms = landms.transpose((0, 2, 1))
        landms = landms.reshape(-1, 10, )

        faces, landmarks = dets / ss, landms / ss
        logger.debug("post-process time: {:.04f}".format(time.time() - tic))
        return faces, landmarks

    def process_prediction(self, image):
        """
        input: image: BGR image
        output: list face box and landmarks
        """
        start_time = time.time()
        # faces, landmarks = [], []
        super(RetinaFaceDetWrapper, self)._check_model_init(logger=logger)
        img, scale, im_width, im_height, ss = self._pre_process(image)

        loc, conf, landms = self._model(img)  # forward pass

        faces, landmarks = self._post_process(loc, conf, landms, img, scale, im_width, im_height, ss)

        elapsed = round(time.time() - start_time, 4)
        logger.debug("Process time: {}".format(elapsed))
        return faces, landmarks, elapsed


# test class
if __name__ == '__main__':
    import sys
    import config

    im = cv2.imread(sys.argv[1])
    detector = RetinaFaceDetWrapper(cfg=config.CONFIG_MAP.retina_faceDet)
    for _ in range(5):
        tic = time.time()
        box, lm, pro_time = detector.process_prediction(im)
        print("Pro time: ", time.time() - tic)
        print(box, lm)

    del detector
