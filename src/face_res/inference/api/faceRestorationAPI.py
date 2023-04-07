"""
aioz.aiar.truongle - Dec 07, 2021
face restoration
input: image
output: image with super resolution
@ref: https://github.com/yangxy/GPEN
"""
import os
import cv2
import time
import logging
import collections
import numpy as np
from inference.api import base_api
from inference.utils.align_faces import warp_and_crop_face, get_reference_facial_points

logger = logging.getLogger("inf.faceResAPI")


class FaceRestorationAPI(base_api.BaseAPI):
    def __init__(self, api_cfg):
        self._init_modules(api_cfg)

        # the mask for pasting restored faces back
        self._mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self._mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self._mask = cv2.GaussianBlur(self._mask, (101, 101), 11)
        # self._mask = cv2.GaussianBlur(self._mask, (101, 101), 11)

        self._gausian_kernel = np.array((
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]), dtype="float32")

    def __del__(self):
        del self._super_resolution
        del self._face_restoration
        del self._face_detector

    def _init_modules(self, api_cfg):
        logger.info("Init a modules ....")
        self._face_detector = api_cfg.face_detector
        self._face_detector.init()
        self._face_restoration = api_cfg.face_restoration
        self._face_restoration.init()
        self._super_resolution = api_cfg.super_resolution
        self._super_resolution.init()
        logger.info("Init DONE")

    @staticmethod
    def _get_reference_5pts(resolution):
        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        return get_reference_facial_points(
                (resolution, resolution), inner_padding_factor, outer_padding, default_square)

    def proceed(self, image, use_sr=None):
        if use_sr is None:
            use_sr = True
        start_time = time.time()
        sr_scale = self._super_resolution.net_scale
        res = cv2.resize(image, (0, 0), fx=sr_scale, fy=sr_scale)
        try:
            # SUPER RESOLUTION
            resolution = self._face_restoration.resolution
            if use_sr:
                img_sr, sr_time = self._super_resolution.process_prediction(image)
                if img_sr is not None:
                    image = cv2.resize(image, img_sr.shape[:2][::-1])
                else:
                    image = cv2.resize(image, (0, 0), fx=sr_scale, fy=sr_scale)
            else:
                image = cv2.resize(image, (0, 0), fx=sr_scale, fy=sr_scale)

            # FACE DETECTION
            face_boxes, land_marks, det_time = self._face_detector.process_prediction(image)

            height, width = image.shape[:2]
            full_mask = np.zeros((height, width), dtype=np.float32)
            full_img = np.zeros(image.shape, dtype=np.uint8)

            for i, (face_box, facial5points) in enumerate(zip(face_boxes, land_marks)):
                fh, fw = (face_box[3] - face_box[1]), (face_box[2] - face_box[0])
                facial5points = np.reshape(facial5points, (2, 5))

                # FACE ALIGNMENT
                of, tfm_inv = warp_and_crop_face(image, facial5points, reference_pts=self._get_reference_5pts(resolution),
                                                 crop_size=(resolution, resolution))

                # FACE RESTORATION
                ef, fgan_time = self._face_restoration.process_prediction(of)

                # AFFINE TRANSFORMATION
                tmp_mask = self._mask
                tmp_mask = cv2.resize(tmp_mask, ef.shape[:2])
                tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

                if min(fh, fw) < 100:  # gaussian filter for small faces
                    ef = cv2.filter2D(ef, -1, self._gausian_kernel)

                tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

                mask = tmp_mask - full_mask
                full_mask[np.where(mask > 0)] = tmp_mask[np.where(mask > 0)]
                full_img[np.where(mask > 0)] = tmp_img[np.where(mask > 0)]

            full_mask = full_mask[:, :, np.newaxis]
            if use_sr and img_sr is not None:
                res = cv2.convertScaleAbs(img_sr * (1 - full_mask) + full_img * full_mask)
            else:
                res = cv2.convertScaleAbs(image * (1 - full_mask) + full_img * full_mask)

        except Exception as e:
            logger.error(e)
            pass

        elapsed = round(time.time() - start_time, 4)
        logger.debug("Process time: {}".format(elapsed))
        return res, elapsed


# test class
if __name__ == '__main__':
    import sys
    import config
    from inference.utils import utils

    use_sr = True if "sr" in sys.argv else False
    save = True if "save" in sys.argv else False
    fr = FaceRestorationAPI(api_cfg=config.API_CFG)
    assert os.path.isfile(sys.argv[1]), "Image file not exists"
    for _ in range(5):
        print("---")
        im = cv2.imread(sys.argv[1])
        out, ti = fr.proceed(im, use_sr=use_sr)
        gpu_usage = utils.get_gpu_memory()
        logger.info("gpu usage: use / total / percent: {} / {} / {}".format(*gpu_usage))
    if save:
        out_pth = sys.argv[1].replace(".", "_sr%s." % str(use_sr))
        cv2.imwrite(out_pth, out)
        print("Saved output image at {}".format(out_pth))
    cv2.imshow("out", out)
    cv2.waitKey()


