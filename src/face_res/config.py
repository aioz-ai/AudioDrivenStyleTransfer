"""
aioz.aiar.truongle - Dec 07, 2021
config
"""
import collections
from inference.wrapper import faceGAN_wrapper
from inference.wrapper import faceGANONNX_wrapper
from inference.wrapper import realESRNet_wrapper
from inference.wrapper import realESRGAN_wrapper
from inference.wrapper import realESRGANONNX_wrapper
from inference.wrapper import retinaFaceDet_wrapper


class Config(collections.namedtuple(
    'Config',
    ['model_path', 'mode', 'hparams'])
):
    def values(self):
        return self._asdict()


Config.__new__.__defaults__ = (None,) * len(Config._fields)


def update_config(config, update_dict):
    config_dict = config.values()
    config_dict.update(update_dict)
    return Config(**config_dict)


CONFIG_MAP = collections.namedtuple(
    "CONFIG_MAP",
    ['retina_faceDet_trace',  # Face Detection
     'GPEN_512', 'GPEN_256', 'GPEN_512_Trace', 'GPEN_512_ONNX',  # Face Restoration
     'real_ESRNet',  # super resolution
     'real_ESRGANx4', 'real_ESRGANx4_trace', 'real_ESRGANx4_ONNX',  # super resolution
     'real_ESRGANx2', 'real_ESRGANx2_trace', 'real_ESRGANx2_ONNX']   # super resolution
)

# Face detection
CONFIG_MAP.retina_faceDet_trace = Config(
    model_path="models/RetinaFace-R50_trace.pt",
    mode=retinaFaceDet_wrapper.MODE.TRACE.value,
    hparams=retinaFaceDet_wrapper.HParams(
        threshold=0.9,
        resize=1,
        nms_threshold=0.4,
        top_k=100,  # 5000
        keep_top_k=50,  # 750
        max_size=512
    )
)

# Face GAN
CONFIG_MAP.GPEN_512 = Config(
    model_path="models/GPEN-BFR-512.pth",
    mode=faceGAN_wrapper.MODE.ORIGIN.value,
    hparams=faceGAN_wrapper.HParams(
        resolution=512,
        is_norm=True,
        n_mlp=8,
        channel_multiplier=2,
        narrow=1
    )
)

CONFIG_MAP.GPEN_512_Trace = Config(
    model_path="models/GPEN-BFR-512_trace.pt",
    mode=faceGAN_wrapper.MODE.TRACE.value,
    hparams=faceGAN_wrapper.HParams(
        resolution=512,
        is_norm=True,
        n_mlp=8,
        channel_multiplier=2,
        narrow=1
    )
)

CONFIG_MAP.GPEN_512_ONNX = Config(
    model_path="models/GPEN-BFR-512.onnx",
    hparams=faceGANONNX_wrapper.HParams(
        resolution=512,
        is_norm=True,
    )
)

CONFIG_MAP.GPEN_256 = Config(
    model_path="models/GPEN-BFR-256.pth",
    hparams=faceGAN_wrapper.HParams(
        resolution=256,
        is_norm=True,
        n_mlp=8,
        channel_multiplier=1,
        narrow=0.5
    )
)


# Super resolution - ESR GAN
CONFIG_MAP.real_ESRNet = Config(
    model_path="models/rrdb_realesrnet_psnr_trace.pt",
    hparams=realESRNet_wrapper.HParams(
        net_scale=2,
    )
)

CONFIG_MAP.real_ESRGANx4 = Config(
    model_path="models/RealESRGAN_x4plus.pth",
    mode=realESRGAN_wrapper.MODE.ORIGIN.value,
    hparams=realESRGAN_wrapper.HParams(
        net_scale=4,
    )
)


CONFIG_MAP.real_ESRGANx4_trace = Config(
    model_path="models/RealESRGAN_x4plus_trace.pt",
    mode=realESRGAN_wrapper.MODE.TRACE.value,
    hparams=realESRGAN_wrapper.HParams(
        net_scale=4,
    )
)

CONFIG_MAP.real_ESRGANx2_ONNX = Config(
    model_path="models/RealESRGAN_x2plus_dynamicAxe.onnx",
    hparams=realESRGANONNX_wrapper.HParams(
        net_scale=2,
    )
)

CONFIG_MAP.real_ESRGANx2 = Config(
    model_path="models/RealESRGAN_x2plus.pth",
    mode=realESRGAN_wrapper.MODE.ORIGIN.value,
    hparams=realESRGAN_wrapper.HParams(
        net_scale=2,
    )
)

CONFIG_MAP.real_ESRGANx2_trace = Config(
    model_path="models/RealESRGAN_x2plus_trace.pt",
    mode=realESRGAN_wrapper.MODE.TRACE.value,
    hparams=realESRGAN_wrapper.HParams(
        net_scale=2,
    )
)

CONFIG_MAP.real_ESRGANx2_ONNX = Config(
    model_path="models/RealESRGAN_x2plus_dynamicAxe.onnx",
    hparams=realESRGANONNX_wrapper.HParams(
        net_scale=2,
    )
)

# API
Api_cfg = collections.namedtuple(
    'Api_cfg',
    ['face_detector', 'face_restoration', 'super_resolution']
)
Api_cfg.__new__.__defaults__ = (None,) * len(Api_cfg._fields)
API_CFG = Api_cfg(
    face_detector=retinaFaceDet_wrapper.RetinaFaceDetWrapper(cfg=CONFIG_MAP.retina_faceDet_trace),
    face_restoration=faceGAN_wrapper.FaceGANWrapper(cfg=CONFIG_MAP.GPEN_512_Trace),
    # face_restoration=faceGANONNX_wrapper.FaceGANWrapper(cfg=CONFIG_MAP.GPEN_512_ONNX),
    super_resolution=realESRGAN_wrapper.RealESRGANWrapper(cfg=CONFIG_MAP.real_ESRGANx2_trace),
    # super_resolution=realESRGANONNX_wrapper.RealESRGANWrapper(cfg=CONFIG_MAP.real_ESRGANx2_ONNX)
)

# Host
Host_info = collections.namedtuple(
    'Host_info',
    ['host', 'port']
)
AI_host = Host_info(host="0.0.0.0", port="8088")
