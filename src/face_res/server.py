"""
aioz.aiar.truongle - Dec 08, 2021
server for using face-restoration api
input:
super_resolution: True / False
file: video file / image file
"""
import os
import time

os.environ['LOGGING_LEVEL'] = "INFO"

import io
import cv2
import flask
import config
import logging
import numpy as np
from PIL import Image
import moviepy.editor as mp
from werkzeug.utils import secure_filename
from inference.api.faceRestorationAPI import FaceRestorationAPI
from communications_utils.communications_utils import HTTPStatus
from communications_utils import communications_utils as cm_utils

logger = logging.getLogger("AIServer")
app = flask.Flask(__name__)

face_restoration = FaceRestorationAPI(api_cfg=config.API_CFG)
# init
for _ in range(5):
    _image = cv2.imread("wiki/test_01.jpg")
    _image = cv2.resize(_image, (256, 256))
    _ = face_restoration.proceed(_image)


def ai_proceed(info, image):
    use_sr = info["super_resolution"] if ("super_resolution" in info.keys()) else None
    if use_sr is not None:
        use_sr = True if use_sr in ['True', '1', 'true'] else False
    logger.info("use super resolution: {}".format(use_sr))
    res, elapsed = face_restoration.proceed(image, use_sr=use_sr)

    return res


def ai_proceed_video(info, video_path):
    use_sr = info["super_resolution"] if ("super_resolution" in info.keys()) else None
    if use_sr is not None:
        use_sr = True if use_sr in ['True', '1', 'true'] else False

    keep_sound = info["keep_sound"] if ("keep_sound" in info.keys()) else False
    if isinstance(keep_sound, str):
        keep_sound = True if keep_sound in ['True', '1', 'true'] else False
    logger.info("use super resolution: {}".format(use_sr))

    ext = os.path.splitext(video_path)[-1]
    vid_out_path = video_path.replace(ext, "_res.mp4")

    cap = cv2.VideoCapture(video_path)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w, vid_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info("Video info: fps: {} / width: {} / height: {} / total frame: {}".format(fps, vid_w, vid_h, total_frame))

    count = 0
    write_init = False
    vid_out = None
    tic = time.time()
    while cap.isOpened() and count <= total_frame:
        ret, frame = cap.read()
        if ret:
            out, elapsed = face_restoration.proceed(frame, use_sr=use_sr)
            if not write_init:
                write_init = True
                # fourcc = cv2.VideoWriter_fourcc(*'VP90')
                # fourcc = cv2.VideoWriter_fourcc(*'H264')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vid_out = cv2.VideoWriter(vid_out_path, fourcc, fps, out.shape[:2][::-1])
            vid_out.write(out)
        count += 1
    cap.release()
    if vid_out:
        vid_out.release()
    logger.info("Process time: {:.04f}".format(time.time() - tic))
    if keep_sound:
        video_clip = mp.VideoFileClip(vid_out_path)
        video_clip.audio = mp.VideoFileClip(video_path).audio
        out_path_final = video_path.replace(".mp4", "_sound.mp4")
        video_clip.write_videofile(out_path_final, audio_codec="aac", codec="libx264")
    else:
        video_clip = mp.VideoFileClip(vid_out_path)
        out_path_final = video_path.replace(".mp4", "_convert.mp4")
        video_clip.write_videofile(out_path_final)
    return out_path_final


@app.route("/", methods=["POST", "GET"])
def main_page():
    return "This is main page"


@app.route("/FaceRestoration", methods=["POST", "GET"])
def FaceRestoration():
    logger.debug("Start receive...")
    data_receive, file = cm_utils.receive()
    response = {"status": HTTPStatus.BadRequest,
                "image": None}
    if file is not None:
        ff = file.read()
        image = np.fromstring(ff, np.uint8)
        image = cv2.imdecode(image, 1)
        logger.debug("image shape: {}".format(image.shape))

        try:
            res = ai_proceed(data_receive, image)
            res = cm_utils.im_to_b64(res)
            response['image'] = res
            response['status'] = HTTPStatus.OK

        except Exception as e:
            logger.error(e)
            response['status'] = HTTPStatus.NotImplemented
    else:
        logger.debug("File is None")

    return response


def serve_pil_image(image_arr):
    pil_im = Image.fromarray(image_arr[:, :, ::-1])
    img_io = io.BytesIO()
    pil_im.save(img_io, 'JPEG', quality=90)
    img_io.seek(0)
    return flask.send_file(img_io, mimetype='image/jpeg')


@app.route("/FaceRestorationImage", methods=["POST", "GET"])
def FaceRestorationImage():
    logger.debug("Start receive...")
    data_receive, file = cm_utils.receive()
    if file is not None:
        ff = file.read()
        image = np.fromstring(ff, np.uint8)
        image = cv2.imdecode(image, 1)
        logger.debug("image shape: {}".format(image.shape))

        try:
            res = ai_proceed(data_receive, image)
            return serve_pil_image(res)

        except Exception as e:
            logger.error(e)
            return flask.abort(404)
    else:
        logger.debug("File is None")
        return flask.abort(400)


@app.route("/FaceRestorationVideo", methods=["POST", "GET"])
def FaceRestorationVideo():
    logger.debug("Start receive...")
    data_receive, video = cm_utils.receive()
    if video is not None:
        filename = secure_filename(video.filename)
        ext = os.path.splitext(filename)[-1]
        video_tmp_path = "tmp/tmp%s" % ext
        os.makedirs(os.path.dirname(video_tmp_path), exist_ok=True)
        video.save(video_tmp_path)
        logger.debug("saved tmp video")
        try:
            out_pth = ai_proceed_video(data_receive, video_tmp_path)
            return flask.send_file(out_pth, attachment_filename="response.mp4", mimetype='video/mp4')
        except Exception as e:
            logger.error(e)
            return flask.abort(404)

    else:
        logger.debug("File is None")
        return flask.abort(400)


if __name__ == "__main__":
    app.run(host=config.AI_host.host,
            port=config.AI_host.port,
            debug=True,
            # threaded=True,
            )
    logger.info("Server is ready on http://{}:{}".format(config.AI_host.host, config.AI_host.port))
