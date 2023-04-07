"""
utils function
"""
import cv2
import flask
import base64
import requests
import numpy as np


class HTTPStatus:
    OK = 200
    BadRequest = 400
    NotFound = 404
    NotImplemented = 501
    ServiceUnavailable = 503


def sent(url=None,
         ip_of_receiver="0.0.0.0", port_of_receiver="7000", channel='cm_channel?',
         json_data=None, files=None, time_out=1):
    """sent request to url or `http://[ip_of_receiver]:[port_of_receiver]/[channel]`
    """
    if url is None:
        url = 'http://' + ip_of_receiver + ':' \
            + port_of_receiver + '/' \
            + channel
    if files is None:
        response = requests.post(
            url=url,
            json=json_data,
            files=None,
            timeout=time_out,
        )
    else:
        # files = {'file': open(file_pth, 'rb')}
        response = requests.post(
            url=url,
            data=json_data,
            files=files,
            timeout=time_out,
        )
    return response


def receive():
    """
    receive data at route (ex: @app.route("/cm_global_channel", methods=["POST", "GET"]) )
    """
    if flask.request.is_json:
        data_receive = flask.request.get_json()
    else:
        data_receive = flask.request.form.to_dict(flat=True)

    if flask.request.files:
        file = flask.request.files['file']
    else:
        file = None

    return data_receive, file


def im_to_b64(im):
    """encode image nd-array to base64
    """
    buffer = cv2.imencode('.png', im)[1]
    im_str = base64.b64encode(buffer).decode('utf-8')
    return im_str


def b64_to_arr(b64_data):
    """decode base64 to image nd-array
    """
    img_original = base64.b64decode(b64_data)
    img_as_np = np.frombuffer(img_original, dtype=np.uint8)
    return cv2.imdecode(img_as_np, flags=1)

