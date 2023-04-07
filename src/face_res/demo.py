"""
aioz.aiar.truongle - Dec 08, 2021
demo face restoration api
"""
import os
os.environ['LOGGING_LEVEL'] = "INFO"
import gc
import torch
gc.collect()
torch.cuda.empty_cache()

import sys
import cv2
import glob
import time
import config
import moviepy.editor as mp
from inference.api.faceRestorationAPI import FaceRestorationAPI
from tqdm import tqdm 

IN_PUT = sys.argv[1]
USE_SR = True if "sr" in sys.argv else False
SAVE_RESULT = True if "save" in sys.argv else False
KEEP_SOUND = True if "sound" in sys.argv else False
if KEEP_SOUND:
    SAVE_RESULT = True

VIDEO_EXT = [".mp4", ".mov", ".wav", ".avi"]


def main():
    face_restoration = FaceRestorationAPI(api_cfg=config.API_CFG)

    # trick
    for _ in range(5):
        image = cv2.imread("wiki/er.jpg")
        image = cv2.resize(image, (256, 256))
        _ = face_restoration.proceed(image, use_sr=USE_SR)

    if os.path.isfile(IN_PUT):  # run with 1 image or video
        ext = os.path.splitext(IN_PUT)[-1]
        if ext in VIDEO_EXT:
            tic = time.time()
            cap = cv2.VideoCapture(IN_PUT)
            total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            vid_w, vid_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            write_init = False
            vid_out = None
            count = 0
            pbar = tqdm(total = total_frame+1)
            while cap.isOpened() and count <= total_frame:
                ret, frame = cap.read()
                if ret:
                    with torch.cuda.amp.autocast():
                        out, elapsed = face_restoration.proceed(frame, use_sr=USE_SR)

                    if SAVE_RESULT:
                        if not write_init:
                            write_init = True
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # vp90
                            vid_out_pth = IN_PUT.replace(ext, "_result_sr%s.mp4" % str(USE_SR))
                            vid_out = cv2.VideoWriter(vid_out_pth, fourcc, fps, out.shape[:2][::-1])
                        vid_out.write(out)
                    
                    # cv2.imshow("res", out)
                count += 1
                pbar.update(1)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

            toc = time.time() - tic
            print("video info: ", vid_w, vid_h, fps, total_frame)
            print("total frame / total time process / fps: {} / {:.04f} / {:.04f}".format(total_frame, toc, total_frame / toc))
            cap.release()
            if vid_out:
                vid_out.release()

            if KEEP_SOUND:
                video_clip = mp.VideoFileClip(vid_out_pth)
                video_clip.audio = mp.VideoFileClip(IN_PUT).audio
                video_clip.write_videofile(vid_out_pth.replace(".mp4", "_sound.mp4"))

        else:
            im_pth = IN_PUT
            im = cv2.imread(im_pth)
            out, elapsed = face_restoration.proceed(im, use_sr=USE_SR)
            if SAVE_RESULT:
                out_pth = im_pth.replace(".", "_result_sr%s." % str(USE_SR))
                cv2.imwrite(out_pth, out)
                print("Saved output image at {}".format(out_pth))
            # cv2.imshow("out", out)
            # cv2.waitKey()

    elif os.path.isdir(IN_PUT):  # run with multi image in dir
        list_file = glob.glob("%s/*.*g" % IN_PUT)
        for im_pth in list_file:
            print(im_pth)
            im = cv2.imread(im_pth)
            ori_h, ori_w, _ = im.shape
            im = cv2.resize(im, (224, 224))
            out, elapsed = face_restoration.proceed(im, use_sr=USE_SR)
            out = cv2.resize(out, (int(ori_w*2), int(ori_h*2)))
            if SAVE_RESULT:
                out_pth = im_pth.replace(".", "_result_sr%s." % str(USE_SR))
                cv2.imwrite(out_pth, out)
                print("Saved output image at {}".format(out_pth))
            cv2.imshow("out", out)
            cv2.waitKey()

    else:
        print("input not exist")


if __name__ == '__main__':
    main()
