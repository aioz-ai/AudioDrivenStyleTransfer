import numpy as np
from torch import normal
import cv2
import pandas as pd
import os, glob
from imutils import face_utils
from tqdm import tqdm
from pathlib import Path 
from typing import *
import subprocess
import shutil
import pandas as pd
import dlib

def split_frame_all_video_in_folder(folder_video: str):
    """To split all videos in the given folder into frames

    Args:
        folder_video (str): _description_
    """
    folder_path = Path(folder_video)
    for vid in folder_path.glob("*.mp4"):
        video2sequence(vid)


def video2sequence(video_path, frame_rate = 29.97):
    """Crop video into sequence frames by normalize them intro the same fps then split them. 

    Args:
        video_path (_type_): _description_
        frame_rate (float, optional): _description_. Defaults to 29.97.

    Returns:
        _type_: _description_
    """
    if not os.path.exists(video_path):
        print("Video path not exists!")
        return
    os.makedirs("tmp", exist_ok=True)
    #Resample the video frame_rate
    command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r %f %s" % (video_path, frame_rate,"tmp/tmp_video.mp4"))
    output = subprocess.call(command, shell=True, stdout=None)
    videofolder = os.path.splitext(video_path)[0]
    if os.path.exists(videofolder):
        shutil.rmtree(videofolder)  #remove all existing files
    os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    vidcap = cv2.VideoCapture("tmp/tmp_video.mp4")
    #vidcap= cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = os.path.join(videofolder, f'{video_name}_frame{count:04d}.jpg')
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    #os.remove("tmp/tmp_video.mp4")
    return imagepath_list


def generating_landmark_lips(video_path):
    """Generate landmark lips npy from 1 video 

    Args:
        video_path (_type_): _description_
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    videofolder = os.path.splitext(video_path)[0]
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    
    for i in tqdm(range(len(glob.glob(f"{videofolder}/*.jpg")) )):
        image_name = f'{video_name}_frame{i:04d}.jpg'
        imagepath = os.path.join(videofolder, image_name)        
        image= cv2.imread(imagepath)
        os.makedirs(os.path.join(videofolder,'landmark'),exist_ok=True)
        lm = os.path.join(videofolder,'landmark', os.path.splitext(image_name)[0] + '.npy' )
        i_lm = os.path.join(videofolder,'landmark', os.path.splitext(image_name)[0] + '.jpg' )
        image = cv2.resize(image, (512,512), interpolation = cv2.INTER_AREA)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(image_gray, 1)
        if rects is None:
            print('--------------------------------')
        for (i,rect) in enumerate(rects):
            shape = predictor(image_gray, rect)
            shape = face_utils.shape_to_np(shape)
            np.save(lm,shape)
            

def generate_landmark_lips_folder(DATA_DIR):
    """To generate landmark npy for a video folder

    Args:
        DATA_DIR (_type_): _description_
    """
    for file in os.listdir(DATA_DIR):
        if os.path.splitext(file)[-1] not in [".mp4", ".avi"]:
            continue
        print(file)
        generating_landmark_lips(os.path.join(DATA_DIR, file))
def min_compare_landmarks_mouth(gen_video_path:str, gt_video_path:str, start=0, stride:int = 1, length:int = 20, max_wh:np.ndarray = np.ones(2), normalize:bool = False, cache_dict:dict = {}):
    """The min term in the style equation 

    Args:
        gen_video_path (str): _description_
        gt_video_path (str): _description_
        start (int, optional): _description_. Defaults to 0.
        stride (int, optional): _description_. Defaults to 1.
        length (int, optional): _description_. Defaults to 20.
        max_wh (np.ndarray, optional): _description_. Defaults to np.ones(2).
        normalize (bool, optional): _description_. Defaults to False.
        cache_dict (dict, optional): _description_. Defaults to {}.

    Returns:
        _type_: _description_
    """
    gen_videofolder = os.path.splitext(gen_video_path)[0]
    gen_video_name = os.path.splitext(os.path.split(gen_video_path)[-1])[0]
    gt_videofolder = os.path.splitext(gt_video_path)[0]
    gt_video_name = os.path.splitext(os.path.split(gt_video_path)[-1])[0]
    distances = []
    lmds = [] 
    Nj = len(glob.glob(f"{gen_videofolder}/*.jpg"))
    for j in range(0, Nj-length+1, stride):
        for i,k in zip(range(start, start+length), range(j, j+length)):
            gt_image_name = f'{gen_video_name}_frame{i:04d}.jpg'
            gen_image_name = f'{gen_video_name}_frame{k:04d}.jpg'
            keygt = f'gt_{gt_image_name}'
            keygen = f'gen_{gen_image_name}'

            try:
                mouth_start_idx, mouth_end_idx = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
                if keygen not in cache_dict.keys():
                    fp = np.load(os.path.join(gen_videofolder,'landmark', os.path.splitext(gen_image_name)[0] + '.npy'))
                    fp = fp[mouth_start_idx:mouth_end_idx]
                    if normalize:
                        fp = (fp ) / max_wh
                    else:
                        fp = fp - np.sum(fp,axis=0) / 20.0
                    cache_dict[keygen] = fp 
                else:
                    fp = cache_dict[keygen]
                
                if keygt not in cache_dict.keys():
                    rp = np.load(os.path.join(gt_videofolder,'landmark', os.path.splitext(gt_image_name)[0] + '.npy'))
                    rp = rp[mouth_start_idx:mouth_end_idx]
                    if normalize:
                        rp = (rp) / max_wh
                    else:
                        rp = rp - np.sum(rp,axis=0) / 20.0
                    cache_dict[keygt] = rp 
                else:
                    rp = cache_dict[keygt]
                dis = (rp-fp)**2
                dis = np.sum(dis,axis=1)
                dis = np.sqrt(dis)
                dis = np.mean(dis,axis=0)
                if normalize:
                    distances.append(dis*100)
                else:
                    distances.append(dis)
            except:
                print("Landmark not found: ",gen_image_name)
        lmds.append(np.mean(distances))
    return np.min(np.array(lmds)), cache_dict

def full_term_compare_landmarks_mouth(gen_video_path:str, gt_video_path:str, stride:int = 1, length:int = 20, normalize:bool = False):
    """Full term style equation

    Args:
        gen_video_path (str): _description_
        gt_video_path (str): _description_
        stride (int, optional): _description_. Defaults to 1.
        length (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """
    gen_videofolder = os.path.splitext(gen_video_path)[0]
    gen_video_name = os.path.splitext(os.path.split(gen_video_path)[-1])[0]
    gt_videofolder = os.path.splitext(gt_video_path)[0]
    gt_video_name = os.path.splitext(os.path.split(gt_video_path)[-1])[0]
    distances = []
    max_wh = np.ones(2)
    # normalize term should be the same for all
    for i in range(len(glob.glob(f"{gen_videofolder}/*.jpg"))):
        image_name = f'{gen_video_name}_frame{i:04d}.jpg'
        fp = np.load(os.path.join(gen_videofolder,'landmark', os.path.splitext(image_name)[0] + '.npy'))
        rp = np.load(os.path.join(gt_videofolder,'landmark', os.path.splitext(image_name)[0] + '.npy'))
        mouth_start_idx, mouth_end_idx = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
        fp,rp = fp[mouth_start_idx:mouth_end_idx], rp[mouth_start_idx:mouth_end_idx]
        left = np.min(rp[:,0]); right = np.max(rp[:,0]); 
        top = np.min(rp[:,1]); bottom = np.max(rp[:,1])
        wh = [right-left, bottom-top]
        max_wh = np.maximum(max_wh, wh) #elment wise max
    lmds = []
    cache_dict={}
    gen_videofolder = os.path.splitext(gen_video_path)[0]
    N = len(glob.glob(f"{gen_videofolder}/*.jpg"))
    for i in tqdm(range(0, N-length+1, stride)):
        res, cache_dict = min_compare_landmarks_mouth(gen_video_path, gt_video_path, start=i, stride=stride, length=length, max_wh=max_wh, normalize=normalize, cache_dict=cache_dict)
        lmds.append(res)
    lmds = np.array(lmds)
    return np.mean(lmds)


def min_compare_landmarks(gen_video_path:str, gt_video_path:str, start=0, stride:int = 1, length:int = 20, max_wh:np.ndarray = np.ones(2), normalize:bool = False, cache_dict:dict = {}):
    """The min term in the style equation 

    Args:
        gen_video_path (str): _description_
        gt_video_path (str): _description_
        start (int, optional): _description_. Defaults to 0.
        stride (int, optional): _description_. Defaults to 1.
        length (int, optional): _description_. Defaults to 20.
        max_wh (np.ndarray, optional): _description_. Defaults to np.ones(2).
        normalize (bool, optional): _description_. Defaults to False.
        cache_dict (dict, optional): _description_. Defaults to {}.

    Returns:
        _type_: _description_
    """
    gen_videofolder = os.path.splitext(gen_video_path)[0]
    gen_video_name = os.path.splitext(os.path.split(gen_video_path)[-1])[0]
    gt_videofolder = os.path.splitext(gt_video_path)[0]
    gt_video_name = os.path.splitext(os.path.split(gt_video_path)[-1])[0]
    distances = []
    lmds = [] 
    Nj = len(glob.glob(f"{gen_videofolder}/*.jpg"))
    for j in range(0, Nj-length+1, stride):
        for i,k in zip(range(start, start+length), range(j, j+length)):
            gt_image_name = f'{gen_video_name}_frame{i:04d}.jpg'
            gen_image_name = f'{gen_video_name}_frame{k:04d}.jpg'
            keygt = f'gt_{gt_image_name}'
            keygen = f'gen_{gen_image_name}'

            try:
                if keygen not in cache_dict.keys():
                    fp = np.load(os.path.join(gen_videofolder,'landmark', os.path.splitext(gen_image_name)[0] + '.npy'))
                    fp = (fp - np.min(fp,axis=0)) / (np.max(fp,axis=0)[0] - np.min(fp,axis=0)[0])
                    cache_dict[keygen] = fp 
                else:
                    fp = cache_dict[keygen]
                
                if keygt not in cache_dict.keys():
                    rp = np.load(os.path.join(gt_videofolder,'landmark', os.path.splitext(gt_image_name)[0] + '.npy'))
                    rp = (rp - np.min(rp,axis=0)) / (np.max(rp,axis=0)[0] - np.min(rp,axis=0)[0])
                    cache_dict[keygt] = rp 
                else:
                    rp = cache_dict[keygt]
                dis = (rp-fp)**2
                dis = np.sum(dis,axis=1)
                dis = np.sqrt(dis)
                dis = np.mean(dis,axis=0)
                distances.append(dis*100)
            except:
                print("Landmark not found: ",gen_image_name)
        lmds.append(np.mean(distances))
    return np.min(np.array(lmds)), cache_dict

def full_term_compare_landmarks(gen_video_path:str, gt_video_path:str, stride:int = 1, length:int = 20, normalize:bool = False):
    """Full term style equation

    Args:
        gen_video_path (str): _description_
        gt_video_path (str): _description_
        stride (int, optional): _description_. Defaults to 1.
        length (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """
    gen_videofolder = os.path.splitext(gen_video_path)[0]
    gen_video_name = os.path.splitext(os.path.split(gen_video_path)[-1])[0]
    gt_videofolder = os.path.splitext(gt_video_path)[0]
    gt_video_name = os.path.splitext(os.path.split(gt_video_path)[-1])[0]
    distances = []
    max_wh = np.ones(2)
    # normalize term should be the same for all
    for i in range(len(glob.glob(f"{gen_videofolder}/*.jpg"))):
        image_name = f'{gen_video_name}_frame{i:04d}.jpg'
        fp = np.load(os.path.join(gen_videofolder,'landmark', os.path.splitext(image_name)[0] + '.npy'))
        rp = np.load(os.path.join(gt_videofolder,'landmark', os.path.splitext(image_name)[0] + '.npy'))
        mouth_start_idx, mouth_end_idx = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
        fp,rp = fp[mouth_start_idx:mouth_end_idx], rp[mouth_start_idx:mouth_end_idx]
        left = np.min(rp[:,0]); right = np.max(rp[:,0]); 
        top = np.min(rp[:,1]); bottom = np.max(rp[:,1])
        wh = [right-left, bottom-top]
        max_wh = np.maximum(max_wh, wh) #elment wise max
    lmds = []
    cache_dict={}
    gen_videofolder = os.path.splitext(gen_video_path)[0]
    N = len(glob.glob(f"{gen_videofolder}/*.jpg"))
    for i in tqdm(range(0, N-length+1, stride)):
        res, cache_dict = min_compare_landmarks(gen_video_path, gt_video_path, start=i, stride=stride, length=length, max_wh=max_wh, normalize=normalize, cache_dict=cache_dict)
        lmds.append(res)
    lmds = np.array(lmds)
    return np.mean(lmds)


def min_compare_landmarks_velocity(gen_video_path:str, gt_video_path:str, start=0, stride:int = 1, length:int = 20, max_wh:np.ndarray = np.ones(2), normalize:bool = False, cache_dict:dict = {}):
    """The min term in the style equation 

    Args:
        gen_video_path (str): _description_
        gt_video_path (str): _description_
        start (int, optional): _description_. Defaults to 0.
        stride (int, optional): _description_. Defaults to 1.
        length (int, optional): _description_. Defaults to 20.
        max_wh (np.ndarray, optional): _description_. Defaults to np.ones(2).
        normalize (bool, optional): _description_. Defaults to False.
        cache_dict (dict, optional): _description_. Defaults to {}.

    Returns:
        _type_: _description_
    """
    gen_videofolder = os.path.splitext(gen_video_path)[0]
    gen_video_name = os.path.splitext(os.path.split(gen_video_path)[-1])[0]
    gt_videofolder = os.path.splitext(gt_video_path)[0]
    gt_video_name = os.path.splitext(os.path.split(gt_video_path)[-1])[0]
    distances = []
    lmds = [] 
    Nj = len(glob.glob(f"{gen_videofolder}/*.jpg"))
    for j in range(0, Nj-length+1, stride):
        for i,k in zip(range(start+1, start+length), range(j+1, j+length)):
            gt_image_name = f'{gen_video_name}_frame{i:04d}.jpg'
            gen_image_name = f'{gen_video_name}_frame{k:04d}.jpg'
            keygt = f'gt_{gt_image_name}'
            keygen = f'gen_{gen_image_name}'
            pre_gt_image_name = f'{gen_video_name}_frame{i-1:04d}.jpg'
            pre_gen_image_name = f'{gen_video_name}_frame{k-1:04d}.jpg'
            pre_keygt = f'gt_{pre_gt_image_name}'
            pre_keygen = f'gen_{pre_gen_image_name}'
            try:
                if keygen not in cache_dict.keys():
                    fp = np.load(os.path.join(gen_videofolder,'landmark', os.path.splitext(gen_image_name)[0] + '.npy'))
                    fp = fp - np.min(fp,axis=0)
                    cache_dict[keygen] = fp 
                else:
                    fp = cache_dict[keygen]
                
                if keygt not in cache_dict.keys():
                    rp = np.load(os.path.join(gt_videofolder,'landmark', os.path.splitext(gt_image_name)[0] + '.npy'))
                    rp = rp - np.min(rp,axis=0)
                    cache_dict[keygt] = rp 
                else:
                    rp = cache_dict[keygt]
                if pre_keygen not in cache_dict.keys():
                    pre_fp = np.load(os.path.join(gen_videofolder,'landmark', os.path.splitext(pre_gen_image_name)[0] + '.npy'))
                    pre_fp = pre_fp - np.min(pre_fp,axis=0)
                    cache_dict[pre_keygen] = pre_fp 
                else:
                    pre_fp = cache_dict[pre_keygen]
                
                if pre_keygt not in cache_dict.keys():
                    pre_rp = np.load(os.path.join(gt_videofolder,'landmark', os.path.splitext(pre_gt_image_name)[0] + '.npy'))
                    pre_rp = pre_rp - np.min(pre_rp,axis=0)
                    cache_dict[pre_keygt] = pre_rp 
                else:
                    pre_rp = cache_dict[pre_keygt]

                fv = fp - pre_fp 
                rv = rp - pre_rp
                fv = fv /  (np.max(fp,axis=0)[0] - np.min(fp,axis=0)[0])
                rv = rv /  (np.max(rp,axis=0)[0] - np.min(rp,axis=0)[0])
                
                dis = (rv-fv)**2
                dis = np.sum(dis,axis=1)
                dis = np.sqrt(dis)
                dis = np.mean(dis,axis=0)
                distances.append(dis*100)
            except:
                print("Landmark not found: ",gen_image_name)
        lmds.append(np.mean(distances))
    return np.min(np.array(lmds)), cache_dict

def full_term_compare_landmarks_velocity(gen_video_path:str, gt_video_path:str, stride:int = 1, length:int = 20, normalize:bool = False):
    """Full term style equation

    Args:
        gen_video_path (str): _description_
        gt_video_path (str): _description_
        stride (int, optional): _description_. Defaults to 1.
        length (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """
    gen_videofolder = os.path.splitext(gen_video_path)[0]
    gen_video_name = os.path.splitext(os.path.split(gen_video_path)[-1])[0]
    gt_videofolder = os.path.splitext(gt_video_path)[0]
    gt_video_name = os.path.splitext(os.path.split(gt_video_path)[-1])[0]
    distances = []
    max_wh = np.ones(2)
    # normalize term should be the same for all
    for i in range(len(glob.glob(f"{gen_videofolder}/*.jpg"))):
        image_name = f'{gen_video_name}_frame{i:04d}.jpg'
        fp = np.load(os.path.join(gen_videofolder,'landmark', os.path.splitext(image_name)[0] + '.npy'))
        rp = np.load(os.path.join(gt_videofolder,'landmark', os.path.splitext(image_name)[0] + '.npy'))
        mouth_start_idx, mouth_end_idx = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
        fp,rp = fp[mouth_start_idx:mouth_end_idx], rp[mouth_start_idx:mouth_end_idx]
        left = np.min(rp[:,0]); right = np.max(rp[:,0]); 
        top = np.min(rp[:,1]); bottom = np.max(rp[:,1])
        wh = [right-left, bottom-top]
        max_wh = np.maximum(max_wh, wh) #elment wise max
    lmds = []
    cache_dict={}
    gen_videofolder = os.path.splitext(gen_video_path)[0]
    N = len(glob.glob(f"{gen_videofolder}/*.jpg"))
    for i in tqdm(range(0, N-length+1, stride)):
        res, cache_dict = min_compare_landmarks_velocity(gen_video_path, gt_video_path, start=i, stride=stride, length=length, max_wh=max_wh, normalize=normalize, cache_dict=cache_dict)
        lmds.append(res)
    lmds = np.array(lmds)
    return np.mean(lmds)
