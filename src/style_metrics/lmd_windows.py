import numpy as np
import cv2
import pandas as pd
import os, glob
from imutils import face_utils
from tqdm import tqdm
from utils import * 
import dlib
from utils import generate_landmark_lips_folder 
import json 
# Running script
GT_DIR = "clips"
DATA_DIRS = ["ravdess_output"]

SAVED_DIR = "standard_lmd"
WINDOWS = list(range(10,101))
STRIDES = list(range(1,20))
# WINDOWS = list(range(10,11))
# STRIDES = list(range(1,2))
os.makedirs(SAVED_DIR, exist_ok=True)
for DATA_DIR in DATA_DIRS:
    for length in WINDOWS:
        for stride in STRIDES:
            try:
                with open(f"{SAVED_DIR}/{DATA_DIR}-{length}-{stride}.json", 'w') as f:
                    results = {'file': [], 'smd': [], 'slv':[], 'sld':[]}
                    for file in os.listdir(GT_DIR):
                        if os.path.splitext(file)[-1] not in [".mp4", ".avi"]:
                            continue
                        vid_smd = full_term_compare_landmarks_mouth(os.path.join(DATA_DIR, file),os.path.join(GT_DIR, file), length=length, stride=stride)
                        vid_slv = full_term_compare_landmarks_velocity(os.path.join(DATA_DIR, file),os.path.join(GT_DIR, file), length=length, stride=stride)
                        vid_sld = full_term_compare_landmarks(os.path.join(DATA_DIR, file),os.path.join(GT_DIR, file), length=length, stride=stride)
                        results['file'].append(file)
                        results['smd'].append(vid_smd.tolist())
                        results['slv'].append(vid_slv.tolist())
                        results['sld'].append(vid_sld.tolist())
                        # break
                        
                    results['file'].append('total')
                    results['smd'].append(np.mean(results['smd']))
                    results['slv'].append(np.mean(results['slv']))
                    results['sld'].append(np.mean(results['sld']))
                    results['length'] = [length]
                    results['stride'] = [stride]

                    json.dump(results, f)
            except Exception as e:
                print("Error: ", e)
