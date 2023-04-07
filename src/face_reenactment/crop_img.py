import cv2 
import face_alignment
import numpy as np
def get_bbox_image(img, fa):    
    preds = fa.get_landmarks(img)
    kpt = preds[0].squeeze()
    left = int(np.min(kpt[:,0]))
    right = int(np.max(kpt[:,0]))
    top = int(np.min(kpt[:,1]))
    bottom = int(np.max(kpt[:,1]))
    return left,top,right,bottom

# @profile
def square_bbox_from_rect_bbox(left,top,right,bottom, k_margin=0.0, shape=(256,256)):
    height = bottom - top
    width = right - left

    more_h = 0 
    more_w = 0
    if width < height:
        more_w += height - width
        width = height
    elif height < width:
        more_h = width - height
        height = width

    k = k_margin
    more_h += int(height*k)
    more_w += int(width*k)

    top = top - int(more_h *0.5)
    more_h -= int(more_h *0.5)
    if top < 0:
        more_h += np.abs(top) 
        top = 0
    bottom = bottom + more_h
    more_h = 0
    if bottom > shape[0]:
        more_h += bottom - shape[0] 
        bottom = shape[0]
    top = top - more_h
    top = max(top,0)

    left = left - int(more_w *0.5)
    more_w -= int(more_w *0.5)
    if left < 0:
        more_w += np.abs(left)
        left = 0 
    right += more_w
    more_w = 0 
    if right > shape[1]:
        more_w += right - shape[1] 
        right = shape[1] 
    left  -= more_w
    left = max(left, 0)
    return left,top,right,bottom


predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device="cuda", flip_input=True)
# img = cv2.imread("inputs/monalisa.jpg")
img = cv2.imread("inputs/ObamaNone.jpg")

bbox = get_bbox_image(img,predictor)
# cai erman tra? hang` là 1.4`
bbox = square_bbox_from_rect_bbox(*bbox,1.,shape=img.shape)
img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
img = cv2.resize(img, (256,256))
cv2.imwrite("inputs/ObamaNone_crop.jpg", img)    