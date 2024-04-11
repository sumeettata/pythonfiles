import copy
import os.path
import uuid

from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.general import non_max_suppression, check_img_size, scale_coords

import torch
import cv2
import numpy as np
import time
import glob
import pandas as pd
import csv
from tqdm import tqdm
from mediapipe_pose_read import get_pose


half = False
augment = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf_thres = 0.1
iou_thres = 0.1
classes = None
agnostic_nms = False
max_det = 1000

# Variables to be input
weights = "yolov5s.pt"
paths = "D:/OneDrive/OneDrive - Tata Insights and Quants/december/Project 6 person detection/person_class/Annoted_images/bahnhof"
#paths = "D:/OneDrive/OneDrive - Tata Insights and Quants/Learning/webscrapping/dataset/person talking on phone"

# model load
model = attempt_load(weights) 
stride = int(model.stride.max())
imgsz = check_img_size(640, s=stride)
names = model.module.names if hasattr(model, 'module') else model.names
image_path = glob.glob(paths+"//*.jpg") + glob.glob(paths+"//*.jpeg")+glob.glob(paths+"//*.png")
print(len(image_path))
ppe_list = []

for path in tqdm(image_path):
    filename = os.path.basename(path)
    image = cv2.imread(path)
    image_height, image_width, _ = image.shape
    img0 = copy.deepcopy(image)
    img = letterbox(img0, 640, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=augment)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            for x1, y1, x2, y2, conf, cls in reversed(det):
                if str(names[int(cls.numpy())]) == "person":
                    positi = get_pose(image[int(y1):int(y2),int(x1):int(x2)])
                    print(positi)
                    if positi < 0.3:
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,0,255), thickness=2)
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x1) + 30, int(y1) - 10), color=(0, 0, 255),thickness=-1)
                        cv2.putText(image, 'Person Talking on phone', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255),  2)
                    else:
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(255, 0,0 ), thickness=2)
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x1) + 30, int(y1) - 10), color=(255, 0, 0),thickness=-1)
                        cv2.putText(image, 'Good to go', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 0),  2)
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("image", 1080, 760)
            cv2.imshow("image",image)
            key = cv2.waitKey(0)
            if key == 27: 
                break
        else:
            break
cv2.destroyAllWindows()