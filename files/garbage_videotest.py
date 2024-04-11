import os
import glob
import copy

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
from ultralytics.yolo.utils import ops


def create_colors(names):
    return [tuple(np.random.randint(low=80, high=256, size=3, dtype=np.dtype(int))) for i in range(len(names))]


def draw_rectangle(image, class_name, x1, y1, x2, y2, color):
    color = (int(color[0]), int(color[1]), int(color[2]))
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=tuple(color), thickness=2)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x1)+(int(x2) - int(x1)), int(y1) - 20), color=tuple(color), thickness=-1)
    cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image

def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def predict(model,video_path,save_path):
    colors_list = create_colors(model.names)
    video_save = save_path + '/'+os.path.basename(video_path).replace('.mp4','.avi')
    cap = cv2.VideoCapture(video_path)
    result = cv2.VideoWriter(video_save, cv2.VideoWriter_fourcc(*'MJPG'),10, (int(cap.get(3)),int(cap.get(4))))
    while True:
        if ret == True: 
            ret, image = cap.read()
            results = model.predict(image, conf=0.4, iou=0.4, boxes=True)
            pred = results[0].boxes.data
            for i, det in enumerate(pred.cpu()):
                image = draw_rectangle(image, model.names[int(det[5].numpy())], int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), colors_list[int(det[5].numpy())])
            result.write(image)
        else:
            break
        
    cap.release()
    result.release() 
        
    
weights_path = 'weights'
video_path = 'videos'


for file in glob.glob(weights_path+'/*.pt'):
    save_path = makedir(video_path+'_predict')
    save_path = makedir(save_path+'/'+os.path.basename(file).replace('.pt',''))
    model = YOLO(file) 
    i = 0
    for video_file in glob.glob(video_path+'/*.mp4'):
        i=i+1
        print(os.path.basename(file) + ' || ' + os.path.basename(video_file) + ' || ' + str(i), end="\r")
        predict(model,video_file,save_path)