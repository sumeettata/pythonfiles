import cv2
from ultralytics import YOLO
import os
import pandas as pd
import glob
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt 
#image_metre = 60
#image_fps = 30


os.environ['KMP_DUPLICATE_LIB_OK']='True'

def angle_cen(init_pt,fin_pt):
    x_pt = fin_pt[0]-init_pt[0]
    y_pt = fin_pt[1]-init_pt[1]
            
    if x_pt == 0 :
        if (y_pt < 0) :
            dir_pt = 'moving straight up'
        elif (y_pt > 0):
            dir_pt = 'moving straight down'
        abs_angle = 90
    
    elif y_pt == 0 :  
        if (x_pt > 0):
            dir_pt = 'moving straight right'
        elif (x_pt < 0):
            dir_pt = 'moving straight left'
        abs_angle = 0
        
    else:    
        slope = abs(y_pt/x_pt)
        abs_angle = math.atan(slope)*(180/math.pi)
        if ((y_pt < 0) and (x_pt > 0)):
            dir_pt = 'moving up right'
        elif ((y_pt > 0) and (x_pt < 0)):
            dir_pt = 'moving down left'
        elif ((y_pt < 0) and (x_pt < 0)):
            dir_pt = 'moving up left'
        elif ((y_pt > 0) and (x_pt > 0)):
            dir_pt = 'moving down right'
        else:
            dir_pt = 'unknown'
        
    return dir_pt,abs_angle

# def speed_cen(init_pt,fin_pt):
#     point1 = np.array(init_pt)
#     point2 = np.array(fin_pt)
#     dist = np.linalg.norm(point1 - point2)
#     speed = dist*image_fps
#     return speed

cap = cv2.VideoCapture("C://Users//SumeetMitra//Downloads//4K Video of Highway Traffic!.mp4")
model = YOLO("weights/vehicle_04_05_2023.pt")

dic_cen = {}
while True:
    ret, frame = cap.read()
    lst = []
    if not ret:
        break
    results = model.track(frame, persist=True)
    class_name = model.names
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    clss = results[0].boxes.cls.cpu().numpy().astype(int)
    for box, id, cl  in zip(boxes, ids,clss):
        if class_name[int(cl)] != 'np':   
            centroid = (int((box[2] + box[0])/2),int((box[3] + box[1])/2))
            if id in dic_cen.keys():
                dic_cen[id].append(centroid)
                dir_cen, ang_cen = angle_cen(dic_cen[id][0],dic_cen[id][-1])
                #sd_cen = speed_cen(dic_cen[id][-2],dic_cen[id][-1])
                frame = cv2.line(frame, dic_cen[id][0], dic_cen[id][-1], (0, 255, 0), 9)
                cv2.putText(frame,f"Angle {round(ang_cen,2)},  direction {dir_cen}",dic_cen[id][-1],cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)
                
            else:
                dic_cen[id] = [centroid]
    
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window", 1080, 720)
    cv2.imshow("Resized_Window", frame)
    if cv2.waitKey(2) & 0xFF == ord("q"):
        break
    
    
    
    
