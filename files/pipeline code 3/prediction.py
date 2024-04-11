import os
import glob
import copy

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
from ultralytics.yolo.utils import ops
import torch
import torchvision.transforms as transforms
from PIL import Image

def create_colors(names):
    return [tuple(np.random.randint(low=80, high=256, size=3, dtype=np.dtype(int))) for i in range(len(names))]


def draw_rectangle(image, class_name, x1, y1, x2, y2, color):
    color = (int(color[0]), int(color[1]), int(color[2]))
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=tuple(color), thickness=2)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x1)+(int(x2) - int(x1)), int(y1) - 20), color=tuple(color), thickness=-1)
    cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image


model = YOLO("D:/Work/Tata Steel/best.pt")
model2 = torch.load('D:/Work/Tata Steel/welding shield detection/model/best.pt',map_location=torch.device('cpu') )
model2_name = {0:'face', 1: 'head',2: 'shield'}

cap = cv2.VideoCapture("D:/Work/Tata Steel/welding shield detection/data/c1.mp4")
result = cv2.VideoWriter('welding_testing6.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, (int(640),int(640)))
colors_list = create_colors(model.names)  
i = 0
while True:
    i=i+1
    ret, image = cap.read()
    if image is not None:
        results = model.predict(image, conf=0.4, iou=0.4, boxes=True)
        pred = results[0].boxes.data
        for i, det in enumerate(pred.cpu()):
            if model.names[int(det[5].numpy())] == 'helmet':
                img_crop = image.copy()[int(det[1].numpy()):int(det[3].numpy())+int(0.75*(int(det[3].numpy())-int(det[1].numpy()))),int(det[0].numpy()):int(det[2].numpy())]
                img_crop = Image.fromarray(img_crop)
                transform = transforms.Compose([transforms.Resize((128,128),transforms.InterpolationMode.BILINEAR),transforms.ToTensor(),transforms.Normalize((0.12,0.11,0.40),(0.89,0.21,0.12))])
                outputs = model2(transform(img_crop).unsqueeze(0))
                _,preds = torch.max(outputs.data,1)
                image = draw_rectangle(image, 'helmet:'+str(model2_name[preds.item()]), int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy())+int(0.75*(int(det[3].numpy())-int(det[1].numpy()))), colors_list[int(det[5].numpy())])
            #else:
                #image = draw_rectangle(image, model.names[int(det[5].numpy())], int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), colors_list[int(det[5].numpy())])
    result.write( cv2.resize(image, (640, 640)))
    cv2.imshow("output_image", cv2.resize(image, (640, 640)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
result.release()