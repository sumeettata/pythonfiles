import os
import glob
import copy

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
from ultralytics.yolo.utils import ops
from datetime import datetime

save_path = "D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/cctv_frontdesk_20072023_again/"

if not os.path.exists(save_path):
    os.mkdir(save_path)
    
def create_colors(names):
    return [tuple(np.random.randint(low=80, high=256, size=3, dtype=np.dtype(int))) for i in range(len(names))]


def draw_rectangle(image, class_name, x1, y1, x2, y2, color):
    color = (int(color[0]), int(color[1]), int(color[2]))
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=tuple(color), thickness=2)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x1)+(int(x2) - int(x1)), int(y1) - 20), color=tuple(color), thickness=-1)
    cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image


model = YOLO("D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage_new/Garbage_weights/WEIGHTS/Garbagefall_30072023.pt")

colors_list = create_colors(model.names)    

#C:/Users/SumeetMitra/Downloads/throw garbage/throw garbage/
#D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage/Roboflow/garbage_detection.v16i.yolov8 (1)/test/images

#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/GCP/garbage/garbage_all_medium/test1_50_80kmph_RC_0002_230525092541.avi")

images_path = glob.glob("D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/garbage_testing_cctv/*")
i=0    
cap = cv2.VideoCapture(0)
while True:
    i=i+1
    if i%30 == 0:
        ret, image = cap.read()
# for image_name in images_path:
#     image = cv2.imread(image_name)
        filename = 'image'
        image_height, image_width, _ = image.shape
        results = model.track(image, conf=0.1, iou=0.1, boxes=True)
        pred = results[0].boxes.data
        for i, det in enumerate(pred.cpu()):
            image = draw_rectangle(image, model.names[int(det[5].numpy())] , int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), colors_list[int(det[5].numpy())])
        cv2.imwrite(save_path+str(filename)+'.png',image)
            #output.append({"filename":filename, "width":image_width, "height":image_height, "class_name":model.names[int(det[5].numpy())], "xmin":int(det[0].numpy()),"ymin":int(det[1].numpy()),"xmax":int(det[2].numpy()),"ymax":int(det[3].numpy())})
        cv2.imshow("output_image", cv2.resize(image, (640, 640)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    #
