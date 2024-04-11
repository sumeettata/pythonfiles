import os
import glob
import copy

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
from ultralytics.yolo.utils import ops

model_path = "D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/GCP/garbage/Test2_2/weights/best.pt"
video_path = "D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage_new/Garbage_testing/videos/CCTV_ Rubbish Thrown Out Car.mp4"
image_path = "D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage/frames"

class Garbage_inference:
    def __init__(self, 
                 model_path = model_path,
                 video_path = video_path,
                 image_path = image_path,
                 split_frame = True
                 ):
        self.video_path = video_path
        self.image_path = image_path
        self.model = YOLO(model_path)
        self.split_frame = split_frame
        
    def create_colors(self,names):
        return [tuple(np.random.randint(low=80, high=256, size=3, dtype=np.dtype(int))) for i in range(len(names))]
    
    def draw_rectangle(self, image, class_name, x1, y1, x2, y2, color):
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=tuple(color), thickness=2)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x1)+(int(x2) - int(x1)), int(y1) - 20), color=tuple(color), thickness=-1)
        cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return image  
    
    def inference(self,image):
        image_height, image_width, _ = image.shape
        results = self.model.predict(image, conf=0.4, iou=0.4, boxes=True)
        pred = results[0].boxes.data
        for i, det in enumerate(pred.cpu()):
            image = self.draw_rectangle(image, self.model.names[int(det[5].numpy())], int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), self.colors_list[int(det[5].numpy())])
        return image
    
    def split_inference(self,image):
        lst1 = []
        for image_1 in np.split(image,4,axis=1):
            lst=[]
            for image_2 in np.split(image_1,4,axis=0):
                results = self.model.predict(image_2, conf=0.4, iou=0.4, boxes=True)
                pred = results[0].boxes.data
                for i, det in enumerate(pred.cpu()):
                    image_2 = self.draw_rectangle(image_2, self.model.names[int(det[5].numpy())], int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), self.colors_list[int(det[5].numpy())])
                lst.append(image_2)
            lst1.append(np.concatenate((lst),axis=0))
        image = np.concatenate((lst1),axis=1)
        return image
    
    def image_inference(self):
        self.colors_list = self.create_colors(self.model.names)  
        images= glob.glob(self.image_path+'/*')
        for image_name in images:
            image = cv2.imread(image_name)
            if self.split_frame == True:
                image = self.split_inference(image)
            else:
                image = self.inference(image)
            cv2.imshow("output_image", cv2.resize(image, (640, 640)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    def video_inference(self):
        self.colors_list = self.create_colors(self.model.names)  
        cap = cv2.VideoCapture(self.video_path)
        result = cv2.VideoWriter(self.video_path.replace('.mp4','.avi'), cv2.VideoWriter_fourcc(*'MJPG'),10, (int(cap.get(3)),int(cap.get(4))))
        while True:
            ret, image = cap.read()
            if ret == True: 
                if self.split_frame == True:
                    image = self.split_inference(image)
                else:
                    image = self.inference(image)
                cv2.imshow("output_image", cv2.resize(image, (640, 640)))
                #result.write(image) 
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break   
                
        cap.release()
        #result.release() 
    

gb = Garbage_inference()
gb.video_inference()