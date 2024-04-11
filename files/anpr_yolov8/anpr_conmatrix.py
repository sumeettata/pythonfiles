import os
import sys
import copy
import glob

import cv2
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
from ultralytics.yolo.utils.metrics import ConfusionMatrix
from tqdm import tqdm 

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class anpr_conmatrix:
    def __init__(self, model_path="D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/GCP/garbage/Garbage_newdataset_31k/train2/weights/best.pt", model_confidence=0.5, iou_threshold=0.6, boxes=True):
        self.boxes = boxes
        self.model_path = model_path
        self.iou_threshold = iou_threshold
        self.model_confidence = model_confidence
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            self.names = self.model.names
        else:
            print("Model not found")
            sys.exit()
    
    def run(self, img_path):
        frame = cv2.imread(img_path)
        output = []
        results = self.model.predict(frame, conf=self.model_confidence, iou=self.iou_threshold, boxes=self.boxes,verbose=False)
        pred = results[0].boxes.data
        
        for i, det in enumerate(pred):

            if len(det):
                class_name = list(dic.values()).index(self.model.names[int(det[5].numpy())])
                output.append([int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), round(float(det[4].numpy()), 2),class_name ])
        return output

    def read_xml(self,label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        lst_bbox = []
        for k,member in enumerate(root.findall('object')):
            class_name = str(member[0].text)
            xmin = int(round(float(member[4][0].text)))
            ymin = int(round(float(member[4][1].text)))
            xmax = int(round(float(member[4][2].text)))
            ymax = int(round(float(member[4][3].text)))
            class_name =  list(dic.values()).index(class_name)
            lst_bbox.append([class_name,xmin,ymin,xmax,ymax])
            
        return lst_bbox
    
anpr_con = anpr_conmatrix()
conf_mat = ConfusionMatrix(nc = 1)  

dic = { 0: 'garbage'}

path = "D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage_test_dataset/images"


files = glob.glob(path + "/*")
files = [x for x in files if not x.endswith('.xml')]

lst1 =[]
lst2 = []
for file in tqdm(files):
    if os.path.exists(file.replace('.jpg','')+'.xml'):
        img_path = file
        label_path = file.replace('.jpg','')+'.xml'
        try:
            lst1.extend(anpr_con.run(img_path))
            lst2.extend(anpr_con.read_xml(label_path))
        except:
            print('error') 
 
preds = torch.tensor(lst1)  
gt_boxes = torch.tensor(lst2)
conf_mat.process_batch(preds, gt_boxes)
conf_mat.print()

conf_mat.plot(normalize=False, save_dir=os.path.dirname(path), names=list(dic.values()))
print(os.path.dirname(path)+'/confusion_matrix.png')