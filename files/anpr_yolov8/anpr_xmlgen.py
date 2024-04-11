import os
import sys
import copy
import glob

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET


class anpr_xmlgen:
    def __init__(self, model_path="weights/vehicle_08_06_2023.pt", model_confidence=0.5, iou_threshold=0.6, boxes=True, classes_to_predict=None):
        self.boxes = boxes
        self.model_path = model_path
        self.iou_threshold = iou_threshold
        self.model_confidence = model_confidence
        self.classes_to_predict = classes_to_predict
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            self.names = self.model.names
        else:
            print("Model not found")
            sys.exit()
    
    def xml_writer(self, frame, file_path):
        original_image = copy.deepcopy(frame)
        
        output = []

        results = self.model.predict(frame, conf=self.model_confidence, iou=self.iou_threshold, boxes=self.boxes)
        pred = results[0].boxes.data
        
        root = Element('annotation')
        SubElement(root, 'folder').text = os.path.dirname(file_path)
        SubElement(root, 'filename').text = os.path.basename(file_path)
        SubElement(root, 'path').text = './images' + os.path.basename(file_path)
        
        source = SubElement(root, 'source')
        SubElement(source, 'database').text = 'Unknown'
        
        size = SubElement(root, 'size')
        SubElement(size, 'width').text = str(original_image.shape[1])
        SubElement(size, 'height').text = str(original_image.shape[0])
        SubElement(size, 'depth').text = '3'
        SubElement(root, 'segmented').text = '0'
        
        for i, det in enumerate(pred):
            if len(det) and self.model.names[int(det[5].numpy())] in self.classes_to_predict:
                
                obj = SubElement(root, 'object')
                SubElement(obj, 'name').text = str(self.model.names[int(det[5].numpy())])
                SubElement(obj, 'pose').text = 'Unspecified'
                SubElement(obj, 'truncated').text = '0'
                SubElement(obj, 'difficult').text = '0'

                bbox = SubElement(obj, 'bndbox')
                SubElement(bbox, 'xmin').text = str(int(det[0].numpy()))
                SubElement(bbox, 'ymin').text = str(int(det[1].numpy()))
                SubElement(bbox, 'xmax').text = str(int(det[2].numpy()))
                SubElement(bbox, 'ymax').text = str(int(det[3].numpy()))
        
        if len(root.findall('object')):
            tree = ElementTree(root)    
            xml_filename = file_path.split('.')[0]+'.xml'
            tree.write(xml_filename)
        

files = glob.glob("C:/Users/SumeetMitra/Downloads/anpr_test/anpr_test/*")
files = [x for x in files if not x.endswith('.xml')]

anpr_xmlgen = anpr_xmlgen(classes_to_predict=["auto", "bus", "car", "motorcycle", "hmv", "truck", "tractor","np"])

for file in files:
    frame = cv2.imread(file)
    anpr_xmlgen.xml_writer(frame, file)