import copy
import os.path
import sys
from tqdm import tqdm
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET

from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.general import non_max_suppression, check_img_size, scale_coords

import torch
import cv2
import numpy as np

import glob
import pandas as pd

# Input the required path for images
img_path = "D://6-03-2023//Daniel_Data_04_03_2023//Daniel_Data_04_03_2023"
# class_needed = ['bicycle', 'car', 'motorcycle', 'bus', 'truck','np']

# Input the required path for models
vehicle_model_path = "D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Python files/downloads/stage_1_1_5k_06_02_2023_large.pt"
human_model_path = "yolov5s.pt"
vehicle_bsf_path = "D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Python files/downloads/vc_bsf (1).pt"


# Inference code for model
class Inference:
    """
    This class performs inference on yolov5 object detection model and returns detection output
    """

    def __init__(self, model_file="yolov5s.pt", image_size=640, confidence_threshold=0.7, iou_threshold=0.7, half=False,
                 augment=False, classes=None, agnostic_nms=False, max_det=1000, detect_particular_classes=False,
                 particular_classes_list=None):
        self.half = half
        self.augment = augment
        self.classes = classes
        self.max_detections = max_det
        self.image_size = image_size
        self.agnostic_nms = agnostic_nms
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.particular_classes_list = particular_classes_list
        self.detect_particular_classes = detect_particular_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(model_file):
            self.model_file = model_file
            self.model = attempt_load(model_file, device=self.device)  # load model
            self.stride = int(self.model.stride.max())  # model stride
            self.image_size = check_img_size(640, s=self.stride)  # check image size
            self.names = self.model.module.names if hasattr(self.model,
                                                            'module') else self.model.names  # get class names
            print(self.names)
        else:
            print(f"Model not found {model_file}, please check the path")
            sys.exit()

    def predictions_on_particular_class(self, detections, filename, image_width, image_height):
        lst = []
        for x1, y1, x2, y2, conf, cls in reversed(detections):
            if self.names[int(cls.numpy())] in self.particular_classes_list:
                lst.append((filename, image_width, image_height, self.names[int(cls.numpy())], int(x1.numpy()), int(y1.numpy()), int(x2.numpy()), (int(y2.numpy())), (round(float(conf.numpy()), 2))))
        return lst

    def predictions(self, detections, filename, image_width, image_height):
        lst = []
        for x1, y1, x2, y2, conf, cls in reversed(detections):
            lst.append((filename, image_width, image_height, self.names[int(cls.numpy())], int(x1.numpy()), int(y1.numpy()), int(x2.numpy()), (int(y2.numpy())), (round(float(conf.numpy()), 2))))
        return lst

    def run(self, image, filename,):
        output = None
        image_height, image_width, _ = image.shape
        original_image = copy.deepcopy(image)
        image = letterbox(image, self.image_size, stride=32)[0]
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.device)
        image = image.half() if self.half else image.float()
        image /= 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        predictions = self.model(image, augment=self.augment)[0]
        predictions = non_max_suppression(predictions, self.confidence_threshold, self.iou_threshold, self.classes,
                                          self.agnostic_nms, max_det=self.max_detections)

        for i, detections in enumerate(predictions):
            if len(detections):
                detections[:, :4] = scale_coords(image.shape[2:], detections[:, :4], original_image.shape).round()
                if self.detect_particular_classes:
                    output = self.predictions_on_particular_class(detections, filename, image_width, image_height)
                elif not self.detect_particular_classes:
                    output = self.predictions(detections, filename, image_width, image_height)
        return output


main_output_list = []
# human_model = Inference(model_file=human_model_path, confidence_threshold=0.5, iou_threshold=0.4,
#                         detect_particular_classes=True,
#                         particular_classes_list=['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'])
vehicle_model = Inference(model_file=vehicle_model_path, confidence_threshold=0.5, iou_threshold=0.4)

# vehicle_bsf_model = Inference(model_file=vehicle_bsf_path, confidence_threshold=0.5, iou_threshold=0.4)


for image_name in tqdm(glob.glob(img_path+"/*.png")+glob.glob(img_path+"/*.jpg")):
    filename = image_name
    image = cv2.imread(image_name)
    # coco_output = human_model.run(image=image, filename=filename)
    # if coco_output is not None:
    #     main_output_list.extend(coco_output)
    vehicle_output = vehicle_model.run(image=image, filename=filename)
    if vehicle_output is not None:
        main_output_list.extend(vehicle_output)
    # vehicle_bsf_output = vehicle_bsf_model.run(image=image, filename=filename)
    # if vehicle_bsf_output is not None:
    #    main_output_list.extend(vehicle_bsf_output)
       
if len(main_output_list):
    df = pd.DataFrame(main_output_list, columns=["filename", "width", "height", "class_name", "x1", "y1", "x2", "y2", "class_confidence"])

#converting all values to str and getting the required class dataframe
df = df.applymap(str)
# df = df[df['class_name'].isin(class_needed)]
    
# Function to convert the dataframe grouped by file_name to xml
def df2xml(group_name,df_group):
    root = Element('annotation')
    SubElement(root, 'folder').text = os.path.dirname(group_name)
    SubElement(root, 'filename').text = os.path.basename(group_name)
    SubElement(root, 'path').text = './images' + os.path.basename(group_name)
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'
    
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = df_group['width'].unique()[0]
    SubElement(size, 'height').text = df_group['height'].unique()[0]
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'
    
    for index, row in df_group.iterrows():
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = row['class_name']
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = row['x1']
        SubElement(bbox, 'ymin').text = row['y1']
        SubElement(bbox, 'xmax').text = row['x2']
        SubElement(bbox, 'ymax').text = row['y2']
    tree = ElementTree(root)    
    xml_filename = group_name.split('.')[0]+'.xml'
    tree.write(xml_filename)
    
# Looping the dataframe grouped by file_name
for file_name, df_grp in tqdm(df.groupby('filename')):
    df2xml(file_name,df_grp)