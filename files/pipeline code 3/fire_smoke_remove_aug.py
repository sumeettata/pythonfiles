import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np


# The input to the variables
image_path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/fire_smoke/final/final/Qube Furniture Detection.v2i.voc'
xml_path = image_path

# Reading the paths and creating dir
dir_list = os.listdir(image_path)
files_jpg = [x for x in dir_list if not x.endswith('.xml')]

for i in tqdm(files_jpg):
    b = []
    label_path = os.path.join(xml_path,i[0:-4]+'.xml')
    file_path = os.path.join(image_path,i)
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        for k,member in enumerate(root.findall('object')):
            class_name = str(member[0].text)
            xmin = int(member[5][0].text)
            ymin = int(member[5][2].text)
            xmax = int(member[5][1].text)
            ymax = int(member[5][3].text)
            img = cv2.imread(file_path)
            img_copy = img.copy()
            img_name = os.path.basename(file_path)
            img = img[ymin:ymax,xmin:xmax]
            b.append(img.mean())
    a = sorted(b)
    if a[0] < 10.00:
        os.remove(label_path)
        os.remove(file_path)
    
            
            