import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np

# The input to the variables
image_path = 'C:/Users/SumeetMitra/Downloads/footwear_tagged'
xml_path = image_path
save_path =  image_path+'/cropped_specific/'


# Reading the paths and creating dir
files = glob.glob(image_path+'/*')
files = [x for x in files if not x.endswith('.xml')]
if not os.path.exists(save_path):
    os.mkdir(save_path)

for file in tqdm(files):
    file_name = os.path.basename(file)
    label_path = os.path.join(xml_path,file_name.split('.')[0]+'.xml')
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        lst1 = []
        lst2 = []
        for k,member in enumerate(root.findall('object')):
            class_name = str(member[0].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            if class_name == 'human':
                lst1.append([[xmin,ymin],[xmax,ymax]])
            if class_name == 'glove':
                lst2.extend([[xmin,ymin],[xmax,ymax],[xmin,ymax],[xmax,ymin]])
        
        pts = np.array(lst2)
        if len(pts):
            for j,each in enumerate(lst1):
                each_init = np.array(each[0])
                each_fin = np.array(each[1])
                inidx = np.all(( each_init<= pts) & (pts <= each_fin), axis=1)
                if np.all(inidx):
                    img_crop = cv2.imread(file)[each[0][1]:each[1][1],each[0][0]:each[1][0]]
                    cv2.imwrite(save_path+file_name.split('.')[0]+str(j)+'.png',img_crop)
                    