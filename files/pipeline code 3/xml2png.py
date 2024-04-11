import os
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# The input to the variables
xml_path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/fire_smoke/Fire and Smoke Segmentation.v4i.voc/valid'
img_path = xml_path
save_path = os.path.dirname(xml_path)+'/PNG-Files/'
if not os.path.exists(save_path):
    os.mkdir(save_path)


files = glob.glob(xml_path+'/*.xml')

df=pd.DataFrame(columns=['x','y'])
for file in tqdm(files):
    tree = ET.parse(file)
    root = tree.getroot()
    for member in root.findall('size'):
        width = int(member[0].text)
        height = int(member[1].text)    
    mask = np.zeros([width,height,1],dtype=np.uint8)    
    for p,member in enumerate(root.findall('object')):
        class_name = str(member[0].text)
        if not os.path.exists(save_path+'/'+str(class_name)):
            os.mkdir(save_path+'/'+str(class_name))
        df=pd.DataFrame(columns=['x','y'])
        xmin = int(member[5][0].text)
        ymin = int(member[5][2].text)
        xmax = int(member[5][1].text)
        ymax = int(member[5][3].text)
        for mem in member[6].iter():
            if (str(mem.tag[0]) == 'x') or (str(mem.tag[0]) == 'y'):
                df.loc[mem.tag[1:],str(mem.tag[0])]=int(float(mem.text))
        df['combined']= df.values.tolist()
        pts = list(df['combined'])
        pts = np.array(pts,np.int32)
        pts = pts.reshape((-1, 1, 2))
        mask = cv2.fillPoly(mask, [pts],255) 
        img = cv2.imread(img_path+'/'+os.path.basename(file).replace('.xml','.jpg'))
        
        img_new = img[ymin:ymax,xmin:xmax]
        mask_new = mask[ymin:ymax,xmin:xmax]
        if (img_new.shape[0] > 50) or (img_new.shape[1] > 50):
            img_max = max(img_new.shape[0],img_new.shape[1])
            if img_max < 300:
                if img_max == img_new.shape[0]:
                    ratio = 300/img_new.shape[0]
                    img_new = cv2.resize(img_new, (int(img_new.shape[1]*ratio), 300))
                    mask_new = cv2.resize(mask_new, (int(mask_new.shape[1]*ratio), 300))
                elif img_max == img_new.shape[1]:
                    ratio = 300/img_new.shape[1]
                    img_new = cv2.resize(img_new, (300,int(img_new.shape[0]*ratio)))
                    mask_new = cv2.resize(mask_new, (300,int(mask_new.shape[0]*ratio)))
            b,g,r = cv2.split(img_new)
            img_new = cv2.merge((b,g,r,mask_new))
            cv2.imwrite(save_path+'/'+str(class_name)+'/'+os.path.basename(file).replace('.xml',str(p)+'.png'),img_new)
