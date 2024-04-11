import os
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# The input to the variables
xml_path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage_new/road_seg/Sidewalk Segmentation.v4i.voc/valid'

save_path = xml_path+'_mask/'
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
    for member in root.findall('object'):
        class_name = str(member[0].text)
        df=pd.DataFrame(columns=['x','y'])
        for mem in member[6].iter():
            if (str(mem.tag[0]) == 'x') or (str(mem.tag[0]) == 'y'):
                df.loc[mem.tag[1:],str(mem.tag[0])]=int(float(mem.text))
        df['combined']= df.values.tolist()
        pts = list(df['combined'])
        pts = np.array(pts,np.int32)
        pts = pts.reshape((-1, 1, 2))
        mask = cv2.fillPoly(mask, [pts],1) 
    cv2.imwrite(save_path+os.path.basename(file).replace('.xml','.png'),mask)
        
        
