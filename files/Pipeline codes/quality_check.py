import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import random


# The input to the variables
image_path = 'D://OneDrive/OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Main//Images'
xml_path = 'D://OneDrive/OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Main//labels_vehicles'
save_path = image_path+'_quality_check'

# Reading the paths and creating dir
dir_list = glob.glob(image_path+'//*')
files_jpg = [x for x in dir_list if not x.endswith('.xml')]
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Generating random samples:
random_list = random.sample(files_jpg, 600)
m=0
print(random_list)
#loop over the list file
for i in tqdm(random_list):
    file_name = os.path.basename(i)
    label_path = os.path.join(xml_path,file_name.split('.')[0]+'.xml')
    file_path = os.path.join(image_path,file_name)
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        for k,member in enumerate(root.findall('object')):
            img = cv2.imread(file_path)
            img_name = os.path.basename(file_path)
            class_name = str(member[0].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            if class_name == 'np':
                img = img[ymin:ymax,xmin:xmax]
                if img.shape[0] < 64 and  img.shape[1] < 64:
                    cv2.imwrite(save_path+'//'+file_name.split('.')[0]+'_'+str(k)+'.png',img)
                    m = m+1 
    if m == 40:
        break  