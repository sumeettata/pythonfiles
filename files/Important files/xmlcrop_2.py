import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm


# The input to the variables
image_path = 'C:/Users/SumeetMitra/Downloads/frames_HJ_casob_1000/frames_HJ_casob_1000'
xml_path = image_path
save_path = 'D:/Work/Tata Steel/Use case 2/hook_cropped'

# Reading the paths and creating dir
dir_list = os.listdir(image_path)
files_jpg = [x for x in dir_list if not x.endswith('.xml')]
if not os.path.exists(save_path):
    os.mkdir(save_path)

for i in tqdm(files_jpg):
    label_path = os.path.join(xml_path,i.split('.')[0]+'.xml')
    file_path = os.path.join(image_path,i)
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        for k,member in enumerate(root.findall('object')):
            class_name = str(member[0].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
#             saving the cropped images into different folders with class number as names
            if class_name == 'hook':
                img = cv2.imread(file_path)
                img_name = os.path.basename(file_path)
                img = img[ymin:ymax,xmin:xmax]
                cv2.imwrite(save_path+'//'+i.split('.')[0]+'_'+str(k)+'.png',img)