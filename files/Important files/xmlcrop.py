import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm


# The input to the variables
image_path = 'C:/Users/SumeetMitra/Downloads/Tagging/Tagging'
xml_path = image_path
save_path = image_path + '_cropped'

# Reading the paths and creating dir
dir_list = os.listdir(image_path)
files_jpg = [x for x in dir_list if not x.endswith(".xml")]
if not os.path.exists(save_path):
    os.mkdir(save_path)

for i in tqdm(files_jpg):
    label_path = os.path.join(xml_path,i.replace('.jpg','.xml').replace('.png','.xml').replace('.jpeg','.xml'))
    file_path = os.path.join(image_path,i)
    if os.path.exists(label_path):
        print(label_path)
        tree = ET.parse(label_path)
        root = tree.getroot()
        for k,member in enumerate(root.findall('object')):
            img = cv2.imread(file_path)
            img_name = os.path.basename(file_path)
            r,c,ch = img.shape
            class_name = str(member[0].text)
            if len(member) >5:
                xmin = int(member[5][0].text)
                xmax = int(member[5][1].text)
                ymin = int(member[5][2].text)
                ymax = int(member[5][3].text)
            else:
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)
            img = img[ymin:ymax,xmin:xmax]
#             saving the cropped images into different folders with class number as names 
            if not os.path.exists(save_path+'/'+str(class_name)):
                os.mkdir(save_path+'/'+str(class_name))
            cv2.imwrite(os.path.join(save_path+'/'+str(class_name),i.split('.')[0]+'_F'+str(k)+'.png'),img)