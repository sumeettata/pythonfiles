import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm


# The input to the variables
image_path = 'C:/Users/SumeetMitra/Downloads/cam2_4_set3_(7th june)/cam2_4_set3'
xml_path = image_path
csv_path = image_path

# Reading the paths and creating dir
dir_list = os.listdir(image_path)
files_jpg = [x for x in dir_list if not x.endswith(".xml")]

df = pd.DataFrame(columns=['filename','class_name','x1','y1','x2','y2','height','width'])
for i in tqdm(files_jpg):
    label_path = os.path.join(xml_path,i.split('.')[0]+'.xml')
    file_path = os.path.join(image_path,i)
    img = cv2.imread(file_path)
    img_name = os.path.basename(file_path)
    r,c,ch = img.shape
    #read images
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        for mem in root.findall('size'):
            width = int(mem[0].text)
            height = int(mem[1].text)
        for member in root.findall('object'):
            class_name = str(member[0].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            df2 = pd.DataFrame([[img_name,class_name,xmin,ymin,xmax,ymax,height,width]],columns=df.columns)
            df = pd.concat([df,df2],ignore_index=True)

#save csv file
df.to_csv(csv_path+'converted.csv',index=False)
