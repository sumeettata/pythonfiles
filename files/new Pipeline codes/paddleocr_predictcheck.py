import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import json
import numpy as np
from paddleocr import PaddleOCR
import re

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ocr = PaddleOCR(use_angle_cls=True,lang='en',show_log=False)#,det_model_dir='',rec_model_dir='')


# The input to the variables
#image_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//ANPR Stage2//good_images//final//Train_data'
#image_path = 'C:/Users/SumeetMitra/Downloads/dataset697/697/New folder (2)_paddleimg'
image_path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/ANPR Stage2/good_images/test_set_data'
xml_path = image_path

lst_1 = []

# Reading the paths and creating dir
dir_list = os.listdir(image_path)
files_jpg = [x for x in dir_list if not x.endswith(".xml")]
k=0
l = 0
for i in tqdm(files_jpg):
    lis = []
    class_name = []
    label_path = os.path.join(xml_path,i.split('.')[0]+'.xml')
    file_path = os.path.join(image_path,i)
    image_file = cv2.imread(file_path)
    
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot() 
        for member in root.findall('object'):
            class_name.append(str(member[0].text))
            
        np_ch = "".join(class_name)
        result = ocr.ocr(image_file, cls=False)
        for idx in range(len(result)):
            res = result[idx]
            txts = [line[1][0] for line in res if len(line[1][0]) > 2]
        
        np_pred = "".join(txts)
        np_pred = re.sub(r'[^\w]', '', np_pred)
        if len(np_pred) > 10:
            np_pred = np_pred.replace('IND','')
            np_pred = np_pred.replace('IN','')
            np_pred = np_pred.replace('I','')
        
        if len(np_pred) > 10:
            np_pred = np_pred[1:]
             
        if np_ch == np_pred:
            k = k+1
        
        lst_1.append(['original/'+i,str(np_pred),str(np_ch)])
        l = l + 1
        
        
df = pd.DataFrame(lst_1)
#df.to_csv(image_path+'.csv')
        
        
print(k/l)
        
        