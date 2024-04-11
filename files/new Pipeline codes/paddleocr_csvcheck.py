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
ocr = PaddleOCR(use_angle_cls=False,lang='en')

# The input to the variables
image_path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/Number_plates_synthetic_test/'
csv_path = image_path+'train_det.csv'


df2 = pd.read_csv(csv_path)
 
l=0
k=0
for index,rows in df2.iterrows():
    img = cv2.imread(image_path+rows['file_name'])
    np_ch = str(rows['plate_number'])
    
    result = ocr.ocr(img, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        txts = [line[1][0] for line in res]
    
    np_pred = "".join(txts)
    np_pred = re.sub(r'[^\w]', '', np_pred)
    if len(np_pred) > 10:
        np_pred = np_pred.replace('IND','')
    
    if len(np_pred) > 10:
        np_pred = np_pred[1:]
        
    df2.loc[index,'predicted']  =   str(np_pred)
    
    if np_ch == np_pred:
        k = k+1
        df2.loc[index,'Results']  =   str(True)  
    else:
        df2.loc[index,'Results']  =   str(False)
    l = l + 1
    print((k/l)*100)
    
print((k/l)*100)

df2.to_csv(os.path.dirname(os.path.dirname(image_path))+"/predicted_labels_test.csv")
        