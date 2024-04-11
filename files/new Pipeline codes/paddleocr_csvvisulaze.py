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
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ocr = PaddleOCR(use_angle_cls=False,lang='en',show_log=False)

# The input to the variables
image_path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/ANPR Stage2/testset_data_final/'
csv_path = image_path+'Results.csv'


df2 = pd.read_csv(csv_path)



def detect_numberplate(image):
    
    img_back = np.ones((image.shape[0],image.shape[1],3), np.uint8)*255
    
    result = ocr.ocr(image, cls=False)
    if len(result[0]):
        for idx in range(len(result)):
            res = result[idx]
            
            for y,line in enumerate(res):
                
                pts = np.int0(line[0])
                
                
                image = cv2.polylines(image, [pts], True, (255, 0, 0) , 1)
                txts = line[1][0]
                
                textSize, baseline = cv2.getTextSize(txts, cv2.FONT_HERSHEY_SIMPLEX , 1, 2)
                img2_txt = cv2.putText(np.ones((textSize[1]+baseline,textSize[0]+baseline,3), np.uint8)*255,str(txts) ,(int(baseline/2),int(textSize[1]+baseline/2)), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,0,0), 2, cv2.LINE_AA)
                if len(res) == 2:
                    if y==0:
                        img_back[0:int(img_back.shape[0]/2),0:int(img_back.shape[1])] = cv2.resize(img2_txt,(int(img_back.shape[1]),int(img_back.shape[0]/2)))
                    elif y==1:
                        img_back[int(img_back.shape[0]/2):int(img_back.shape[0]),0:int(img_back.shape[1])] = cv2.resize(img2_txt,(int(img_back.shape[1]),int(img_back.shape[0]-int(img_back.shape[0]/2))))
                else:
                    img_back = cv2.resize(img2_txt,(img_back.shape[1],img_back.shape[0]))
    
    return img_back,image 
          
def makedir(name):
    if not os.path.exists(name):
        os.mkdir(name)

     
l=0
k=0
for index,rows in tqdm(df2.iterrows()):
    img = cv2.imread(image_path+rows['file_name'])
    makedir(os.path.dirname(image_path+rows['file_name'])+'_predicted') 
    makedir(os.path.dirname(image_path+rows['file_name'])+'_predicted/incorrect') 
    makedir(os.path.dirname(image_path+rows['file_name'])+'_predicted/correct') 
    np_ch = str(rows['plate_number'])

    np_pred = str(rows['predicted'])
    
    img_back,image = detect_numberplate(img)
    img_back = cv2.vconcat([image,img_back])
    
    
    if np_ch == np_pred:
        cv2.imwrite(os.path.dirname(image_path+rows['file_name'])+'_predicted/correct/'+os.path.basename(rows['file_name']),img_back)
        k = k+1 
    else:
        cv2.imwrite(os.path.dirname(image_path+rows['file_name'])+'_predicted/incorrect/'+os.path.basename(rows['file_name']),img_back)
    l = l + 1
    
print((k/l)*100)
