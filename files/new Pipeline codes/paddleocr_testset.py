import cv2
import glob
import os
from paddleocr import PaddleOCR
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from tqdm import tqdm 

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ocr = PaddleOCR(use_angle_cls=False,lang='en')

path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/New folder/testing_split/test'

if not os.path.exists(path+'_predict'):
    os.mkdir(path+'_predict')
    
if not os.path.exists(path+'_predict/incorrect'):
    os.mkdir(path+'_predict/incorrect')

if not os.path.exists(path+'_predict/correct'):
    os.mkdir(path+'_predict/correct')
    
lst = []  
for file in tqdm(glob.glob(path+'/*')):
    file_base = os.path.basename(file)
    np_ch = file_base.split('_')[1]
    
    img = cv2.imread(file)
    result = ocr.ocr(img, cls=False,det=False)
    
    
    textSize, baseline = cv2.getTextSize(result[0][0][0], cv2.LINE_AA , 1, 1)
    img2_txt = cv2.putText(np.ones((textSize[1]+baseline,textSize[0]+baseline,3), np.uint8)*255,str(result[0][0][0]) ,(int(baseline/2),int(textSize[1]+baseline/2)), cv2.LINE_AA, 1 , (0,0,0), 1, cv2.LINE_AA)
    
    final = cv2.vconcat([img,cv2.resize(img2_txt,(img.shape[1],img.shape[0]))])
    
    if str(np_ch) == re.sub(r'[^\w]', '',str(result[0][0][0])):
        cv2.imwrite(path+'_predict/correct/'+file_base,final)
    else:
        cv2.imwrite(path+'_predict/incorrect/'+file_base,final)
        
    lst.append([file_base,np_ch,result[0][0][0],re.sub(r'[^\w]', '',str(result[0][0][0]))])

df = pd.DataFrame(lst,columns=['filename','groundtruth','predicted','predicted_removespace'])
df.to_csv(path+'_predict.csv')