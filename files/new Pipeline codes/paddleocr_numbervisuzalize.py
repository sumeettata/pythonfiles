import os
import glob
import copy
import re

import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=False,det_model_dir='',rec_model_dir='',lang='en',show_log=False)



def detect_numberplate(image):
    
    img_back = np.ones((image.shape[0],image.shape[1],3), np.uint8)*255
    
    result = ocr.ocr(image, cls=False)
    if len(result[0]):
        for idx in range(len(result)):
            res = result[idx]
            
            for y,line in enumerate(res):
                
                print(line)
                
                txts = line[1][0]
                print(txts)
                
                textSize, baseline = cv2.getTextSize(txts, cv2.FONT_HERSHEY_SIMPLEX , 1, 2)
                img2_txt = cv2.putText(np.ones((textSize[1]+baseline,textSize[0]+baseline,3), np.uint8)*255,str(txts) ,(int(baseline/2),int(textSize[1]+baseline/2)), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,0,0), 2, cv2.LINE_AA)
                if len(res) == 2:
                    if y==0:
                        img_back[0:int(img_back.shape[0]/2),0:int(img_back.shape[1])] = cv2.resize(img2_txt,(int(img_back.shape[1]),int(img_back.shape[0]/2)))
                    elif y==1:
                        img_back[int(img_back.shape[0]/2):int(img_back.shape[0]),0:int(img_back.shape[1])] = cv2.resize(img2_txt,(int(img_back.shape[1]),int(img_back.shape[0]-int(img_back.shape[0]/2))))
                else:
                    img_back = cv2.resize(img2_txt,(img_back.shape[1],img_back.shape[0]))
    
    return img_back            
                
path = "C:/Users/SumeetMitra/Downloads/cropped_np_04-04-2023_paddleimg (2)/cropped_np_04-04-2023_paddleimg/cropped_np_04-04-2023_paddleimg"              

images_path = glob.glob(path+"/*.png")

if not os.path.exists(path+'_crop_only'):
    os.mkdir(path+'_crop_only')
    
for image_name in images_path[0:300]:
    print(len(images_path))
    filename = os.path.basename(image_name)
    image_crop = cv2.imread(image_name)
    image = cv2.imread(image_name)

    image_back = detect_numberplate(image_crop)
    print(image_back.shape)
    final = cv2.vconcat([image,image_back])
    cv2.imwrite(path+'_crop_only/'+filename,final)