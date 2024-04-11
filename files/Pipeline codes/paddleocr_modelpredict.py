from paddleocr import PaddleOCR,draw_ocr
import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import json
import numpy as np
import shutil
# # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# # You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# # to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = r"D:\OneDrive\OneDrive - Tata Insights and Quants\Work\TATA Communications\Data_1\ANPR Stage2\good_images\final\Train_data\1IMG_20230209_145419_1.png"
result = ocr.ocr(img_path,rec=False)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# # draw result
# from PIL import Image
# result = result[0]
# image = Image.open(img_path).convert('RGB')
# im_show = draw_ocr(image, result, txts=None, scores=None, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result1.jpg')

img_path = r"D:\OneDrive\OneDrive - Tata Insights and Quants\Work\TATA Communications\Data_1\ANPR Stage2\good_images\final\Train_data\1IMG_20230209_145419_1.png"
label_path = r"D:\OneDrive\OneDrive - Tata Insights and Quants\Work\TATA Communications\Data_1\ANPR Stage2\good_images\final\Train_data\1IMG_20230209_145419_1.xml"

tree = ET.parse(label_path)
root = tree.getroot() 

x_min = []
y_min = []
x_max = []
y_max = []
class_name = []
e = 0
for member in root.findall('object'):
    class_name.append(str(member[0].text))
    x_min.append(int(member[4][0].text))
    y_min.append(int(member[4][1].text))
    x_max.append(int(member[4][2].text))
    y_max.append(int(member[4][3].text))
    
if len(x_min) > 1:
    np_ch = class_name   
    dic = {}         
    dic["points"] = [[x_min[0],y_min[0]], [x_max[-1],y_min[-1]], [x_max[-1],y_max[-1]], [x_min[0],y_max[0]]]
    dic["transcription"] = str("".join(np_ch))
    
pts = np.array([[x_min[0],y_min[0]], [x_max[-1],y_min[-1]], [x_max[-1],y_max[-1]], [x_min[0],y_max[0]]],np.int32)

pts = pts.reshape((-1, 1, 2))
isClosed = True
color = (255, 0, 0)
thickness = 2
image_2 = cv2.polylines(cv2.imread(img_path), [pts],isClosed, color, thickness)
rect = cv2.minAreaRect(pts)


box = cv2.boxPoints(rect)
box = np.int0(box)
img2 = cv2.drawContours(cv2.imread(img_path), [box], 0, (0, 0, 255), 2)

img_mask = np.zeros(cv2.imread(img_path,0).shape, np.uint8)
img_mask = cv2.drawContours(img_mask, [box], 0, 255, -1)

rotate_matrix = cv2.getRotationMatrix2D(center=rect[0], angle=rect[-1], scale=1)
  
# rotate the image using cv2.warpAffine 
# 90 degree anticlockwise
rotated_image = cv2.warpAffine(src=cv2.imread(img_path), M=rotate_matrix, dsize=(max(int(rect[1][0]),cv2.imread(img_path).shape[0]),max(int(rect[1][1]),cv2.imread(img_path).shape[1])))
rotated_image_mask = cv2.warpAffine(src=img_mask, M=rotate_matrix, dsize=(max(int(rect[1][0]),cv2.imread(img_path).shape[0]),max(int(rect[1][1]),cv2.imread(img_path).shape[1])))
coor = cv2.boundingRect(rotated_image_mask)

final_image =   rotated_image[coor[1]:(coor[1]+coor[3]),coor[0]:(coor[0]+coor[2])]      
cv2.imwrite('result_trail_4.png',final_image)
#cv2.imwrite('result_trail_3.png',img2)
print(box)