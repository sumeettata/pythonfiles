import os
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm 

path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage_new/Roboflow/GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/train/images'

for file in tqdm(glob.glob(path+'/*')):
    img = cv2.imread(file)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2[img2>200] = 0
    img2[img2!=0] = 255
    if 100*(img.size-np.count_nonzero(img))/img.size > 10:
        r,g,b = cv2.split(img)
        img = cv2.merge((r,g,b,img2))
        cv2.imwrite('D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage_new/Roboflow/GARBAGE CLASSIFICATION 3.v2-gc1.yolov8/train/'+os.path.basename(file).replace('.jpg','.png'),img)