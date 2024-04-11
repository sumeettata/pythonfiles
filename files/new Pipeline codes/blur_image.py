import glob
import random
import cv2
import numpy as np
from tqdm import tqdm 
import os

path = 'Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/Number_plates_synthetic/train_rec'

files = glob.glob(path+"/*")
files = random.choices(files,k=20000)


for file in tqdm(files):
    img = cv2.imread(file)
    a = random.choice(range(20))*0.1
    b = random.choice(range(20))
    c = random.choice([3,5,7,15])
    img =  cv2.blur(img,(c,c))
    img = cv2.convertScaleAbs(img, alpha=a, beta=b)
    cv2.imwrite(file,img)
    # cv2.imshow('img', img)
    # k = cv2.waitKey(0)
    # if k == 27: 
    #     break 
    # cv2.destroyAllWindows()