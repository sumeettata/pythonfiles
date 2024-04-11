import os
import cv2
import glob
import argparse
from tqdm import tqdm

path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Learning//yolov5//runs//detect//exp11'

files = glob.glob(path+'//*.png')+glob.glob(path+'//*.jpg')

for file in files:
    img = cv2.imread(file)
    cv2.namedWindow("window_name", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("window_name", 1300, 700)
    cv2.imshow('window_name', img)
    cv2.imshow("window_name", img)
    k = cv2.waitKey(50)
    if k == 27:
        break
cv2.destroyAllWindows()