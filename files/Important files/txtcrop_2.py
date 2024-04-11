import cv2
import os
import xml.etree.ElementTree as ET
import random
import pandas as pd
import glob
from tqdm import tqdm

# The input to the variables 
root_path = 'D://vehicle_data//New_folder//New folder (4)//vehicle-orientation-1//vehicle-orientation-1'
txt2_path = 'D://vehicle_data//New_folder//New folder (4)//vehicle-orientation-1//vehicle-orientation-1'
save_path = 'C://Users//SumeetMitra//PycharmProjects'
cropped_need = 3000 #cropped needed for each class


# Reading the paths and creating dir
if not os.path.exists(save_path):
    os.mkdir(save_path)
dir_list = os.listdir(root_path)
files_jpg = [x.split('.')[0] for x in dir_list if x.endswith(".jpg")]

# Looping over the files
for i in tqdm(files_jpg):
    txt_path = os.path.join(txt2_path,i+'.txt')
    file_path = os.path.join(root_path,i+'.jpg')
    lst = []
#     Reading the txt file
    with open(txt_path, "r") as text_file:
        for k,text_line in enumerate(text_file):
            img = cv2.imread(file_path)
            c,r,ch = img.shape
            text_contents = text_line.replace('\n','').split(' ')
            p = [eval(q) for q in text_contents]
            xmin = int((p[1] - (p[3]/2))*r)
            ymin = int((p[2] - (p[4]/2))*c)
            xmax = int((p[1] + (p[3]/2))*r)
            ymax = int((p[2] + (p[4]/2))*c)
            img = img[ymin:ymax,xmin:xmax]
#             saving the cropped images into different folders with class number as names 
            if img.shape[0] >= 100 and img.shape[1] >= 100:
                if not os.path.exists(save_path+'//class_'+str(p[0])):
                    os.mkdir(save_path+'//class_'+str(p[0]))
                cv2.imwrite(os.path.join(save_path+'//class_'+str(p[0]),i+'_'+str(k)+'.jpg'),img)