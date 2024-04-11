import cv2
import os
import xml.etree.ElementTree as ET
import random
import pandas as pd
import glob
from tqdm import tqdm

# The input to the variables 
root_path = 'D:/Work/Tata Steel/Use case 2/Hook_DB/Hook_DB/Hook_img'
txt2_path = root_path

if not os.path.exists(root_path+'_pred'):
    os.mkdir(root_path+'_pred')
    
# Reading the paths
dir_list = os.listdir(root_path)
files_jpg = [x for x in dir_list if x.endswith(".png")]
factor = 0.5

# Looping over the files
for i in tqdm(files_jpg):
    txt_path = os.path.join(txt2_path,i[:-4]+'.txt')
    if os.path.exists(txt_path):
        file_path = os.path.join(root_path,i)
        img = cv2.imread(file_path)
        lst = []
    #     Reading the txt file
        with open(txt_path, "r") as text_file:
            for k,text_line in enumerate(text_file):
                c,r,ch = img.shape
                text_contents = text_line.replace('\n','').split(' ')
                p = [eval(q) for q in text_contents]
                xmin = ((p[1] - (p[3]/2))*r)
                ymin = ((p[2] - (p[4]/2))*c)
                xmax = ((p[1] + (p[3]/2))*r)
                ymax = ((p[2] + (p[4]/2))*c)

                x1min = int(xmin - factor*(xmax-xmin)*0.5)
                y1min = int(ymin - factor*(ymax-ymin)*0.5)
                x1max = int(xmax + factor*(xmax-xmin)*0.5)
                y1max = int(ymax + factor*(ymax-ymin)*0.5)

                #img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),255,2)
                img = img[y1min:y1max,x1min:x1max]
                if not os.path.exists(root_path+'_pred'+'//class_'+str(p[0])):
                    os.mkdir(root_path+'_pred'+'//class_'+str(p[0]))
                try:
                    cv2.imwrite(os.path.join(root_path+'_pred'+'//class_'+str(p[0]),i.replace('.png','')+'_'+str(k)+'.jpg'),img)
                except:
                    print('error')
            