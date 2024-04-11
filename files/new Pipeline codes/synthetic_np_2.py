import glob
import os
import cv2
import random
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

path =  'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/Paddleocr/Train_data'
files =  glob.glob(path+'/*.png')

letters_path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/cropped_ANPR'


def get_letter(let_in):
    let_file =  glob.glob(letters_path+'/'+let_in+'/*')
    let_random = random.choice(let_file)
    return let_random

state_code = ['AN','AP','AR','AS','BH','BR','CH','CG','DD','DL','GA','GJ','HR','HP','JK','JH','KA','KL','LA','LD','MP','MH','MN','ML','MZ','NL','OD','PY','PB','RJ','SK','TN','TS','TR','UP','UK','WB']
number_code = [str(x) for x in range(10)]
alphabet_code = [chr(v).upper() for v in range(97, 123)]

def get_number():
    plate_no = str(random.choice(state_code)+random.choice(number_code)+random.choice(number_code)+random.choice(alphabet_code)+random.choice(alphabet_code)+random.choice(number_code)+random.choice(number_code)+random.choice(number_code)+random.choice(number_code))
    return plate_no

if not os.path.exists('trail_images'):
    os.mkdir('trail_images')

for i in tqdm(files):
    label_path = os.path.join(i.split('.')[0]+'.xml')
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        img = cv2.imread(i)
        min_color = int(np.min(cv2.imread(i,0)))
        print(min_color)
        img_original = cv2.imread(i)
        plate_number = get_number()
        for k,member in enumerate(root.findall('object')):
            class_name = str(member[0].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            img_mask = np.zeros(img.shape[:2], np.uint8)
            cv2.rectangle(img_mask, (xmin,ymin),(xmax,ymax), 255, -1)
            img = cv2.inpaint(img, img_mask, 7, cv2.INPAINT_TELEA)
            img2 = img[ymin:ymax,xmin:xmax]
            img_ori = img_original[ymin:ymax,xmin:xmax]
            kernel = np.array([[0, -1,  0],[-1,  5, -1],[0, -1,  0]])
            img2 = cv2.filter2D(img2, ddepth=-1, kernel=kernel)
            if k < len(plate_number):
                textSize, baseline = cv2.getTextSize(plate_number[k], cv2.FONT_HERSHEY_TRIPLEX , 1, 2)
                img2_txt = cv2.putText(cv2.resize(img2,(textSize[1]+baseline,textSize[0]+baseline)),plate_number[k] ,(0,textSize[0]), cv2.FONT_HERSHEY_TRIPLEX, 1 , (min_color, min_color, min_color), 2, cv2.LINE_AA)
                img2_txt = cv2.resize(img2_txt, ((xmax-xmin),(ymax-ymin)))
                img_random = np.random.random((img2_txt.shape[0],img2_txt.shape[1]))
                img2[:,:,0][img_random<0.9] = 0
                img2[:,:,1][img_random<0.9] = 0
                img2[:,:,2][img_random<0.9] = 0
                img2_txt[:,:,0][img_random>0.9] = 0
                img2_txt[:,:,1][img_random>0.9] = 0
                img2_txt[:,:,2][img_random>0.9] = 0
                img_letter = img2 + img2_txt
                img[ymin:ymax,xmin:xmax] = img_letter
        cv2.imwrite('trail_images/'+os.path.basename(i),img)