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

if not os.path.exists('trail_images1'):
    os.mkdir('trail_images1')
    

lst_csv = []
for i in tqdm(files):
    label_path = os.path.join(i.split('.')[0]+'.xml')
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        img = cv2.imread(i)
        img_original = cv2.imread(i)
        plate_number = get_number()
        for k,member in enumerate(root.findall('object')):
            
            if k < len(plate_number):
                #try:
                    #random_file = get_letter(plate_number[k])
                #except:
                    #print('error occured')
                    #break
                #img_letter = cv2.imread(random_file)
                class_name = str(member[0].text)
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)
                img_mask = np.zeros(img.shape[:2], np.uint8)
                cv2.rectangle(img_mask, (xmin,ymin),(xmax,ymax), 255, -1)
                img = cv2.inpaint(img, img_mask, 7, cv2.INPAINT_TELEA)
                img2 = img[ymin:ymax,xmin:xmax]
                kernel = np.array([[0, -1,  0],[-1,  5, -1],[0, -1,  0]])
                img2 = cv2.filter2D(img2, ddepth=-1, kernel=kernel)
                #img_txt = cv2.resize(img_letter, ((xmax-xmin),(ymax-ymin)))
                #ret,img_txt_fore = cv2.threshold(cv2.cvtColor(img_txt, cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
                #img_txt_gray = cv2.cvtColor(img_txt_fore, cv2.COLOR_GRAY2BGR)
                #img2[img_txt_fore > 0] = 0
                #img_final = img2+img_txt_gray
                #img_letter = cv2.add(img_letter, cv2.randn(img_ori, 0 , 180))
                img[ymin:ymax,xmin:xmax] = img2
        cv2.imwrite('trail_images1/'+os.path.basename(i),img)