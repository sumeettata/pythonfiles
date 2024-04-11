import glob
import os
import cv2
import random
import xml.etree.ElementTree as ET
import numpy as np


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

for i in files[0:5]:
    label_path = os.path.join(i.split('.')[0]+'.xml')
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        img = cv2.imread(i)
        plate_number = get_number()
        print(plate_number)
        for k,member in enumerate(root.findall('object')):
            random_file = get_letter(plate_number[k])
            img_letter = cv2.imread(random_file)
            class_name = str(member[0].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            img2 = img[ymin:ymax,xmin:xmax]
            img_letter = cv2.resize(img_letter, ((xmax-xmin),(ymax-ymin)))
            kernel = np.ones((5, 5), np.uint8)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ret,img2_mask = cv2.threshold(img2_gray,127,255,cv2.THRESH_BINARY_INV)
            img_dilation = cv2.dilate(img2_mask, kernel, iterations=1)
            img2_inpainted = cv2.inpaint(img2, img_dilation, 7, cv2.INPAINT_TELEA)
            img_letter_gray = cv2.cvtColor(img_letter, cv2.COLOR_BGR2GRAY)
            ret,img_letter_mask = cv2.threshold(img_letter_gray ,150,255,cv2.THRESH_BINARY)
            img_letter_mask_inv = cv2.bitwise_not(img_letter_mask)
            img_background = cv2.bitwise_and(img2_inpainted ,img2_inpainted ,mask=img_letter_mask)
            img_foreground = cv2.bitwise_and(img_letter,img_letter,mask=img_letter_mask_inv)
            result_image = cv2.add(img_background,img_foreground )
            #img2_txt = cv2.putText(img2_inpainted, plate_number[k] , (0,img2_inpainted.shape[0]-10), cv2.FONT_HERSHEY_COMPLEX, 3 , (0, 0, 0), 5, cv2.LINE_AA)
            #img[ymin:ymax,xmin:xmax] = img2_txt
            img[ymin:ymax,xmin:xmax] = result_image
        cv2.imwrite(os.path.basename(i),cv2.blur(img,(10,10)))