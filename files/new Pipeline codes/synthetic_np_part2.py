import glob
import os
import cv2
import random
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from num2words import num2words


txt_path = "Work/TATA Communications/Python files/ANPR_syntheticnp/final_text/final_text/plate_digits_1"
plate_path = "Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/used_gimp"
plate_labelpath = "Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/Train_data"

state_code = ['AN','AP','AR','AS','BH','BR','CH','CG','DD','DL','GA','GJ','HR','HP','JK','JH','KA','KL','LA','LD','MP','MH','MN','ML','MZ','NL','OD','PY','PB','RJ','SK','TN','TS','TR','UP','UK','WB']

number_code = [str(x) for x in range(10)]
alphabet_code = [chr(v).upper() for v in range(97, 123)]

def get_number():
    plate_no = str(random.choice(state_code)+random.choice(number_code)+random.choice(number_code)+random.choice(alphabet_code)+random.choice(alphabet_code)+random.choice(number_code)+random.choice(number_code)+random.choice(number_code)+random.choice(number_code))
    return plate_no

def get_letter(let_in):
    if let_in in [str(x) for x in range(10)]:
        let_in = num2words(int(let_in))
    let_img = cv2.imread(txt_path+"/"+str(let_in)+'.png',cv2.IMREAD_UNCHANGED)
    return let_img
    
if not os.path.exists('trail_images36'):
    os.mkdir('trail_images36')

j = 0
while j < 100:
    plate_pth = glob.glob(plate_path+"/*")[0]
    label_pth = plate_labelpath+"/"+os.path.basename(plate_pth).split(".")[0]+".xml"
    if os.path.exists(label_pth):
        tree = ET.parse(label_pth)
        root = tree.getroot()
        img = cv2.imread(plate_pth)
        plate_number = get_number()
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        for k,member in enumerate(root.findall('object')):
            class_name = str(member[0].text)
            xmin.append(int(member[4][0].text))
            ymin.append(int(member[4][1].text))
            xmax.append(int(member[4][2].text))
            ymax.append(int(member[4][3].text))
            img_let_resiz = cv2.resize(get_letter(plate_number[k]), ((xmax[0]-xmin[0]),(ymax[0]-ymin[0])))
            img_let = cv2.cvtColor(img_let_resiz[:,:,:3], cv2.COLOR_BGR2GRAY )
            img_random = np.random.random((img_let.shape[0],img_let.shape[1]))
            img_let[img_random>0.5] = img_let[img_random>0.5]- 253
            img_let[img_random>0.5] = img_let[img_random>0.5]- 250
            img_let[img_random>0.5] = img_let[img_random>0.5]- 253
            img_let_mask = img_let_resiz[:,:,3]
            kernel = np.ones((1,1),np.uint8)
            img_let_border = cv2.morphologyEx(img_let_mask, cv2.MORPH_GRADIENT, kernel)
            img_crop= img[ymin[0]:ymax[0],xmin[-1]:(xmin[-1]+(xmax[0]-xmin[0]))]
            img_crop[img_let_mask > 127] = 0
            img_let = cv2.cvtColor(img_let, cv2.COLOR_GRAY2BGR)
            img_let[img_let_mask==0] = 0
            img_final = img_crop+img_let
            img_final = cv2.inpaint(img_final,img_let_border,1,cv2.INPAINT_TELEA)
            img[ymin[0]:ymax[0],xmin[-1]:(xmin[-1]+(xmax[0]-xmin[0]))] = img_final
            
    cv2.imwrite('trail_images36/'+"np_sync"+str(j)+".png",img)
    j = j +1 
            
