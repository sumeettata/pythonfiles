import glob
import os
import cv2
import random
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from num2words import num2words
import json

txt_path = "Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/cropped_fonts/final_font"
plate_path = "Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/used_gimp"
plate_labelpath = "Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/plate_labels"

state_code = ['AN','AP','AR','AS','BH','BR','CH','CG','DD','DL','GA','GJ','HR','HP','JK','JH','KA','KL','LA','LD','MP','MH','MN','ML','MZ','NL','OD','PY','PB','RJ','SK','TN','TS','TR','UP','UK','WB']

#get the json file containg all the rto code and load its values only 
rto_code = json.load(open("D:/OneDrive/OneDrive - Tata Insights and Quants/Rto_vehicle_list.json")).values()
rto_code = sum(list(rto_code),[]) # convert the dic values to list and then convert 2d list to 1d list 

number_code = [str(x) for x in range(10)]
alphabet_code = [chr(v).upper() for v in range(97, 123)]

def get_number():
    #plate_no = 
    plate_no = str(random.choice(state_code)+random.choice(number_code[0:6])+random.choice(number_code)+random.choice(alphabet_code)+random.choice(alphabet_code)+random.choice(number_code)+random.choice(number_code)+random.choice(number_code)+random.choice(number_code))
    return plate_no

def get_letter(let_in):
    #if let_in in [str(x) for x in range(10)]:
        #let_in = num2words(int(let_in))
    if (let_in == "I"):
        let_in = 'P'
    let_img = cv2.imread(txt_path+"/"+str(let_in)+'.png',cv2.IMREAD_UNCHANGED)
    return let_img
    
if not os.path.exists('trail_images387'):
    os.mkdir('trail_images387')

j = 0
#for rto_number in rto_code:
    
while j < 100:
    lst_label = []
    plate_pth = glob.glob(plate_path+"/*")[6]
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
            xmin.append(int(member[4][0].text)+3)
            ymin.append(int(member[4][1].text)+3)
            xmax.append(int(member[4][2].text)-3)
            ymax.append(int(member[4][3].text)-3)
            print(plate_number[k])
            img_let_resiz = cv2.resize(get_letter(plate_number[k]), ((xmax[0]-xmin[0]),(ymax[0]-ymin[0])))
            #img_let_org = np.ones([(ymax[0]-ymin[0]),(xmax[0]-xmin[0]),3],dtype=np.uint8)*255
            img_let_org = img_let_resiz[:,:,:3]
            img_let = cv2.cvtColor(img_let_resiz[:,:,:3], cv2.COLOR_BGR2GRAY)
            ret , img_back_inv = cv2.threshold(img_let_resiz[:,:,3], 127 , 255, cv2.THRESH_BINARY)
            kernel = np.ones((1, 1), np.uint8)
            img_back_inv = cv2.erode(img_back_inv, kernel, iterations=3)
            #ret, img_back = cv2.threshold(img_let, , 255, cv2.THRESH_BINARY)
            img_back = cv2.bitwise_not(img_back_inv)
            img_random = np.random.random((img_let.shape[0],img_let.shape[1]))
            img_let_org[img_random>0.7] = img_let_org[img_random>0.7]- 251
            img_random = np.random.random((img_let.shape[0],img_let.shape[1]))
            img_let_org[img_random>0.7] = img_let_org[img_random>0.7]- 250
            img_random = np.random.random((img_let.shape[0],img_let.shape[1]))
            img_let_org[img_random>0.7] = img_let_org[img_random>0.7]- 252
            img_crop= img[ymin[0]:ymax[0],xmin[-1]:(xmin[-1]+(xmax[0]-xmin[0]))]
            img_background = cv2.bitwise_and(img_crop,img_crop,mask=img_back)
            img_let_foreground = cv2.bitwise_and(img_let_org,img_let_org,mask=img_back_inv)
            img_final2 = cv2.add(img_background,img_let_foreground)
            #kernel = np.ones((2,2),np.uint8)
            #img_let_border = cv2.morphologyEx(img_back, cv2.MORPH_GRADIENT, kernel)
            #img_final2[img_let_border != 0] = 0
            img_final = cv2.addWeighted(img_crop,0.3,img_final2,0.7,1)
            #kernel = np.ones((2,2),np.uint8)
            #img_let_border = cv2.morphologyEx(img_back, cv2.MORPH_GRADIENT, kernel)
            #img_final = cv2.inpaint(img_final,img_let_border,1,cv2.INPAINT_TELEA)
            img[ymin[0]:ymax[0],xmin[-1]:(xmin[-1]+(xmax[0]-xmin[0]))] = img_final
            if k == 4:
                xmin = []
                ymin = []
                xmax = []
                ymax = []
        lst.append[]        
    cv2.imwrite('trail_images387/'+"np_sync"+str(j)+".png",img)
    j = j +1 
            