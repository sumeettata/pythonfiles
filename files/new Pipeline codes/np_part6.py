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
import largestinteriorrectangle as lir

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
    
if not os.path.exists('trail_images349'):
    os.mkdir('trail_images349')
j = 0
#for rto_number in rto_code:
    
while j < 100:
    lst_label = []
    plate_pth = glob.glob(plate_path+"/*")[4]
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
            #if k == 4:
                #xmin = []
                #ymin = []
                #xmax = []
                #ymax = []
        #lst.append[]
        
        img_mask = np.ones([img.shape[0],img.shape[1]],dtype=np.uint8)*255
    
        txt_mask = np.zeros([img.shape[0],img.shape[1]],dtype=np.uint8)
        txt_mask = cv2.rectangle(txt_mask, (xmin[0],ymin[0]), (xmax[-1],ymax[-1]), (255,255,255), -1)
        
        bordr = 500
        txt_mask = cv2.copyMakeBorder(txt_mask, bordr, bordr, bordr, bordr, cv2.BORDER_CONSTANT, value=[0, 0, 0]) 
        img_mask = cv2.copyMakeBorder(img_mask, bordr, bordr, bordr, bordr, cv2.BORDER_CONSTANT, value=[0, 0, 0])   
        img = cv2.copyMakeBorder(img, bordr, bordr, bordr, bordr, cv2.BORDER_CONSTANT, value=[200, 200, 200])
        
        #rows, cols, dim = img.shape
        angle = np.radians(random.randrange(-10,10))
        M_rotate = np.float32([[np.cos(angle), -(np.sin(angle)), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
        a1 = random.randrange(-2,2)*0.05
        b1 = random.randrange(-2,2)*0.05
        M_shear = np.float32([[1, a1, 0],[b1, 1  , 0],[0, 0  , 1]])
        a2 = random.randrange(5,10)*0.1
        b2 = random.randrange(5,10)*0.1
        M_scale = np.float32([[a2, 0  , 0],[0,   b2, 0],[0,   0,   1]])
        M_transition = np.float32([[1, 0, -bordr],[0, 1  , -bordr],[0, 0  , 1]])
        M_transition_rev = np.float32([[1, 0, bordr],[0, 1  , bordr],[0, 0  , 1]])
        M = M_transition_rev@M_shear@M_rotate@M_transition
        
        img_changed_mask = cv2.warpPerspective(img_mask,M,(img_mask.shape[1],img_mask.shape[0]))
        contours,_= cv2.findContours(img_changed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_pts2  = cv2.goodFeaturesToTrack(img_changed_mask, 27, 0.01, 10)
        corners = np.int0(img_pts2.reshape((img_pts2.shape[0],-1)))
        cor_xmin = sorted(corners[:,0])[1]
        cor_xmax = sorted(corners[:,0])[-2]
        cor_ymin = sorted(corners[:,1])[1]
        cor_ymax = sorted(corners[:,1])[-2]
        dim_mask = cv2.boundingRect(contours[0])
        
        txt_mask = cv2.warpPerspective(txt_mask,M,(txt_mask.shape[1],txt_mask.shape[0]))
        #txt_mask2 = cv2.rectangle(txt_mask, (cor_xmin,cor_xmax), (cor_ymin,cor_ymax), 0, -1)
        txt_mask = txt_mask[cor_ymin:cor_ymax,cor_xmin:cor_xmax]
        #contours4  = np.int0(cv2.goodFeaturesToTrack(txt_mask, 27, 0.01, 10)).reshape(-1,2)
        #dim_mask2 = cv2.minAreaRect(contours2[0])
        #dim_mask2 = np.int0(cv2.boxPoints(dim_mask2))
        contours4,_= cv2.findContours(txt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        dim_mask2 = cv2.minAreaRect(contours4[0])
        dim_mask2 = np.int0(cv2.boxPoints(dim_mask2))
        print(contours4)
        if np.min(dim_mask2) < 0:
            dim_mask2 = cv2.boundingRect(contours4)
        
        print(dim_mask2)
        #print(contours4) 
        #contours2,_= cv2.findContours(txt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #dim_mask2 = cv2.minAreaRect(contours2[0])
        #dim_mask2 = np.int0(cv2.boxPoints(dim_mask2))
        #txt_mask = cv2.polylines(txt_mask, [dim_mask2],True, 0, 8)
        #rectangle = lir.lir(np.array([np.int0(contours4.reshape(-1,2))]))
        #txt_mask = cv2.rectangle(txt_mask, lir.pt2(rectangle), lir.pt1(rectangle), 0, 1)
        #print(dim_mask2)
        #plt.imshow(txt_mask)
        #plt.show()        
        img_changed = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
        #img_final = img_changed[dim_mask[1]:(dim_mask[1]+dim_mask[3]),dim_mask[0]:(dim_mask[0]+dim_mask[2])]
        img_final = img_changed[cor_ymin:cor_ymax,cor_xmin:cor_xmax]  
        img_final = cv2.polylines(img_final, [dim_mask2], True, 0, 2) 
        plt.imshow(img_final)
        plt.show()
    dic = {}
    dic["points"] = [dim_mask2]  
    dic["transcription"] = str(plate_number)  
    break         
    cv2.imwrite('trail_images349/'+"np_sync"+str(j)+".png",img_final)
    j = j +1 
            