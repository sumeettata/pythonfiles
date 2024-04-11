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
plate_path = "Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/Train_data"
plate_labelpath = plate_path


#if not os.path.exists('trail_images387'):
    #os.mkdir('trail_images387')
    
# def get_dim(pts,M):
#     pts2 = np.append(pts,[[1],[1],[1],[1]],axis=1)
#     pts2 = np.transpose(np.matmul(M,np.transpose(pts2)))
#     pts2 = np.delete(pts2,-1,1).astype(int)
#     dim = np.max(pts2,axis=0) - np.min(pts2,axis=0)
#     return dim
    
    
plate_pth = glob.glob(plate_path+"/*")[6]
label_pth = plate_labelpath+"/"+os.path.basename(plate_pth).split(".")[0]+".xml"
if os.path.exists(label_pth):
    tree = ET.parse(label_pth)
    root = tree.getroot()
    img = cv2.imread(plate_pth)
    print(img.shape)
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
    #img_pts =  np.array([[0,0],[img.shape[0],0],[0,img.shape[1]],[img.shape[0],img.shape[1]]])  
    #pts = np.array([[xmin[0],ymin[0]], [xmax[-1],ymin[-1]], [xmax[-1],ymax[-1]], [xmin[0],ymax[0]]],np.int32)
    #img2 = cv2.rectangle(img.copy(), (xmin[0],ymin[0]), (xmax[-1],ymax[-1]), (255,255,255), 5) 
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
    a1 = random.randrange(-5,5)*0.1
    b1 = random.randrange(-5,5)*0.1
    M_shear = np.float32([[1, a1, 0],[b1, 1  , 0],[0, 0  , 1]])
    a2 = random.randrange(8,10)*0.1
    b2 = random.randrange(8,10)*0.1
    M_scale = np.float32([[a2, 0  , 0],[0,   b2, 0],[0,   0,   1]])
    M_transition = np.float32([[1, 0, -bordr],[0, 1  , -bordr],[0, 0  , 1]])
    M_transition_rev = np.float32([[1, 0, bordr],[0, 1  , bordr],[0, 0  , 1]])
    M = M_transition_rev@M_shear@M_rotate@M_transition
    
    img_changed_mask = cv2.warpPerspective(img_mask,M,(img_mask.shape[1],img_mask.shape[0]))
    contours,_= cv2.findContours(img_changed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dim_mask = cv2.boundingRect(contours[0])
    
    txt_mask = cv2.warpPerspective(txt_mask,M,(txt_mask.shape[1],txt_mask.shape[0]))
    contours2,_= cv2.findContours(txt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dim_mask2 = cv2.minAreaRect(contours[0])
    dim_mask2 = np.int0(cv2.boxPoints(dim_mask2))
            
    img_changed = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
    plt.imshow(cv2.rectangle(img_changed.copy(), (dim_mask[0],dim_mask[1]), ((dim_mask[0]+dim_mask[2]),(dim_mask[1]+dim_mask[3])), (255,255,255), 5) )
    plt.show()
    img_final = img_changed[dim_mask[1]:(dim_mask[1]+dim_mask[3]),dim_mask[0]:(dim_mask[0]+dim_mask[2])]
    plt.imshow(img_final)
    plt.show()
            
    di = cv2.drawContours(img_changed, [dim_mask2], 0, (0, 0, 255), 2)
    plt.imshow(di)
    plt.show()