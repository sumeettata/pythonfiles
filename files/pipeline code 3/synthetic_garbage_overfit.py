import pandas as pd
import glob
import cv2
from tqdm import tqdm
import numpy as np
import os
import random
import xml.etree.ElementTree as ET

path_image = '/CV/data/garbage/Road_segment/Road_imgs/*/*/'
path_save = '/CV/data/garbage/Road_segment/Garbage_synthetic_created_overfit/'
mask_path = '/CV/data/garbage/Road_segment/'

files_garbage = glob.glob('/CV/data/garbage/Road_segment/PNG_files/*/*.png')

files = glob.glob(path_image+'/*.jpg')
if not os.path.exists(path_save):
    os.mkdir(path_save)
    
if not os.path.exists(path_save+'images'):
    os.mkdir(path_save+'images')

if not os.path.exists(path_save+'labels'):
    os.mkdir(path_save+'labels')

        
def add_defect(steel_image,defect_path,location_defect):
    defect_image1 = cv2.imread(defect_path,cv2.IMREAD_UNCHANGED)
    r1,c1,ch1 = steel_image.shape
    l2, l1 = location_defect
    resize_factor = random.choice(range(2,40))*0.01
    try:
    	defect_image = cv2.resize(defect_image1,(0, 0), fx = resize_factor, fy = resize_factor)
    except:
        defect_image = cv2.resize(defect_image1,(0, 0), fx = 0.1, fy = 0.1)
        print('pass')
    defect_image_back1 = defect_image[:,:,3]
    defect_image_bgr_original = defect_image[:,:,:3]
    defect_image_bgr = defect_image_bgr_original
    if random.randrange(0,100) > 70:
        defect_image_bgr = cv2.convertScaleAbs(defect_image_bgr_original, alpha=random.randrange(5,10)*0.1, beta=random.randrange(0,10))
    
    r2,c2,ch2 = defect_image.shape
    x1 = int(l1-(r2/2))
    x2 = int(l1+(r2/2))
    y1 = int(l2-(c2/2))
    y2 = int(l2+(c2/2))
    if (x1 > 0) and (y1 > 0) and (x2 < r1) and (y2 < c1):
        steel_mask = steel_image[x1:x2,y1:y2]
        ret, defect_image_back = cv2.threshold(defect_image_back1, 127, 255, cv2.THRESH_BINARY)
        defect_image_back_inv = cv2.bitwise_not(defect_image_back)
        steel_background = cv2.bitwise_and(steel_mask,steel_mask,mask=defect_image_back_inv)
        defect_foreground = cv2.bitwise_and(defect_image_bgr,defect_image_bgr,mask=defect_image_back)
        result_image = cv2.add(steel_background,defect_foreground)
        steel_image[x1:x2,y1:y2] = result_image
    else:
        x1=y1=x2=y2=0
        
    return steel_image,[y1,x1,y2,x2]

def csv2txt(lst_g,lst_txt):
    x_cen = ((int(lst_g[0]) + int(lst_g[2]))/2)/int(img.shape[1])
    y_cen = ((int(lst_g[1]) + int(lst_g[3]))/2)/int(img.shape[0])
    w = abs(int(lst_g[0]) - int(lst_g[2]))/int(img.shape[1])
    h = abs(int(lst_g[1]) - int(lst_g[3]))/int(img.shape[0])

    lst_txt.append('0'+' '+str(format(x_cen, '.6f'))+' '+str(format(y_cen, '.6f'))+' '+str(format(w, '.6f'))+' '+str(format(h, '.6f')))
    return lst_txt

lst = []
no = 0
for i in tqdm(files):
    no = no+1
    lst_garbage = []
    lop = 0
    img = cv2.imread(i)
    img = cv2.resize(img,(640,640))
    img_copy = img.copy()
    mask_path = mask_path
    img_mask = cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)[:,:,3]
    img_mask = cv2.resize(img_mask,(640,640))

    img_mask[img_mask == 8] = 255
    img_mask[img_mask != 255] = 0
    
    img_mask_inv = cv2.bitwise_not(img_mask)
    p=0
    garb_count = random.choice(range(2,10))
    while lop < garb_count:
        gar_im = random.choice(files_garbage)
        a = random.randrange(0, img.shape[1])
        b = random.randrange(0, img.shape[0])
        p = p +1
        if img_mask[b,a] == 255:
            if not (a-b) == 0:
                img,gar_loc = add_defect(img,gar_im,[a,b])
                if sum(gar_loc) != 0:
                    lst.append(['garbagesync_'+str(no)+'.png','garbage',img.shape[1],img.shape[0]]+gar_loc)
                    lst_garbage.append(gar_loc)
                    lop = lop+1
        if p >100:
            break
    img1_bg = cv2.bitwise_and(img_copy,img_copy,mask = img_mask_inv)
    img2_fg = cv2.bitwise_and(img,img,mask = img_mask)
    dst = cv2.add(img1_bg,img2_fg)
    cv2.imwrite(path_save+'images/'+'garbagesync_'+str(no)+'.png',dst)
    dst2 = dst.copy()
    lst_txt = []
    for lst_g in lst_garbage:
        dst2 = cv2.rectangle(dst2, (lst_g[0],lst_g[1]), (lst_g[2],lst_g[3]), [0,0,255], 5)
        lst_txt = csv2txt(lst_g,lst_txt)
            
df = pd.DataFrame(lst,columns=['File_name','Class_name','width','height','x1','y1','x2','y2'])
df.to_csv(path_save+'Garbage_tagged_seg_new.csv')

