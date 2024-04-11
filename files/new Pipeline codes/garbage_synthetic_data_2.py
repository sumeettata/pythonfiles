import pandas as pd
import glob
import cv2
from tqdm import tqdm
import numpy as np
import os
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

path_image = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage_new/Road_imgs/bdd100k/*/'
path_save = 'Garbage_synthetic_final1/'


files_garbage = glob.glob('D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage_new/PNG_files/*/*.png')
files = glob.glob(path_image+'/*.jpg')


if not os.path.exists(path_save):
    os.mkdir(path_save)
    
if not os.path.exists(path_save+'images'):
    os.mkdir(path_save+'images')

if not os.path.exists(path_save+'labels'):
    os.mkdir(path_save+'labels')

def xml2mask(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    if "Road Detection Segmentation.v9i.voc" in xml_file:
        for member in root.findall('size'):
            width = int(member[1].text)
            height = int(member[0].text)
            print('check')
    else:
        for member in root.findall('size'):
            width = int(member[0].text)
            height = int(member[1].text)
        
    mask = np.zeros([width,height,1],dtype=np.uint8)    
    for member in root.findall('object'):
        class_name = str(member[0].text)
        df=pd.DataFrame(columns=['x','y'])
        for mem in member[6].iter():
            if (str(mem.tag[0]) == 'x') or (str(mem.tag[0]) == 'y'):
                df.loc[mem.tag[1:],str(mem.tag[0])]=int(float(mem.text))
        df['combined']= df.values.tolist()
        pts = list(df['combined'])
        pts = np.array(pts,np.int32)
        pts = pts.reshape((-1, 1, 2))
        mask = cv2.fillPoly(mask, [pts],1) 
    return mask    
        
def add_defect(steel_image,defect_path,location_defect):
    defect_image1 = cv2.imread(defect_path,cv2.IMREAD_UNCHANGED)
    r1,c1,ch1 = steel_image.shape
    l2, l1 = location_defect
    resize_factor = random.choice(range(5,20))*0.01
    defect_image = cv2.resize(defect_image1,(0, 0), fx = resize_factor, fy = resize_factor)
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
        x1=x2=y1=y2=0
        
    return steel_image,[y1,x1,y2,x2]

def csv2txt(lst_g,lst_txt):
    x_cen = ((int(lst_g[0]) + int(lst_g[2]))/2)/int(img.shape[1])
    y_cen = ((int(lst_g[1]) + int(lst_g[3]))/2)/int(img.shape[0])
    w = abs(int(lst_g[0]) - int(lst_g[2]))/int(img.shape[1])
    h = abs(int(lst_g[1]) - int(lst_g[3]))/int(img.shape[0])

    lst_txt.append('0'+' '+str(format(x_cen, '.6f'))+' '+str(format(y_cen, '.6f'))+' '+str(format(w, '.6f'))+' '+str(format(h, '.6f')))
    return lst_txt

lst = []
for i in files:
    lst_garbage = []
    lop = 0
    img = cv2.imread(i)
    height_img = int(img.shape[0])
    img = cv2.resize(img,(640,640))
    img_copy = img.copy()
    xml_path = i.replace('.jpg','.xml')
    mask_path = i.replace('.jpg','.png')
    mask_path2 = i.replace('.jpg','_mask.png')
    if os.path.exists(xml_path):
        img_mask = xml2mask(xml_path)[:,:,0]
    elif os.path.exists(mask_path):
        img_mask = cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)
        if img_mask.shape[2] == 4:
            img_mask = img_mask[:,:,3]
    elif os.path.exists(mask_path2):
        img_mask = cv2.imread(mask_path2,cv2.IMREAD_UNCHANGED)
        img_mask[img_mask == 2] = 255
    else:
        print('error')
    img_mask = cv2.resize(img_mask,(640,640))
    img_mask[img_mask == 1] = 255
    img_mask[img_mask == 8] = 255
    img_mask[img_mask == 7] = 255
    img_mask[img_mask != 255] = 0
    img_mask_inv = cv2.bitwise_not(img_mask)
    p=0
    garb_count = random.choice(range(2,6))
    while lop < garb_count:
        gar_im = random.choice(files_garbage)
        a = random.randrange(0, img.shape[1])
        b = random.randrange(0, img.shape[0])
        p = p +1
        if img_mask[b,a] == 255:
            if not (a-b) == 0:
                img,gar_loc = add_defect(img,gar_im,[a,b])
                if sum(gar_loc) != 0:
                    lst.append([os.path.basename(i),'garbage',img.shape[1],img.shape[0]]+gar_loc)
                    lst_garbage.append(gar_loc)
                    lop = lop+1
        if p >100:
            break
    img1_bg = cv2.bitwise_and(img_copy,img_copy,mask = img_mask_inv)
    img2_fg = cv2.bitwise_and(img,img,mask = img_mask)
    dst = cv2.add(img1_bg,img2_fg)
    cv2.imwrite(path_save+'images/'+os.path.basename(i),dst)
    
    dst2 = dst.copy()
    lst_txt = []
    for lst_g in lst_garbage:
        dst2 = cv2.rectangle(dst2, (lst_g[0],lst_g[1]), (lst_g[2],lst_g[3]), [0,0,255], 5)
        lst_txt = csv2txt(lst_g,lst_txt)
   
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window", 640 , 640 )
    cv2.imshow("Resized_Window", np.hstack((dst2,cv2.cvtColor(img_mask,cv2.COLOR_GRAY2BGR))))
    cv2.waitKey(1)
    
df = pd.DataFrame(lst,columns=['File_name','Class_name','width','height','x1','y1','x2','y2'])
df.to_csv(path_save+'Garbage_tagged_seg_new1.csv')

