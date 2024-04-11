import cv2
import os
import xml.etree.ElementTree as ET
import random
import pandas as pd
import glob
import statistics
import numpy as np
import glob

def add_defect(steel_image,defect_path,location_defect):
    defect_image1 = cv2.imread(defect_path,cv2.IMREAD_UNCHANGED)
    r1,c1,ch1 = steel_image.shape
    l2, l1 = location_defect
    if (r1 > 2000) or (c1 > 2000) : 
        resize_factor = random.choice(range(5,50))*0.1
    else:
        resize_factor = random.choice(range(2,50))*0.01
    defect_image = cv2.resize(defect_image1,(0, 0), fx = resize_factor, fy = resize_factor)
    defect_image_back1 = defect_image[:,:,3]
    defect_image_bgr_original = defect_image[:,:,:3]
    defect_image_bgr = defect_image_bgr_original
    #defect_image_bgr = cv2.convertScaleAbs(defect_image_bgr_original, alpha=defect_color[0], beta=defect_color[1])
    
    r2,c2,ch2 = defect_image.shape
    x1 = int(l1-(r2/2))
    x2 = int(l1+(r2/2))
    y1 = int(l2-(c2/2))
    y2 = int(l2+(c2/2))
    if (x1 > 0) and (y1 > 0) and (x2 < r1) and (y2 < c1) and (x2 > 0) and (y2 > 0) and (x1 < r1) and (y1 < c1) :
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

def Read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    list_xml = []
    for member in root.findall('object'):
        class_name = member[0].text
        xmin = int(member[5][0].text)
        ymin = int(member[5][2].text)
        xmax = int(member[5][1].text)
        ymax = int(member[5][3].text)
        list_xml.append([xmin,ymin,xmax,ymax])
    return list_xml

def location_defect(defect_path,rod_location_list):
    xx1,yy1,xx2,yy2 = rod_location_list
    defect_image_ = cv2.imread(defect_path)
    rr2,cc2,chh2 = defect_image_.shape
    final_x1 = int(xx1+(cc2/2))
    final_x2 = int(xx2-(cc2/2))
    final_y1 = int(yy1+(rr2/2))
    final_y2 = int(yy2-(rr2/2))
    range_list = [final_x1,final_y1,final_x2,final_y2]
    return range_list

files = glob.glob('D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/fire_smoke/fire_smoke_anurag/*/*/*/*')
files_jpg = [x for x in files if not x.endswith('.xml')]
files_garbage = glob.glob('D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/fire_smoke/Fire and Smoke Segmentation.v4i.voc/PNG-Files/fire/*')
save_path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/fire_smoke/final/fire_smoke_synthetic5'

if not os.path.exists(save_path):
    os.mkdir(save_path)
    
i=0
lst = []
for file in files_jpg:
    xml_path = file[:-4]+'.xml'
    if os.path.exists(xml_path):
        rod_list = Read_xml(xml_path)
        steel_image = cv2.imread(file)
        if len(rod_list):
            j = 0
            while j < 3:
                garbage_block = random.choice(rod_list)
                garbage_file = random.choice(files_garbage)
                range_ls = garbage_block
                if abs(range_ls[0] - range_ls[2]) >1:
                    a = random.randrange(range_ls[0], range_ls[2])
                else:
                    a = (range_ls[0]+range_ls[2])/2
                if abs(range_ls[1] - range_ls[3]) >1:
                    b = random.randrange(range_ls[1], range_ls[3])
                else:
                    b = (range_ls[1]+range_ls[3])/2    
                
                
                steel_image,position_list= add_defect(steel_image,garbage_file,[a,b])
                
                if sum(position_list) != 0:
                    j = j+1
                    class_name = os.path.basename(os.path.dirname(garbage_file))
                    lst.append(['sync_firesmoke_'+str(i)+'.jpg',str(class_name),steel_image.shape[1],steel_image.shape[0]]+position_list)
            cv2.imwrite(save_path+'/sync_firesmoke_'+str(i)+'.jpg', steel_image)
            cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Resized_Window", 640 , 640 )
            cv2.imshow("Resized_Window", steel_image)
            cv2.waitKey(1)
            
            i = i+1
    
df = pd.DataFrame(lst,columns=['File_name','Class_name','width','height','x1','y1','x2','y2'])
df.to_csv(save_path+'.csv')
        
   