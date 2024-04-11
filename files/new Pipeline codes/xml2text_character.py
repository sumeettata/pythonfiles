import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import json
import numpy as np
import shutil
import matplotlib.pyplot as plt

def rotate_text(point_s,img_s):
    point_s = point_s.reshape((-1, 1, 2))
    rect = cv2.minAreaRect(point_s)
    
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box_need = [box[1].tolist(),box[2].tolist(),box[3].tolist(),box[0].tolist()]
    j=0
    for i in box_need:
        if i[0] < 0:
            box_need[j][0] = 0
        if i[1] < 0:
            box_need[j][1] = 0
        j=j+1
            
    img2 = cv2.drawContours(img_s.copy(), [box], 0, (0, 0, 255), 2)
    img_mask = np.zeros(img_s.shape[:2], np.uint8)
    img_mask = cv2.drawContours(img_mask, [box], 0, 255, -1)
    if point_s[0][0][1] >= point_s[1][0][1] :
        rotate_matrix = cv2.getRotationMatrix2D(center=rect[0], angle=(90-rect[-1]), scale=1)
        rotated_image = cv2.warpAffine(src=img_s, M=rotate_matrix, dsize=(max(int(rect[1][1]),img_s.shape[1]),max(int(rect[1][0]),img_s.shape[0])))
        rotated_image_mask = cv2.warpAffine(src=img_mask, M=rotate_matrix, dsize=(max(int(rect[1][1]),img_s.shape[1]),max(int(rect[1][0]),img_s.shape[0])))    
    else:    
        rotate_matrix = cv2.getRotationMatrix2D(center=rect[0], angle=rect[-1], scale=1)
        rotated_image = cv2.warpAffine(src=img_s, M=rotate_matrix, dsize=(max(int(rect[1][0]),img_s.shape[1]),max(int(rect[1][1]),img_s.shape[0])))
        rotated_image_mask = cv2.warpAffine(src=img_mask, M=rotate_matrix, dsize=(max(int(rect[1][0]),img_s.shape[1]),max(int(rect[1][1]),img_s.shape[0])))    
        
    coor = cv2.boundingRect(rotated_image_mask)
    final_image = rotated_image[int(coor[1]):int(coor[1]+coor[3]),int(coor[0]):int(coor[0]+coor[2])] 
    
    if final_image.shape[0]>2*final_image.shape[1]:
        final_image = cv2.rotate(final_image, cv2.ROTATE_90_CLOCKWISE)
    
    return final_image,box_need,img2
    
# The input to the variables
image_path = 'C:/Users/SumeetMitra/Downloads/cropped_np_04-04-2023_paddleimg (2) (1)/cropped_np_04-04-2023_paddleimg/test_set'
xml_path = image_path

# Reading the paths and creating dir
dir_list = os.listdir(image_path)
files_jpg = [x for x in dir_list if not x.endswith(".xml")]

data_line =[]
class_lst = []
data_line_det = []

if not os.path.exists(os.path.join(os.path.dirname(image_path)+'//train_det')):
    os.mkdir(os.path.join(os.path.dirname(image_path)+'//train_det'))

for no,i in tqdm(enumerate(files_jpg)):
    lis = []
    label_path = os.path.join(xml_path,i.split('.')[0]+'.xml')
    file_path = os.path.join(image_path,i)
    image_file = cv2.imread(file_path)
    #image_2 = image_file.copy()
    
    #read images
    if os.path.exists(label_path):
        shutil.copy(file_path,os.path.join(os.path.dirname(image_path)+'//train_det//',i))
        tree = ET.parse(label_path)
        root = tree.getroot() 
        np_name = 'train/'+str(i)
        x_min = []
        y_min = []
        x_max = []
        y_max = []
        class_name = []
        e = 0
        for member in root.findall('object'):
            class_name.append(str(member[0].text))
            x_min.append(int(member[4][0].text))
            y_min.append(int(member[4][1].text))
            x_max.append(int(member[4][2].text))
            y_max.append(int(member[4][3].text))
            if len(x_min) > 1:
                if x_min[-1] < x_min[-2]:
                    np_ch = class_name[:-1]
                    
                    pts = np.array([[x_min[0],y_min[0]], [x_max[-2],y_min[-2]], [x_max[-2],y_max[-2]], [x_min[0],y_max[0]]],np.int32)
                    
                    x,y,w,h = cv2.boundingRect(pts)
                    rotate_image = image_file[y:y+h,x:x+w]
                    
                    box_image = pts
                    contoured_image = cv2.rectangle(image_file.copy(),(x,y),(x+w,y+h),(0,255,0),2)
                    
                    #rotate_image,box_image,contoured_image = rotate_text(pts,image_file)
                    dic = {}
                    dic["points"] = [box_image]  
                    dic["transcription"] = str("".join(np_ch))
                    
                    if not os.path.exists(os.path.dirname(image_path)+'//train'):
                        os.mkdir(os.path.dirname(image_path)+'//train')
                        
                    if not os.path.exists(os.path.dirname(image_path)+'//polygon_check//'):
                        os.mkdir(os.path.dirname(image_path)+'//polygon_check//')
                    cv2.imwrite(os.path.dirname(image_path)+'//polygon_check//'+i,contoured_image) 
                        
                    image_save = os.path.dirname(image_path)+'//train//'+'origin'+str(no)+'_'+str("".join(np_ch))+str(e)+'.png'
                    cv2.imwrite(image_save,rotate_image)
                    
                    lis.append(dic)
                    class_lst = "".join(np_ch)
                    data_line.append('train/'+'origin'+str(no)+'_'+str("".join(np_ch))+'_'+str(e)+'.png'+'\t'+str(class_lst)+'\n')
                    class_name = [class_name[-1]]
                    x_min = [x_min[-1]]
                    y_min = [y_min[-1]]
                    x_max = [x_max[-1]]
                    y_max = [y_max[-1]]
                    e = e+1
        
        if len(x_min) > 1:
            np_ch = class_name

            pts = np.array([[x_min[0],y_min[0]], [x_max[-1],y_min[-1]], [x_max[-1],y_max[-1]], [x_min[0],y_max[0]]],np.int32)
            # try:
            #    rotate_image,box_image,contoured_image = rotate_text(pts,image_file)
            # except:    
            #     rotate_image,box_image,contoured_image = rotate_text(pts,image_file)
            
            x,y,w,h = cv2.boundingRect(pts)
            rotate_image = image_file[y:y+h,x:x+w]
            
            box_image = pts
            contoured_image = cv2.rectangle(image_file.copy(),(x,y),(x+w,y+h),(0,255,0),2)
            
            
            dic = {}         
            dic["points"] = [box_image] 
            dic["transcription"] = str("".join(np_ch))
            
            if not os.path.exists(os.path.dirname(image_path)+'//train'):
                os.mkdir(os.path.dirname(image_path)+'//train')
                        
            if not os.path.exists(os.path.dirname(image_path)+'//polygon_check//'):
                os.mkdir(os.path.dirname(image_path)+'//polygon_check//')
            cv2.imwrite(os.path.dirname(image_path)+'//polygon_check//'+i,contoured_image) 
                
            image_save = os.path.dirname(image_path)+'//train//'+'origin'+str(no)+'_'+str("".join(np_ch))+str(e)+'.png'
            cv2.imwrite(image_save,rotate_image)
            
        
            lis.append(dic)
            class_lst = "".join(np_ch)
            data_line.append('train/'+'origin'+str(no)+'_'+str("".join(np_ch))+'_'+str(e)+'.png'+'\t'+str(class_lst)+'\n')
            
        
        if len(lis):
            print(lis)          
            #lis = json.dumps(lis)
        
            data_line_det.append(str(np_name)+'\t'+str(lis)+'\n')
            
data_str = ''.join(data_line)
data_det = ''.join(data_line_det)                

save_path = os.path.dirname(image_path)+'//train_list.txt'
with open(save_path, 'w', newline='') as f:
    f.write(data_str)
    f.close()
    
save_path2 = os.path.dirname(image_path)+'//train_list_det.txt'
with open(save_path2, 'w', newline='') as f:
    f.write(data_det)
    f.close()

