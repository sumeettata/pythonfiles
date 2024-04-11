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

    
# The input to the variables
image_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//ANPR Stage2//good_images//final//Train_data'
xml_path = image_path

# Reading the paths and creating dir
dir_list = os.listdir(image_path)
files_jpg = [x for x in dir_list if not x.endswith(".xml")]


data_line =[]
class_lst = []
data_line_det = []
lst_csv = []
for i in tqdm(files_jpg):
    lis = []
    label_path = os.path.join(xml_path,i.split('.')[0]+'.xml')
    file_path = os.path.join(image_path,i)
    image_file = cv2.imread(file_path)
    image_2 = image_file.copy()
    
    #read images
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot() 
        np_name = str(i)
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
                    
                    pts = pts.reshape((-1, 1, 2))
                    rect = cv2.minAreaRect(pts)
                    
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    image_2 = cv2.drawContours(image_2, [box], 0, (0, 0, 255), 2)
                    
                    dic = {}
                    dic["points"] = box.tolist()
                    dic["transcription"] = str("".join(np_ch))
                    
                    
                    lis.append(dic)
                    class_lst = "".join(np_ch)
                    
                    
                    data_line.append(str(np_name.split(".")[0]+str(e)+'.'+np_name.split(".")[1]+'\t'+str(class_lst)+'\n'))
                    class_name = [class_name[-1]]
                    x_min = [x_min[-1]]
                    y_min = [y_min[-1]]
                    x_max = [x_max[-1]]
                    y_max = [y_max[-1]]
                    e = e+1
        
        
        if len(x_min) > 1:
            np_ch = class_name

            pts = np.array([[x_min[0],y_min[0]], [x_max[-1],y_min[-1]], [x_max[-1],y_max[-1]], [x_min[0],y_max[0]]],np.int32)
            
            
            pts = pts.reshape((-1, 1, 2))
            rect = cv2.minAreaRect(pts)
            
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            image_2 = cv2.drawContours(image_2, [box], 0, (0, 0, 255), 2)
            
            dic = {}         
            dic["points"] = box.tolist()
            dic["transcription"] = str("".join(np_ch))
            
            
            lis.append(dic)
            class_lst = "".join(np_ch)
            data_line.append(str(np_name.split(".")[0]+str(e)+'.'+np_name.split(".")[1]+'\t'+str(class_lst)+'\n'))
            
        
        if len(lis):
            print(lis)          
            lis = json.dumps(lis)
        
            data_line_det.append(str(np_name)+'\t'+str(lis)+'\n')    
            
            lst_csv.append([str(np_name),str(class_lst),str(lis)])
                        
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resized_Window", 500 , 500 )
        cv2.imshow("Resized_Window", image_2)
        cv2.waitKey(1) 

         
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

save_path2 = os.path.dirname(image_path)+'//original_label.csv'
df = pd.DataFrame(lst_csv,columns=['filename','ground_truth','location'])
df.to_csv(save_path2)