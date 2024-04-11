import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import json

# The input to the variables
image_path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/ANPR Stage2/testset_data_final/original'
xml_path = image_path
save_path = image_path+'_cropped'

# Reading the paths and creating dir
dir_list = os.listdir(image_path)
df = pd.read_csv('D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/ANPR Stage2/testset_data_final/Results.csv')

files_jpg = [x for x in dir_list if not x.endswith('.xml')]
if not os.path.exists(save_path):
    os.mkdir(save_path)

lst = []
lst1 = []
#for i in tqdm(files_jpg):
for index,row in tqdm(df.iterrows()):
    i= os.path.basename(row['file_name'])
    label_path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/ANPR Stage2/testset_data_final/original_labels/'+os.path.basename(row['file_name']).split('.')[0]+'.xml'
    file_path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/ANPR Stage2/testset_data_final/'+ row['file_name']
    #label_path = os.path.join(xml_path,i.split('.')[0]+'.xml')
    #file_path = os.path.join(image_path,i)
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        dic_lst = []
        for k,member in enumerate(root.findall('object')):
            class_name = str(member[0].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
#             saving the cropped images into different folders with class number as names
            #if class_name == 'np':
            img = cv2.imread(file_path)
            img_name = os.path.basename(file_path)
            img = img[ymin:ymax,xmin:xmax]
            cv2.imwrite(save_path+'//'+i.split('.')[0]+'_'+str(k)+'.png',img)
            lst.append(str(i.split('.')[0]+'_'+str(k)+'.png')+'\t'+str(class_name)+'\n')
            
            dic1 = {}
            dic1["points"] = [[xmin,ymin],[xmax,ymin],[xmin,ymax],[xmax,ymax]]
            dic1["transcription"] = str(class_name)
            dic_lst.append(dic1)
            lst2 = json.dumps(dic_lst)
            
            lst1.append(str(row['file_name'])+'\t'+str(lst2)+'\n')
            # if not os.path.exists(save_path+'//'+str(class_name)):
            #     os.mkdir(save_path+'//'+str(class_name))
            # cv2.imwrite(os.path.join(save_path+'//'+str(class_name),i+'_'+str(k)+file_extension),img)
            
with open('D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/ANPR Stage2/testset_data_final/train_rec.txt', 'w', newline='') as f:
    f.write(''.join(lst))
    f.close()
    
with open('D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/ANPR Stage2/testset_data_final/train_det.txt', 'w', newline='') as f:
    f.write(''.join(lst1))
    f.close()