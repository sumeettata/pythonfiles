import os
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
import pandas as pd
import cv2


class xml_converter:
    def __init__(self,img_path,label_path=None,save_path=None):
        self.img_path = img_path
        self.path_base = os.path.basename(self.img_path)
        self.path_dir = os.path.dirname(self.img_path)
        if label_path:
            self.label_path = label_path
        else:
            self.label_path = self.img_path.split(".")[0]+".xml"
        if save_path:
            self.save_path = save_path
        else:
            self.save_path = self.path_dir
        self.lst_box = self.reader()
    
        
    def reader(self):
        tree = ET.parse(self.label_path)
        root = tree.getroot()
        lst_bbox = []
        for mem in root.findall('size'):
            width = int(mem[0].text)
            height = int(mem[1].text)
        for k,member in enumerate(root.findall('object')):
            class_name = str(member[0].text)
            xmin = int(round(float(member[4][0].text)))
            ymin = int(round(float(member[4][1].text)))
            xmax = int(round(float(member[4][2].text)))
            ymax = int(round(float(member[4][3].text)))
            lst_bbox.append([self.path_base,class_name,xmin,ymin,xmax,ymax,height,width])
            
        return lst_bbox
    
    def folder_check(self,check_path):
        if not os.path.exists(check_path):
            os.mkdir(check_path)
        return check_path 
               
    def crop(self,class_crop=None,img_no = 1):
        if class_crop:
            for box in self.lst_box:
                if str(bbox[1]) in class_crop:
                    img = cv2.imread(self.img_path)
                    img = img[box[3]:box[5],box[2]:box[4]]
                    new_path = self.folder_check(self.path_dir+'/'+str(box[1]))
                    cv2.imwrite(new_path+'/'+box[0].split('.')[0]+'_'+str(img_no)+'.png',img)
        else:
            for box in self.lst_box:
                img = cv2.imread(self.img_path)
                img = img[box[3]:box[5],box[2]:box[4]] 
                new_path = self.folder_check(self.path_dir+'/'+str(box[1]))
                cv2.imwrite(new_path+'/'+box[0].split('.')[0]+'_'+str(img_no)+'.png',img)  
                        
           
        
    def convert_csv(self):
        df = pd.DataFrame(self.lst_box,columns=['filename', 'class_name','x1', 'y1', 'x2', 'y2','image_width', 'image_height'])
        df.to_csv(self.path_dir+"/"+self.path_base.split('.')[0]+".csv")
        
    def 
        
test = xml_converter("D:/OneDrive/OneDrive - Tata Insights and Quants/ANPR stage 2/downloaded plates/archive/Indian_Number_Plates/Sample_Images/Datacluster_number_plates (1).jpg")
test.crop()
test.convert_csv()
