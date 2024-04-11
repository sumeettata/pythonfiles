import os
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
import shutil

#code used to get the cropped pic of car which contained labelled number plates only 



# The input to the variables
image_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Main//Images'
xml_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Main//labels_vehicles'
save_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//cropped_vehicle'

# Reading the paths and creating dir
files_jpg = glob.glob(image_path+"//*")
files_jpg = [x for x in files_jpg if not x.endswith('.xml')]

    
def create_path(crt_path):
    if not os.path.exists(crt_path):
        os.mkdir(crt_path) 
    return crt_path
    
def write_jpg(file_path,veh_l,save_file_1):
    img = cv2.imread(file_path)
    img = img[veh_l[2]:veh_l[4],veh_l[1]:veh_l[3]]
    cv2.imwrite(save_file_1+'.jpg',img)    

def lst2xml(file_path,new_np,veh_l,save_file_2):
    root = Element('annotation')
    SubElement(root, 'folder').text = os.path.dirname(file_path)
    SubElement(root, 'filename').text = os.path.basename(file_path)
    SubElement(root, 'path').text = './images' + os.path.basename(file_path)
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'
    
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(veh_l[3] - veh_l[1])
    SubElement(size, 'height').text = str(veh_l[4] - veh_l[2])
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'
    
    obj = SubElement(root, 'object')
    SubElement(obj, 'name').text = 'np'
    SubElement(obj, 'pose').text = 'Unspecified'
    SubElement(obj, 'truncated').text = '0'
    SubElement(obj, 'difficult').text = '0'

    bbox = SubElement(obj, 'bndbox')
    SubElement(bbox, 'xmin').text = str(new_np[0])
    SubElement(bbox, 'ymin').text = str(new_np[1])
    SubElement(bbox, 'xmax').text = str(new_np[2])
    SubElement(bbox, 'ymax').text = str(new_np[3])
    
    tree = ElementTree(root)    
    tree.write(save_file_2+'.xml')
    
    
save_path = create_path(save_path)
xml_save_path = create_path(save_path+"//labels") 
jpg_save_path = create_path(save_path+"//images")    
    
    
for file in tqdm(files_jpg):
    np_lst =[]
    vehicle_lst =[]
    file_name = os.path.basename(file)
    label_path = os.path.join(xml_path,file_name.split('.')[0]+'.xml')
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        for k,member in enumerate(root.findall('object')):
            class_name = str(member[0].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            lst = [class_name,xmin,ymin,xmax,ymax]
#             saving the cropped images into different folders with class number as names
            if class_name == 'np':
                np_lst.append(lst)
            else:
                vehicle_lst.append(lst)
        i=0
        for np_0 in np_lst:
            for veh_0 in vehicle_lst:
                if ((np_0[1] > veh_0[1]) and (np_0[2] > veh_0[2]) and (np_0[3] < veh_0[3]) and (np_0[4] < veh_0[4])):
                    new_np_0 = np.subtract(np.array(np_0[1:5]),np.array(veh_0[1:3]*2))
                    save_file = file_name.split('.')[0]+"_"+str(veh_0[0])+"_"+str(i)
                    write_jpg(file,veh_0,jpg_save_path+"//"+save_file)
                    lst2xml(file,new_np_0,veh_0,xml_save_path+"//"+save_file)
                    vehicle_lst.remove(veh_0)
                    


shutil.make_archive(save_path+'//cropped_vehicle', 'zip', save_path)
