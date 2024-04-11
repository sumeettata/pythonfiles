import os
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
import numpy as np
import cv2
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree

# The input to the variables
image_path = 'C://Users//SumeetMitra//Downloads//nbm_rod_sumeet//nbm_rod_sumeet'
xml_path = image_path

files = glob.glob(image_path+'//*')
files = [x for x in files if not x.endswith('.xml')]

for file in files:
    img = cv2.imread(file,0)
    ret,img_thres = cv2.threshold(img,100,1,cv2.THRESH_BINARY)
    img_numpy = np.array(img_thres)
    img_numpy_2 = np.array(img)
    img_where = np.where(np.sum(img_numpy,axis=0) > 50)[0]
    img_where_2 = np.where(np.sum(img_numpy_2,axis=0) == np.sum(img_numpy_2,axis=0).max())[0]
    xml_file = file.split('.')[0]+'.xml'
    if os.path.exists(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            class_name = str(member[0].text)
            if class_name == 'rod':
                member[4][0].text = str(img_where[3]-6)
                member[4][2].text = str(img_where[-3]+6)
                
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = 'rib'
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = str(img_where_2[0]-14)
        SubElement(bbox, 'ymin').text = '0'
        SubElement(bbox, 'xmax').text = str(img_where_2[-1]+14)
        SubElement(bbox, 'ymax').text = str(img.shape[0])
        tree.write(xml_file)