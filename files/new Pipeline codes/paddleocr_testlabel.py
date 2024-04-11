import cv2
import glob
import os
from paddleocr import PaddleOCR
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from tqdm import tqdm 
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm 

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ocr = PaddleOCR(use_angle_cls=False,lang='en',show_log=False)


path = 'C:/Users/SumeetMitra/Downloads/anpr_test/cropped'

def makedir(name):
    if not os.path.exists(name):
        os.mkdir(name)

makedir(path+'_paddlexml')
makedir(path+'_paddleimg')

for file in tqdm(glob.glob(path+'/*')):
    file_base = os.path.basename(file)
    
    img = cv2.imread(file)
    
    root = Element('annotation')
    SubElement(root, 'folder').text = os.path.dirname(file)
    SubElement(root, 'filename').text = os.path.basename(file)
    SubElement(root, 'path').text = './images/' + os.path.basename(file)
    
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'
    
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(img.shape[1])
    SubElement(size, 'height').text = str(img.shape[0])
    SubElement(size, 'depth').text = '3'
    SubElement(root, 'segmented').text = '0'
    
    result = ocr.ocr(img, cls=False)
        
    for idx in range(len(result)):
        res = result[idx]
        
        for line in res:
            txts = line[1][0]
            box = line[0]
            box = np.int0(box)
            
            box = cv2.boundingRect(box)
            obj = SubElement(root, 'object')
            SubElement(obj, 'name').text = str(txts)
            SubElement(obj, 'pose').text = 'Unspecified'
            SubElement(obj, 'truncated').text = '0'
            SubElement(obj, 'difficult').text = '0'

            bbox = SubElement(obj, 'bndbox')
            SubElement(bbox, 'xmin').text = str(box[0])
            SubElement(bbox, 'ymin').text = str(box[1])
            SubElement(bbox, 'xmax').text = str(box[0]+box[2])
            SubElement(bbox, 'ymax').text = str(box[1]+box[3])
            
    if len(root.findall('object')):
            tree = ElementTree(root)    
            xml_filename = path+'_paddlexml/'+file_base.split('.')[0]+'.xml'
            #tree.write(xml_filename)
            img_filename = path+'_paddleimg/'+file_base
            #shutil.copy(file,img_filename)
            
    