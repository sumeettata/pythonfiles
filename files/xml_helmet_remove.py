import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm


# The input to the variables
xml_path = 'C:/Users/SumeetMitra/Downloads/Tagging/Tagging'
image_path = 'C:/Users/SumeetMitra/Downloads/Tagging/images'

# Reading the paths and creating dir
dir_list = os.listdir(image_path)
files_jpg = [x for x in dir_list if not x.endswith(".xml")]

for i in tqdm(files_jpg):
    label_path = os.path.join(xml_path,i.replace('.jpg','.xml').replace('.png','.xml').replace('.jpeg','.xml'))
    file_path = os.path.join(image_path,i)
    if os.path.exists(file_path):
        print(label_path)
        tree = ET.parse(label_path)
        root = tree.getroot()
        human_lst = []
        for k,member in enumerate(root.findall('object')):
            class_name = str(member[0].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            if xmin < 881:
                root.remove(member)
            elif ymin < 135:
                root.remove(member)

        tree.write(label_path)


        
