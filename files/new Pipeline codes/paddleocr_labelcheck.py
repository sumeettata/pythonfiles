import glob
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET

jpg_path = "Work//TATA Communications//Data_1//ANPR Stage2//good_images//final//Train_data"
xml_path = jpg_path

y=0
c=0
r=0
for file in tqdm([x for x in glob.glob(jpg_path+'//*') if x.endswith('.xml')]):
    class_name = []
    xmin = []
    j = 0
    tree = ET.parse(file)
    root = tree.getroot() 
    for member in root.findall('object'):
        if len(str(member[0].text)) == 1:
            class_name.append(str(member[0].text))
        else:
            print(file)
        xmin.append(int(member[4][0].text))
        if len(xmin) > 1:
            if xmin[-1]< xmin[-2]:
                j = j+1
    if len(class_name) != 10:
        r = r+1
    
    if j == 0:
        y=y+1
    if j == 1:
        c = c+1           
    if  j > 1:
        print(file)   
           
print(y,c,r)