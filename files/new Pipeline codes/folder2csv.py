import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import time

# The input to the variables
image_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Main//Images'
xml_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Main//labels_vehicles'
csv_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Csv Files'

# Reading the paths and creating dir
files = glob.glob(image_path+'//*')
files = [x for x in files if not x.endswith('.xml')]

df = pd.DataFrame(columns=['filename','label_created','label_modified','class_name','x1','y1','x2','y2','height','width']) 
for i in tqdm(files):
    file_name = os.path.basename(i).split('.')[0]
    label_path = xml_path+'//'+file_name+'.xml'
    ti_create = os.path.getctime(label_path)
    ti_modify = os.path.getmtime(label_path)
    c_ti = time.ctime(ti_create)
    m_ti = time.ctime(ti_modify)
    tree = ET.parse(label_path)
    root = tree.getroot()
    for mem in root.findall('size'):
        width = int(mem[0].text)
        height = int(mem[1].text)
    for member in root.findall('object'):
        class_name = str(member[0].text)
        xmin = int(member[4][0].text)
        ymin = int(member[4][1].text)
        xmax = int(member[4][2].text)
        ymax = int(member[4][3].text)
        df2 = pd.DataFrame([[os.path.basename(i),c_ti,m_ti,class_name,xmin,ymin,xmax,ymax,height,width]],columns=df.columns)
        df = pd.concat([df,df2],ignore_index=True)
        
#save csv file
df.to_csv(csv_path+'//main.csv',index=False)
