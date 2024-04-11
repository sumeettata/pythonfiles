import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm


# The input to the variables
xml_path = r'D:\OneDrive\OneDrive - Tata Insights and Quants\Work\TATA Communications\Data_1\Vehicle orentation\old data\front_images\Trucks_pre'


# Reading the paths and creating dir
files = glob.glob(xml_path+'\*.xml')
print(files)
df = pd.DataFrame(columns=['filename','class_name','x1','y1','x2','y2','height','width'])
for i in tqdm(files):
    tree = ET.parse(i)
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
        df2 = pd.DataFrame([[i,class_name,xmin,ymin,xmax,ymax,height,width]],columns=df.columns)
        df = pd.concat([df,df2],ignore_index=True)
            
#change the class_name to numeric
df['class_name'] = df['class_name'].replace(df['class_name'].unique(),range(len(df['class_name'].unique())))

# itterrating thought the csv file 
for group_name, df_group in tqdm(df.groupby('filename')):
    txt_path = group_name.split('.')[0]+'.txt'
    lst = []
    for index, row in df_group.iterrows():
        # changing the points of csv file into yolo format (xcen,ycen,w,h)
        class_lst = str(int(row['class_name']))
        x_cen = ((int(row['x1']) + int(row['x2']))/2)/int(row['width'])
        y_cen = ((int(row['y1']) + int(row['y2']))/2)/int(row['height'])
        w = abs(int(row['x1']) - int(row['x2']))/int(row['width'])
        h = abs(int(row['y1']) - int(row['y2']))/int(row['height'])
        lst.append(class_lst+' '+str(x_cen)+' '+str(y_cen)+' '+str(w)+' '+str(h))
        #  Saving the list formed into txt format
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lst))
            f.close()