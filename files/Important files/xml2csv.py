import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
import argparse
from tqdm import tqdm

# taking the input from command prompt
parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, required =True, help= 'The Folder path containing the images and xml files')
args = parser.parse_args()

# getting the files from folder
files_images = glob.glob(args.path+'//*')
files_images = [x for x in files_images if not x.endswith('.xml')] 

# Making the empty dataframe and itterating on images
df = pd.DataFrame(columns=['Image_name','Class_name','x_min','y_min','x_max','y_max','Height','Width'])
for file in tqdm(files_images):
    #read images
    img = cv2.imread(file)
    img_name = os.path.basename(file)
    r,c,ch = img.shape
    xml_path = str(file.split('.')[0]+'.xml')
    #read xml file 
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for member in root.findall('object'):
            class_name = str(member[0].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            df2 = pd.DataFrame([[img_name,class_name,xmin,ymin,xmax,ymax,r,c]],columns=df.columns)
            df = pd.concat([df,df2],ignore_index=True)

#save csv file
df.to_csv(args.path+'.csv',index=False)