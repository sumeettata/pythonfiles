import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


# The input to the variables
xml_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Main//labels_vehicles'
# xml_path = 'D://6-03-2023//Daniel_Data_04_03_2023_old//Daniel_Data_04_03_2023'

# Reading the paths and creating dir
files = glob.glob(xml_path+'//*.xml')
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
        if class_name == 'np':
            df2 = pd.DataFrame([[os.path.basename(i),class_name,xmin,ymin,xmax,ymax,height,width]],columns=df.columns)
            df = pd.concat([df,df2],ignore_index=True)
            
#change the class_name to numeric
df = df.assign(x_centroid = lambda x : (x['x2'] + x['x1'])/(2*x['width']))
df = df.assign(y_centroid = lambda x : (x['y2'] + x['y1'])/(2*x['height']))
df = df.assign(Area = lambda x : ((x['x2'] - x['x1'])*(x['y2'] - x['y1']))/(x['height']*x['width'])*100)

# save the csv file
df.to_csv('D://6-03-2023//Daniel_Data_04_03_2023_old'+'//'+xml_path.split('//')[-1]+'.csv',index=False)

# Save the scatter plot
plt.scatter(x = df['x_centroid'], y = df['y_centroid'])
plt.title('Centroid Distribution')
plt.xlabel('x_Centroid')
plt.ylabel('y_Centroid')
plt.savefig('D://6-03-2023//Daniel_Data_04_03_2023_old'+'//'+xml_path.split('//')[-1]+'.png')