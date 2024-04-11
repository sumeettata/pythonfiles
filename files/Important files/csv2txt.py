import pandas as pd
import os
import cv2
import glob
import argparse
from tqdm import tqdm

# taking the input from command prompt
parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, required =True, help= 'The path containing csv file')
args = parser.parse_args()

file = args.path

# Read CSV file 
df = pd.read_csv(file)
folder_path = file.split('.')[0]
#  making the folder with same name as csv file name 
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

#change the class_name to numeric
df['Class_name'] = df['Class_name'].replace(df['Class_name'].unique(),range(len(df['Class_name'].unique())))

# itterrating thought the csv file 
for group_name, df_group in tqdm(df.groupby('Image_name')):
    txt_path = os.path.join(folder_path,group_name.split('.')[0]+'.txt')
    lst = []
    for index, row in df_group.iterrows():
        # changing the points of csv file into yolo format (xcen,ycen,w,h)
        class_lst = str(int(row['Class_name']))
        x_cen = ((int(row['x_min']) + int(row['x_max']))/2)/int(row['Width'])
        y_cen = ((int(row['y_min']) + int(row['y_max']))/2)/int(row['Height'])
        w = abs(int(row['x_min']) - int(row['x_max']))/int(row['Width'])
        h = abs(int(row['y_min']) - int(row['y_max']))/int(row['Height'])
        lst.append(class_lst+' '+str(x_cen)+' '+str(y_cen)+' '+str(w)+' '+str(h))
        #  Saving the list formed into txt format
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lst))
            f.close()