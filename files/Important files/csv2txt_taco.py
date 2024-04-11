import pandas as pd
import os
import cv2
import glob
from tqdm import tqdm

path='D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage/Taco Dataset'

file = path+'/'+'meta_df.csv'

df = pd.read_csv(file)


for group_name, df_group in tqdm(df.groupby('img_file')):
    txt_path = os.path.join(path+'/data/',group_name.split('.')[0]+'.txt')
    lst = []
    for index, row in df_group.iterrows():
        # changing the points of csv file into yolo format (xcen,ycen,w,h)
        x_cen = float(row['x']+(row['width']/2))/float(row['img_width'])
        y_cen = float(row['y']+(row['height']/2))/float(row['img_height'])
        w = float(row['width'])/float(row['img_width'])
        h = float(row['height'])/float(row['img_height'])
        lst.append(str(0)+' '+str(x_cen)+' '+str(y_cen)+' '+str(w)+' '+str(h))
        #  Saving the list formed into txt format
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lst))
            f.close()
