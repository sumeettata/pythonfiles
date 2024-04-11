import pandas as pd
import os
import cv2
import glob
from tqdm import tqdm

# taking the input from command prompt
csv_path = "D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Csv Files//Images_1_03_202301032023_0116.csv"
image_folder = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Temp//Images_1_03_2023'

df = pd.read_csv(csv_path)

# visualizing the csv file
for group_name, df_group in df.groupby('filename'):
    img_path = os.path.join(image_folder,group_name)
    img = cv2.imread(img_path)
    for index, row in df_group.iterrows():
        start_point = (int(row['x1']),int(row['y1']))
        end_point = (int(row['x2']),int(row['y2']))               
        img = cv2.rectangle(img, start_point, end_point, [255,0,0], 5)
        img = cv2.putText(img, row['class_name'],  start_point,cv2.FONT_HERSHEY_SIMPLEX, 5,(255, 0, 0), 5, cv2.LINE_AA)
    cv2.namedWindow("window_name", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("window_name", 1300, 700)
    cv2.imshow('window_name', img)
    k = cv2.waitKey(0)
    if k == 27:
        break
cv2.destroyAllWindows()