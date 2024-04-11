import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import natsort

# The input to the variables
image_path = 'D:/Work/Tata Communications/python files/testing/ppe_aditya/ppe_aditya'
xml_path = image_path
save_path = image_path+'_predict'
logo_path = "C:/Users/SumeetMitra/Downloads/PNG Logos/PNG Logos/TATA-Group-and-TATA-Communications-Logo-Lockup-White.png"



# Reading the paths and creating dir
dir_list = os.listdir(image_path)

files_jpg = [x for x in dir_list if not x.endswith('.xml')]
files_jpg = natsort.natsorted(files_jpg,reverse=False)

files_jpg = ['00000'+str(x)+'.jpg' for x in range(1494)]
if not os.path.exists(save_path):
    os.mkdir(save_path)


dic = {
       "helmet":(0, 90, 132),
       "nohelmet":(140, 80, 140),
         "vest":(1, 90, 132),
         "novest":(149, 80, 140), 
         "glove":(0, 20, 132),
         "noglove":(0, 20, 132), 
         "footwear":(169, 80, 140), 
         "nofootwear":(169, 80, 140),
         }

output_fps = 30

result = cv2.VideoWriter(save_path+'.avi', cv2.VideoWriter_fourcc(*'MJPG'),output_fps, (int(2560),int(1920)))
result2 = cv2.VideoWriter(save_path+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'),output_fps, (int(2560),int(1920)))

def draw_rectangle(image, class_name, x1, y1, x2, y2, color):
    color = (int(color[0]), int(color[1]), int(color[2]))
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=tuple(color), thickness=2)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x1)+(int(x2) - int(x1)), int(y1) - 20), color=tuple(color), thickness=-1)
    cv2.putText(image, str(class_name), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image



for i in tqdm(files_jpg):
    label_path = os.path.join(xml_path,i.split('.')[0]+'.xml')
    file_path = os.path.join(image_path,i)
    image = cv2.imread(file_path)
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        for k,member in enumerate(root.findall('object')):
            class_name = str(member[0].text)
            if class_name in dic.keys():
                if class_name=="nofootwear":
                    class_name="footwear"
                elif class_name=="glove":
                    class_name="noglove"
                image = draw_rectangle(image, class_name, int(member[4][0].text), int(member[4][1].text), int(member[4][2].text), int(member[4][3].text), dic[class_name])
    

    result.write(image)
    result2.write(image)
    cv2.imwrite(save_path+'/'+os.path.basename(i),image)

result.release()
result2.release()

import moviepy.editor as mp
video2 = mp.VideoFileClip(save_path+'.mp4')

logo = (mp.ImageClip(logo_path)
          .set_duration(video2.duration)
          .resize(height=200) # if you need to resize...
          .margin(left=10, bottom=15, opacity=0) # (optional) logo-border padding
          .set_pos(("right","bottom")))

final = mp.CompositeVideoClip([video2, logo])
final.write_videofile(save_path+'_2'+'.mp4')

