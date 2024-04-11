import os
import glob
import copy

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/cctv_frontdesk'

files = glob.glob(path+'/*.png')


result = cv2.VideoWriter('demo_cctv2.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, (int(1280),int(960)))

for file in tqdm(files):
    img = cv2.imread(file)
    result.write(cv2.resize(img, (1280, 960)))
    
result.release() 
    
