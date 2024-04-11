import numpy as np
import cv2
import pi_heif
import glob
import os
from tqdm import tqdm

save_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//ANPR//Images_24022023_jpg'

for file in tqdm(glob.glob("D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//ANPR//Images_24022023//*.HEIC")):
    file_name = os.path.basename(file).split('.')[0]
    heif_file = pi_heif.open_heif(file, convert_hdr_to_8bit=False, bgr_mode=True)
    np_array = np.asarray(heif_file)
    cv2.imwrite(save_path+'//'+file_name+".png", np_array)