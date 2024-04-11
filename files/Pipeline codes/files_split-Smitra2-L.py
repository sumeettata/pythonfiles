import glob
import numpy as np
import shutil
from tqdm import tqdm
import os

# Input to the variables
path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/jindal_data/images/cam3_images'
split_no = 3

# loading the files and creating folder
files = glob.glob(path+'//*')
files = [x for x in files if not x.endswith('.xml')]
save_path = path+'_splited'
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
# split the list
files_list = np.array_split(files, split_no)
# loop over the list
for i,file in enumerate(files_list):
    folder_save = os.path.join(save_path,"dataset_"+str(i))
    if not os.path.exists(folder_save):
        os.mkdir(folder_save)
    for f in tqdm(file):
        file_name = os.path.basename(f)
        xml_name = os.path.basename(f).split('.')[0]+'.xml'
        file_save = os.path.join(folder_save,file_name)
        xml_save = os.path.join(folder_save,xml_name)
        shutil.copy(f,file_save)
        if os.path.exists(f.split('.')[0]+'.xml'):
            shutil.copy(f.split('.')[0]+'.xml',xml_save)