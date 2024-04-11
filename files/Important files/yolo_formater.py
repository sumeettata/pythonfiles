import numpy as np
import glob
import os
import yaml
from tqdm import tqdm
import shutil

# input the variables
files = glob.glob('D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//Vehicle orentation//front//*')
files = [x for x in files if not x.endswith('.xml') ]
ratio_needed = [0.7,0.15]
length = len(files)
save_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//Vehicle orentation//front'


# The train test val ratios are used 
train_end = int(ratio_needed[0]*length)
val_end = int((ratio_needed[1]+ratio_needed[0])*length)

#the folder containts are splited by ratio got 
folder_split = np.split(files, [train_end,val_end])

# function to made path
def make_path(path_need):
    if not os.path.exists(path_need):
        os.mkdir(path_need)
    return path_need

# Make the folders inside the folders
save_path = make_path(save_path+'_Dataset')
images_path = make_path(save_path+'//images')
label_path = make_path(save_path+'//labels')
images_train_path = make_path(images_path+'//train')
images_val_path = make_path(images_path+'//val')
images_test_path = make_path(images_path+'//test')
label_train_path = make_path(label_path+'//train')
label_val_path = make_path(label_path+'//val')
label_test_path = make_path(label_path+'//test')

# make yaml file
dic_file = {'train' : str(images_train_path), 'val' : str(images_val_path), 'test' : str(images_test_path)}

with open(save_path+'//positions.yaml', 'w') as file:
    documents = yaml.dump(dic_file, file)
    
image_location = [images_train_path,images_val_path,images_test_path]
label_location = [label_train_path,label_val_path,label_test_path]

# copy data to required folder
for i,file in enumerate(folder_split):
    for f in tqdm(file):
        file_name = os.path.basename(f)
        txt_name = os.path.basename(f).split('.')[0]+'.txt'
        file_save = os.path.join(image_location[i],file_name)
        txt_save = os.path.join(label_location[i],txt_name)
        shutil.copy(f,file_save)
        shutil.copy(f.split('.')[0]+'.txt',txt_save)