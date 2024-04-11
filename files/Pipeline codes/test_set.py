import os
import glob
from tqdm import tqdm
import random
import shutil

# The input to the variables
root_path = 'D://OneDrive/OneDrive - Tata Insights and Quants//Work//TATA Communications//'
image_path = root_path+'Data//Temp//Images//'
xml_path = root_path+'Data//Temp//labels_vehicles//'
save_image = root_path+'Data//Test//Images//'
save_xml = root_path+'Data//Test//labels_vehicles//'
percent_test = 0.05

#read the folder contents
dir_list = glob.glob(image_path+'*')
files_image = [x for x in dir_list if not x.endswith('.xml')]
files_random = random.sample(files_image, int(percent_test*len(files_image)))

for file in tqdm(files_random):
    file_name = os.path.basename(file)
    shutil.copy(file,save_image+file_name)
    shutil.copy(xml_path+file_name.split('.')[0]+'.xml',save_xml+file_name.split('.')[0]+'.xml')