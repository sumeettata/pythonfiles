import glob
import shutil
import os
from tqdm import tqdm

files = glob.glob('D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//ANPR//27-02_mov//*//*.jpg')
save_p = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Temp//Images_1_03_2023//'
for file in tqdm(files):
    file_name = os.path.basename(file)
    save_path = os.path.join(save_p,file_name)
    shutil.copy(file,save_path)