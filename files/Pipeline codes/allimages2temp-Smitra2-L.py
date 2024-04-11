import glob
import shutil
import os
from tqdm import tqdm

files = glob.glob('D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//16-03-2023_ANPR//juminsir_data//22April_mov//*//*.jpg')
save_p = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//16-03-2023_ANPR//juminsir_data//21-Apr-2023_run'

if not os.path.exists(save_p):
    os.mkdir(save_p)
    
    
print(len(files))
for file in tqdm(files):
    file_name = os.path.basename(file)
    save_path = os.path.join(save_p,file_name)
    shutil.copy(file,save_path)