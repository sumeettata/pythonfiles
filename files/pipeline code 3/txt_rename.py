import os
import glob
from tqdm import tqdm

files = glob.glob('C:/Users/SumeetMitra/Downloads/TACO dataset.v15i.yolov8/*/images/*.jpg')
j=0
i=0
for file in tqdm(files):
    j=j+1
    file_name = os.path.basename(file)
    image_path = os.path.dirname(file)
    txt_file = os.path.basename(file)[:-4]+'.txt'
    txt_path = os.path.dirname(file).replace('images','labels')
    if os.path.exists(txt_path+'/'+txt_file):
        os.rename(file,image_path+'/modified_'+str(j)+'.jpg')
        os.rename(txt_path+'/'+txt_file,txt_path+'/modified_'+str(j)+'.txt')
        i=i+1
        
print(i)
print(j)