import glob
import numpy as np
import os
import imagehash
from PIL import Image
from tqdm import tqdm
import shutil

files = glob.glob('D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//cropped//*.png')

hashes = []
location =  []
for i,file in enumerate(tqdm(files)):
    with Image.open(file) as img:
        temp_hash = str(imagehash.average_hash(img, hash_size = 8))
        location.append(file)
        if temp_hash in hashes:
            index = hashes.index(temp_hash)
            dublicate = location[index]
            shutil.copy(file,'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Trash//'+os.path.basename(file))
        hashes.append(temp_hash)
        
print(len(hashes))
print(len(list(set(hashes))))