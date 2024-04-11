import glob
import numpy as np
import os
import imagehash
from PIL import Image
from tqdm import tqdm
import shutil

files = glob.glob('D://6-03-2023//Front//*.jpg')

hashes = []
location =  []
for i,file in enumerate(tqdm(files)):
    with Image.open(file) as img:
        temp_hash = str(imagehash.average_hash(img, hash_size = 8))
        location.append(file)
        if temp_hash in hashes:
            index = hashes.index(temp_hash)
            dublicate = location[index]
            shutil.move(file,'D://6-03-2023//Trash//'+os.path.basename(file))
        hashes.append(temp_hash)
        
print(len(hashes))
print(len(list(set(hashes))))