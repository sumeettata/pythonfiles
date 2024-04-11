import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import glob

file = "D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage/Synthetic_garbage_data/images"
gpu_path = '/home/danielv10/work/sumeet/data/Garbage/Synthetic_garbage_data/images'

files = glob.glob(file+'/*')
files = [gpu_path+'/'+os.path.basename(x) for x in files]
lst_train, lst_val = train_test_split(files, test_size=0.2)

for lst_yolo in lst_train:
    with open(os.path.dirname(file)+'/train.txt', 'w') as f:
        f.write('\n'.join(lst_train))
        f.close()

for lst_yolo in lst_val:
    with open(os.path.dirname(file)+'/val.txt', 'w') as f:
        f.write('\n'.join(lst_val))
        f.close()