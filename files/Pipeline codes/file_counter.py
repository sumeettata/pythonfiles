import glob
import os
import matplotlib.pyplot as plt
import shutil

files= glob.glob(r"D:\OneDrive\OneDrive - Tata Insights and Quants\Work\TATA Communications\Data\Main\*\*")
non_tagged_path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Trash'

file = [os.path.basename(x).split('.')[0] for x in files]
file_1 = list(set(file))

j = 0
for x in files:
    base_name = os.path.basename(x)
    file_name = base_name.split('.')[0] 
    if file.count(file_name) == 1:
        print(x)
        # shutil.move(x,non_tagged_path+'/'+base_name)
        j = j+1
        
file_counts = [file.count(x) for x in file_1]

total_count = [('tagged '+str(x-1),file_counts.count(x)) for x in list(set(file_counts))]
print(total_count)
print('total file moved : ' + str(j))