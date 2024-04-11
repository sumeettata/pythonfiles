import os
import glob
from tqdm import tqdm
import shutil


# The input to the variables
image_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Desktop//documents//Divyanshi_videos_1_26-04-2023_DIV26042023 (2)//Divyanshi_videos_1_26-04-2023_DIV26042023//dataset_1'
xml_path = image_path
image_save = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Temp2//Images'
xml_save = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Temp2//labels_vehicles'

#image_save = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Main//Images'
#xml_save = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Main//labels_vehicles'

# Reading the paths and creating dir
files = glob.glob(image_path+'//*')
files = [x for x in files if not x.endswith('.xml')]
# Function to check for dublicates exists and to rename the file
def check_file(file_check,label_check):
    if not os.path.exists(file_check):
        fin_file = file_check
        fin_label = label_check 
    else: 
        file_rename = file_check.split('.')[0]+'0.'+file_check.split('.')[1]
        label_rename = label_check.split('.')[0]+'0.'+label_check.split('.')[1]
        #fin_file, fin_label = check_file(file_rename,label_rename)
        fin_file = fin_label = 404
       
    return fin_file,fin_label 
    
i = j = 0
for file in tqdm(files):
    file_basename = os.path.basename(file)
    file_name = os.path.splitext(file_basename)[0]
    label_path = os.path.join(xml_path,file_name+'.xml')
    if os.path.exists(label_path):
        file_save = os.path.join(image_save,file_basename)
        label_save = os.path.join(xml_save,file_name+'.xml')
        checked_file, checked_label = check_file(file_save,label_save)
        if checked_file != 404:
            i = i+1
            shutil.move(file,checked_file)
            shutil.move(label_path,checked_label)
        else:
            print(file_save)
            print(file)
            j = j+1
    
print('copied : {}, non-copied : {}'.format(i,j))