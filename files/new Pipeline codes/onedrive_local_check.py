import glob
import shutil
import os
from tqdm import tqdm 

# input the root path
path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//'


#input the required paths 
image_path_main = path+'//Data//Main//Images//'
xml_path_main = path+'Data//Main//labels_vehicles//'
image_path_temp = path+'//Data//Temp//Images//'
xml_path_temp = path+'Data//Temp//labels_vehicles//'

one_drive_xml_path = 'C://Users//SumeetMitra//Downloads//OneDrive_1_27-3-2023'

#check the file 
onedrive_xml = [os.path.basename(x).split('.')[0] for x in glob.glob(one_drive_xml_path+'//*')]
main_images = glob.glob(image_path_temp+'*')



i=0
for file in tqdm(main_images):
    file_name = os.path.basename(file).split('.')[0]
    if file_name not in onedrive_xml:
        i = i+1
        shutil.copy(file,xml_path_temp+os.path.basename(file))
        #shutil.copy(xml_path_main+file_name+'.xml',xml_path_temp+file_name+'.xml')

print(len(onedrive_xml))
print(len(main_images)-i)
