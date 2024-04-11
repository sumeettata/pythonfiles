import os
import glob
import shutil

image_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Python files//anpr_yolov8//test_dataset'
xml_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Temp//labels_vehicles'

files = glob.glob(image_path+'//*')
files = [x for x in files if not x.endswith('.xml')]
print(len(files))
for file in files:
    xml_name = os.path.basename(file).split('.')[0]+'.xml'
    xml_name_path = xml_path+'//'+xml_name
    if os.path.exists(xml_name_path):
        shutil.copy(xml_name_path,image_path+'//'+xml_name)