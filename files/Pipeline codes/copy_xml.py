import os
import glob
import shutil

files = glob.glob('Work//TATA Communications//Data_1//16-03-2023_ANPR//OneDrive_2023-04-05 (1)//Vehicles + Number Plates//*')
files = [x for x in files if not x.endswith('.xml')]
path = 'Work//TATA Communications//Data_1//16-03-2023_ANPR//test_data'

print(len(files))

for file in files:
    file_name = os.path.basename(file)
    xml_name = file_name.split('.')[0]+'.xml'
    if os.path.exists(file.split('.')[0]+'.xml'):
        shutil.move(file,path+'//'+file_name)
        shutil.move(file.split('.')[0]+'.xml',path+'//'+xml_name)