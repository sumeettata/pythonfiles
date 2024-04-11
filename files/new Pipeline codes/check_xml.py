import glob
import os

files = glob.glob('D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Python files//anpr_yolov8//test_dataset//*')

files = [x for x in files if not x.endswith('.xml')]
for file in files:
    xml_name = file.split('.')[0]+'.xml'
    if not os.path.exists(xml_name):
        os.remove(file)