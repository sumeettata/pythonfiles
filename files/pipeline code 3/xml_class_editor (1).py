import os
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm

# # The input to the variables
# xml_path = 'C://Users//SumeetMitra//Downloads//footwear'

# #replace string
# class_old1 = 'helmet'
# class_new1 = 'nohelmet'
# class_old2 = 'vest'
# class_new2 = 'novest'

# i=0
# j=0
# files = glob.glob(xml_path+'//*.xml')
# for file in tqdm(files):
#     tree = ET.parse(file)
#     root = tree.getroot()
#     for member in root.findall('object'):
#         class_name = str(member[0].text)
#         if class_name == class_old1:
#             i=i + 1
#             member[0].text = class_new1
#         if class_name == class_old2:
#             member[0].text = class_new2
#             j=j+1
            
#     tree.write(file)

# print(class_old1,i)
# print(class_old2,j)


# deleting particular class
# The input to the variables
xml_path = 'C:/Users/SumeetMitra/Downloads/200_images/New folder (2)'

#replace string
class_new1 = 'head'
class_old2 = 'helmet'
class_old3 = 'nohelmet'
class_old1 = 'human'
class_new2 = 'person'

i=0
j=0
files = glob.glob(xml_path+'//*.xml')
for file in tqdm(files):
    tree = ET.parse(file)
    root = tree.getroot()
    for member in root.findall('object'):
        class_name = str(member[0].text)
        ymax = int(member[4][3].text)
        ymax_new = ymax + int(0.2*ymax)
        if class_name != class_new1:
            if class_name == class_old2:
                i=i + 1
                member[0].text = class_new1
                member[4][3].text = str(ymax_new)
            elif class_name == class_old3:
                i=i + 1
                member[0].text = class_new1
                member[4][3].text = str(ymax_new)
            elif class_name == class_old1:
                i = i+1
                member[0].text = class_new2
            else:
                root.remove(member)
            
    tree.write(file)
