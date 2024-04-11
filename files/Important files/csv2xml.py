import os
import pandas as pd
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Put the required the values on the variables
csv_name = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Csv Files//Images_1_03_202301032023_0116.csv'
save_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Temp'
class_needed = ['bicycle', 'car', 'motorcycle', 'bus', 'truck','np']

# Creating the pandas dataframe and making paths
df = pd.read_csv(csv_name)
folder_name  = os.path.dirname(csv_name)
save_path = save_path +"//"+ csv_name.split('//')[-1].split('.')[0]
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
#converting all values to str and getting the required class dataframe
df = df.applymap(str)
df = df[df['class_name'].isin(class_needed)]

# Function to convert the dataframe grouped by file_name to xml
def df2xml(group_name,df_group,save_path):
    root = Element('annotation')
    SubElement(root, 'folder').text = save_path
    SubElement(root, 'filename').text = group_name
    SubElement(root, 'path').text = './images' + group_name
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'
    
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = df_group['width'].unique()[0]
    SubElement(size, 'height').text = df_group['height'].unique()[0]
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'
    
    for index, row in df_group.iterrows():
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = row['class_name']
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = row['x1']
        SubElement(bbox, 'ymin').text = row['y1']
        SubElement(bbox, 'xmax').text = row['x2']
        SubElement(bbox, 'ymax').text = row['y2']
    tree = ElementTree(root)    
    xml_filename = os.path.join(save_path, os.path.splitext(group_name)[0] + '.xml')
    tree.write(xml_filename)

# Looping the dataframe grouped by file_name
for file_name, df_grp in tqdm(df.groupby('filename')):
    df2xml(file_name,df_grp,save_path)