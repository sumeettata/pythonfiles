# import pandas as pd
# import numpy as np
# from lxml import etree
# import xmlAnnotation.etree.cElementTree as ET
#
# fields = ['filename', 'height', 'width', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
# df = pd.read_csv('merged_data.csv', usecols=fields)
#
#
# # Change the name of the file.
# # This will replace the / with -
# def nameChange(x):
#     x = x.replace("/", "-")
#     return x
#
#
# df['filename'] = df['filename'].apply(nameChange)
# print(df['filename'])
# for i in range(0, 2):
#     height = df['height'].iloc[i]
#     width = df['width'].iloc[i]
#     depth = 3
#
#     annotation = ET.Element('annotation')
#     ET.SubElement(annotation, 'folder').text = 'images'
#     ET.SubElement(annotation, 'filename').text = str(df['filename'].iloc[i])
#     ET.SubElement(annotation, 'segmented').text = '0'
#     size = ET.SubElement(annotation, 'size')
#     ET.SubElement(size, 'width').text = str(width)
#     ET.SubElement(size, 'height').text = str(height)
#     ET.SubElement(size, 'depth').text = str(depth)
#     ob = ET.SubElement(annotation, 'object')
#     ET.SubElement(ob, 'name').text = str(df['class'].iloc[i])
#     ET.SubElement(ob, 'pose').text = 'Unspecified'
#     ET.SubElement(ob, 'truncated').text = '0'
#     ET.SubElement(ob, 'difficult').text = '0'
#     bbox = ET.SubElement(ob, 'bndbox')
#     ET.SubElement(bbox, 'xmin').text = str(df['xmin'].iloc[i])
#     ET.SubElement(bbox, 'ymin').text = str(df['ymin'].iloc[i])
#     ET.SubElement(bbox, 'xmax').text = str(df['xmax'].iloc[i])
#     ET.SubElement(bbox, 'ymax').text = str(df['ymax'].iloc[i])
#
#     fileName = str(df['filename'].iloc[i])
#     tree = ET.ElementTree(annotation)
#     tree.write(fileName + ".xml", encoding='utf8')


from collections import defaultdict
import os
import csv

from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET

save_root2 = "D:/vehicle_data/front_images/merged_output_212"
class_need = ["traffic_sign", "car", "bike"]
csv_name = 'D:/vehicle_data/front_images/merged_output_2.csv'

if not os.path.exists(save_root2):
    os.mkdir(save_root2)


def write_xml(folder, filename, bbox_list):
    root = Element('annotation')
    SubElement(root, 'folder').text = folder
    SubElement(root, 'filename').text = filename
    SubElement(root, 'path').text = './images' + filename
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'

    # Details from first entry
    e_filename, e_width, e_height, e_class_name, e_xmin, e_ymin, e_xmax, e_ymax = bbox_list[0]

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = e_width
    SubElement(size, 'height').text = e_height
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'

    for entry in bbox_list:
        e_filename, e_width, e_height, e_class_name, e_xmin, e_ymin, e_xmax, e_ymax = entry

        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = e_class_name
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = e_xmin
        SubElement(bbox, 'ymin').text = e_ymin
        SubElement(bbox, 'xmax').text = e_xmax
        SubElement(bbox, 'ymax').text = e_ymax

    # indent(root)
    tree = ElementTree(root)

    xml_filename = os.path.join('.', folder, os.path.splitext(filename)[0] + '.xml')
    tree.write(xml_filename)


entries_by_filename = defaultdict(list)

with open(csv_name , 'r', encoding='utf-8') as f_input_csv:
    csv_input = csv.reader(f_input_csv)
    header = next(csv_input)
    print(header)
    for row in csv_input:
        filename, width, height, class_name, xmin, ymin, xmax, ymax = row
        if class_name in class_need:
            entries_by_filename[filename].append(row)
        #entries_by_filename[filename].append(row)     

for filename, entries in entries_by_filename.items():
    print(filename, len(entries))
    print(save_root2)
    write_xml(save_root2, filename, entries)