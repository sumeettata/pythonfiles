import glob
import os
import json
import pandas as pd


path = r'D:\OneDrive\OneDrive - Tata Insights and Quants\Work\TATA Communications\Data_1\Garbage\bdd100k\bdd100k_labels_images_val\bdd100k_labels_images_val.json'

f = open(path)
content = json.load(f)

lst = []
for i in content:
    file = i['name']
    for j in i['labels']:
        if j['category'] == 'drivable area':
            class_name = j['category']
            for h in j['poly2d']:
                vertices = h['vertices']
                type_c = h['types']
                closed = h['closed']
                lst.append([file,class_name,vertices,type_c,closed])
                
                
df = pd.DataFrame(lst,columns=['File_name','Class_name','vertices','type','closed'])
df.to_csv('drivable_area.csv')