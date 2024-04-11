import xml.etree.ElementTree as ET
import pandas as pd
import glob2 as glob
import os
import cv2
from tqdm import tqdm 


folder_path = 'C:/Users/SumeetMitra/Downloads/cam3_sumeet/cam3_sumeet'


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()
    df = pd.DataFrame(columns=["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        #classes = root.find('class').text

        ymin, xmin, ymax, xmax, classes = None, None, None, None, None
        classes = str(boxes.find("name").text)
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        df=df.append([{"filename":filename, "width":width, "height":height, "class":classes, "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax}])
        #list_with_single_boxes = [filename, width, height,classes, xmin, ymin, xmax, ymax]
        #list_with_all_boxes.append(list_with_single_boxes)

    return df

dfall = pd.DataFrame()
for filename in glob.glob(folder_path+'/*.xml'): 
    #D:/PData/TataSteel/PPE/OtherClass12Jul/Working/Vikash/
    dft = read_content(filename)
    dfall=dfall.append(dft)
    
print(dfall.groupby('class').count())

# if not os.path.exists(os.path.dirname(folder_path)+'/glove'):
#     os.mkdir(os.path.dirname(folder_path)+'/glove')
# #For Human, Helmet and vest
# #df = pd.DataFrame(columns=["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
# dfh = dfall[dfall['class'].isin(['footwear', 'nofootwear', 'glove', 'noglove'])]
# class_names = {"footwear":(0, 255, 0), "nofootwear":(0, 255, 255), "glove":(255, 0, 0), "noglove":(0, 0, 255)}
# for filename in tqdm(glob.glob(folder_path+'/*.jpg')):
#     fname = os.path.basename(filename).replace('.JPG','.jpg')
#     fori = os.path.basename(filename)
#     image = cv2.imread(filename)
#     df = dfh[dfh['filename'].isin([fname,fori])]
#     for i in range(0, len(df)):
#         image = cv2.rectangle(image, (df.iloc[i]['xmin'], df.iloc[i]['ymin']), (df.iloc[i]['xmax'],df.iloc[i]['ymax']), class_names[df.iloc[i]['class']], 3)
#     if(len(df)>0):
#         cv2.imwrite(os.path.dirname(folder_path)+'/glove/'+fname, image)
        
        
if not os.path.exists(os.path.dirname(folder_path)+'/human'):
    os.mkdir(os.path.dirname(folder_path)+'/human')
#For Human, Helmet and vest
#df = pd.DataFrame(columns=["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
dfh = dfall[dfall['class'].isin(['human', 'helmet', 'nohelmet', 'vest', 'novest'])]
class_names = {"human":(0, 255, 255), "vest":(255, 0, 0), "novest":(0, 0, 255), "helmet":(255, 0, 0), "nohelmet":(0, 0, 255)}
for filename in tqdm(glob.glob(folder_path+'/*.jpg')):
    fname = os.path.basename(filename).replace('.JPG','.jpg')
    fori = os.path.basename(filename)
    image = cv2.imread(filename)
    df = dfh[dfh['filename'].isin([fname,fori])]
    for i in range(0, len(df)):
        image = cv2.rectangle(image, (df.iloc[i]['xmin'], df.iloc[i]['ymin']), (df.iloc[i]['xmax'],df.iloc[i]['ymax']), class_names[df.iloc[i]['class']], 1)
    cv2.imwrite(os.path.dirname(folder_path)+'/human/'+fname, image)