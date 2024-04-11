import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def xml2txt(xml_path,class_label):
    
    xml_path = xml_path+'/images'
    
    #creating a folder for txt
    folder_path = os.path.dirname(xml_path)+'/labels'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # Reading the paths and creating dir
    files = glob.glob(xml_path+'/*.xml')
    df = pd.DataFrame(columns=['filename','class_name','x1','y1','x2','y2','height','width'])
    for i in tqdm(files):
        tree = ET.parse(i)
        root = tree.getroot()
        for mem in root.findall('size'):
            width = int(mem[0].text)
            height = int(mem[1].text)
        for member in root.findall('object'):
            class_name = str(member[0].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            df2 = pd.DataFrame([[os.path.basename(i).replace(".xml",".jpg"),class_name,xmin,ymin,xmax,ymax,height,width]],columns=df.columns)
            df = pd.concat([df,df2],ignore_index=True)

    #saving the csv file
    df.to_csv(xml_path+"/label.csv")       

    #change the class_name to numeric
    df = df[df['class_name'] != 'bicycle']
    df['class_name'] = df['class_name'].replace(class_label.values(),class_label.keys())

    # itterrating thought the csv file 
    for group_name, df_group in tqdm(df.groupby('filename')):
        txt_name = group_name.split('.')[0]+'.txt'
        lst = []
        for index, row in df_group.iterrows():
            # changing the points of csv file into yolo format (xcen,ycen,w,h)
            class_lst = str(int(row['class_name']))
            x_cen = ((int(row['x1']) + int(row['x2']))/2)/int(row['width'])
            y_cen = ((int(row['y1']) + int(row['y2']))/2)/int(row['height'])
            w = abs(int(row['x1']) - int(row['x2']))/int(row['width'])
            h = abs(int(row['y1']) - int(row['y2']))/int(row['height'])
            lst.append(class_lst+' '+str(x_cen)+' '+str(y_cen)+' '+str(w)+' '+str(h))
            #  Saving the list formed into txt format
        with open(folder_path+'/'+txt_name, 'a') as f:
            f.write('\n'+'\n'.join(lst))
            f.close()


    #getting the file names into the list
    files_names = df['filename'].unique().tolist()
    #files_names = natsort.natsorted(files_names,reverse=False)

    #split the files into test and train set
    files_names = [str(os.path.dirname(xml_path)+'/images/'+x) for x in files_names]
    lst_train, lst_val = train_test_split(files_names, test_size=0.1)

    #creating the txt files containing the file name 
    for lst_yolo in lst_train:
        with open(os.path.dirname(xml_path)+'/train.txt', 'a+') as f1:
            f1.write(lst_yolo+"\n")

    for lst_yolo in lst_val:
        with open(os.path.dirname(xml_path)+'/val.txt', 'a+') as f2:
            f2.write(lst_yolo+"\n")