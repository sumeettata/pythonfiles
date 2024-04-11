import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# The input to the variables
image_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Temp//Images'
xml_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Temp//labels_vehicles'

#image_path = xml_path = "D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//13-03-2023_ANPR//Jamshedpur 12_March_2023_splited//dataset_1_jam_DIV02042023"
csv_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Csv Files'

# Input the required values to variable
Human_class = ['person']
Vehicle_class = ['vehicle_back','vehicle_side','vehicle_front','bicycle', 
                 'car', 'motorcycle', 'bus', 'truck','vehicle','np']

# get the time and date of file run
file_time = '-'+datetime.now().strftime("%d%m%Y_%H%M")

# Reading the paths and creating dir
dir_list = os.listdir(image_path)
files_jpg = [x for x in dir_list if not x.endswith(".xml")]

#read previous data from file
df = pd.read_csv(csv_path+'//main.csv')
df = df.drop(['label_created','label_modified'],axis=1)

#df = pd.DataFrame(columns=['filename','class_name','x1','y1','x2','y2','height','width'])
for i in tqdm(files_jpg):
    label_path = os.path.join(xml_path,i.split('.')[0]+'.xml')
    file_path = os.path.join(image_path,i)
    img = cv2.imread(file_path)
    img_name = os.path.basename(file_path)
    r,c,ch = img.shape
    #read images
    if os.path.exists(label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        for member in root.findall('object'):
            class_name = str(member[0].text)
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            if class_name == 'motarcycle':
                print(img_name)
            
            df2 = pd.DataFrame([[img_name,class_name,xmin,ymin,xmax,ymax,r,c]],columns=df.columns)
            df = pd.concat([df,df2],ignore_index=True)

#save csv file
df.to_csv(csv_path+'//'+xml_path.split('//')[-1]+file_time+'.csv',index=False)

# adding the extra class_type column
df['Class_type'] = df['class_name'].apply(lambda x: 'Human_Class' if x in Human_class else ('Vehicle_class' if x in Vehicle_class else 'New_class'))
df['Aspect_ratio'] = (df['y2']-df['y1'])/(df['x2'] - df['x1'])


# Setting the figure title and font
y = len(df['Class_type'].unique())
fig, axes = plt.subplots(2,y, figsize=(75,50))
plt.rcParams.update({'font.size': 50})

# Plotting the Pie chart
i=0
for grp_name,df_grp in df.groupby('Class_type'):
    df2 = df_grp['class_name'].value_counts()
    explode = [0.05]*len(df2.index)
    def fmt(x):
        return '{:.0f}'.format(df2.values.sum()*x/100)
    axes[0,i].pie(df2,labels=df2.index,explode = explode,pctdistance=0.9, labeldistance=1.1,autopct=fmt)
    axes[0,i].legend(bbox_to_anchor=(1.0, 1.0),loc=0)
    axes[0,i].set_title(grp_name+' value counts')
    i = i+1

# plotting the box plot
i=0   
for grp_name,df_grp in df.groupby('Class_type'):
    plt.rcParams.update({'font.size': 50})
    df_grp.boxplot(column=['Aspect_ratio'],by='class_name',fontsize = 50,
                   flierprops=dict(markerfacecolor='b',linewidth=5, markersize=15),
                   boxprops = dict(linestyle='-', linewidth=5),
                   whiskerprops = dict(linestyle='--' , linewidth=5),
                   medianprops = dict(linestyle='-', linewidth=6, color = 'r'),
                   capprops=dict(linestyle='-', linewidth=5),
                   ax =axes[1,i], rot=45)
    axes[1,i].set_title(grp_name+' Aspect ratio')
    axes[1,i].set_xlabel("")
    i= i+1

# saving the plot
fig.suptitle("Ths Analysis of labelled images present based on class name ")
fig.savefig('D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//data_analysis//'+xml_path.split('//')[-1]+file_time+'.png')
print(df['class_name'].unique())