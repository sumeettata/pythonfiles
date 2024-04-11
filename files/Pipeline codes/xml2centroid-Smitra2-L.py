import os
import xml.etree.ElementTree as ET
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# get the time and date of file run
file_time = '_'+datetime.now().strftime("%d%m%Y_%H%M")

# The input to the variables
xml_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data//Main//labels_vehicles'
save_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Centoid_data'
# xml_path = 'D://6-03-2023//Daniel_Data_04_03_2023_old//Daniel_Data_04_03_2023'

# making the directory to save file 
save_path = save_path+'//'+xml_path.split('//')[-1]+file_time
if not os.path.exists(save_path ):
    os.mkdir(save_path)
    
# Reading the paths and creating dir
files = glob.glob(xml_path+'//*.xml')
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
        if class_name == 'np':
            df2 = pd.DataFrame([[os.path.basename(i),class_name,xmin,ymin,xmax,ymax,height,width]],columns=df.columns)
            df = pd.concat([df,df2],ignore_index=True)
            
#change the class_name to numeric
df = df.assign(x_centroid = lambda x : (x['x2'] + x['x1'])/2)
df = df.assign(y_centroid = lambda x : (x['y2'] + x['y1'])/2)
df = df.assign(Area = lambda x : ((x['x2'] - x['x1'])*(x['y2'] - x['y1']))/(x['height']*x['width'])*100)

# save the csv file
df.to_csv(save_path+'//'+xml_path.split('//')[-1]+'.csv',index=False)

# # Save the scatter plot
# y = len(df['height'].unique())
# fig, axes = plt.subplots(y,1, figsize=(50,50),dpi=1)

# # Plotting the scatter plot
# i=0
# for grp_name,df_grp in df.groupby('height'):
#     height = int(grp_name)
#     width = int(df_grp['width'].unique()[0])
#     plt.rcParams.update({'font.size': 50})
#     axes[i].minorticks_on()
#     axes[i].set_xlim(0,width)
#     axes[i].set_ylim(0,height)
#     axes[i].tick_params(axis='both', which='major', labelsize=50,length=10, width=5)
#     axes[i].tick_params(axis='both', which='minor', labelsize=50,length=10, width=5)
#     axes[i].scatter(x = df_grp['x_centroid'], y = df_grp['y_centroid'], s=100,linewidths = 10)
#     # axes[i].xlabel('x_Centroid')
#     # axes[i].ylabel('y_Centroid')
#     axes[i].set_title(str(grp_name)+' height')
#     i = i+1   
# fig.savefig('D://OneDrive//OneDrive - Tata Insights and Quants//Work'+'//'+xml_path.split('//')[-1]+'.png')

y = len(df['height'].unique())
for grp_name,df_grp in df.groupby('height'):
    print('Running' + str(grp_name))
    height = int(grp_name)
    width = int(df_grp['width'].unique()[0])
    df_grp['y_centroid'] = df_grp['y_centroid'].apply(lambda x : height-x)
    plt.rcParams.update({'font.size': int((50/3024)*height)})
    plt.figure(figsize=(width/100,height/100))
    plt.xlim(0,width)
    plt.ylim(height,0)
    plt.minorticks_on()
    plt.scatter(x = df_grp['x_centroid'], y = df_grp['y_centroid'].apply(lambda x : height-x), s=int((100/3024)*height),linewidths = int((10/3024)*height) )
    plt.grid(which='both',color='r', linestyle='-', linewidth=1)
    plt.title(str(height)+' X '+str(width))
    plt.xlabel('x_centroid')
    plt.ylabel('y_centroid')
    plt.savefig(save_path+'//'+xml_path.split('//')[-1]+'_inverted_'+str(grp_name)+'.png')
    plt.close()