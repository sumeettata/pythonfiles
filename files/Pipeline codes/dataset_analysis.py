import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Input the required values to variable
csv_name = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Csv Files//labels_vehicles.csv'
Human_class = ['person']
Vehicle_class = ['vehicle_back','vehicle_side','vehicle_front','bicycle', 
                 'car', 'motorcycle', 'bus', 'truck','vehicle','np']

# get the time and date of file run
file_time = '-'+datetime.now().strftime("%d%m%Y_%H%M")


# Reading the csv file
df = pd.read_csv(csv_name)
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
fig.savefig('D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//data_analysis//'+csv_name.split('//')[-1].split('.')[0]+file_time+'.png')

