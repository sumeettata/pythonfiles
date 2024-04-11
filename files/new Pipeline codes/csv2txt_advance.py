import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import natsort

file = "C:/Users/SumeetMitra/garbage_fall2.csv"
gpu_path = '/CV/data/garbage/garbage_synthetic_fall'

df = pd.read_csv(file)
folder_path = file.split('.')[0]
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

#change the class_name to numeric
df.loc[df['category'] != 'garbage', 'category'] = 1
df.loc[df['category'] == 'garbage', 'category'] = 0
df[['x1','x2','y1','y2']] = df[['x1','x2','y1','y2']].apply(pd.to_numeric)
num = df._get_numeric_data()
num[num < 0] = 0
num['x1'][num['x1']>1280] =1280
num['x2'][num['x2']>1280] =1280
num['y1'][num['y1']>720] =720
num['y2'][num['y2']>720] =720
#df['category'] = df['category'].replace(df['category'].unique(),range(len(df['category'].unique())))


for group_name, df_group in tqdm(df.groupby('filename')):
    txt_path = os.path.join(folder_path,group_name.split('.')[0]+'.txt')
    lst = []
    df_group = df_group.drop_duplicates(['x1','y1','x2','y2'],keep='first')
    for index, row in df_group.iterrows():
        # changing the points of csv file into yolo format (xcen,ycen,w,h)
        class_lst = str(int(row['category']))
        x_cen = ((int(row['x1']) + int(row['x2']))/2)/int(row['image_width'])
        y_cen = ((int(row['y1']) + int(row['y2']))/2)/int(row['image_height'])
        w = abs(int(row['x1']) - int(row['x2']))/int(row['image_width'])
        h = abs(int(row['y1']) - int(row['y2']))/int(row['image_height'])

        lst.append(class_lst+' '+str(format(x_cen, '.6f'))+' '+str(format(y_cen, '.6f'))+' '+str(format(w, '.6f'))+' '+str(format(h, '.6f')))
    #  Saving the list formed into txt format
        
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lst))
        f.close()

files_names = df['filename'].unique().tolist()
files_names = natsort.natsorted(files_names,reverse=False)
files_names = [str(gpu_path+'/images/'+x) for x in files_names]
#lst_train, lst_val = train_test_split(files_names, test_size=0.2)

for lst_yolo in files_names[0:int(len(files_names)*0.9)]:
    with open(os.path.dirname(file)+'/train.txt', 'a+') as f:
        f.write(lst_yolo+"\n")
f.close()

for lst_yolo in files_names[int(len(files_names)*0.9):len(files_names)]:
    with open(os.path.dirname(file)+'/val.txt', 'a+') as f:
        f.write(lst_yolo+"\n")
f.close()


# import os
# import glob

# images_path = glob.glob("/CV/data/Generalised_PPE/images/val/*")

# for image_path in images_path:
#     with open("val.txt", "a+") as file:
#         file.write(image_path+"\n")
# file.close()


# images_path = glob.glob("/CV/data/Generalised_PPE/images/train/*")

# for image_path in images_path:
#     with open("train.txt", "a+") as file:
#         file.write(image_path+"\n")
# file.close()