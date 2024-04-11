import pandas as pd
import glob
import cv2
from tqdm import tqdm
import numpy as np
import os
import random
import json
import copy

path_image = 'C://Users//SumeetMitra//Downloads//bdd100k_images_100k//bdd100k//images//100k//val//'
path_garbage = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//Garbage//'
path_save = path_garbage+'Garbage_synthetic_advance//'
path_json = 'C://Users//SumeetMitra//Downloads//bdd100k_labels_images_val.json'


files_garbage = glob.glob(path_garbage+'PNG-FILES//*//*')


if not os.path.exists(path_save):
    os.mkdir(path_save)

def add_defect(steel_image,defect_path,location_defect,area):
    defect_image1 = cv2.imread(defect_path,cv2.IMREAD_UNCHANGED)
    r1,c1,ch1 = steel_image.shape
    l2, l1 = location_defect
    resize_factor = (0.25/177016)*area
    loc_box = []
    if resize_factor > 0.005:
        defect_image = cv2.resize(defect_image1,(0,0), fx = resize_factor, fy = resize_factor)
    
        defect_image_back1 = defect_image[:,:,3]
        defect_image_bgr_original = defect_image[:,:,:3]
        defect_image_bgr = defect_image_bgr_original
        #defect_image_bgr = cv2.convertScaleAbs(defect_image_bgr_original, alpha=defect_color[0], beta=defect_color[1])
    
        r2,c2,ch2 = defect_image.shape
        x1 = int(l1-(r2/2))
        x2 = int(l1+(r2/2))
        y1 = int(l2-(c2/2))
        y2 = int(l2+(c2/2))
        if (x1 > 0) and (y1 > 0) and (x2 < r1) and (y2 < c1):
            steel_mask = steel_image[x1:x2,y1:y2]
            ret, defect_image_back = cv2.threshold(defect_image_back1, 127, 255, cv2.THRESH_BINARY)
            defect_image_back_inv = cv2.bitwise_not(defect_image_back)
            steel_background = cv2.bitwise_and(steel_mask,steel_mask,mask=defect_image_back_inv)
            defect_foreground = cv2.bitwise_and(defect_image_bgr,defect_image_bgr,mask=defect_image_back)
            result_image = cv2.add(steel_background,defect_foreground)
            steel_image[x1:x2,y1:y2] = result_image
            
        loc_box = [y1,x1,y2,x2] 
           
    return steel_image,loc_box

def mask_cut(ori_img,copy_img,mask_img):
    img_mask_inv = cv2.bitwise_not(mask_img)
    img1_bg = cv2.bitwise_and(copy_img,copy_img,mask = img_mask_inv)
    img2_fg = cv2.bitwise_and(ori_img,ori_img,mask = mask_img)
    dst = cv2.add(img1_bg,img2_fg)
    return dst

def parabolic_path(lst1):
    df = pd.DataFrame(lst1,columns=['category','x1','y1','x2','y2','image_width','image_height'])     
    df = df.assign(x2_x1 = lambda x : (x['x2'] - x['x1']))
    df = df.assign(y2_y1 = lambda x : (x['y2'] - x['y1']))
    df = df.assign(Area = lambda x : (x['x2_x1']*x['y2_y1']))
    df = df.sort_values(by=['Area'], ascending=False)
    df = df.reset_index(drop = True)
    df = df.assign(loc_0y_l = lambda x :(x['y1']+0.1*x['y2_y1']))
    df = df.assign(loc_0x_l = lambda x :(x['x1']))
    df = df.assign(loc_1y_l = lambda x :(x['y1']+0.3*x['y2_y1']))
    df = df.assign(loc_1x_l = lambda x :(x['x1']-((x['y2_y1']*(x['loc_1y_l']-x['loc_0y_l']))**0.5)))
    df = df.assign(loc_2y_l = lambda x :(x['y1']+0.5*x['y2_y1']))
    df = df.assign(loc_2x_l = lambda x :(x['x1']-((x['y2_y1']*(x['loc_2y_l']-x['loc_0y_l']))**0.5)))
    df = df.assign(loc_3y_l = lambda x :(x['y1']+0.7*x['y2_y1']))
    df = df.assign(loc_3x_l = lambda x :(x['x1']-((x['y2_y1']*(x['loc_3y_l']-x['loc_0y_l']))**0.5)))
    df = df.assign(loc_4y_l = lambda x :(x['y1']+0.9*x['y2_y1']))
    df = df.assign(loc_4x_l = lambda x :(x['x1']-((x['y2_y1']*(x['loc_4y_l']-x['loc_0y_l']))**0.5)))
    df = df.assign(loc_0y_r = lambda x :(x['y1']+0.1*x['y2_y1']))
    df = df.assign(loc_0x_r = lambda x :(x['x2']))
    df = df.assign(loc_1y_r = lambda x :(x['y1']+0.3*x['y2_y1']))
    df = df.assign(loc_1x_r = lambda x :(x['x2']+((x['y2_y1']*(x['loc_1y_r']-x['loc_0y_r']))**0.5)))
    df = df.assign(loc_2y_r = lambda x :(x['y1']+0.5*x['y2_y1']))
    df = df.assign(loc_2x_r = lambda x :(x['x2']+((x['y2_y1']*(x['loc_2y_r']-x['loc_0y_r']))**0.5)))
    df = df.assign(loc_3y_r = lambda x :(x['y1']+0.7*x['y2_y1']))
    df = df.assign(loc_3x_r = lambda x :(x['x2']+((x['y2_y1']*(x['loc_3y_r']-x['loc_0y_r']))**0.5)))
    df = df.assign(loc_4y_r = lambda x :(x['y1']+0.9*x['y2_y1']))
    df = df.assign(loc_4x_r = lambda x :(x['x2']+((x['y2_y1']*(x['loc_4y_r']-x['loc_0y_r']))**0.5)))
    return df  

def check_dir(df2,m):
    gar_veh = m 
    print("Pass check")
    if (df2['loc_4x_l'].values[m] > 0) or (df2['loc_4x_r'].values[m] < df2['image_width'].values[m]):
        gar_dir = str(random.choice(['l','r']))
        if gar_dir == 'l':
            if df2['loc_4x_l'].values[m] > 0:
                gar_fin = 'l'
            else:
                gar_fin,gar_veh =  check_dir(df2,m)
    
        elif gar_dir == 'r':
            if df2['loc_4x_r'].values[m] < df2['image_width'].values[m]:
                gar_fin = 'r'
            else:
                gar_fin,gar_veh = check_dir(df2,m)    
            
    else:
        if len(df2) <= m+1:
            gar_fin,gar_veh = check_dir(df2,m+1) 
        else:
            gar_fin = gar_veh = 404 
    
    return gar_fin,gar_veh 

def check_dir2(df2,m):
    gar_veh = m 
    print("Pass check")
    if df2['loc_4x_l'].values[m] > 0:
        gar_fin = 'l'
    elif df2['loc_4x_r'].values[m] < df2['image_width'].values[m]:
        gar_fin = 'r'
    else:
        print(len(df2))
        if len(df2) <= m+1:
            gar_fin,gar_veh = check_dir(df2,m+1) 
        else:
            gar_fin = gar_veh = 404    
    
    return gar_fin,gar_veh 

    
def save_images(img,df,n,y,save_name,files_garbage):
    save_dir = path_save
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    lst_img = []
    ny = str(n)+str(y)
    gar_im = str(random.choice(files_garbage))
    img_mask = np.zeros(img.shape[:2], np.uint8)
    for rang in range(n):
        cv2.rectangle(img_mask, (int(df['x1'].values[n])-1,int(df['y1'].values[n])-1), (int(df['x2'].values[n])+1,int(df['y2'].values[n])+1), 255, -1)
    cv2.imwrite(save_dir+save_name+str(ny)+'_0.png',img) 
    img_copy = copy.deepcopy(img)
    img_copy2,gar_loc = add_defect(img_copy,gar_im,[df['loc_0x_'+str(y)].values[n],df['loc_0y_'+str(y)].values[n]],df['Area'].values[n])
    if len(gar_loc):
        dst2 = mask_cut(img,img_copy2,img_mask)
        cv2.imwrite(save_dir+save_name+str(ny)+'_1.png',dst2) 
        #lst_img.append([str(save_name+str(ny)+'_1.png'),str(df['category'].values[n]),str(df['x1'].values[n]),str(df['y1'].values[n]),str(df['x2'].values[n]),str(df['y2'].values[n]),str(df['image_width'].values[n]),str(df['image_height'].values[n])])
        lst_img.append([str(save_name+str(ny)+'_1.png'),str('garbage'),str(gar_loc[0]),str(gar_loc[1]),str(gar_loc[2]),str(gar_loc[3]),str(df['image_width'].values[n]),str(df['image_height'].values[n])])
        img_copy = copy.deepcopy(img)
        img_copy3,gar_loc = add_defect(img_copy,gar_im,[df['loc_1x_'+str(y)].values[n],df['loc_1y_'+str(y)].values[n]],df['Area'].values[n])
        dst3 = mask_cut(img,img_copy3,img_mask)
        cv2.imwrite(save_dir+save_name+str(ny)+'_2.png',dst3)
        #lst_img.append([str(save_name+str(ny)+'_2.png'),str(df['category'].values[n]),str(df['x1'].values[n]),str(df['y1'].values[n]),str(df['x2'].values[n]),str(df['y2'].values[n]),str(df['image_width'].values[n]),str(df['image_height'].values[n])])
        lst_img.append([str(save_name+str(ny)+'_2.png'),str('garbage'),str(gar_loc[0]),str(gar_loc[1]),str(gar_loc[2]),str(gar_loc[3]),str(df['image_width'].values[n]),str(df['image_height'].values[n])])
        img_copy = copy.deepcopy(img)
        img_copy4,gar_loc = add_defect(img_copy,gar_im,[df['loc_2x_'+str(y)].values[n],df['loc_2y_'+str(y)].values[n]],df['Area'].values[n])
        dst4 = mask_cut(img,img_copy4,img_mask)
        cv2.imwrite(save_dir+save_name+str(ny)+'_3.png',dst4)
        #lst_img.append([str(save_name+str(ny)+'_3.png'),str(df['category'].values[n]),str(df['x1'].values[n]),str(df['y1'].values[n]),str(df['x2'].values[n]),str(df['y2'].values[n]),str(df['image_width'].values[n]),str(df['image_height'].values[n])])
        lst_img.append([str(save_name+str(ny)+'_3.png'),str('garbage'),str(gar_loc[0]),str(gar_loc[1]),str(gar_loc[2]),str(gar_loc[3]),str(df['image_width'].values[n]),str(df['image_height'].values[n])])
        img_copy = copy.deepcopy(img)
        img_copy5,gar_loc = add_defect(img_copy,gar_im,[df['loc_3x_'+str(y)].values[n],df['loc_3y_'+str(y)].values[n]],df['Area'].values[n])
        dst5 = mask_cut(img,img_copy5,img_mask)
        cv2.imwrite(save_dir+save_name+str(ny)+'_4.png',dst5)
        #lst_img.append([str(save_name+str(ny)+'_4.png'),str(df['category'].values[n]),str(df['x1'].values[n]),str(df['y1'].values[n]),str(df['x2'].values[n]),str(df['y2'].values[n]),str(df['image_width'].values[n]),str(df['image_height'].values[n])])
        lst_img.append([str(save_name+str(ny)+'_4.png'),str('garbage'),str(gar_loc[0]),str(gar_loc[1]),str(gar_loc[2]),str(gar_loc[3]),str(df['image_width'].values[n]),str(df['image_height'].values[n])])
        img_copy = copy.deepcopy(img)
        img_copy6,gar_loc = add_defect(img_copy,gar_im,[df['loc_4x_'+str(y)].values[n],df['loc_4y_'+str(y)].values[n]],df['Area'].values[n])
        dst6 = mask_cut(img,img_copy6,img_mask)
        cv2.imwrite(save_dir+save_name+str(ny)+'_5.png',dst6)
        #lst_img.append([str(save_name+str(ny)+'_5.png'),str(df['category'].values[n]),str(df['x1'].values[n]),str(df['y1'].values[n]),str(df['x2'].values[n]),str(df['y2'].values[n]),str(df['image_width'].values[n]),str(df['image_height'].values[n])])
        lst_img.append([str(save_name+str(ny)+'_5.png'),str('garbage'),str(gar_loc[0]),str(gar_loc[1]),str(gar_loc[2]),str(gar_loc[3]),str(df['image_width'].values[n]),str(df['image_height'].values[n])])
    
    return lst_img
    
f = open(path_json )
content = json.load(f)

lst_final = []
for i in content[0:100]:
    lst = []

    file_path = cv2.imread(path_image+i['name'])
    for j in i['labels']:
        if (j['category'] == 'car') or (j['category'] == 'truck') or (j['category'] == 'bus'):
            lst.append([j['category'],j['box2d']['x1'],j['box2d']['y1'],j['box2d']['x2'],j['box2d']['y2'],int(file_path.shape[1]),int(file_path.shape[0])])
    df1 = parabolic_path(lst)
    print(path_image+i['name'])
    gar_fin1,gar_veh1 = check_dir2(df1,0) 
    if not (gar_fin1 == 404) :
        lst_init = save_images(file_path,df1,int(gar_veh1),gar_fin1,i['name'].split('.')[0],files_garbage) 
        lst_final.extend(lst_init) 
                
df_fin = pd.DataFrame(lst_final,columns=['category','class_name','x1','y1','x2','y2','image_width','image_height'])  
df_fin.to_csv("final.csv")  