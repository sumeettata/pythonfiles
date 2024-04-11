import cv2
from ultralytics import YOLO
import os
import pandas as pd
import glob
import cv2
from tqdm import tqdm
import numpy as np
import os
import random
import json
import copy
import matplotlib.pyplot as plt


os.environ['KMP_DUPLICATE_LIB_OK']='True'


def add_defect(steel_image,defect_path,location_defect,area):
    defect_image1 = cv2.imread(defect_path,cv2.IMREAD_UNCHANGED)
    r1,c1,ch1 = steel_image.shape
    l2, l1 = location_defect
    resize_factor = (0.2/177016)*area
    loc_box = []
    defect_image = cv2.resize(defect_image1,(0,0), fx = resize_factor, fy = resize_factor)

    defect_image_back1 = defect_image[:,:,3]
    defect_image_bgr_original = defect_image[:,:,:3]
    defect_image_bgr = defect_image_bgr_original

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

def parabolic_contin(df,no,sid):
    for i in range(no):
        df['loc_'+str(i+1)+'y_'+str(sid)] = df.apply(lambda x :(x['y1']+0.1*(i+2)*x['y2_y1']),axis=1)
        if str(sid) == 'l':
            df['loc_'+str(i+1)+'x_'+str(sid)] = df.apply(lambda x :(x['x1']-((x['y2_y1']*(x['loc_'+str(i+1)+'y_'+str(sid)]-x['loc_0y_'+str(sid)]))**0.5)),axis=1)
        elif str(sid) == 'r':
            df['loc_'+str(i+1)+'x_'+str(sid)] = df.apply(lambda x :(x['x2']+((x['y2_y1']*(x['loc_'+str(i+1)+'y_'+str(sid)]-x['loc_0y_'+str(sid)]))**0.5)),axis=1)
    
    return df

def parabolic_path(lst1):
    df = pd.DataFrame(lst1,columns=['category','id','x1','y1','x2','y2','image_width','image_height'])     
    df = df.assign(x2_x1 = lambda x : (x['x2'] - x['x1']))
    df = df.assign(y2_y1 = lambda x : (x['y2'] - x['y1']))
    df = df.assign(Area = lambda x : (x['x2_x1']*x['y2_y1']))
    df = df.assign(resize_Area = lambda x : (x['Area']*(0.2/177016)))
    df = df[df['resize_Area']>0.005]
    df = df.sort_values(by=['Area'], ascending=False)
    df = df.reset_index(drop = True)
    df = df.assign(loc_0y_l = lambda x :(x['y1']+0.1*x['y2_y1']))
    df = df.assign(loc_0x_l = lambda x :(x['x1']))
    df = parabolic_contin(df,8,'l')
    df = df.assign(loc_0y_r = lambda x :(x['y1']+0.1*x['y2_y1']))
    df = df.assign(loc_0x_r = lambda x :(x['x2']))
    df = parabolic_contin(df,8,'r')
    return df  


def check_dir2(df2,m):
    gar_veh = m 
    print("Pass check")
    if df2['loc_8x_l'].values[m] > 0:
        gar_fin = 'l'
    elif df2['loc_8x_r'].values[m] < df2['image_width'].values[m]:
        gar_fin = 'r'
    else:
        print(len(df2))
        if len(df2) <= m+1:
            gar_fin,gar_veh = check_dir2(df2,m+1) 
        else:
            gar_fin = gar_veh = 404    
    
    return gar_fin,gar_veh    
            

model = YOLO("weights/vehicle_04_05_2023.pt")
path_garbage = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//Garbage//'
files_garbage = glob.glob(path_garbage+'PNG-FILES//*//*')

video_path = "Videos"

for w,path in enumerate(glob.glob(video_path+'/*.mp4')):

    print('video '+ str(w)+" of " + str(len(glob.glob(video_path+'/*.mp4'))))
    lst_final = []

    cap = cv2.VideoCapture(path)

    if os.path.exists(path.split('.')[0]):
        print('Duplicates')
        continue
    
    if not os.path.exists(path.split('.')[0]):
        os.mkdir(path.split('.')[0])
        
    result = cv2.VideoWriter(path.split('.')[0]+'//'+os.path.basename(path).split('.')[0]+'.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, (int(1920),int(1080)))
    

    k = 0
    i = 0
    id_no_lst = []
    while True:
        k = k + 1
        i = i + 1 
        file_name = os.path.basename(path).split('.')[0]+'_'+str(k)+'.png'
        ret, frame = cap.read()
        #frame = cv2.rotate(frame, cv2.ROTATE_180)
        lst = []
        if not ret:
            break
        if frame is not None:
            frame = cv2.resize(frame, (int(1920), int(1080)))
        
        try:
               
            results = model.track(frame, persist=True)
        
        
            if results[0].boxes:
                print(results[0].boxes.id)
                if results[0].boxes.id is not None:
                    class_name = model.names
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    clss = results[0].boxes.cls.cpu().numpy().astype(int)
                
                    if i == 1:
                            
                        for box, id, cl  in zip(boxes, ids,clss):
                            if (class_name[int(cl)] != 'np'): 
                                lst_final.append([file_name,class_name[cl],box[0], box[1],box[2], box[3],frame.shape[1],frame.shape[0]])
                            if (class_name[int(cl)] != 'np') and (id not in id_no_lst) :
                                lst.append([cl,id,box[0], box[1],box[2], box[3],frame.shape[1],frame.shape[0]])
                        df1 = parabolic_path(lst)
                                
                        if len(df1) == 0:
                            for box, id, cl  in zip(boxes, ids,clss):
                                if (class_name[int(cl)] != 'np'):        
                                    lst.append([cl,id,box[0], box[1],box[2], box[3],frame.shape[1],frame.shape[0]])
                                
                        df1 = parabolic_path(lst)
                        
                        if len(df1):
                            gar_fin1,gar_veh1 = check_dir2(df1,0)
                            n = int(gar_veh1)
                            y = str(gar_fin1)
                            if not (gar_fin1 == 404) :
                                id_no = df1['id'].values[n]
                                id_no_lst.append(id_no)
                                lst_img = []
                                if i == 1:
                                    gar_im = str(random.choice(files_garbage))
                                img_mask = np.zeros(frame.shape[:2], np.uint8)
                                for rang in range(n):
                                    cv2.rectangle(img_mask, (int(df1['x1'].values[n])-1,int(df1['y1'].values[n])-1), (int(df1['x2'].values[n])+1,int(df1['y2'].values[n])+1), 255, -1)
                                plt.imshow(img_mask)
                                plt.show()
                                img_copy = copy.deepcopy(frame)
                                img_copy2,gar_loc = add_defect(img_copy,gar_im,[df1['loc_'+str(i-1)+'x_'+y].values[n],df1['loc_'+str(i-1)+'y_'+y].values[n]],df1['Area'].values[n])
                                if len(gar_loc):
                                    lst_final.append([file_name,'garbage',gar_loc[0], gar_loc[1],gar_loc[2], gar_loc[3],frame.shape[1],frame.shape[0]])
                                    frame = mask_cut(frame,img_copy2,img_mask)  
                                    frame2 = cv2.rectangle(frame.copy(), (gar_loc[0], gar_loc[1]),(gar_loc[2], gar_loc[3]) , [0,0,255], 10)
                                    for vehicles in lst:
                                        frame2 = cv2.rectangle(frame2, (vehicles[2], vehicles[3]),(vehicles[4], vehicles[5]) , [0,255,0], 8)
                                    
                        print(lst_final)
                                    
                    elif i == 9:
                        i = 0    
                    
                    else:
                        for box, id, cl  in zip(boxes, ids,clss):
                            if (class_name[int(cl)] != 'np'): 
                                lst_final.append([file_name,class_name[cl],box[0], box[1],box[2], box[3],frame.shape[1],frame.shape[0]])
                            if (class_name[int(cl)] != 'np') and id == id_no:
                                lst.append([cl,id,box[0], box[1],box[2], box[3],frame.shape[1],frame.shape[0]])
                                
                        df1 = parabolic_path(lst)
                        if len(df1):
                            if not (gar_fin1 == 404) :        
                                lst_img = []
                                img_mask = np.zeros(frame.shape[:2], np.uint8)
                                for rang in range(n):
                                    cv2.rectangle(img_mask, (int(df1['x1'].values[n])-1,int(df1['y1'].values[n])-1), (int(df1['x2'].values[n])+1,int(df1['y2'].values[n])+1), 255, -1)  
                                img_copy = copy.deepcopy(frame)
                                img_copy2,gar_loc = add_defect(img_copy,gar_im,[df1['loc_'+str(i-1)+'x_'+y].values[n],df1['loc_'+str(i-1)+'y_'+y].values[n]],df1['Area'].values[n])
                                if len(gar_loc):
                                    lst_final.append([file_name,'garbage',gar_loc[0], gar_loc[1],gar_loc[2], gar_loc[3],frame.shape[1],frame.shape[0]])
                                    frame = mask_cut(frame,img_copy2,img_mask)    
                                    frame2 = cv2.rectangle(frame.copy(), (gar_loc[0], gar_loc[1]),(gar_loc[2], gar_loc[3]) , [0,0,255], 10)
                                    for vehicles in lst:
                                        frame2 = cv2.rectangle(frame2, (vehicles[2], vehicles[3]),(vehicles[4], vehicles[5]) , [0,255,0], 8)
                                          
                        else:
                            i = 0
        
        except:
            print('Skip The Frame')   
             
        cv2.imwrite(path.split('.')[0]+'//'+file_name,frame)  
        resize_window = 'video '+ str(w)+" of " + str(len(glob.glob(video_path+'/*.mp4')))      
        cv2.namedWindow(resize_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(resize_window, 1080, 720)
        try:
            cv2.imshow(resize_window, frame2)
            result.write(frame2) 
        except:
            cv2.imshow(resize_window, frame)
            result.write(frame) 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cap.release()
    result.release()  
    cv2.destroyAllWindows()
    df5 = pd.DataFrame(lst_final,columns=['filename','category','x1','y1','x2','y2','image_width','image_height'] )
    df5.to_csv(path.split('.')[0]+'//'+os.path.basename(path).split('.')[0]+'.csv')