import glob
import os
import cv2
import random
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from num2words import num2words


os.chdir('Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/')

class generate_np:
    def __init__(self,font_path="cropped_fonts1/final_font",
                 plate_path="used_gimp/20220725_11_43_56_412_000_hxH0nNGamnPRFROZtlLa0JHSbvD3_F_3264_2448.jpg",
                 ind_path = "ind_symbols",
                 label_path="plate_labels",json_path="Rto_vehicle_list.json",
                 chnge_lin=3, seqn=1, font_colr= (0,0,0), folder_wise = True, noise = True, rotate_need=True,rotate_rec=False,
                 nospacetxt = False
                 ):
        
        self.font_path = font_path
        self.plate_path = plate_path
        self.foldername = os.path.basename(plate_path).split('.')[0]
        self.label_path = label_path+ "/"+os.path.basename(self.plate_path).split(".")[0]+".xml"
        self.json_path = json_path
        self.chnge_lin = chnge_lin
        self.colr = font_colr
        self.rotate_need = rotate_need
        self.rotate_rec = rotate_rec
        self.nospacetxt = nospacetxt
        self.seqn = seqn
        self.noise = noise
        self.ind_path = ind_path
        if folder_wise == True:
            self.folderpath = self.make_dir(self.foldername)
        else:
            self.folderpath = self.make_dir('Number_plates_synthetic_test')
        self.img = cv2.imread(self.plate_path)
        self.make_dir(self.folderpath+'/train_det')
        self.make_dir(self.folderpath+'/train_rec')
        self.lst = [] 
        self.lst2 = []
        self.lst3 = []
        self.load_gen()
        self.load_gen_bharat()
        self.save_csv()
        self.save_csv('train_rec')
    
    def load_gen(self):  
         
        for start_code in random.choices(self.load_json(),k=15):
            tree = ET.parse(self.label_path)
            root = tree.getroot()
            self.plate_number = plate_number = self.get_number(start_code)
            img = self.img.copy()
            print(plate_number)
            xmin,ymin,xmax,ymax = [],[],[],[]
            self.xmin_prev,self.xmax_prev,self.ymin_prev,self.ymax_prev = [],[],[],[]
            for k,member in enumerate(root.findall('object')):
                
                xmin.append(int(member[4][0].text))
                ymin.append(int(member[4][1].text))
                xmax.append(int(member[4][2].text))
                ymax.append(int(member[4][3].text))
                
                #if not plate_number[k] == '#':
                img_let_resiz = cv2.resize(self.get_letter(plate_number[k]), ((xmax[0]-xmin[0]),(ymax[0]-ymin[0])))
                img_final = self.image_put(img_let_resiz,xmin,xmax,ymin,ymax)
                if self.nospacetxt == True:
                    if len(xmin) < 2:
                        img[ymin[0]:ymax[0],xmin[-1]:(xmin[-1]+(xmax[0]-xmin[0]))] = img_final
                        self.xlastpt = (xmin[-1]+(xmax[0]-xmin[0]))
                    else:
                        try:
                            img[ymin[0]:ymax[0],(self.xlastpt+1):((self.xlastpt+1)+(xmax[0]-xmin[0]))] = img_final
                            self.xlastpt = ((self.xlastpt+1)+(xmax[0]-xmin[0])) 
                            
                        except:
                            print("removed") 
                else:
                    img[ymin[0]:ymax[0],xmin[-1]:(xmin[-1]+(xmax[0]-xmin[0]))] = img_final   
                    
                if k == self.chnge_lin:
                    self.xmin_prev,self.xmax_prev,self.ymin_prev,self.ymax_prev = xmin_prev,xmax_prev,ymin_prev,ymax_prev = xmin,xmax,ymin,ymax
                    xmin,ymin,xmax,ymax = [],[],[],[]
            
            self.dim_plate2 = plate_number[:(self.chnge_lin+1)].replace('#','')
            self.dim_plate1 = plate_number[(self.chnge_lin+1):].replace('#','')
            img = self.rotate_shear(img,xmax,xmin,ymax,ymin)
            self.save_file = str('train_det/gen_'+str(self.seqn)+'_'+str(self.plate_number.replace('#',''))+'.png')
            cv2.imwrite(self.folderpath+'/'+self.save_file,img)
            self.save_dic()
            self.lst.append(str(self.save_file)+'\t'+str(self.lis)+'\n')
            self.lst2.append([self.save_file,self.plate_number.replace('#',''),self.lis])
            
    def load_gen_bharat(self):
        
        j = 0
        while j < 5:
            tree = ET.parse(self.label_path)
            root = tree.getroot()
            self.plate_number = plate_number = self.get_bharat_number()
            img = self.img.copy()
            print(plate_number)
            xmin,ymin,xmax,ymax = [],[],[],[]
            self.xmin_prev,self.xmax_prev,self.ymin_prev,self.ymax_prev = [],[],[],[]
            for k,member in enumerate(root.findall('object')):
                
                xmin.append(int(member[4][0].text))
                ymin.append(int(member[4][1].text))
                xmax.append(int(member[4][2].text))
                ymax.append(int(member[4][3].text))
                
                img_let_resiz = cv2.resize(self.get_letter(plate_number[k]), ((xmax[0]-xmin[0]),(ymax[0]-ymin[0])))
                img_final = self.image_put(img_let_resiz,xmin,xmax,ymin,ymax)
                if self.nospacetxt == True:
                    if len(xmin) < 2:
                        img[ymin[0]:ymax[0],xmin[-1]:(xmin[-1]+(xmax[0]-xmin[0]))] = img_final
                        self.xlastpt = (xmin[-1]+(xmax[0]-xmin[0]))
                    else:
                        try:
                            img[ymin[0]:ymax[0],(self.xlastpt+1):((self.xlastpt+1)+(xmax[0]-xmin[0]))] = img_final
                            self.xlastpt = ((self.xlastpt+1)+(xmax[0]-xmin[0])) 
                        except:
                            print("removed") 
                else:
                    img[ymin[0]:ymax[0],xmin[-1]:(xmin[-1]+(xmax[0]-xmin[0]))] = img_final 
                    
                if k == self.chnge_lin:
                    self.xmin_prev,self.xmax_prev,self.ymin_prev,self.ymax_prev = xmin_prev,xmax_prev,ymin_prev,ymax_prev = xmin,xmax,ymin,ymax
                    xmin,ymin,xmax,ymax = [],[],[],[]
                        
            self.dim_plate2 = plate_number[:(self.chnge_lin+1)]
            self.dim_plate1 = plate_number[(self.chnge_lin+1):]
            img = self.rotate_shear(img,xmax,xmin,ymax,ymin)
            self.save_file = str('train_det/gen_'+str(self.seqn)+'_'+str(self.plate_number.strip('#'))+'.png')
            cv2.imwrite(self.folderpath+'/'+self.save_file,img)
            self.save_dic()
            self.lst.append(str(self.save_file)+'\t'+str(self.lis)+'\n')
            self.lst2.append([self.save_file,self.plate_number.strip('#'),self.lis])
            j = j +1
                
    def load_json(self):
        rto_code = json.load(open(self.json_path)).values()
        rto_code = sum(list(rto_code),[])
        return rto_code
 
    def get_number(self,start_code):
        number_code = [str(x) for x in range(10)]
        alphabet_code = [chr(v).upper() for v in range(97, 123) if v not in [111,105]]
        plate_no = str(start_code)+random.choice(alphabet_code)+random.choice(alphabet_code)+random.choice(number_code)+random.choice(number_code)+random.choice(number_code)+random.choice(number_code)
        return plate_no
    
    def get_bharat_number(self):
        number_code = [str(x) for x in range(10)]
        alphabet_code = [chr(v).upper() for v in range(97, 123) if v not in [111,105]]
        plate_no = str(random.choice([x for x in range(20,30)]))+str('BH')+random.choice(number_code)+random.choice(number_code)+random.choice(number_code)+random.choice(number_code)+random.choice(alphabet_code)+random.choice(alphabet_code)
        return plate_no
        
    def get_letter(self,let_in):
        if let_in in [str(x) for x in range(10)]:
            if os.path.basename(self.font_path) == 'final_font':
                if str(let_in) == '4':
                    let_in = random.choice(['4','4_1'])
                let_img = cv2.imread(self.font_path+"/"+str(let_in)+'.png',cv2.IMREAD_UNCHANGED)  
            else:
                let_img = cv2.imread(self.font_path+"/"+str(num2words(int(let_in)))+'.png',cv2.IMREAD_UNCHANGED)   
        else:
            let_img = cv2.imread(self.font_path+"/"+str(let_in)+'.png',cv2.IMREAD_UNCHANGED)   
        return let_img
    
    def make_dir(self,path_name):
        if not os.path.exists(path_name):
            os.mkdir(path_name) 
        return path_name  
    
    def image_noise(self,img_let_org):
        if self.colr == (10,10,10):
            val = 250
        elif self.colr:
            val = 250
        else:
            val = 5
        img_random = np.random.random((img_let_org.shape[0],img_let_org.shape[1]))
        img_let_org[img_random>0.7] = img_let_org[img_random>0.7]- val-1
        img_random = np.random.random((img_let_org.shape[0],img_let_org.shape[1]))
        img_let_org[img_random>0.7] = img_let_org[img_random>0.7]- val
        img_random = np.random.random((img_let_org.shape[0],img_let_org.shape[1]))
        img_let_org[img_random>0.7] = img_let_org[img_random>0.7]- val-2
        return img_let_org
    
    def image_put(self,img_let_resiz,xmin,xmax,ymin,ymax):
        if self.colr :
            img_let_org = np.zeros([(ymax[0]-ymin[0]),(xmax[0]-xmin[0]),3],dtype=np.uint8)
            img_let_org = cv2.rectangle(img_let_org,(0,0),((xmax[0]-xmin[0]),(ymax[0]-ymin[0])),self.colr,-1)
        else:
            img_let_org = img_let_resiz[:,:,:3]
        img_let = cv2.cvtColor(img_let_org, cv2.COLOR_BGR2GRAY)
        ret , img_back_inv = cv2.threshold(img_let_resiz[:,:,3], 127 , 255, cv2.THRESH_BINARY)
        kernel = np.ones((1, 1), np.uint8)
        img_back_inv = cv2.erode(img_back_inv, kernel, iterations=3)
        img_back = cv2.bitwise_not(img_back_inv)
        if (self.noise):
            img_let_org = self.image_noise(img_let_org)
        img_crop= self.img[ymin[0]:ymax[0],xmin[-1]:(xmin[-1]+(xmax[0]-xmin[0]))]
        img_background = cv2.bitwise_and(img_crop,img_crop,mask=img_back)
        img_let_foreground = cv2.bitwise_and(img_let_org,img_let_org,mask=img_back_inv)
        img_final2 = cv2.add(img_background,img_let_foreground)
        img_final = cv2.addWeighted(img_crop,0.2,img_final2,0.8,1)
        return img_final
    
    def transform_matrix(self,bordr,shr=10,scal=70):
        if len(self.xmin_prev):
            shr,scal = 15,70
        a1 = random.randrange(-shr,shr)*0.01
        b1 = random.randrange(-shr,shr)*0.01
        M_shear = np.float32([[1, a1, 0],[b1, 1  , 0],[0, 0  , 1]])
        a2 = random.randrange(scal,100)*0.01
        b2 = random.randrange(scal,100)*0.01
        M_scale = np.float32([[a2, 0  , 0],[0,   b2, 0],[0,   0,   1]])
        M_transition = np.float32([[1, 0, -bordr],[0, 1  , -bordr],[0, 0  , 1]])
        M_transition_rev = np.float32([[1, 0, bordr],[0, 1  , bordr],[0, 0  , 1]])
        M = M_transition_rev@M_scale@M_shear@M_transition
        return M
    
    def rotate_shear(self,img,xmax,xmin,ymax,ymin,bordr = 500,bgrnd_color=200,bgrnd_extra=0):
        if self.rotate_need == True:
            M = self.transform_matrix(bordr)
        else:
            M = np.float32([[1, 0, 0],[0, 1  , 0],[0, 0  , 1]])
        if len(self.xmin_prev):
            xmin_prev,ymin_prev,xmax_prev,ymax_prev = self.xmin_prev,self.ymin_prev,self.xmax_prev,self.ymax_prev
            bgrnd_extra = 0
        img_mask = np.ones([img.shape[0],img.shape[1]],dtype=np.uint8)*255
        img_mask = cv2.copyMakeBorder(img_mask, bordr, bordr, bordr, bordr, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img_changed_mask = cv2.warpPerspective(img_mask,M,(img_mask.shape[1],img_mask.shape[0]))
        img_pts2  = cv2.goodFeaturesToTrack(img_changed_mask, 4, 0.01, 10)
        corners = np.int0(img_pts2.reshape((img_pts2.shape[0],-1)))
        self.cor_xmin =  cor_xmin = sorted(corners[:,0])[1]-bgrnd_extra
        self.cor_xmax = cor_xmax = sorted(corners[:,0])[-2]+bgrnd_extra
        self.cor_ymin = cor_ymin = sorted(corners[:,1])[1]-bgrnd_extra
        self.cor_ymax = cor_ymax = sorted(corners[:,1])[-2]+bgrnd_extra
        
        img = cv2.copyMakeBorder(img, bordr, bordr, bordr, bordr, cv2.BORDER_CONSTANT, value=[bgrnd_color, bgrnd_color, bgrnd_color])   
        img_changed = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
        img_final = img_changed[cor_ymin:cor_ymax,cor_xmin:cor_xmax]
        self.p = 0
        if len(self.xmin_prev):
            self.dim_mask_1 = self.txt_label(img,M,xmin_prev,ymin_prev,xmax_prev,ymax_prev,bordr,img_final.copy())
            save_file_name1 = self.save_file2
            
        self.dim_mask_2 = self.txt_label(img,M,xmin,ymin,xmax,ymax,bordr,img_final.copy())
        save_file_name2 = self.save_file2
        
        if len(self.xmin_prev):
            self.lst3.append([save_file_name1,self.plate_number.strip('#'),self.dim_plate2.strip('#')])
            self.lst3.append([self.save_file2,self.plate_number.strip('#'),self.dim_plate1.strip('#')])
        else:
             self.lst3.append([save_file_name2,self.plate_number.strip('#'),self.dim_plate2.strip('#')])   
            
        img_final2 = cv2.polylines(img_final.copy(), [self.dim_mask_2], True, 0, 2)
        if len(self.xmin_prev):
            img_final2 = cv2.polylines(img_final2, [self.dim_mask_1], True, 0, 2)
        self.show_image(img_final2)
        return img_final
    
    def txt_label(self,img,M,xmin,ymin,xmax,ymax,bordr,img_final):
        txt_mask = np.zeros([img.shape[0],img.shape[1]],dtype=np.uint8)
        if self.nospacetxt == True:
            txt_mask = cv2.rectangle(txt_mask, (xmin[0],ymin[0]), (self.xlastpt,ymax[0]), (255,255,255), -1)
        else:
            txt_mask = cv2.rectangle(txt_mask, (xmin[0],ymin[0]), ((xmin[-1]+(xmax[0]-xmin[0])),ymax[0]), (255,255,255), -1)
        txt_mask = cv2.copyMakeBorder(txt_mask, bordr, bordr, bordr, bordr, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        txt_mask = cv2.warpPerspective(txt_mask,M,(txt_mask.shape[1],txt_mask.shape[0])) 
        txt_mask = txt_mask[self.cor_ymin:self.cor_ymax,self.cor_xmin:self.cor_xmax]
        contours4,_= cv2.findContours(txt_mask.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        dim_mask2 = cv2.minAreaRect(contours4[0])
        self.txt_mask = txt_mask
        self.dim_mask = dim_mask2
        dim_mask2 = np.int0(cv2.boxPoints(dim_mask2))
        if np.min(dim_mask2) < 0:
            dim_mask2 = cv2.boundingRect(contours4[0])
            dim_mask2 = np.array([[dim_mask2[0],dim_mask2[1]],[dim_mask2[0]+dim_mask2[2],dim_mask2[1]],[dim_mask2[0]+dim_mask2[2],dim_mask2[1]+dim_mask2[3]],[dim_mask2[0],dim_mask2[1]+dim_mask2[3]]])      
        self.save_recimg(img_final)
        return dim_mask2
        
    def show_image(self,img_new):
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resized_Window", 500 , 500 )
        cv2.imshow("Resized_Window", img_new)
        cv2.waitKey(1)
    
    def save_dic(self):
        if len(self.xmin_prev):
            dic1 = {}
            dic1["points"] = self.dim_mask_1.tolist()
            dic1["transcription"] = str(self.dim_plate2.strip('#'))
            dic_lst = [dic1]
            dic2 = {}
            dic2["points"] = self.dim_mask_2.tolist() 
            dic2["transcription"] = str(self.dim_plate1.strip('#'))
            dic_lst.extend([dic2])
        else:
            dic1 = {}
            dic1["points"] = self.dim_mask_2.tolist()
            dic1["transcription"] = str(self.dim_plate2.strip('#'))
            dic_lst = [dic1]
            
        print(dic_lst)
        self.lis = json.dumps(dic_lst)
    
    def save_txt(self):
        with open(self.folderpath+'/train_det.txt', 'w', newline='') as f:
            f.write(''.join(self.lst))
            f.close()
    
    def save_csv(self,nam='train_det'):
        if nam == 'train_det':
            df1 = pd.DataFrame(self.lst2,columns=['file_name','plate_number','locations'])
        else:
            df1 = pd.DataFrame(self.lst3,columns=['file_name','plate_number','locations'])
        if os.path.exists(self.folderpath+'/'+nam+'.csv'):
            df2 = pd.read_csv(self.folderpath+'/'+nam+'.csv')
            df1 = pd.concat([df2, df1])
        self.df1 =  df1
        df1.to_csv(self.folderpath+'/'+nam+'.csv',index=False)
        
    def ind_save(self):
        ind_lst = []
        for n, fil in enumerate(glob.glob(self.ind_path+'/*')):
            img = cv2.imread(fil)
            cv2.imwrite(self.folderpath+'/'+str('train_rec/gen_ind_'+str(n)+'.png'),img)
            ind_lst.append(str('train_rec/gen_ind_'+str(n)+'.png')+'\t'+str('IND')+'\n')
        self.ind_lst = ind_lst
                
    def csv2txt_det(self,nam='det'):
        df = pd.read_csv(self.folderpath+'/'+'train_'+nam+'.csv')
        df_train, df_test = train_test_split(df, test_size=0.2)
        lstcsv1 = []
        lstcsv2 = []
            
        for index,rows in tqdm(df_train.iterrows()):
            lstcsv1.append(str(rows['file_name'])+'\t'+str(rows['locations'])+'\n')
        
        if len(self.ind_path) and nam == 'rec':
            self.ind_save()
            lstcsv1.extend(self.ind_lst)
                
        with open(self.folderpath+'/'+'train_'+nam+'.txt', 'w', newline='') as f:
            f.write(''.join(lstcsv1))
            f.close()
            
        for index,rows in tqdm(df_test.iterrows()):
            lstcsv2.append(str(rows['file_name'])+'\t'+str(rows['locations'])+'\n')
            
        with open(self.folderpath+'/'+'val_'+nam+'.txt','w', newline='') as f:
            f.write(''.join(lstcsv2))
            f.close()
        
    def save_recimg(self,img):
        if self.rotate_rec == True:
            if self.dim_mask[-1] != 90.0:
                rotated_mask = self.rotateImage(self.txt_mask,self.dim_mask[-1])
                rotated_img = self.rotateImage(img,self.dim_mask[-1])
            else:
                rotated_mask = self.txt_mask
                rotated_img = img
        else:
            rotated_mask = self.txt_mask
            rotated_img = img
        rotated_mask_pt = cv2.boundingRect(rotated_mask)
        rotated_img = rotated_img[rotated_mask_pt[1]:rotated_mask_pt[1]+rotated_mask_pt[3],rotated_mask_pt[0]:rotated_mask_pt[0]+rotated_mask_pt[2]]
        self.save_file2 = str('train_rec/gen_'+str(self.seqn)+'_'+str(self.plate_number)+'_'+str(self.p)+'.png')
        self.p = self.p+1
        randnum = random.randrange(5,11)*0.1
        rotated_img = cv2.resize(rotated_img, (0, 0), fx = randnum, fy = randnum)
        cv2.imwrite(self.folderpath+'/'+self.save_file2,rotated_img)
    
    
    
    def rotateImage(self,image, angle):
        image = cv2.copyMakeBorder(image, 500, 500,500,500, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center,angle,1.0)
        rotated_image = cv2.warpAffine(image, M, (w,h))
        if angle >= 45:
            rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image  
    
r = 0        
for font_pth in glob.glob('cropped_fonts1/*'):
    print(font_pth)
    r=r+1         
    gt = generate_np(font_path=font_pth,
                plate_path="used_gimp/1Cars44.png",
                chnge_lin=15, seqn=r, font_colr= (10,10,10), folder_wise = False, noise = True)
    r =r+1
    generate_np(font_path=font_pth,
                plate_path="used_gimp/20230228_080552537_iOS_3.png",
                chnge_lin=4, seqn=r, font_colr= (10,10,10), folder_wise = False, noise = True)
    r =r+1
    generate_np(font_path=font_pth,
                plate_path="used_gimp/IMG_3264_6.png",
                chnge_lin=4, seqn=r, font_colr= (10,10,10), folder_wise = False, noise = True)
    r =r+1
    generate_np(font_path=font_pth,
                plate_path="used_gimp/VID20230225115813_9_4.png",
                chnge_lin=3, seqn=r, font_colr= (10,10,10), folder_wise = False, noise = True)
    r =r+1
    generate_np(font_path=font_pth,
               plate_path="used_gimp/IMG_20230227_103312.jpg",
               chnge_lin=15, seqn=r, font_colr= (10,10,10), folder_wise = False, noise = True)
    r =r+1
    generate_np(font_path=font_pth,
                plate_path="used_gimp/VID20230227142228_105_3.png",
                chnge_lin=15, seqn=r, font_colr= (10,10,10), folder_wise = False, noise = True)
    r =r+1
    generate_np(font_path=font_pth,
                plate_path="used_gimp/20220725_11_43_56_412_000_hxH0nNGamnPRFROZtlLa0JHSbvD3_F_3264_2448.jpg",
                chnge_lin=15, seqn=r, font_colr= (200,200,200), folder_wise = False, noise = True)
    r=r+1
    generate_np(font_path=font_pth,
                plate_path="used_gimp/20220629_17_28_46_267_000_kxrpwvclQwUQ89FI9T5qOvsth6h2_T_4000_3000.jpg",
                chnge_lin=4, seqn=r, font_colr= (200,200,200), folder_wise = False, noise = True)
    r =r+1
    generate_np(font_path=font_pth,
               plate_path="used_gimp/20220629_17_28_46_267_000_kxrpwvclQwUQ89FI9T5qOvsth6h2_T_4000_3000.jpg",
               chnge_lin=4, seqn=r, font_colr= (0,200,200), folder_wise = False, noise = False)
    r=r+1
    gt = generate_np(font_path=font_pth,
                   plate_path="used_gimp/20220725_11_43_56_412_000_hxH0nNGamnPRFROZtlLa0JHSbvD3_F_3264_2448.jpg",
                   chnge_lin=15, seqn=r, font_colr= (0,200,200), folder_wise = False, noise = False)
    r =r+1


gt.csv2txt_det('det')
gt.csv2txt_det('rec')