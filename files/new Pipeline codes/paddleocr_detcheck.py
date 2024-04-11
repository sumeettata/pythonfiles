import numpy as np
import cv2
from paddleocr import PaddleOCR
import re
import os

img_loc = 'Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/Train_data'
#path = 'Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/Train_data/train_list_det.txt'
path2 = 'Work/TATA Communications/GCP/paddle_ocr/checkpoints/checkpoints/det_db/predicts_db.txt'
path = 'Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/predicts_db.txt'

#os.chdir(os.path.basename(path))

#os.environ['KMP_DUPLICATE_LIB_OK']='True'
#ocr = PaddleOCR(use_angle_cls=True,lang='en')

dic = {}
with open(path2,'rb') as g:
    lines2 = g.readlines()
    for line in lines2:
        line = line.decode('utf-8')
        line = line.strip("\n").split("\t")
        img_path = os.path.basename(line[0])
        label = line[1]
        dic[img_path] = label
        

with open(path,'rb') as f:
    lines = f.readlines()
    for line in lines:
        line = line.decode('utf-8')
        line = line.strip("\n").split("\t")
        img_path = line[0]
        label = line[1]
        
        img = cv2.imread(img_loc+'/'+img_path)
        img_2 = img.copy()
        for idx1 in eval(label):
            idxpt = np.int0(np.array(idx1['points'],np.int32))
            img_2 = cv2.polylines(img_2, [idxpt],True, (255, 0, 0) , 2)
        
        for idx1 in eval(dic[img_path]):
            idxpt = np.int0(np.array(idx1['points'],np.int32))
            img_2 = cv2.polylines(img_2, [idxpt],True, (0, 255, 0) , 2)
        
        
        # result = ocr.ocr(img, cls=True,rec=False)
        # print(result)
        # for idx in range(len(result)):
        #     print(np.array(idx))
        #     idx = np.int0(np.array(idx,np.int32))
        #     #img_2 = cv2.polylines(img_2, [idx],True, (0, 0, 255) , 2)
        
        #cv2.imwrite('Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/checkdata/'+img_path,img_2)    
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resized_Window", 500 , 500 )
        cv2.imshow("Resized_Window", img_2)
        key = cv2.waitKey(0)
        if key == 27:
            break  

cv2.destroyAllWindows()           