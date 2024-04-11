
import cv2
from paddleocr import PaddleOCR
import re
import os


path = 'Work/TATA Communications/Data_1/ANPR Stage2/good_images/final/Number_plates_synthetic_2/train_rec.txt'

#os.chdir(os.path.basename(path))

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ocr = PaddleOCR(use_angle_cls=True,lang='en')

l=0
k=0
with open(path,'rb') as f:
    lines = f.readlines()
    for line in lines:
        line = line.decode('utf-8')
        line = line.strip("\n").split("\t")
        img_path = line[0]
        label = line[1]
        img = cv2.imread(os.path.dirname(path)+'/'+img_path)
        np_ch = label
        
        result = ocr.ocr(img, cls=True,det=False)
        for idx in range(len(result)):
            res = result[idx]
            txts = [line[0] for line in res]
            
        np_pred = "".join(txts)
        #np_pred = re.sub(r'[^\w]', '', np_pred)
        #if len(np_pred) > 10:
            #np_pred = np_pred.strip('IND')
            
        if (np_ch == np_pred) and (np_ch == 1):
            k = k+1
            
        l = l+1
        print(np_ch,np_pred)
        print(k/l)
    
    