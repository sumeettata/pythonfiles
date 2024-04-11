import cv2
import pytesseract
import os
import glob
import numpy as np

path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/ANPR stage 2/downloaded plates/archive (8)/Indian_Number_Plates/Sample_Images'

pytesseract.pytesseract.tesseract_cmd = "C:/Users/SumeetMitra/AppData/Local/Programs/Tesseract-OCR/tesseract"
#img = cv2.imread("C:/Users/SumeetMitra/Downloads/License-Plate/LicensePlate.png")

for file in glob.glob(path+"/*"):
    result = ocr.ocr(file,rec=False)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)
            x,y,w,h = cv2.boundingRect(np.array(line))
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            
    cv2.imshow('show',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_size(img1):
    
    boxes = pytesseract.image_to_boxes(img1)
    h1,w1,_= img1.shape
    for b in boxes.splitlines():
        b = b.split(' ')
        x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
        cv2.rectangle(img1,(x,h1-y),(w,h1-h),(0,0,255),3)
    return img1

#img6 = get_size(img)    
#cv2.imshow('show',img6)
#cv2.waitKey(0)
#cv2.destroyAllWindows()