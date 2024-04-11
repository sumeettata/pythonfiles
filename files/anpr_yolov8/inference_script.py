import os
import glob
import copy
import re

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
from ultralytics.yolo.utils import ops
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=False,lang='en',show_log=False)

def create_colors(names):
    return [tuple(np.random.randint(low=80, high=256, size=3, dtype=np.dtype(int))) for i in range(len(names))]


def draw_rectangle(image, class_name, x1, y1, x2, y2, color):
    color = (int(color[0]), int(color[1]), int(color[2]))
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=tuple(color), thickness=2)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x1)+(int(x2) - int(x1)), int(y1) - 20), color=tuple(color), thickness=-1)
    cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    return image

    
def detect_numberplate(image):
    
    img_back = np.ones((image.shape[0],image.shape[1],3), np.uint8)*255
    
    result = ocr.ocr(image, cls=False)
    if len(result[0]):
        for idx in range(len(result)):
            res = result[idx]
            
            for line in res:
                txts = line[1][0]
                box = line[0]
                box = np.int0(box)
                
                box = cv2.boundingRect(box)
                
                img = img_back.copy()[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
                
                textSize, baseline = cv2.getTextSize(txts, cv2.FONT_HERSHEY_SIMPLEX , 1, 2)
                img2_txt = cv2.putText(np.ones((textSize[1]+baseline,textSize[0]+baseline,3), np.uint8)*255,str(txts) ,(int(baseline/2),int(textSize[1]+baseline/2)), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,0,0), 2, cv2.LINE_AA)
        
                img_back[box[1]:box[1]+box[3],box[0]:box[0]+box[2]] = cv2.resize(img2_txt,(img.shape[1],img.shape[0]))
                
                
    
    # else:
    #     result_rec = ocr.ocr(image, det=False, cls=False)
        
    #     for idx in range(len(result_rec)):
    #         res = result_rec[idx]
    #         for line in res:
    #             txts = str(line[0])
                
    #             if len(txts) > 3:
    #                 img = img_back.copy()
                    
    #                 textSize, baseline = cv2.getTextSize(txts, cv2.FONT_HERSHEY_SIMPLEX , 1, 2)
    #                 img2_txt = cv2.putText(np.ones((textSize[1]+baseline,textSize[0]+baseline,3), np.uint8)*255,str(txts) ,(int(baseline/2),int(textSize[1]+baseline/2)), cv2.FONT_HERSHEY_SIMPLEX, 1 , (255,0,0), 2, cv2.LINE_AA)
    #                 img_back = cv2.resize(img2_txt,(img.shape[1],img.shape[0]))
                
                
            
    return img_back
    
            
# def detect_numberplate(image):
#     result = ocr.ocr(image, cls=False)
#     print(result)
#     if len(result[0]):
#         for idx in range(len(result)):
#             res = result[idx]
#             txts = [line[1][0] for line in res if len(line[1][0]) > 2]
#             np_pred = re.sub(r'[^\w]', '',"".join(txts))
#             if len(np_pred) > 10:
#                 np_pred = np_pred.replace('IND','')
#     else:
#         np_pred = 'np' 
        
#     return np_pred       



model = YOLO(r"weights/vehicle_08_06_2023.pt")
colors_list = create_colors(model.names)    

path = "D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/16-03-2023_ANPR/videos/videos_mov/VID20230318110013"

if not os.path.exists(path+'_predict'):
    os.mkdir(path+'_predict')
    
if not os.path.exists(path+'_crop'):
    os.mkdir(path+'_crop')
      

images_path = glob.glob(path+"/*.jpg")
output = []
for image_name in images_path:
    print(len(images_path))
    filename = os.path.basename(image_name)
    image = cv2.imread(image_name)
    image_height, image_width, _ = image.shape
    results = model.predict(image, conf=0.4, iou=0.4, boxes=True)
    pred = results[0].boxes.data
    for i, det in enumerate(pred.cpu()):
        try:
            if str(model.names[int(det[5].numpy())]) == 'np':
                image_crop = image.copy()[int(det[1].numpy()):int(det[3].numpy()),int(det[0].numpy()):int(det[2].numpy())]
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                image_crop[:,:,0] = clahe.apply(image_crop[:,:,0])
                image_crop[:,:,1] = clahe.apply(image_crop[:,:,1])
                image_crop[:,:,2] = clahe.apply(image_crop[:,:,2])
                image_back = detect_numberplate(image_crop)
                
                #final = cv2.vconcat([image_crop,image_back])
                #cv2.imwrite(path+'_crop/'+str(i)+filename,final)
                
                image = draw_rectangle(image, 'np', int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), colors_list[int(det[5].numpy())])
                image[int(det[1].numpy())-image_back.shape[0]:int(det[1].numpy()),int(det[0].numpy()):int(det[0].numpy())+image_back.shape[1]] = image_back
        
            else:
                image = draw_rectangle(image, model.names[int(det[5].numpy())], int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), colors_list[int(det[5].numpy())])
        
        except:
            print('pass')
        
        #image = draw_rectangle(image, model.names[int(det[5].numpy())], int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), colors_list[int(det[5].numpy())])
        output.append({"filename":filename, "width":image_width, "height":image_height, "class_name":model.names[int(det[5].numpy())], "xmin":int(det[0].numpy()),"ymin":int(det[1].numpy()),"xmax":int(det[2].numpy()),"ymax":int(det[3].numpy())})
    cv2.imshow("output_image", cv2.resize(image, (1080, 720)))
    cv2.waitKey(1)
    #cv2.imwrite(path+'_predict/'+filename,image)
ppe_output = pd.DataFrame(output)
#ppe_output.to_csv("phone.csv", columns=["filename", "height", "width", "class_name", "xmin", "ymin", "xmax", "ymax"])