import os
import glob
import copy
import re

import cv2
import numpy as np
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from paddleocr import PaddleOCR

## C:\Users\SumeetMitra\.conda\envs\paddle_env\Lib\site-packages\paddleocr\tools\infer\predict_det.py
# class TextDetector(object):
#     def __init__(self, args):
#         self.args = args
#         self.det_algorithm = args.det_algorithm
#         self.use_onnx = args.use_onnx
#         pre_process_list = [{
#             'DetResizeForTest': {
#                 'image_size': [160,240],

#pip install paddleocr
#pip install paddlepaddle
#python3 -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

class number:
    def __init__(self,det_model,rec_model):
        self.ocr = PaddleOCR(use_angle_cls=False,lang='en',show_log=False,det_model_dir=det_model,rec_model_dir=rec_model)

    def detect_numberplate(self,image):
        result = self.ocr.ocr(image, cls=False)
        if len(result[0]):
            for idx in range(len(result)):
                res = result[idx]
                txts = [line[1][0] for line in res if len(line[1][0]) > 2]
                np_pred = "".join(txts)
                np_pred = re.sub(r'[^\w]', '',np_pred)
                if len(np_pred) > 10:
                    np_pred = np_pred.replace('IND','')
                if len(np_pred) > 10:
                    np_pred = np_pred.replace('IN','')
                if len(np_pred) > 10:
                    np_pred = np_pred.replace('I','')
                if len(np_pred) > 10:
                    np_pred = np_pred[1:]            
        else:
            np_pred = '' 
            
        return np_pred     
    
    
a=number()

image = cv2.imread("MicrosoftTeams-image (10).png")
print(a.detect_numberplate(image))


