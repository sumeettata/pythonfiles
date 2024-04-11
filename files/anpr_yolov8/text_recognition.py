import os
import copy
import glob

import cv2
import numpy as np

from keras.models import load_model
from paddleocr import PaddleOCR

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# #keras model load and prediction function
ocr = PaddleOCR(use_angle_cls=True, lang='en')
# model = load_model("weights/keras_model_v5.h5")

img_size = (112,32)
class_name = {0:'bad',1:'good'}

def predict_np(img): 
    data = []
    np_img = copy.deepcopy(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resized_img = cv2.resize(img, (112,32)) # Reshaping images to preferred size
    # resized_img = np.expand_dims(resized_img, axis=0)
    # pred = model.predict(np.array(resized_img))
    # pred1 = np.argmax(pred,axis = 1)
    result = ocr.ocr(np_img, cls=True)
    for idx in range(len(result)): 
        res = result[idx]
        txts = [line[1][0] for line in res]
    return "hhhj", "".join(txts)


# images_path = glob.glob("D:/OneDrive/OneDrive - Tata Insights and Quants (1)/teams_downloads/selected/*.jpg")
# for image_name in images_path:
# image = cv2.imread("D:/OneDrive/OneDrive - Tata Insights and Quants (1)/teams_downloads/selected/174px-Saudi_Arabia_-_License_Plate_-_Private.png", 0)
# print(image.shape)
# _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow("image", thresh)
# cv2.waitKey()
# if key == 27:
#     break

    # _, output = predict_np(image)
    # print(output)
    # cv2.imshow("image", cv2.putText(image, str(output), (int(10), int(100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2))
    # key = cv2.waitKey()
    # if key == 27:
    #     break
# images_path = glob.glob("D:/OneDrive/OneDrive - Tata Insights and Quants (1)/teams_downloads/selected/*.jpg") + glob.glob("D:/OneDrive/OneDrive - Tata Insights and Quants (1)/teams_downloads/selected/*.png")
# for image_name in images_path:
#     image = cv2.imread(image_name)
#     # Convert image to grayscale
#     gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#     # Use canny edge detection
#     edges = cv2.Canny(gray,127,255,apertureSize=3)

#     cv2.imshow('image', edges)
#     cv2.waitKey()