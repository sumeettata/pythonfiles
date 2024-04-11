import os
import sys
import copy
import glob

import cv2
import numpy as np
import pandas as pd

from text_recognition import predict_np
from ultralytics import YOLO


class anpr_inference:
    def __init__(self, model_path="weights/vehicle.pt", model_confidence=0.5, iou_threshold=0.6, boxes=True, classes_to_predict=None):
        self.boxes = boxes
        self.model_path = model_path
        self.iou_threshold = iou_threshold
        self.model_confidence = model_confidence
        self.classes_to_predict = classes_to_predict
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            self.names = self.model.names
            self.colors_list = self.create_colors(self.names)
        else:
            print("Model not found")
            sys.exit()
    
    def create_colors(self, names):
        return [tuple(np.random.randint(low=100, high=256, size=3, dtype=np.dtype(int))) for i in range(len(names))]
        

    @staticmethod
    def draw_rectangle(image, class_name, x1, y1, x2, y2, color):
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=tuple(color), thickness=2)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x1)+(int(x2) - int(x1)), int(y1) - 20), color=tuple(color), thickness=-1)
        cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return image

    
    def run(self, frame, filename):
        output = []
        number_plate_output = []
        original_image = copy.deepcopy(frame)
        
        results = self.model.predict(frame, conf=self.model_confidence, iou=self.iou_threshold, boxes=self.boxes)
        pred = results[0].boxes.data
        
        for i, det in enumerate(pred):

            if len(det) and self.model.names[int(det[5].numpy())] in self.classes_to_predict:
                original_image = self.draw_rectangle(original_image, self.model.names[int(det[5].numpy())], int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), self.colors_list[int(det[5].numpy())])
                output.append([filename, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), round(float(det[4].numpy()), 2), self.model.names[int(det[5].numpy())], int(original_image.shape[0]),int(original_image.shape[1])])
           

            if self.model.names[int(det[5].numpy())] == "np":
                np_image = original_image[int(det[1].numpy()): int(det[3].numpy()), int(det[0].numpy()):int(det[2].numpy())]
                np_image_cls, output_str = predict_np(np_image)
                original_image = self.draw_rectangle(original_image, self.model.names[int(det[5].numpy())]+"_"+output_str, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), self.colors_list[int(det[5].numpy())])
                number_plate_output.append([filename, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), round(float(det[4].numpy()), 2), self.model.names[int(det[5].numpy())], output_str,int(original_image.shape[0]),int(original_image.shape[1])])
        return output, number_plate_output, original_image




base_path = "C:/Users/SumeetMitra/Downloads/"
video ="VIDEO_13042023100250_1681360370290.mp4"
filename = os.path.basename(os.path.join(base_path, video))
cap = cv2.VideoCapture(os.path.join(base_path, video))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# out = cv2.VideoWriter("13_apr_2023/"+filename[:-4]+"_v8_"+"output.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

anpr_inference = anpr_inference(classes_to_predict=["auto", "bus", "car", "motorcycle", "hmv", "truck", "tractor"])

# while cap.isOpened():
#     ret, frame = cap.read()
#     anpr_output, number_plate_output, output_image = anpr_inference.run(frame, None)
#     print(f"anpr_output:{anpr_output}, number_plate_output:{number_plate_output}")
#     cv2.imshow("image", cv2.resize(output_image, (640, 640)))
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
# cap.release()
# # out.release()
# cv2.destroyAllWindows()


images_path = glob.glob(r"D:\OneDrive\OneDrive - Tata Insights and Quants\Work\TATA Communications\Python files\anpr_yolov8\test_dataset\*")
images_path = [x for x in images_path if not x.endswith('.xml')]

lst = []
lst_np = []
for image_name in images_path:
    frame = cv2.imread(image_name)
    filename = os.path.basename(image_name)
    anpr_output, number_plate_output, output_image = anpr_inference.run(frame, filename)
    lst.extend(anpr_output)
    lst_np.extend(number_plate_output)
    cv2.imshow("image", cv2.resize(output_image, (640, 640)))
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()

import pandas as pd

df = pd.DataFrame(lst,columns=['filename', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class_name', 'image_width', 'image_height'])
df.to_csv('testing_ANPR.csv')
df1 = pd.DataFrame(lst_np,columns=['filename', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class_name', 'number_plate','image_width', 'image_height'])
df1.to_csv('testing_np.csv')

