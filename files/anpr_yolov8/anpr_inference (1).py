import os
import sys
import copy
import glob

import cv2
import torch
import numpy as np
import pandas as pd


from text_recognition import predict_np
from ultralytics import YOLO


class anpr_inference:
    def __init__(self, model_path="weights/vehicle_08_06_2023.pt", model_confidence=0.5, iou_threshold=0.5, boxes=True, classes_to_predict=None):
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
        return [tuple(np.random.randint(low=10, high=256, size=3, dtype=np.dtype(int))) for i in range(len(names))]
        

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
        count_of_vehicles = 0
        results = self.model.predict(frame, conf=self.model_confidence, iou=self.iou_threshold, boxes=self.boxes, verbose=False)
        # print(dir(results))
        pred = results[0].boxes.data
        # print(type(pred), )
        if pred.shape[1] > 6:
            for i, det in enumerate(pred.cpu()):
                x1, y1, x2, y2 = int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy())
                area = (x2-x1)*(y2-y1)
                
                if len(det) and self.model.names[int(det[6].numpy())] in self.classes_to_predict and int(area) > 3000 :
                    class_name = self.model.names[int(det[6].numpy())] +"_ID:"+ str(int(det[5].numpy()))
                    original_image = self.draw_rectangle(original_image, class_name, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), self.colors_list[int(det[6].numpy())])
                    output.append((filename, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), round(float(det[4].numpy()), 2), self.model.names[int(det[6].numpy())]))


                if self.model.names[int(det[6].numpy())] == "np":
                    np_image = original_image[int(det[1].numpy()): int(det[3].numpy()), int(det[0].numpy()):int(det[2].numpy())]
                    np_image_cls, output_str = predict_np(np_image)
                    class_name = self.model.names[int(det[6].numpy())] +"_ID:"+ str(int(det[5].numpy())) +"_"+ output_str
                    original_image = self.draw_rectangle(original_image, class_name, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), self.colors_list[int(det[6].numpy())])
                    number_plate_output.append((filename, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), round(float(det[4].numpy()), 2), self.model.names[int(det[6].numpy())], output_str))
            
            return output, number_plate_output, original_image
        else:
            for i, det in enumerate(pred.cpu()):
                x1, y1, x2, y2 = int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy())
                area = (x2-x1)*(y2-y1)
                if len(det) and self.model.names[int(det[5].numpy())] in self.classes_to_predict:
                    count_of_vehicles+=1
                    class_name = self.model.names[int(det[5].numpy())]
                    original_image = self.draw_rectangle(original_image, class_name, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), self.colors_list[int(det[5].numpy())])
                    output.append((filename, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), round(float(det[4].numpy()), 2), self.model.names[int(det[5].numpy())]))
            

                if self.model.names[int(det[5].numpy())] == "np":
                    np_image = original_image[int(det[1].numpy()): int(det[3].numpy()), int(det[0].numpy()):int(det[2].numpy())]
                    np_image_cls, output_str = predict_np(np_image)
                    class_name = self.model.names[int(det[5].numpy())] +"_"+ output_str
                    original_image = self.draw_rectangle(original_image, class_name, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), self.colors_list[int(det[5].numpy())])
                    number_plate_output.append((filename, int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), round(float(det[4].numpy()), 2), self.model.names[int(det[5].numpy())], output_str))
            print(f"COUNT OF VEHICLES:{count_of_vehicles}")
            original_image = cv2.putText(original_image, f"COUNT OF VEHICLES:{count_of_vehicles}", (int(100), int(60)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            return output, number_plate_output, original_image





anpr_inference = anpr_inference(classes_to_predict=["auto", "bus", "car", "motorcycle", "hmv", "truck", "tractor"])

# base_path = "D:/Work/data/dashcam_videos/"
# videos = os.listdir("D:/Work/data/dashcam_videos/")
# print(videos)
# for video in videos:
#     # video = "NVR_ch8_main_20230508165649_20230508171000.mp4"
#     filename = os.path.basename(os.path.join(base_path, video)) 
#     cap = cv2.VideoCapture(os.path.join(base_path, filename))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
#     out = cv2.VideoWriter(base_path+"/output/"+filename[:-4]+"_v8_"+"output.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
#     count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             anpr_output, number_plate_output, output_image = anpr_inference.run(frame, None)
#             print(f"anpr_output:{anpr_output}, number_plate_output:{number_plate_output}")
#             cv2.imshow("image", cv2.resize(output_image, (640, 640)))
#             out.write(output_image)
#             key = cv2.waitKey(1)
#             if key == 27:
#                 break
#         else:
#             out.release()
#             cap.release()
#             continue
#             # count+=1
#     cv2.destroyAllWindows()





filename = os.path.basename("C:/Users/SumeetMitra/Downloads/Littering on the road #4 - Sibu Stupid Drivers (1).mp4") 
cap = cv2.VideoCapture("C:/Users/SumeetMitra/Downloads/Littering on the road #4 - Sibu Stupid Drivers (1).mp4")
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# out = cv2.VideoWriter(filename[:-4]+"_v8_"+"output.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        anpr_output, number_plate_output, output_image = anpr_inference.run(frame, None)
        print(f"anpr_output:{anpr_output}, number_plate_output:{number_plate_output}")
        cv2.imshow("image", cv2.resize(output_image, (640, 640)))
        # out.write(output_image)
        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()



