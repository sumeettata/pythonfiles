import os
import sys
import copy
import glob
import math



import cv2
import numpy as np

from ultralytics import YOLO
# from text_recognition import predict_np

image_metre = 60
image_fps = 30

dict =  {}
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def angle_cen(init_pt,fin_pt):
    x_pt = fin_pt[0]-init_pt[0]
    y_pt = fin_pt[1]-init_pt[1]
    dir_pt = "unknown"
    abs_angle = 0
    if x_pt == 0 :
        if (y_pt < 0) :
            dir_pt = 'MSU'
        elif (y_pt > 0):
            dir_pt = 'MSD'
        abs_angle = 90
    
    elif y_pt == 0 :  
        if (x_pt > 0):
            dir_pt = 'MSR'
        elif (x_pt < 0):
            dir_pt = 'MSL'
        abs_angle = 0
        
    else:    
        slope = abs(y_pt/x_pt)
        abs_angle = math.atan(slope)*(180/math.pi)
        if ((y_pt < 0) and (x_pt > 0)):
            dir_pt = 'MUR'
        elif ((y_pt > 0) and (x_pt < 0)):
            dir_pt = 'MDL'
        elif ((y_pt < 0) and (x_pt < 0)):
            dir_pt = 'MUL'
        elif ((y_pt > 0) and (x_pt > 0)):
            dir_pt = 'MDR'
        else:
            dir_pt = 'Unknown'
        
    return dir_pt,abs_angle


def create_colors(names):
    return [tuple(np.random.randint(low=10, high=256, size=3, dtype=np.dtype(int))) for i in range(len(names))]
    

def calculate_speed(self, linea_point, lineb_point, center_x, center_y, direction):
    
    if direction in ["MUL", "MUR", "MSU"]:
        if int(linea_point[1]) <= center_y:
            print("entered_line 1")
        elif center_y >= int(lineb_point[1]):
            print("crossed line 2")
    
    if direction in ["MDL", "MDR", "MSD"]:
        if center_y >= int(lineb_point[1]):
            print("entered line 2")
        elif center_y >= int(linea_point[1]):
            print("crossed line 1")   
 
    else:
        print("vehicle not entered any lines")
    return None


def draw_rectangle(image, class_name, x1, y1, x2, y2, color):
    color = (int(color[0]), int(color[1]), int(color[2]))
    
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=tuple(color), thickness=2)
    # cv2.rectangle(image, (int(x1), int(y1)), (int(x1)+(int(x2) - int(x1)), int(y1) - 10), color=tuple(color), thickness=-1)
    cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return image

def draw_rectangle_(image, x1, y1, x2, y2, color):
    color = (int(color[0]), int(color[1]), int(color[2]))
    
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=tuple(color), thickness=2)
    # cv2.rectangle(image, (int(x1), int(y1)), (int(x1)+(int(x2) - int(x1)), int(y1) - 30), color=tuple(color), thickness=-1)
    # cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    return image

def check_up_cross(line1_y1, line2_y1, y2, id):
    count = 0
    if y2 >= line1_y1 :
        if id in dict.keys():
            dict[id] = dict[id]+1 
        else:
            dict[id] = count+1       
        if y2 > line2_y1:
            print(f"ID:{id} and Count:{dict[id]}")

    

# def check_down_cross(line_one_[1], line_two_[1], int(box[3])):
#     if 
#     return 

def check_up_down(line1_y1, line2_y1, y2):
    d1 = abs(line1_y1 - y2)
    d2 = abs(line2_y1 - y2)
    
    k = min(d1, d2)
    if k == d1:
        return "up"
    elif k == d2:
        return "down"

cap = cv2.VideoCapture("D:/Work/data/dashcam_videos/30_up_down.mp4")
filename = os.path.basename("D:/Work/data/dashcam_videos/30_up_down.mp4") 

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter("D:/Work/data/dashcam_videos/"+filename[:-4]+"_v8_"+"output_vs.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))


model = YOLO("weights/vehicle_04_05_2023.pt")
class_name = model.names
print(class_name)
dic_cen = {}
new_dic = {}
i =0
colors_list =create_colors(class_name)
unique_ids = []
unique_ids_direction = []
while True:
    ret, frame = cap.read()
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    image = copy.deepcopy(frame)
    line_one = (290, 165), (560, 180)
    line_one_ = [290, 165, 560, 180]
    line_two = (45, 235), (600, 280)
    line_two_ = [45, 235, 600, 280]
    lst = []
    if not ret:
        break
    frame = frame[200:600, 800:1500]
    cv2.line(frame, line_one[0], line_one[1], color=(0,255, 0), thickness=2)
    cv2.line(frame, line_two[0], line_two[1], color=(0,255, 0), thickness=2)
    
    results = model.track(frame, persist=True, conf=0.4, iou=0.4, verbose=False, classes=[0, 1, 2, 3, 4, 6, 7])
    # print(results[0].boxes.keys)
    # sys.exit()
    pred = results[0].boxes.data
    ids = results[0].boxes.id
    if ids is not None:
        boxes = results[0].boxes.xyxy.numpy().astype(int)
        clss = results[0].boxes.cls.numpy().astype(int)
        # print(clss)
        ids = results[0].boxes.id.numpy().astype(int)
        for box, id, cl  in zip(boxes, ids, clss):
            if unique_ids is not None and id in unique_ids:
                k = 0
            elif model.names[cl] in ["car", "auto", "bus", "truck", "motorcycle", "tractor"]:
                unique_ids.append(id)
                unique_ids_direction.append([id, check_up_down(line_one_[1], line_two_[1], int(box[3]))])
                
            # print(unique_ids_direction)
            frame = draw_rectangle_(frame, int(box[0]), int(box[1]), int(box[2]), int(box[3]), colors_list[cl])
            if unique_ids_direction:
                for j in unique_ids_direction:
                    if j[0] in ids and j[1] == "up":
                        check_up_cross(line_one_[1], line_two_[1], int(box[3]), id)
    
    else:
        for i, det in enumerate(pred.cpu()):
            x1, y1, x2, y2 = int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy())
            frame = draw_rectangle(frame, model.names[int(det[5].numpy())],  int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), colors_list[int(det[5].numpy())])
    out.write(frame)
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window", 640, 640)
    cv2.imshow("Resized_Window", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
out.release()



'''
# if :
                # np_image = image[int(box[1]): int(box[3]), int(box[0]):int(box[2])]
                # np_image_cls, output_str = predict_np(np_image)
                # class_name ="NP" +"_"+ output_str
                # frame = draw_rectangle(frame, class_name, int(box[0]), int(box[1]), int(box[2]), int(box[3]), colors_list[cl])
'''