import os
import copy
import glob
import math
import logging


import cv2
import numpy as np

from ultralytics import YOLO
# from text_recognition import predict_np

image_metre = 60
image_fps = 30

logging.basicConfig(filename="testing_vehicle.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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
        logger.info([y_pt,x_pt])   
        slope = abs(y_pt/x_pt)
        abs_angle = math.atan(slope)*(180/math.pi)
        if ((y_pt < 0) and (x_pt > 0)):
            dir_pt = 'MUR'
        elif ((y_pt > 0) and (x_pt < 0)):
            dir_pt = 'MDL'
            logger.info("here at MDL")
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


cap = cv2.VideoCapture("30_up_down.mp4")
filename = os.path.basename("30_up_down.mp4") 

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(filename[:-4]+"_v8_"+"output_vs.avi", fourcc, 20, (int(cap.get(3)), int(cap.get(4))))


model = YOLO("weights/vehicle_04_05_2023.pt")
class_name = model.names
print(class_name)
dic_cen = {}
new_dic = {}
i =0
colors_list =create_colors(class_name)

while True:
    ret, frame = cap.read()
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    image = copy.deepcopy(frame)
    line_one = (840, 430), (1390, 470)
    line_one_ = [840, 430, 1390, 470]
    line_two = (1090, 365), (1375, 377)
    line_two_ = [1090, 365, 1375, 377]
    cv2.line(frame, line_one[0], line_one[1], color=(0,255, 0), thickness=2)
    cv2.line(frame, line_two[0], line_two[1], color=(0,255, 0), thickness=2)
    lst = []
    if not ret:
        break
    frame = frame[200:600, 800:1500]
    results = model.track(frame, persist=True, conf=0.4, iou=0.4, verbose=False)
    pred = results[0].boxes.data
    ids = results[0].boxes.id
    if ids is not None:
        boxes = results[0].boxes.xyxy.numpy().astype(int)
        clss = results[0].boxes.cls.numpy().astype(int)
        ids = results[0].boxes.id.numpy().astype(int)
        logger.info("######################################################################################################################################################change frame")
        for box, id, cl  in zip(boxes, ids, clss):
            frame = draw_rectangle_(frame, int(box[0]), int(box[1]), int(box[2]), int(box[3]), colors_list[cl])
            centroid = [int((box[2] + box[0])/2),int((box[3] + box[1])/2)]
            logger.info(centroid)
            i+=1
            # dir_cen, ang_cen = angle_cen([box[2], box[3]], [box[0], box[1]])
            if id in dic_cen.keys():
                dic_cen[id].append(centroid)
                logger.info(dic_cen)
                if len(dic_cen[id]) < 5 :
                    dir_cen, ang_cen = angle_cen(dic_cen[id][0],dic_cen[id][-1])
                else:
                    dir_cen, ang_cen = angle_cen(dic_cen[id][-4],dic_cen[id][-1])
                logger.info(dir_cen)
                txt = str(model.names[cl].upper())+" "+dir_cen 
                frame = draw_rectangle(frame, txt, int(box[0]), int(box[1]), int(box[2]), int(box[3]), colors_list[cl])
                #frame = cv2.line(frame, dic_cen[id][0], dic_cen[id][-1], (0, 255, 0), 2)
            else:
                dic_cen[id] = [centroid]
    
    else:
        for i, det in enumerate(pred.cpu()):
            x1, y1, x2, y2 = int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy())
            frame = draw_rectangle(frame, model.names[int(det[5].numpy())],  int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), colors_list[int(det[5].numpy())])
    out.write(frame)
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window", 1080, 720)
    cv2.imshow("Resized_Window", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
out.release()

cv2.destroyAllWindows()
'''
# if :
                # np_image = image[int(box[1]): int(box[3]), int(box[0]):int(box[2])]
                # np_image_cls, output_str = predict_np(np_image)
                # class_name ="NP" +"_"+ output_str
                # frame = draw_rectangle(frame, class_name, int(box[0]), int(box[1]), int(box[2]), int(box[3]), colors_list[cl])
'''