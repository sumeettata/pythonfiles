import os
import glob
import copy

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
from ultralytics.yolo.utils import ops
from datetime import datetime


if not os.path.exists("D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/cctv_frontdesk"):
    os.mkdir("D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/cctv_frontdesk")
    
def create_colors(names):
    return [tuple(np.random.randint(low=80, high=256, size=3, dtype=np.dtype(int))) for i in range(len(names))]


def draw_rectangle(image, class_name, x1, y1, x2, y2, color):
    color = (int(color[0]), int(color[1]), int(color[2]))
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=tuple(color), thickness=2)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x1)+(int(x2) - int(x1)), int(y1) - 20), color=tuple(color), thickness=-1)
    cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image
# model = YOLO("yolov8s.yaml")

#model = YOLO(r"D:/OneDrive\OneDrive - Tata Insights and Quants\Work\TATA Communications\GCP\garbage\train08\weights\best.pt")
# model = YOLO(r"D:/OneDrive\OneDrive - Tata Insights and Quants\Work\TATA Communications\GCP\garbage\train9\weights\best.pt")
#model = YOLO(r"D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/GCP/garbage/Garbagefall_retrain/train5/weights/best.pt")
model = YOLO('D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage_new/Garbage_weights/WEIGHTS/Garbagefall_30072023.pt')
#model = YOLO("D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/GCP/garbage_newgpu/garbage_31k_11092023.pt")

colors_list = create_colors(model.names)    

#C:/Users/SumeetMitra/Downloads/throw garbage/throw garbage/
#D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage/Roboflow/garbage_detection.v16i.yolov8 (1)/test/images

#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/GCP/garbage/garbage_all_medium/test1_50_80kmph_RC_0002_230525092541.avi")

images_path = glob.glob("D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/cctv_frontdesk_24072023/*")
#print(images_path)
output = []

username = 'admin'

password = 'Gupdate@4427'

ip_address = '182.48.248.71'

port_number = '554'

rtsp_url = f'rtsp://{username}:{password}@{ip_address}:{port_number}/live.sdp/'

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'


try:

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

except:

    print(rtsp_url + ' not opening ')
    
# result = cv2.VideoWriter('video_garbage_test3_2.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, (int(640),int(640)))

#cap = cv2.VideoCapture('D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage_new/Garbage_testing/videos/cut_testing_garbage.mp4')

i=0
while True:
    i=i+1
    if i%30 == 0:
        ret, image = cap.read()
        if image is not None:
            image1 = image.copy()
            image1 = cv2.resize(image1, (0, 0), fx = 0.5, fy = 0.5)
            #image = cv2.imread('D:/OneDrive/OneDrive - Tata Insights and Quants/Pictures/Screenshots/Screenshot 2023-07-27 121154.png')
    # for image_name in images_path:

    #     image = cv2.imread(image_name)
            #image_height, image_width, _ = image.shape
            filename = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            print(filename)
            #results = model.predict(image, conf=0.1, iou=0.1, boxes=True)
            #pred = results[0].boxes.data
            #for i, det in enumerate(pred.cpu()):
                #image = draw_rectangle(image, model.names[int(det[5].numpy())], int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), colors_list[int(det[5].numpy())])
            #if len(pred):
            #cv2.imwrite("D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/cctv_frontdesk_new/"+str(filename)+'.png',image)
            cv2.imwrite("D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/cctv_frontdesk_original/"+str(filename)+'.png',image1)
                #output.append({"filename":filename, "width":image_width, "height":image_height, "class_name":model.names[int(det[5].numpy())], "xmin":int(det[0].numpy()),"ymin":int(det[1].numpy()),"xmax":int(det[2].numpy()),"ymax":int(det[3].numpy())})
            cv2.imshow("output_image", cv2.resize(image, (640, 640)))
            # result.write(cv2.resize(image, (720, 1080))) 
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            #cv2.imwrite("D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage_new/Results/Garbage_noclub/"+str(filename),image)
            #ppe_output = pd.DataFrame(output)
            #ppe_output.to_csv("phone.csv", columns=["filename", "height", "width", "class_name", "xmin", "ymin", "xmax", "ymax"])

cap.release()
# result.release() 