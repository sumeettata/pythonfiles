import os
import glob
import copy

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
from ultralytics.yolo.utils import ops


def create_colors(names):
    return [tuple(np.random.randint(low=80, high=256, size=3, dtype=np.dtype(int))) for i in range(len(names))]


def draw_rectangle(image, class_name, x1, y1, x2, y2, color):
    color = (int(color[0]), int(color[1]), int(color[2]))
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=tuple(color), thickness=2)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x1)+(int(x2) - int(x1)), int(y1) - 20), color=tuple(color), thickness=-1)
    cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image
# model = YOLO("yolov8s.yaml")
model = YOLO("tata_steel_ppe_05_05_2023.pt")
colors_list = create_colors(model.names)    

images_path = glob.glob("D:/OneDrive/OneDrive - Tata Insights and Quants (1)/teams_downloads/phone/*.jpg")
output = []
for image_name in images_path:
    filename = os.path.basename(image_name)
    image = cv2.imread(image_name)
    image_height, image_width, _ = image.shape
    results = model.predict(image, conf=0.4, iou=0.4, boxes=True)
    pred = results[0].boxes.data
    for i, det in enumerate(pred.cpu()):
        image = draw_rectangle(image, model.names[int(det[5].numpy())], int(det[0].numpy()), int(det[1].numpy()), int(det[2].numpy()), int(det[3].numpy()), colors_list[int(det[5].numpy())])
        output.append({"filename":filename, "width":image_width, "height":image_height, "class_name":model.names[int(det[5].numpy())], "xmin":int(det[0].numpy()),"ymin":int(det[1].numpy()),"xmax":int(det[2].numpy()),"ymax":int(det[3].numpy())})
    cv2.imshow("output_image", cv2.resize(image, (640, 640)))
    cv2.waitKey(1)
ppe_output = pd.DataFrame(output)
ppe_output.to_csv("phone.csv", columns=["filename", "height", "width", "class_name", "xmin", "ymin", "xmax", "ymax"])