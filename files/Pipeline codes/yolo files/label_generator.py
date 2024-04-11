import copy
import os.path
import uuid

from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.general import non_max_suppression, check_img_size, scale_coords

import torch
import cv2
import numpy as np
import time
import glob
import pandas as pd
import csv

half = False
augment = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf_thres = 0.1
iou_thres = 0.1
classes = None
agnostic_nms = False
max_det = 1000

# Loading the model
# model = attempt_load("./weights/combined_model_tata_com.pt", map_location="cpu")
model = attempt_load("yolov5s.pt") #
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check image size
names = model.module.names if hasattr(model, 'module') else model.names  # get class names
# print(names)
# Loading the images from dataset
# image_path = glob.glob("C:/Users/DanielVadranapu/Downloads/test_NBM (1)/test_NBM/*.jpg")
path = "D:/swami_new/*.jpg"
image_path = glob.glob(path+"*.jpg") + glob.glob(path+"*.jpeg")+glob.glob(path+"*.png")
print(len(image_path))
ppe_list = []

for path in image_path:
    # filename = path
    filename = os.path.basename(path)
    image = cv2.imread(path)
    image_height, image_width, _ = image.shape
    img0 = copy.deepcopy(image)
    img = letterbox(img0, 640, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=augment)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # print(pred) 1000
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            # print(type(det))
            # sys.exit()
            # print(det)
            for x1, y1, x2, y2, conf, cls in reversed(det):
                # print(det)
                # if int(y1) > 30:
                if str(names[int(cls.numpy())]) == "person":
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(255, 255, 0), thickness=2)
                    # cv2.rectangle(image, (int(384), int(155)), (int(420), int(182)), color=(255, 255, 0), thickness=2)
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x1) + 30, int(y1) - 10), color=(255, 255, 0),
                                  thickness=-1)
                    cv2.putText(image, str(names[int(cls.numpy())]), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0),  2)
                    ppe_list.append((filename,  image_height, image_width, str(names[int(cls.numpy())]), int(x1), int(y1), int(x2), int(y2)))
                # ppe_list.append({"filename": filename, "height": image_height, "width": image_width,
                #                  "class": str(names[int(cls.numpy())]), "xmin": int(384), "ymin": int(155),
                #                  "xmax": int(420), "ymax": int(182)})
            cv2.imshow("image", cv2.resize(image, (640, 640)))
            # cv2.imwrite("D:/Work/data/temp/"+filename, image)
            key = cv2.waitKey(1)
            if key == 27:  # escape key
                break

                # image = cv2.resize(image, (1280, 720))
                # cv2.imshow("image", cv2.resize(image, (640, 640)))
                # # cv2.imwrite("D:/Work/data/tata_com/camera_31/test_images_output/"+filename, image)
                # key = cv2.waitKey(1)
                # if key == 27:  # escape key
                #     break
# cv2.destroyAllWindows()

if len(ppe_list) > 0:
    ppe_output = pd.DataFrame(ppe_list, columns=["filename", "height", "width", "class", "xmin", "ymin", "xmax", "ymax"])
    ppe_output.to_csv("D:/test.csv", index=False)
