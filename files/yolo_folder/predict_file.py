from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load a model
model = YOLO('yolov8s.pt')  # load a custom model

# Validate the model
metrics = model.val(data="dataset.yaml")  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category