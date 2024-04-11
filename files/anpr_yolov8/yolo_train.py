import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="garbage.yaml", epochs=3)  # train the model
# export the model to ONNX format