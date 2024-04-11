import cv2
from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

cap = cv2.VideoCapture("VID20230318110013.mp4")
model = YOLO("weights/vehicle.pt")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True)
    print(results[0].boxes.data)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    for box, id in zip(boxes, ids):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Id {id}",
            (box[0], box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window", 1080, 720)
    cv2.imshow("Resized_Window", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break