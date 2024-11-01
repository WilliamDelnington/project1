from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

res = model("E:/Works/AI/images/data/train/tree/0JMQUBAGC20X.jpg", show=True)
cv2.waitKey(0)