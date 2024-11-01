import cv2 as cv

img = cv.imread("E:/Works/AI/_Opencv_/Photos/tree-1.jpg")

if img is not None:
    cv.imshow("Image", img)
else:
    print("Image not found or cannot be loaded")

capture = cv.VideoCapture(1)

cv.waitKey(0)