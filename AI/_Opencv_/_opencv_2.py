import cv2

img = cv2.imread("E:/Works/AI/images/data/kagglecatsanddogs_3367a/PetImages/Cat/2.jpg", 0)
_, segmented_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

print(len(segmented_img[0]))