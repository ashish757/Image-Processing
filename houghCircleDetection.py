import cv2 as cv
import numpy as np
from utils import displayImage

image = cv.imread("sample/rxlr-4-2.jpg")



circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, dp=3, minDist=30, param1=10, param2=80, minRadius=3, maxRadius=30)
circles = np.uint16(np.around(circles))
for circle in circles[0,:]:
    cv.circle(image, (circle[0], circle[1]), circle[2], (255, 0,0), 2)


displayImage(image)


