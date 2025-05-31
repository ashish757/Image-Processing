import cv2 as cv
import numpy as np


def houghCircleDetection(outputImage, inputImage):
    img = outputImage.copy()
    circles = cv.HoughCircles(inputImage, cv.HOUGH_GRADIENT, dp=4, minDist=20, param1=20, param2=80, minRadius=5, maxRadius=25)
    circles = np.uint16(np.around(circles))

    for circle in circles[0,:]:
        cv.circle(img, (circle[0], circle[1]), circle[2], (255, 0,0), 2)

    return [img, len(circles[0])]

