import cv2 as cv
from utils import displayImage

image = cv.imread("sample/rxlr-4-2.jpg")

img = image.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


displayImage(gray)

# blurred = cv.GaussianBlur(gray, (5,5,), 0)

edges = cv.Canny(gray, 1, 180)

kernal = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
closed = cv.dilate(edges, kernal, iterations=1)

# addText(closed, "Dilated")
displayImage(closed)

