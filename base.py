import cv2 as cv
from houghCircleDetection import houghCircleDetection
from contourDetection import contourDetection
from utils import displayImage, addText

from preProcessing import preProcessedEdges


image = cv.imread("sample/rxlr-4-2.jpg")


edges = preProcessedEdges(image)
# displayImage(edges)

hough, count = houghCircleDetection(image, edges)
addText(hough, f"Hough Detection: {count} cells")
displayImage(hough)

# contour, count = contourDetection(image, edges)
# addText(contour, f'Contour Detection : {count} cells')
# displayImage(contour)
