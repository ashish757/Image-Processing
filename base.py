import cv2 as cv
from houghCircleDetection import houghCircleDetection
from contourDetection import contourDetection
from utils import displayImage, addText

from preProcessing import preProcessedEdges
import os

def load_images_from_folder(folder):
    images = []
    i = 0
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            edges = preProcessedEdges(img)
            # displayImage(edges)

            hough, count = houghCircleDetection(img, edges)
            addText(hough, f"Hough Detection: {count} cells")
            displayImage(hough)

            contour, count = contourDetection(img, edges)
            addText(contour, f'Contour Detection : {count} cells')
            displayImage(contour)

            # cv.imwrite(os.path.join("output", f"[Hough-{i}] {filename}"), hough)
            # cv.imwrite(os.path.join("output", f"[Contour-{i}] {filename}"), contour)

            i += 1


            images.append(img)
    return images

images = load_images_from_folder("sample")

# i = 0
# for image in images:
#     edges = preProcessedEdges(image)
#     displayImage(edges)
#
#     hough, count = houghCircleDetection(image, edges)
#     addText(hough, f"Hough Detection: {count} cells")
#     displayImage(hough)
#
#
#     contour, count = contourDetection(image, edges)
#     addText(contour, f'Contour Detection : {count} cells')
#     displayImage(contour)
#     i+=1

