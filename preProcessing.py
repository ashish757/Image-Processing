import cv2 as cv



def preProcessedEdges(img):
    imgCopy = img.copy()
    gray = cv.cvtColor(imgCopy, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5,5,), 0)


    edges = cv.Canny(blurred, 1, 100)
    kernal = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    closed = cv.dilate(edges, kernal, iterations=1)
    return closed
