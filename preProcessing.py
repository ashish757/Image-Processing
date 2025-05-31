import cv2 as cv



def preProcessedEdges(img):
    imgCopy = img.copy()
    imgCopy = cv.cvtColor(imgCopy, cv.COLOR_BGR2GRAY)
    imgCopy = cv.GaussianBlur(imgCopy, (5,5), 0)


    ##### Thresholding
    # imgCopy = cv.adaptiveThreshold(imgCopy, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 10)


    ####### Canny edge
    # imgCopy = cv.Canny(imgCopy, 10, 90)
    imgCopy = cv.Canny(imgCopy, 10, 80, 3, L2gradient=True)
    kernal = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    # imgCopy = cv.dilate(imgCopy, kernal, iterations=1)
    return imgCopy
