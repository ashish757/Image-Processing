import cv2 as cv
import numpy as np
from numpy.ma.testutils import approx

image = cv.imread("sample/rxlr-4-2.jpg")

def displayImage(image, screen_res = (1280, 720)):
    (h, w) = image.shape[:2]
    scale_width = screen_res[0] / w
    scale_height = screen_res[1] / h
    scale = min(scale_width, scale_height)
    window_width = int(w * scale)
    window_height = int(h * scale)

    resized_img = cv.resize(image, (window_width, window_height))

    cv.imshow('Image', resized_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def preProcessedEdges(img):
    imgCopy = img.copy()
    gray = cv.cvtColor(imgCopy, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5,5,), 0)


    edges = cv.Canny(blurred, 1, 180)
    kernal = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    closed = cv.dilate(edges, kernal, iterations=2)
    return closed

def generateImages(image, rng=10):
    for i in range(5):
        img = cv.Canny(image, threshold1=10, threshold2=100)
        location = "preProcessing/image" + str(i) + ".jpg"
        cv.imwrite(location, img)

def getContours(edges):
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def drawContours(image, ovals):
    img = image.copy()
    for ellipse in ovals:
        cv.ellipse(img, ellipse, (255, 0, 0), 2)
    return img

def getCircularContours(edge):
    ovals = []
    for cnt in getContours(edge):
        cnt = cv.convexHull(cnt)
        # APPROXIMATION
        epsilon = 0.01*cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        cnt = approx

        if (len(cnt) >= 5 and cv.contourArea(cnt) > 10):

            ellipse = cv.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            aspect_ratio = max(MA, ma) / min(MA, ma)

            # Keep only reasonable ovals (e.g., AR between 1.2 and 3)
            if 0.5 <= aspect_ratio <= 2:
                print(ellipse)
                ovals.append(ellipse)
    return ovals

def houghCircle(output, image):

    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, dp=3, minDist=30, param1=10, param2=80, minRadius=3, maxRadius=30)
    circles= np.uint16(np.around(circles))
    for circle in circles[0,:]:
        cv.circle(output, (circle[0], circle[1]), circle[2], (255, 0,0), 2)


edges = preProcessedEdges(image)
displayImage(edges)
houghCircle(image, edges)
displayImage(image)

# ovals = getCircularContours(edges)
# print(contours)
# displayImage(drawContours(image, ovals))

# circularity = 4*pi*(area/sq(preimeter))

# displayImage(image)

# generateImages(grey)

