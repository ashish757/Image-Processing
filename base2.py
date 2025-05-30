import cv2 as cv
import numpy as np

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
    edges = cv.Canny(blurred, 10, 100)
    kernal = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    closed = cv.dilate(edges, kernal, iterations=1)
    return closed

def generateImages(image, rng=10):
    for i in range(5):
        img = cv.Canny(image, threshold1=10, threshold2=100)
        location = "preProcessing/image" + str(i) + ".jpg"
        cv.imwrite(location, img)

def getContours(edges):
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def drawContours(image, detections):
    img = image.copy()
    cv.drawContours(img, detections[0], -1, (255, 0, 0), 2)
    # for ellipse in detections[1]:
    #     cv.ellipse(img, ellipse, (255, 0, 0), 2)

    # cv.imwrite("detections.jpg", image)
    return img

def getCircularContours(edge):
    circularContours = []
    ovals = []
    for cnt in getContours(edge):
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt , True)
        if area == 0:
            print(area, perimeter)
        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area )/(perimeter * perimeter)
        # if len(cnt) >= 5:
        # ellipse = cv.fitEllipse(cnt)
        # (x, y), (MA, ma), angle = ellipse
        # aspect_ratio = max(MA, ma) / min(MA, ma)
        #
        # # Keep only reasonable ovals (e.g., AR between 1.2 and 3)
        # if 0.5 <= aspect_ratio <= 2:
        #     print(ellipse)
        #     ovals.append(ellipse)

        # print(circularity)
        if 0.5 <= circularity <= 1:
            circularContours.append(cnt)

    # for cnt in circularContours:
    #     ellipse = cv.fitEllipse(cnt)
    return [circularContours, ovals]



edges = preProcessedEdges(image)
displayImage(edges)
[contours, ovals] = getCircularContours(edges)
# print(contours)
displayImage(drawContours(image, [contours, ovals]))

# circularity = 4*pi*(area/sq(preimeter))

# displayImage(image)

# generateImages(grey)

