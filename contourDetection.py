import cv2 as cv


def getContours(edges):
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def drawContours(image, ovals):
    for ellipse in ovals:
        cv.ellipse(image, ellipse, (255, 0, 0), 2)
    return image

def getEllipticalContours(edge):
    ovals = []
    for cnt in getContours(edge):
        cnt = cv.convexHull(cnt)
        # APPROXIMATION
        epsilon = 0.001*cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        cnt = approx

        if len(cnt) >= 5 and cv.contourArea(cnt) > 6:

            ellipse = cv.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            aspect_ratio = max(MA, ma) / min(MA, ma)

            # Keep only reasonable ovals (e.g., AR between 1.2 and 3)
            if 1 <= aspect_ratio <= 2:
                # print(ellipse)
                ovals.append(ellipse)
    return ovals


def contourDetection(image, edges):
    contours = getEllipticalContours(edges)

    img = image.copy()
    return [drawContours(img, contours), len(contours)]

