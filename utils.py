import cv2 as cv

def saveImage(img, location):
    cv.imwrite(location, img)

def displayImage(img, title="Image", screen_res = (1280, 720)):
    (h, w) = img.shape[:2]
    scale_width = screen_res[0] / w
    scale_height = screen_res[1] / h
    scale = min(scale_width, scale_height)
    window_width = int(w * scale)
    window_height = int(h * scale)

    resized_img = cv.resize(img, (window_width, window_height))

    cv.imshow(title, resized_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def addText(img, text, pos=(5, 100), font=cv.FONT_HERSHEY_SIMPLEX, scale=3, color=(255, 0, 0), thickness=5):
    cv.putText(img, text, pos, font, scale, color, thickness)

