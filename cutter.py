import cv2
import numpy as np
import imutils
from PIL import ImageEnhance, Image


def cutImage(stitched_img):
    # make border to the image to find the requested rectangle
    stitched_img = cv2.copyMakeBorder(stitched_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, (0, 0, 0))
    # convert to gray scale
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    # try to find where is the "black pixel" using binary 0 and 1
    # to get a clean image without the black space
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # find the contours on the binary image to find the circle corners and substract those later
    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find the maximum area of the image
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)
    # find mask with the same shape with the required image
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    minRectangle = mask.copy()
    sub = mask.copy()
    # subtract the minimum rectangle from the image
    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)
    # now find the minimum rectangle in the contours
    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    # take the requested image of contour
    x, y, w, h = cv2.boundingRect(areaOI)
    # give the image without the black space
    cut_stitched_img = stitched_img[y:y + h, x:x + w]

    return cut_stitched_img[:, :, ::-1]

"""
def cutImage(stitched_img):
    # make border to the image to find the requested rectangle
    stitched_img = cv2.copyMakeBorder(stitched_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, (0, 0, 0))
    # convert to gray scale
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    # try to find where is the "black pixel" using binary 0 and 1
    # to get a clean image without the black space
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    plt.figure()
    plt.imshow(thresh_img)
    plt.axis('off')
    plt.show()
    # find the contours on the binary image to find the circle corners and substract those later
    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find the maximum area of the image
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)
    # find mask with the same shape with the required image
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    minRectangle = mask.copy()
    sub = mask.copy()
    # subtract the minimum rectangle from the image
    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)
    # now find the minimum rectangle in the contours
    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    plt.figure()
    plt.imshow(minRectangle)
    plt.axis('off')
    plt.show()
    # take the requested image of contour
    x, y, w, h = cv2.boundingRect(areaOI)
    # give the image without the black space
    stitched_img = stitched_img[y:y + h, x:x + w]

    cv2.imwrite("stitchedOutputProcessed.png", stitched_img)
    # make the image sharper
    Im = Image.open('stitchedOutputProcessed.png')
    enhancer = ImageEnhance.Sharpness(Im)
    enhanced = enhancer.enhance(5.0)
    enhanced.save('enhanced.png')
    plt.figure()
    plt.imshow(stitched_img[:, :, ::-1])
    plt.axis('off')
    plt.show()
    plt.imshow(enhanced)

    plt.axis('off')
    plt.show()


"""
